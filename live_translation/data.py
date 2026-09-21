"""MuST-C manifests and immutable, causally encoded speech batches.

No corpus is downloaded here. MuST-C must be obtained under its own access terms.
Manifests preserve the official segment order and split names.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class EncodedBatch:
    """Aligned audio chunks and translations; B is batch size and T is chunk count."""

    states: Tensor                  # [B, T, features]; one fixed encoding per chunk
    lengths: Tensor                 # [B]; observed chunks before padding
    target_tokens: Tensor           # [B, tokens]; translation includes end token
    target_lengths: Tensor          # [B]; translation length including end token
    chunk_end_ms: Tensor            # [B, T]; time each audio chunk becomes available
    durations_ms: Tensor            # [B]; source audio duration
    references: list[str]
    ids: list[str]
    encoder_lengths: Tensor | None = None  # Encoder states before padding the final chunk.
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.states.ndim != 3:
            raise ValueError("states must have shape [batch, chunks, dimension]")
        b, t, _ = self.states.shape
        if self.lengths.shape != (b,) or not bool(((self.lengths > 0) & (self.lengths <= t)).all()):
            raise ValueError("Every example needs 1..T observed chunks")
        if self.chunk_end_ms.shape != (b, t) or self.durations_ms.shape != (b,):
            raise ValueError("Invalid audio timing shape")
        if self.target_tokens.ndim != 2 or self.target_tokens.shape[0] != b:
            raise ValueError("Invalid target shape")
        if self.target_lengths.shape != (b,) or not bool(((self.target_lengths > 0) & (self.target_lengths <= self.target_tokens.shape[1])).all()):
            raise ValueError("Invalid target lengths")
        if len(self.ids) != b or len(self.references) != b:
            raise ValueError("Ids/references must align with batch")
        for i, length in enumerate(self.lengths.tolist()):
            ends = self.chunk_end_ms[i, :length]
            if not bool(torch.isfinite(ends).all()) or bool((ends <= 0).any()) or bool((ends[1:] <= ends[:-1]).any()):
                raise ValueError("Chunk availability times must increase strictly")
            if float(ends[-1]) > float(self.durations_ms[i]) + 1e-3:
                raise ValueError("Chunk available after source duration")

    @property
    def mask(self) -> Tensor:
        return torch.arange(self.states.shape[1], device=self.states.device)[None] < self.lengths[:, None]

    def to(self, device: str | torch.device) -> "EncodedBatch":
        return EncodedBatch(**{f.name: (getattr(self, f.name).to(device) if isinstance(getattr(self, f.name), Tensor) else getattr(self, f.name)) for f in fields(self)})

    def subset(self, indices: Sequence[int] | Tensor) -> "EncodedBatch":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.states.device)
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            values[f.name] = value.index_select(0, index.to(value.device)) if isinstance(value, Tensor) else ([value[i] for i in index.tolist()] if f.name in {"ids", "references"} else value)
        return EncodedBatch(**values)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    ids = set()
    for record in records:
        for key in ("id", "audio", "target_text"):
            if key not in record:
                raise ValueError(f"Manifest record missing {key}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate utterance id: {record['id']}")
        ids.add(record["id"])
        audio = Path(record["audio"])
        record["audio"] = str(audio if audio.is_absolute() else path.parent / audio)
        if not Path(record["audio"]).is_file():
            raise FileNotFoundError(record["audio"])
        if float(record.get("offset", 0)) < 0 or float(record.get("duration", 1)) <= 0:
            raise ValueError("Audio offset must be nonnegative and duration positive")
    return records


def prepare_mustc(root: str | Path, output: str | Path, split: str, language: str = "de", limit: int | None = None) -> Path:
    """Write a JSONL manifest for a supplied MuST-C en-LANG directory.

    Official YAML segment rows and parallel text lines are aligned by position;
    do not sort talks or collapse repeated audio filenames.
    """
    import yaml

    root, output = Path(root).resolve(), Path(output).resolve()
    if (root / f"en-{language}").is_dir():
        root = root / f"en-{language}"
    base = root / "data" / split
    segments = yaml.safe_load((base / "txt" / f"{split}.yaml").read_text())
    source = (base / "txt" / f"{split}.en").read_text().splitlines()
    target = (base / "txt" / f"{split}.{language}").read_text().splitlines()
    if len(segments) != len(source) or len(source) != len(target):
        raise ValueError("MuST-C YAML/English/translation row counts differ")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    records = []
    for i, (segment, src, tgt) in enumerate(zip(segments, source, target)):
        if limit is not None and i >= limit:
            break
        audio = base / "wav" / segment["wav"]
        if not audio.is_file():
            raise FileNotFoundError(audio)
        records.append(dict(id=f"{split}:{i}:{Path(segment['wav']).stem}", audio=str(audio), offset=float(segment["offset"]), duration=float(segment["duration"]), source_text=src, target_text=tgt, split=split, corpus="MuST-C", language=f"en-{language}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    return output


def read_audio(record: dict[str, Any]) -> tuple[Tensor, int]:
    import soundfile as sf

    with sf.SoundFile(record["audio"]) as stream:
        rate = stream.samplerate
        start = round(float(record.get("offset", 0)) * rate)
        count = round(float(record["duration"]) * rate) if "duration" in record else len(stream) - start
        if start < 0 or count <= 0 or start + count > len(stream) + 1:
            raise ValueError(f"Invalid audio segment bounds: {record['id']}")
        stream.seek(start)
        waveform = stream.read(count, dtype="float32", always_2d=True).mean(axis=1)
    if rate != 16000:
        raise ValueError(f"Expected MuST-C 16000 Hz, got {rate}; resample explicitly before preparing data")
    return torch.from_numpy(waveform), rate


def export_simuleval(manifest: str | Path, output: str | Path) -> Path:
    """Materialize official segments as WAV plus aligned source/target lists."""
    import soundfile as sf

    records, output = read_manifest(manifest), Path(output).resolve()
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for i, record in enumerate(records):
        waveform, rate = read_audio(record)
        destination = audio_dir / f"{i:06d}.wav"
        sf.write(destination, waveform.numpy(), rate, subtype="FLOAT")
        sources.append(str(destination))
    (output / "source.txt").write_text("\n".join(sources) + "\n")
    (output / "target.txt").write_text("\n".join(r["target_text"] for r in records) + "\n")
    (output / "ids.json").write_text(json.dumps([r["id"] for r in records], ensure_ascii=False, indent=2) + "\n")
    return output


def make_toy_batch(n: int = 16, state_dim: int = 8, seed: int = 0, min_chunks: int = 4, max_chunks: int = 8, vocab_size: int = 11) -> EncodedBatch:
    """Make synthetic vectors and toy German labels for a quick code check."""
    if min_chunks <= 0 or max_chunks < min_chunks or state_dim < 2 or vocab_size < 5:
        raise ValueError("Invalid synthetic batch dimensions")
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(min_chunks, max_chunks + 1, (n,), generator=gen)
    states = torch.randn(n, max_chunks, state_dim, generator=gen)
    times = torch.arange(1, max_chunks + 1).float()[None].expand(n, -1).clone() * 280
    targets = torch.zeros(n, max_chunks + 1, dtype=torch.long)
    words = ToyVocabulary.words(vocab_size)
    references = []
    for i, length in enumerate(lengths.tolist()):
        labels = states[i, :length, :min(state_dim - 1, vocab_size - 3)].argmax(-1) + 3
        states[i, :length, -1] = 0
        states[i, length - 1, -1] = 1
        states[i, length:] = 0
        times[i, length:] = times[i, length - 1]
        targets[i, :length] = labels
        targets[i, length] = 2
        references.append(" ".join(words[j] for j in labels.tolist()))
    return EncodedBatch(states, lengths, targets, lengths + 1, times, lengths.float() * 280, references, [f"synthetic:{seed}:{i}" for i in range(n)], metadata={"corpus": "synthetic_gaussian_audit", "not_a_translation_benchmark": True, "seed": seed})


class ToyVocabulary:
    @staticmethod
    def words(size: int = 11) -> list[str]:
        base = ["<pad>", "<bos>", "<eos>", "ich", "wir", "das", "ist", "ein", "gut", "heute", "hier"]
        return base[:size] + [f"wort{i}" for i in range(len(base), size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mustc-root", required=True,
                        help="Extracted MuST-C root, or its en-LANG subdirectory")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--split", choices=["train", "dev", "tst-COMMON", "tst-HE"],
                        required=True, help="Official MuST-C split")
    parser.add_argument("--language", default="de", help="Target language code (default: de)")
    parser.add_argument("--limit", type=int, help="Write only the first N segments")
    parser.add_argument("--simuleval-output",
                        help="Also export segment WAVs and aligned source/target lists")
    args = parser.parse_args()
    try:
        manifest = prepare_mustc(args.mustc_root, args.output, args.split,
                                 args.language, args.limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(manifest)
    if args.simuleval_output:
        print(export_simuleval(manifest, args.simuleval_output))


if __name__ == "__main__":
    main()
