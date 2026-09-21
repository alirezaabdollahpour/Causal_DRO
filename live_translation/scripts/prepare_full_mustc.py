#!/usr/bin/env python3
"""Prepare the full English–Arabic MuST-C splits for the benchmark.

All training rows can supply continuation examples for the causal attack.
The test manifest combines both official test splits and keeps each row's
original split, position, and utterance ID.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile


EXPECTED_COUNTS = {"train": 212085, "dev": 1073, "tst-COMMON": 2019, "tst-HE": 578}
SPLITS = tuple(EXPECTED_COUNTS)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    temporary.chmod(0o660)
    os.replace(temporary, path)


def _symlink(source: Path, destination: Path) -> None:
    relative = os.path.relpath(source, destination.parent)
    if destination.is_symlink() and os.readlink(destination) == relative:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace existing manifest: {destination}")
    destination.symlink_to(relative)


def prepare(source_dir: Path, output_dir: Path, *, validate_audio: bool = True) -> dict:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    official_audit = json.loads((source_dir / "manifest_audit.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, talks, manifest_hashes = {}, {}, {}
    audio_paths: set[Path] = set()
    latest_end: dict[Path, tuple[float, str]] = {}
    test_rows: list[dict] = []
    combined = output_dir / "benchmark_test.jsonl"
    with tempfile.NamedTemporaryFile("w", dir=output_dir, prefix=".benchmark_test.",
                                     suffix=".tmp", encoding="utf-8", delete=False) as writer:
        temporary = Path(writer.name)
        for split in SPLITS:
            path = source_dir / f"{split}.jsonl"
            actual_hash = digest(path)
            expected_hash = official_audit["splits"][split]["manifest_sha256"]
            if actual_hash != expected_hash:
                raise ValueError(f"Official {split} manifest differs from its upstream audit")
            for relative, expected in official_audit["splits"][split].get("input_sha256", {}).items():
                upstream = source_dir.parent / relative
                if digest(upstream) != expected:
                    raise ValueError(f"Official {split} upstream text/alignment changed: {relative}")
            manifest_hashes[split] = actual_hash
            seen, split_talks = set(), set()
            with path.open(encoding="utf-8") as source:
                for row_number, line in enumerate(source):
                    row = json.loads(line)
                    if (row.get("official_row") != row_number or row.get("split") != split
                            or row.get("language") != "en-ar"
                            or row.get("id") != f"{split}:{row_number}:{row.get('talk_id')}"
                            or not row.get("source_text", "").strip()
                            or not row.get("target_text", "").strip()):
                        raise ValueError(f"Invalid official {split} row {row_number}")
                    if row["id"] in seen:
                        raise ValueError(f"Duplicate official utterance ID: {row['id']}")
                    seen.add(row["id"])
                    split_talks.add(row["talk_id"])
                    audio = Path(row["audio"])
                    audio_paths.add(audio)
                    end = float(row["offset"]) + float(row["duration"])
                    if end > latest_end.get(audio, (-1.0, ""))[0]:
                        latest_end[audio] = (end, row["id"])
                    if split.startswith("tst-"):
                        test_rows.append(row)
            count = len(seen)
            if count != EXPECTED_COUNTS[split] or count != official_audit["splits"][split]["segments"]:
                raise ValueError(f"Incomplete official {split} split: {count} rows")
            counts[split], talks[split] = count, split_talks
        talk_counts = Counter(row["talk_id"] for row in test_rows)
        talk_positions = Counter()
        for row in test_rows:
            row["segment_index"] = talk_positions[row["talk_id"]]
            row["expected_talk_segments"] = talk_counts[row["talk_id"]]
            talk_positions[row["talk_id"]] += 1
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if talks[left] & talks[right]:
                raise ValueError(f"Official talks overlap across {left} and {right}")
    missing = [str(path) for path in audio_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} official audio files are missing; first: {missing[0]}")
    audio_audit: dict = {"files": len(audio_paths), "headers_validated": validate_audio}
    if validate_audio:
        import soundfile as sf

        rates, channels, frames = Counter(), Counter(), {}
        for path in sorted(audio_paths):
            info = sf.info(path)
            rates[info.samplerate] += 1
            channels[info.channels] += 1
            frames[str(path)] = info.frames
            end_seconds, last_id = latest_end[path]
            if round(end_seconds * info.samplerate) > info.frames + 1:
                raise ValueError(f"Official segment exceeds audio bounds: {last_id}")
        if set(rates) != {16000}:
            raise ValueError(f"Unexpected source audio sample rate(s): {dict(rates)}")
        # Audio headers let us check segment bounds without decoding every clip.
        audio_audit.update(sample_rates=dict(rates), channels=dict(channels),
                           total_audio_frames=sum(frames.values()))
    temporary.chmod(0o660)
    os.replace(temporary, combined)
    aliases = {"train": "train", "bank": "train", "dev": "dev"}
    for name, split in aliases.items():
        _symlink(source_dir / f"{split}.jsonl", output_dir / f"benchmark_{name}.jsonl")
    result = {
        "schema_version": 1,
        "dataset": official_audit["dataset"],
        "dataset_version": official_audit["version"],
        "population": "every official row; no duration, talk, text, or score filtering",
        "counts": {**counts, "test_combined": counts["tst-COMMON"] + counts["tst-HE"],
                   "bank": counts["train"]},
        "talks": {split: len(value) for split, value in talks.items()},
        "official_manifest_sha256": manifest_hashes,
        "combined_test_sha256": digest(combined),
        "test_order": ["tst-COMMON", "tst-HE"],
        "bank_is_complete_train": True,
        "audio": audio_audit,
    }
    _atomic_json(output_dir / "full_manifest_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("live_translation/data/kaggle_mustc_en_ar/manifests"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("live_translation/data/kaggle_mustc_en_ar/full_manifests"))
    parser.add_argument("--skip-audio-headers", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output_dir,
                             validate_audio=not args.skip_audio_headers), indent=2))


if __name__ == "__main__":
    main()
