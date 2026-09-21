#!/usr/bin/env python3
"""Create English–Arabic MuST-C manifests from Kaggle's file layout.

The main manifests keep every aligned row in each official split. Optional
samples use a fixed seed, segment duration, and talk identity; they do not
select rows using model results or translation quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import yaml

SPLITS = ("train", "dev", "tst-COMMON", "tst-HE")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, records):
    part = path.with_suffix(".jsonl.part")
    part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    part.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_for_split(root, split, known_files):
    base = root / split
    src = base / "txt" / ("train_en.txt" if split == "train" else f"{split}.en")
    tgt = base / "txt" / ("train_ar.txt" if split == "train" else f"{split}.ar")
    alignment = base / "txt" / f"{split}.yaml"
    source, target = src.read_text(encoding="utf-8-sig").splitlines(), tgt.read_text(encoding="utf-8-sig").splitlines()
    segments = yaml.load(alignment.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not len(source) == len(target) == len(segments):
        raise ValueError(f"Unaligned row counts in {split}: {len(source)}, {len(target)}, {len(segments)}")
    records, blank_source, blank_target = [], 0, 0
    for index, (segment, english, arabic) in enumerate(zip(segments, source, target)):
        relative = f"{split}/wav/{segment['wav']}"
        if relative not in known_files:
            raise ValueError(f"YAML references audio absent from upstream inventory: {relative}")
        if float(segment["offset"]) < 0 or float(segment["duration"]) <= 0:
            raise ValueError(f"Invalid segment timing: {split}:{index}")
        blank_source += not bool(english.strip())
        blank_target += not bool(arabic.strip())
        records.append({
            "id": f"{split}:{index}:{Path(segment['wav']).stem}",
            "audio": str(base / "wav" / segment["wav"]),
            "offset": float(segment["offset"]), "duration": float(segment["duration"]),
            "source_text": english, "target_text": arabic,
            "talk_id": Path(segment["wav"]).stem,
            "speaker_id": str(segment.get("speaker_id", "")),
            "official_row": index, "split": split, "corpus": "MuST-C",
            "language": "en-ar", "dataset": "sebaeymohamed/must-c-en-ar", "dataset_version": 1,
        })
    audit = {"segments": len(records), "talks": len({r["talk_id"] for r in records}), "duration_seconds": sum(r["duration"] for r in records), "blank_sources": blank_source, "blank_targets": blank_target, "input_sha256": {str(p.relative_to(root)): sha256(p) for p in (src, tgt, alignment)}}
    return records, audit


def sampled(records, seed, minimum, maximum, limit=None, per_talk=None):
    grouped = defaultdict(list)
    for record in records:
        if minimum <= record["duration"] <= maximum:
            grouped[record["talk_id"]].append(record)
    rng = random.Random(seed)
    talks = sorted(grouped)
    rng.shuffle(talks)
    for talk in talks:
        rng.shuffle(grouped[talk])
    selected = []
    depth = 0
    while True:
        available = False
        for talk in talks:
            if depth < len(grouped[talk]) and (per_talk is None or depth < per_talk):
                selected.append(grouped[talk][depth])
                available = True
                if limit is not None and len(selected) >= limit:
                    return selected
        if not available:
            return selected
        depth += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="live_translation/data/kaggle_mustc_en_ar")
    parser.add_argument("--output", default="live_translation/data/kaggle_mustc_en_ar/manifests")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--minimum-duration", type=float, default=1.0)
    parser.add_argument("--maximum-duration", type=float, default=12.0)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--dev-per-talk", type=int, default=8)
    parser.add_argument("--test-per-talk", type=int, default=8)
    parser.add_argument("--validate-audio", action="store_true", help="Require all waveforms and audit every segment against 16 kHz audio headers")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((root / "kaggle_files.json").read_text())
    known_files = {r["name"]: r for r in inventory}
    audits, split_records = {}, {}
    for split in SPLITS:
        records, audit = records_for_split(root, split, known_files)
        write_jsonl(output / f"{split}.jsonl", records)
        audit["manifest_sha256"] = sha256(output / f"{split}.jsonl")
        split_records[split], audits[split] = records, audit
    talk_sets = {split: {r["talk_id"] for r in records} for split, records in split_records.items()}
    overlap = {f"{left}|{right}": sorted(talk_sets[left] & talk_sets[right]) for i, left in enumerate(SPLITS) for right in SPLITS[i+1:]}
    if any(overlap.values()):
        raise ValueError(f"Official splits share talks: {overlap}")
    selections = {}
    for split in ("train", "dev", "tst-COMMON"):
        selected = sampled(split_records[split], args.seed, args.minimum_duration, args.maximum_duration, limit=args.train_limit if split == "train" else None, per_talk=(args.dev_per_talk if split == "dev" else args.test_per_talk) if split != "train" else None)
        sample_path = output / f"{split}.sample.jsonl"
        write_jsonl(sample_path, selected)
        selections[split] = {"segments": len(selected), "talks": len({r["talk_id"] for r in selected}), "duration_seconds": sum(r["duration"] for r in selected), "sha256": sha256(sample_path), "min_duration": min(r["duration"] for r in selected), "max_duration": max(r["duration"] for r in selected)}
    # Save a few available clips for quick backend checks during the download.
    sanity = [r for r in split_records["dev"] if Path(r["audio"]).is_file() and 1 <= r["duration"] <= 8][:3]
    if sanity:
        write_jsonl(output / "dev.backend_sanity.jsonl", sanity)
    audio_audit = {"performed": False}
    if args.validate_audio:
        import soundfile as sf
        headers = {}
        invalid = []
        for split, records in split_records.items():
            for record in records:
                path = record["audio"]
                if path not in headers:
                    info = sf.info(path)
                    headers[path] = (info.samplerate, info.frames, info.channels)
                rate, frames, channels = headers[path]
                if rate != 16000 or round(record["offset"] * rate) + round(record["duration"] * rate) > frames + 1:
                    invalid.append({"id": record["id"], "sample_rate": rate, "frames": frames, "channels": channels})
        audio_audit = {"performed": True, "audio_files": len(headers), "sample_rates": sorted({x[0] for x in headers.values()}), "channel_counts": dict(Counter(x[2] for x in headers.values())), "invalid_segments": invalid}
        if invalid:
            write_json(output / "invalid_audio_segments.json", invalid)
    audit = {"dataset": "sebaeymohamed/must-c-en-ar", "version": 1, "splits": audits, "talk_overlap": overlap, "sample_policy": {"seed": args.seed, "minimum_duration": args.minimum_duration, "maximum_duration": args.maximum_duration, "train_limit": args.train_limit, "dev_per_talk": args.dev_per_talk, "test_per_talk": args.test_per_talk, "method": "Seeded uniform shuffle within each talk, seeded talk order, round-robin selection; duration only; official split preserved; no score or label filtering"}, "samples": selections, "audio_audit": audio_audit}
    write_json(output / "manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
