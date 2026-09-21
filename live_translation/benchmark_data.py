"""Resumable, provenance-checked CPU caches for committed speech encodings.

Each artifact contains one utterance and plain tensor/dict data, so it can be
read with ``torch.load(weights_only=True)``. No corpus-sized tensor is ever
placed on the GPU. The manifest identity, source file identities, and backend
provenance must match before an existing cache is resumed.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Sequence
import uuid

import torch
from torch.utils.data import Dataset

from .data import EncodedBatch, read_manifest


SCHEMA_VERSION = 1


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backend_provenance_matches(cached: dict, current: dict) -> bool:
    """Reuse cached encodings only with the same model and encoder settings."""
    return cached == current


def _atomic_json(path: Path, value: dict):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical(value) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_payload(batch: EncodedBatch) -> dict[str, Any]:
    payload = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            if field.name == "states":
                value = value.float()
        payload[field.name] = value
    return payload


def _read_item(path: Path, expected_id: str | None = None) -> EncodedBatch:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(EncodedBatch)}:
        raise ValueError(f"Invalid encoded-cache schema: {path}")
    batch = EncodedBatch(**payload)
    if len(batch.ids) != 1 or (expected_id is not None and batch.ids != [expected_id]):
        raise ValueError(f"Cached utterance identity mismatch: {path}")
    if batch.states.dtype != torch.float32 or not bool(torch.isfinite(batch.states).all()):
        raise ValueError(f"Cached states must be finite float32: {path}")
    return batch


def _identity(backend, manifest: str | Path | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).resolve()
        records = read_manifest(path)
        manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path = str(path)
    else:
        records = [dict(row) for row in manifest]
        if not records:
            raise ValueError("Cannot cache an empty manifest")
        if len({row["id"] for row in records}) != len(records):
            raise ValueError("Duplicate manifest utterance IDs")
        for row in records:
            row["audio"] = str(Path(row["audio"]).resolve())
        manifest_sha = hashlib.sha256(_canonical(records)).hexdigest()
        manifest_path = None
    audio_identity = {}
    for row in records:
        path = Path(row["audio"])
        stat = path.stat()
        # Audio byte hashes belong in download provenance. Size+mtime here
        # detects local replacement without rereading multi-GB talk archives.
        audio_identity[str(path)] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "schema_version": SCHEMA_VERSION, "manifest_sha256": manifest_sha,
        "manifest_path": manifest_path, "records": records,
        "backend_provenance": dict(backend.provenance),
        "source_file_identity": audio_identity,
        "state_dim": int(backend.state_dim), "pad_id": int(backend.pad_id),
    }


def cache_manifest(backend, manifest: str | Path | list[dict[str, Any]],
                   cache_dir: str | Path) -> "EncodedDataset":
    """Encode one source at a time; resume valid items after interruption.

    Reusing ``cache_dir`` with another manifest or backend is an error, rather
    than silent cache reuse. Missing/corrupt individual items are regenerated.
    The return value is a lazy CPU ``EncodedDataset``. ``backend.encode`` must
    implement the committed-prefix encoder and return one aligned utterance.
    """
    import fcntl

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity(backend, manifest)
    fingerprint = hashlib.sha256(_canonical(identity)).hexdigest()
    index_path = cache_dir / "index.json"
    progress_path = cache_dir / "progress.jsonl"
    with (cache_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        new_index = not index_path.exists()
        if not new_index:
            index = json.loads(index_path.read_text())
            if index.get("fingerprint") != fingerprint or index.get("identity") != identity:
                raise ValueError("Encoded cache provenance differs; use a new cache_dir for this manifest/backend")
        else:
            index = {"fingerprint": fingerprint, "identity": identity, "complete": False,
                     "items": [{"file": f"utterance_{i:07d}.pt", "id": row["id"], "sha256": None}
                               for i, row in enumerate(identity["records"])]}
            # An orphan journal without its index has no trusted identity.
            # Existing item files are still validated and rediscovered below.
            progress_path.unlink(missing_ok=True)
        if len(index.get("items", ())) != len(identity["records"]):
            raise ValueError("Encoded cache index/manifest lengths differ")
        # Each item is committed atomically.  A small append-only progress log
        # records its checksum so interruption never forces a re-encode.  The
        # former per-item rewrite of index.json was quadratic I/O on the full
        # 212,085-row training manifest (the index itself exceeds 100 MB).
        if progress_path.exists():
            with progress_path.open("r+", encoding="utf-8") as progress:
                header = progress.readline()
                try:
                    if json.loads(header) != {"fingerprint": fingerprint}:
                        raise ValueError("Encoded cache progress provenance differs")
                except json.JSONDecodeError as error:
                    raise ValueError("Invalid encoded cache progress header") from error
                line_number = 1
                while True:
                    offset = progress.tell()
                    line = progress.readline()
                    if not line:
                        break
                    line_number += 1
                    try:
                        if not line.endswith("\n"):
                            raise json.JSONDecodeError("Torn final append", line, len(line))
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final append can be rediscovered from its
                        # already-committed item file during the scan below.
                        if not progress.read(1):
                            progress.seek(offset)
                            progress.truncate()
                            break
                        raise ValueError(f"Invalid encoded cache progress line {line_number}")
                    position, digest = event.get("position"), event.get("sha256")
                    if (not isinstance(position, int) or position < 0 or position >= len(index["items"])
                            or not isinstance(digest, str) or len(digest) != 64):
                        raise ValueError(f"Invalid encoded cache progress line {line_number}")
                    index["items"][position]["sha256"] = digest
        else:
            temporary = progress_path.with_name(f".{progress_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps({"fingerprint": fingerprint}) + "\n")
                os.replace(temporary, progress_path)
            finally:
                temporary.unlink(missing_ok=True)
        index["complete"] = False
        _atomic_json(index_path, index)
        with progress_path.open("a", encoding="utf-8", buffering=1) as progress:
          for position, (entry, record) in enumerate(zip(index["items"], identity["records"])):
            destination = cache_dir / entry["file"]
            reusable = False
            if destination.exists():
                try:
                    actual_sha = _sha256_file(destination)
                    if entry["sha256"] is not None and entry["sha256"] != actual_sha:
                        raise ValueError("Cached artifact checksum mismatch")
                    cached = _read_item(destination, record["id"])
                    if cached.references != [record["target_text"]] or cached.states.shape[-1] != identity["state_dim"]:
                        raise ValueError("Cached reference or dimension mismatch")
                    if (cached.metadata or {}).get("cache_fingerprint") != fingerprint:
                        raise ValueError("Cached artifact belongs to another manifest/backend")
                    reusable = True
                except (ValueError, RuntimeError, OSError, EOFError, KeyError, IndexError, TypeError, pickle.UnpicklingError):
                    reusable = False
            if not reusable:
                batch = backend.encode([record]).to("cpu")
                if batch.ids != [record["id"]] or batch.references != [record["target_text"]]:
                    raise ValueError("Backend returned misaligned source IDs/references")
                if batch.states.shape[-1] != identity["state_dim"]:
                    raise ValueError("Backend state dimension differs from cache provenance")
                batch = collate_batches([batch], identity["pad_id"])
                batch.metadata = {**(batch.metadata or {}), "cache_fingerprint": fingerprint,
                                  "records": [dict(record)], "original_records": [dict(record)]}
                temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                try:
                    torch.save(_batch_payload(batch), temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                actual_sha = _sha256_file(destination)
            if entry["sha256"] != actual_sha:
                entry["sha256"] = actual_sha
                progress.write(json.dumps({"position": position, "sha256": actual_sha}) + "\n")
        index["complete"] = True
        _atomic_json(index_path, index)
        progress_path.unlink(missing_ok=True)
    return EncodedDataset(cache_dir)


prepare_cache = cache_manifest


class EncodedDataset(Dataset):
    """Lazy per-utterance CPU tensors plus immutable original manifest rows."""

    def __init__(self, cache_dir: str | Path, *, verify_checksums: bool = True):
        self.cache_dir = Path(cache_dir).resolve()
        self.index = json.loads((self.cache_dir / "index.json").read_text())
        identity = self.index.get("identity", {})
        if identity.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported encoded cache schema")
        if hashlib.sha256(_canonical(identity)).hexdigest() != self.index.get("fingerprint"):
            raise ValueError("Encoded cache index provenance checksum failed")
        if not self.index.get("complete"):
            raise ValueError("Encoded cache is incomplete; resume cache_manifest before opening")
        self.records = identity["records"]
        self.provenance = identity["backend_provenance"]
        self.fingerprint = self.index["fingerprint"]
        self.pad_id = identity["pad_id"]
        self.state_dim = identity["state_dim"]
        self.verify_checksums = verify_checksums
        if len(self.index["items"]) != len(self.records):
            raise ValueError("Cache index/manifest lengths differ")
        for entry, record in zip(self.index["items"], self.records):
            if entry["id"] != record["id"] or not entry["sha256"] or not (self.cache_dir / entry["file"]).is_file():
                raise ValueError("Cache has a missing or misaligned item; resume cache_manifest")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> EncodedBatch:
        entry = self.index["items"][index]
        path = self.cache_dir / entry["file"]
        if self.verify_checksums and _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Encoded cache item checksum failed: {path}")
        batch = _read_item(path, entry["id"])
        if (batch.metadata or {}).get("cache_fingerprint") != self.fingerprint:
            raise ValueError("Cached artifact provenance differs from its index")
        return batch


def load_cache(cache_dir: str | Path) -> EncodedDataset:
    return EncodedDataset(cache_dir)


def collate_batches(items: Sequence[EncodedBatch], pad_id: int) -> EncodedBatch:
    """Trim per-utterance padding, then collate CPU items into one CPU batch."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    dimensions = {item.states.shape[-1] for item in items}
    if len(dimensions) != 1:
        raise ValueError("Cannot collate different source state dimensions")
    encoder_presence = {item.encoder_lengths is not None for item in items}
    if len(encoder_presence) != 1:
        raise ValueError("Cannot mix known and unknown encoder lengths")
    states, targets, times, durations, lengths, target_lengths, encoder_lengths = [], [], [], [], [], [], []
    references, ids, records = [], [], []
    for item in items:
        if item.states.device.type != "cpu":
            raise ValueError("collate_batches accepts CPU cache items; move the collated batch to GPU afterward")
        original = (item.metadata or {}).get("records", (item.metadata or {}).get("original_records"))
        if original is not None and len(original) != len(item.ids):
            raise ValueError("Original manifest rows must align with cached batch")
        for i, (length, target_length) in enumerate(zip(item.lengths.tolist(), item.target_lengths.tolist())):
            states.append(item.states[i, :length].detach().float())
            targets.append(item.target_tokens[i, :target_length])
            times.append(item.chunk_end_ms[i, :length])
            durations.append(item.durations_ms[i])
            lengths.append(length); target_lengths.append(target_length)
            if item.encoder_lengths is not None:
                encoder_lengths.append(item.encoder_lengths[i])
            references.append(item.references[i]); ids.append(item.ids[i])
            if original is not None:
                records.append(dict(original[i]))
    padded_times = nn_pad(times)
    for i, length in enumerate(lengths):
        padded_times[i, length:] = times[i][-1]
    metadata = dict(items[0].metadata or {})
    # Different train/dev cache fingerprints can coexist only in deliberately
    # constructed donor batches; preserve all identities without pretending the
    # collated batch came from the first manifest alone.
    fingerprints = sorted({(item.metadata or {}).get("cache_fingerprint", "") for item in items} - {""})
    if len(fingerprints) > 1:
        metadata.pop("cache_fingerprint", None)
        metadata["cache_fingerprints"] = fingerprints
    if records:
        if len(records) != len(ids):
            raise ValueError("Cannot mix cached original rows with untracked examples")
        metadata["records"] = records
        metadata["original_records"] = records
    return EncodedBatch(nn_pad(states), torch.tensor(lengths), nn_pad(targets, pad_id),
                        torch.tensor(target_lengths), padded_times, torch.stack(durations), references, ids,
                        torch.stack(encoder_lengths) if encoder_lengths else None, metadata)


def nn_pad(values: list[torch.Tensor], padding_value: float = 0):
    return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=padding_value)
