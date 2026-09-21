#!/usr/bin/env python3
"""Download the public MuST-C English–Arabic files from Kaggle.

Downloads can resume file by file. The script checks each file's size and saves
its SHA-256 hash locally. Kaggle's file listing does not provide an upstream
hash, so these hashes help verify a local copy but cannot authenticate it.
Text files download first, followed by development, test, and training audio.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time
import zipfile
import shutil
from urllib.parse import quote

import requests

DATASET = "sebaeymohamed/must-c-en-ar"
API = "https://www.kaggle.com/api/v1/datasets"
LOCAL = threading.local()
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def request_slot():
    """Stay below three requests/second across all download workers."""
    global NEXT_REQUEST
    with REQUEST_LOCK:
        delay = max(0.0, NEXT_REQUEST - time.monotonic())
        if delay:
            time.sleep(delay)
        NEXT_REQUEST = time.monotonic() + 0.4


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def atomic_json(path, value):
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    part.replace(path)


def inventory(root):
    root.mkdir(parents=True, exist_ok=True)
    meta_path, files_path = root / "kaggle_metadata.json", root / "kaggle_files.json"
    if not meta_path.exists():
        response = session().get(f"{API}/view/{DATASET}", timeout=60)
        response.raise_for_status()
        atomic_json(meta_path, response.json())
    if files_path.exists():
        return json.loads(meta_path.read_text()), json.loads(files_path.read_text())
    files, token = [], None
    while True:
        params = {"pageSize": 200}
        if token:
            params["pageToken"] = token
        response = session().get(f"{API}/list/{DATASET}", params=params, timeout=60)
        response.raise_for_status()
        value = response.json()
        if "datasetFiles" not in value:
            raise RuntimeError("Kaggle listing omitted datasetFiles")
        files.extend(value["datasetFiles"])
        token = value.get("nextPageToken")
        if not token:
            break
    names = [f["name"] for f in files]
    if len(names) != len(set(names)):
        raise RuntimeError("Kaggle pagination returned duplicate names")
    atomic_json(files_path, files)
    return json.loads(meta_path.read_text()), files


def download_one(root, entry, version):
    name, size = entry["name"], int(entry["totalBytes"])
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Unsafe upstream path")
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    hash_path = root / ".sha256" / (name + ".json")
    if destination.is_file() and destination.stat().st_size == size and hash_path.exists():
        value = json.loads(hash_path.read_text())
        if value.get("size") == size:
            return value
    if destination.exists() and destination.stat().st_size != size:
        raise RuntimeError(f"Existing completed file has wrong size: {name}")
    part = destination.with_suffix(destination.suffix + ".part")
    def materialize():
        if not part.exists():
            return False
        if part.stat().st_size == size:
            part.replace(destination)
            return True
        # Kaggle may wrap a single large file in a ZIP. Check its size and CRC.
        if not zipfile.is_zipfile(part):
            return False
        with zipfile.ZipFile(part) as archive:
            members = archive.infolist()
            if len(members) != 1 or PurePosixPath(members[0].filename).name != rel.name or members[0].file_size != size:
                raise RuntimeError(f"Unexpected individual ZIP payload: {name}")
            unpacked = destination.with_suffix(destination.suffix + ".unpacked.part")
            with archive.open(members[0]) as source, unpacked.open("wb") as target:
                shutil.copyfileobj(source, target, length=2**20)
            if unpacked.stat().st_size != size:
                raise RuntimeError(f"Uncompressed size mismatch: {name}")
            unpacked.replace(destination)
        part.unlink()
        return True
    if not destination.exists():
        for attempt in range(8):
            if materialize():
                break
            offset = part.stat().st_size if part.exists() else 0
            try:
                request_slot()
                response = session().get(
                    f"{API}/download/{DATASET}/{quote(name, safe='')}",
                    params={"datasetVersionNumber": version},
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                    stream=True, timeout=(30, 120),
                )
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Kaggle returned HTML instead of dataset content")
                mode = "ab" if offset and response.status_code == 206 else "wb"
                if mode == "ab" and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Invalid resumed Content-Range")
                with response, part.open(mode) as stream:
                    for chunk in response.iter_content(2**20):
                        stream.write(chunk)
                if not materialize():
                    raise RuntimeError(f"Download size mismatch for {name}")
                break
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == 7:
                    raise RuntimeError(f"Failed after 8 attempts: {name}: {type(exc).__name__}") from exc
                time.sleep(60 if getattr(getattr(exc, "response", None), "status_code", None) == 429 else min(2 ** attempt, 30))
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(chunk)
    value = {"name": name, "size": size, "sha256": digest.hexdigest(), "version": version}
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(hash_path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="live_translation/data/kaggle_mustc_en_ar")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--audio-only", action="store_true", help="May run concurrently with --metadata-only")
    parser.add_argument("--priority-manifest", action="append", default=[], help="Prioritize audio paths in these existing JSONL manifests")
    args = parser.parse_args()
    if args.metadata_only and args.audio_only:
        parser.error("--metadata-only and --audio-only are mutually exclusive")
    root = Path(args.root).resolve()
    metadata, files = inventory(root)
    version = metadata["currentVersionNumber"]
    if version != 1:
        raise RuntimeError("The audited release is version 1; review a new release explicitly")
    print(json.dumps({"dataset": DATASET, "version": version, "files": len(files), "listed_bytes": sum(f["totalBytes"] for f in files), "metadata_bytes": metadata["totalBytes"], "root": str(root)}), flush=True)
    priority = {}
    for manifest in args.priority_manifest:
        for line in Path(manifest).read_text().splitlines():
            if line.strip():
                name = str(Path(json.loads(line)["audio"]).resolve().relative_to(root))
                priority.setdefault(name, len(priority))
    def key(entry):
        name = entry["name"]
        if not name.endswith(".wav"):
            group = 0
        elif name in priority:
            group = 1
        elif name.startswith(("dev/", "tst-COMMON/", "tst-HE/")):
            group = 2
        else:
            group = 3
        return group, priority.get(name, 0), name
    files.sort(key=key)
    if args.metadata_only:
        files = [f for f in files if not f["name"].endswith(".wav")]
    elif args.audio_only:
        files = [f for f in files if f["name"].endswith(".wav")]
    start, last_print, completed, nbytes = time.monotonic(), 0.0, [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, root, entry, version): entry for entry in files}
        for future in as_completed(futures):
            value = future.result()
            completed.append(value)
            nbytes += value["size"]
            now = time.monotonic()
            if now - last_print >= 15 or len(completed) == len(files):
                print(json.dumps({"completed_files": len(completed), "total_files": len(files), "completed_bytes": nbytes, "elapsed_seconds": round(now-start, 1), "last_file": value["name"]}), flush=True)
                last_print = now
    audit = {"dataset": DATASET, "source_url": "https://www.kaggle.com/datasets/" + DATASET, "version": version, "downloaded_utc": datetime.now(timezone.utc).isoformat(), "metadata_only": args.metadata_only, "audio_only": args.audio_only, "files": sorted(completed, key=lambda f: f["name"]), "total_bytes": nbytes, "hash_provenance": "SHA-256 computed locally after expected-byte-size validation; upstream hashes unavailable"}
    audit_name = "download_metadata_audit.json" if args.metadata_only else ("download_audio_audit.json" if args.audio_only else "download_audit.json")
    atomic_json(root / audit_name, audit)


if __name__ == "__main__":
    main()
