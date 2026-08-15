#!/usr/bin/env python3
"""Resumable downloader for a public Google Drive image folder."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from collections import Counter
from pathlib import Path

import gdown
import requests


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        signature = handle.read(12)
    return signature.startswith(b"\xff\xd8\xff") or signature.startswith(b"\x89PNG\r\n\x1a\n")


def download_one(file_id: str, target: Path, retries: int = 8) -> tuple[str, str]:
    if valid_image(target):
        return "skipped", target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    last_error = "unknown error"
    for attempt in range(retries):
        try:
            with requests.get(params=params, url=url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError("Google returned HTML instead of image bytes")
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not valid_image(partial):
                raise RuntimeError("download did not have a JPEG/PNG signature")
            partial.replace(target)
            return "downloaded", target.name
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if partial.exists():
                partial.unlink()
            time.sleep(min(60, (2**attempt) + random.random()))
    return "failed", f"{target.name}: {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    files = gdown.download_folder(
        id=args.folder_id,
        output=str(args.output),
        quiet=True,
        skip_download=True,
    )
    original_names = [Path(item.path).name for item in files]
    name_counts = Counter(original_names)
    manifest = []
    for item, original_name in zip(files, original_names):
        path = Path(original_name)
        if name_counts[original_name] > 1:
            target_name = f"{path.stem}__{item.id[:8]}{path.suffix}"
        else:
            target_name = original_name
        manifest.append(
            {"id": item.id, "name": target_name, "original_name": original_name}
        )
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"enumerated={len(manifest)}", flush=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, item["id"], args.output / item["name"]): item
            for item in manifest
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            status, detail = future.result()
            counts[status] += 1
            if status == "failed":
                print(f"FAILED {detail}", flush=True)
            if index % 25 == 0 or index == len(futures):
                print(f"progress={index}/{len(futures)} counts={counts}", flush=True)

    print(json.dumps(counts), flush=True)
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
