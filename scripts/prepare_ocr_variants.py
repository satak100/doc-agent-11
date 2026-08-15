#!/usr/bin/env python3
"""Create fast held-out preprocessing/layout variants for an OCR ablation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from doc_agent.ingest.preprocess import enhance_image

LOC_RE = re.compile(r"<\|LOC_(\d+)\|>")


def rx_anchor(text: str) -> tuple[float, float] | None:
    for line in text.splitlines():
        plain = LOC_RE.sub("", line).strip().casefold()
        if plain not in {"rx", "r", "rₓ", "rα", "r_c"}:
            continue
        coords = [int(value) for value in LOC_RE.findall(line)]
        if len(coords) >= 2:
            return coords[0] / 1000.0, coords[1] / 1000.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--spotting-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = {json.loads(line)["relative_path"] for line in args.labels.read_text().splitlines()}
    spotting = {}
    for line in args.spotting_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        spotting[row.get("relative_path")] = row.get("text", "")
    enhanced_dir, crop_dir = args.output / "enhanced", args.output / "layout_crop"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in sorted(selected):
        with Image.open(args.images / name) as source:
            image = source.convert("RGB")
        enhanced = enhance_image(image)
        enhanced.save(enhanced_dir / name, quality=94)
        anchor = rx_anchor(spotting.get(name, ""))
        if anchor:
            x, y = anchor
            left = max(0, int((x - 0.08) * image.width))
            top = max(0, int((y - 0.04) * image.height))
        else:
            left, top = 0, int(image.height * 0.18)
        crop = image.crop((left, top, image.width, image.height))
        crop.save(crop_dir / name, quality=95)
        manifest.append({"relative_path": name, "rx_anchor": anchor, "crop": [left, top, image.width, image.height]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(manifest), "anchored": sum(row["rx_anchor"] is not None for row in manifest)}))


if __name__ == "__main__":
    main()
