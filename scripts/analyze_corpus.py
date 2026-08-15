#!/usr/bin/env python3
"""Fast, dependency-light corpus EDA for photographed document pages."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    at = lambda q: ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]
    return {"min": ordered[0], "p25": at(0.25), "median": at(0.5), "p75": at(0.75), "max": ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ocr-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in SUFFIXES)
    widths, heights, brightness, contrast, edge_energy = [], [], [], [], []
    total_bytes = 0
    for path in paths:
        total_bytes += path.stat().st_size
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("L")
        widths.append(image.width)
        heights.append(image.height)
        thumb = image.copy()
        thumb.thumbnail((512, 512))
        stats = ImageStat.Stat(thumb)
        brightness.append(stats.mean[0])
        contrast.append(stats.stddev[0])
        edge_energy.append(ImageStat.Stat(thumb.filter(ImageFilter.FIND_EDGES)).mean[0])
    records = [json.loads(line) for line in args.ocr_jsonl.read_text(encoding="utf-8").splitlines()]
    texts = [row.get("text", "") for row in records if row.get("status") == "ok"]
    report = {
        "pages": len(paths),
        "bytes": total_bytes,
        "width_px": quantiles(widths),
        "height_px": quantiles(heights),
        "brightness_0_255": quantiles(brightness),
        "contrast_stddev": quantiles(contrast),
        "edge_energy": quantiles(edge_energy),
        "portrait_fraction": sum(h >= w for w, h in zip(widths, heights)) / len(paths),
        "ocr_nonempty_pages": sum(bool(text.strip()) for text in texts),
        "ocr_word_occurrences_no_overlap": sum(len(text.split()) for text in texts),
        "mean_ocr_words_per_page": statistics.mean(len(text.split()) for text in texts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
