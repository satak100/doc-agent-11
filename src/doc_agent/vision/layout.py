"""Stage 2 — inexpensive page layout used before cached OCR is attached."""
from __future__ import annotations

from PIL import Image, ImageOps

from ..contracts import Page, Region


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect heading and body regions cheaply; OCR supplies line polygons.

    Prescription templates consistently reserve the upper band for letterhead.
    We expose that region separately from the body so a live reader can crop or
    route it.  Cached Paddle spotting remains the fine-grained line detector.
    """
    regions: list[Region] = []
    for page in pages:
        with Image.open(page.image_path) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
        heading_end = max(1, int(height * float(cfg.get("layout", {}).get("heading_ratio", 0.22))))
        regions.append(Region(page_id=page.id, bbox=(0, 0, width, heading_end), kind="heading"))
        regions.append(Region(page_id=page.id, bbox=(0, heading_end, width, height), kind="text"))
    return regions
