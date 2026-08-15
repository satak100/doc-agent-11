"""Stage 1 — optional, cached document-photo preprocessing."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..contracts import Page


def enhance_image(image: Image.Image) -> Image.Image:
    """Correct orientation, stretch contrast gently, denoise and sharpen ink.

    We intentionally do not hard-binarize: the held-out experiment showed that
    faint blue handwriting can be lost.  This function is an ablatable option,
    not silently applied to the source corpus.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return ImageEnhance.Sharpness(image).enhance(1.35)


def run(pages: list[Page], cfg: dict) -> list[Page]:
    options = cfg.get("preprocess", {})
    if not options.get("enabled", False):
        return pages
    cache = Path(os.getenv("DOC_AGENT_PREPROCESS_DIR", options.get("cache_dir", "artifacts/preprocessed")))
    cache.mkdir(parents=True, exist_ok=True)
    output: list[Page] = []
    for page in pages:
        destination = cache / f"{page.id}.jpg"
        if not destination.exists():
            with Image.open(page.image_path) as source:
                enhance_image(source).save(destination, quality=94, optimize=True)
        output.append(page.model_copy(update={"image_path": str(destination.resolve())}))
    return output
