"""Stage 1 — deterministic, duplicate-safe page-image ingestion."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..contracts import Page

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def load_pages(cfg: dict) -> list[Page]:
    """Load page images in stable order; an environment override keeps configs portable."""
    root = Path(os.getenv("DOC_AGENT_IMAGE_DIR", cfg["paths"]["images"]))
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    pages: list[Page] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        relative = path.relative_to(root).as_posix()
        page_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
        pages.append(Page(id=page_id, image_path=str(path.resolve()), doc_id=relative))
    if not pages:
        raise ValueError(f"no supported page images found under {root}")
    return pages
