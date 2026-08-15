"""Stage 4 — batched embedding."""
from __future__ import annotations

from ..contracts import Chunk
from .embedding_models import encode_texts


def encode(chunks: list[Chunk], cfg: dict):
    return encode_texts([chunk.text for chunk in chunks], cfg, kind="passage")
