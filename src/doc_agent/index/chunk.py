"""Stage 4 — line-aware, overlapping Unicode word chunking."""
from __future__ import annotations

import re
import unicodedata

from ..contracts import Chunk

TOKEN_RE = re.compile(r"\S+", re.UNICODE)


def script_profile(text: str) -> set[str]:
    """Lightweight language/script ID required for the code-switched corpus."""
    scripts: set[str] = set()
    for char in text:
        codepoint = ord(char)
        if 0x0980 <= codepoint <= 0x09FF:
            scripts.add("bengali")
        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            scripts.add("latin")
    return scripts


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    limit = int(cfg["index"]["chunk_tokens"])
    overlap = int(cfg["index"]["overlap"])
    if not 0 <= overlap < limit:
        raise ValueError("index.overlap must be >= 0 and < index.chunk_tokens")
    output: list[Chunk] = []
    for source in chunks:
        tokens = TOKEN_RE.findall(unicodedata.normalize("NFKC", source.text))
        step = limit - overlap
        for ordinal, start in enumerate(range(0, len(tokens), step)):
            part = tokens[start : start + limit]
            if not part:
                break
            output.append(
                Chunk(
                    id=f"{source.id}:{ordinal}",
                    doc_id=source.doc_id,
                    text=" ".join(part),
                    page_ids=source.page_ids,
                )
            )
            if start + limit >= len(tokens):
                break
    return output
