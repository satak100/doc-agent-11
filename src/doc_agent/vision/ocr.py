"""Stage 3 — cached PaddleOCR-VL 1.6 text spotting adapter."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..contracts import Chunk, Region

LOC_RE = re.compile(r"<\|LOC_(\d+)\|>")


def _clean(text: str) -> str:
    """Preserve reading order while removing model-specific polygon tokens."""
    text = LOC_RE.sub("", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class Reader:
    """Reads a resumable OCR JSONL cache produced by the GPU runner."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        cache_path = Path(os.getenv("DOC_AGENT_OCR_JSONL", self.cfg["cache_jsonl"]))
        self.records: dict[str, str] = {}
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "ok":
                    self.records[row["relative_path"]] = _clean(row.get("text", ""))

    def transcribe_region(self, region: Region) -> str:
        # Region.page_id is resolved by transcribe(), which owns the page mapping.
        return self.records.get(region.page_id, "")


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Attach cached OCR once per page and expose it as page-level chunks."""
    reader = Reader(cfg)
    id_to_doc = cfg.get("_page_id_to_doc", {})
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for region in regions:
        if region.page_id in seen:
            continue
        seen.add(region.page_id)
        doc_id = id_to_doc.get(region.page_id, region.page_id)
        text = reader.records.get(doc_id, "")
        if text:
            chunks.append(
                Chunk(
                    id=f"page:{region.page_id}",
                    doc_id=doc_id,
                    text=text,
                    page_ids=[region.page_id],
                )
            )
    return chunks
