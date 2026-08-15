"""Stage 4 — compact exact cosine vector index for a 300-page corpus."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contracts import Chunk
from .chunk import script_profile


def _root(cfg: dict) -> Path:
    return Path(os.getenv("DOC_AGENT_INDEX_DIR", cfg["paths"]["index_dir"]))


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Persist normalized float32 vectors and metadata atomically enough for reruns."""
    root = _root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != len(chunks):
        raise ValueError("one embedding is required for every chunk")
    np.save(root / "vectors.npy", vectors)
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    profiles = [script_profile(chunk.text) for chunk in chunks]
    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_type": "numpy-flat-cosine",
        "embedding_backend": cfg["embed"]["backend"],
        "embedding_model": cfg["embed"].get("model"),
        "documents": len({chunk.doc_id for chunk in chunks}),
        "chunks": len(chunks),
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "words": sum(len(chunk.text.split()) for chunk in chunks),
        "vector_bytes": int(vectors.nbytes),
        "script_chunks": {
            "bengali": sum("bengali" in value for value in profiles),
            "latin": sum("latin" in value for value in profiles),
            "mixed_bengali_latin": sum({"bengali", "latin"} <= value for value in profiles),
        },
    }
    (root / "index_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load(cfg: dict) -> tuple[list[Chunk], np.ndarray]:
    root = _root(cfg)
    vectors = np.load(root / "vectors.npy", mmap_mode="r")
    with (root / "chunks.jsonl").open(encoding="utf-8") as handle:
        chunks = [Chunk.model_validate_json(line) for line in handle]
    return chunks, vectors
