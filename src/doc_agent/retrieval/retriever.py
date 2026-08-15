"""Stage 5 — dense cosine retrieval."""
from __future__ import annotations

import numpy as np

from ..contracts import Chunk
from ..index import store
from ..index.embedding_models import encode_texts

class Retriever:
    def __init__(self, cfg: dict) -> None:
        self.full_cfg = cfg
        self.cfg = cfg["retrieve"]
        self.chunks, self.vectors = store.load(cfg)
    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return top-k chunks; dot product is cosine because both sides are normalized."""
        limit = min(k or int(self.cfg["k"]), len(self.chunks))
        query_vector = encode_texts([query], self.full_cfg, kind="query")[0]
        scores = np.asarray(self.vectors @ query_vector)
        indices = np.argpartition(-scores, limit - 1)[:limit]
        indices = indices[np.argsort(-scores[indices])]
        return [self.chunks[int(i)].model_copy(update={"score": float(scores[i])}) for i in indices]


class HybridRetriever:
    """Fuse dense and exact-token ranks; no score calibration is required."""

    def __init__(self, dense_cfg: dict, lexical_cfg: dict) -> None:
        self.dense = Retriever(dense_cfg)
        self.lexical = Retriever(lexical_cfg)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        depth = max(20, k * 4)
        rankings = [self.dense.retrieve(query, depth), self.lexical.retrieve(query, depth)]
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, 1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (60 + rank)
                chunks[chunk.id] = chunk
        ordered = sorted(fused, key=fused.get, reverse=True)[:k]
        return [chunks[key].model_copy(update={"score": fused[key]}) for key in ordered]


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)

def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]

def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None
