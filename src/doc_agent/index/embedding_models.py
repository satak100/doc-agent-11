"""Embedding backends: a fast lexical baseline and pretrained multilingual E5."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Iterable

import numpy as np

TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def hash_encode(texts: Iterable[str], dim: int, prefix: str = "") -> np.ndarray:
    """Signed feature hashing baseline; deterministic and dependency-free."""
    rows = []
    for text in texts:
        vector = np.zeros(dim, dtype=np.float32)
        for token in TOKEN_RE.findall((prefix + text).casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % dim] += 1.0 if value & 1 else -1.0
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return np.stack(rows) if rows else np.empty((0, dim), dtype=np.float32)


@lru_cache(maxsize=2)
def _load_e5(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def e5_encode(
    texts: list[str], model_name: str, batch_size: int, prefix: str
) -> np.ndarray:
    """Mean-pooled E5 embeddings using the published model's required prefixes."""
    import torch
    import torch.nn.functional as functional
    tokenizer, model, device = _load_e5(model_name)
    rows: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {text}" for text in texts[start : start + batch_size]]
        encoded = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = functional.normalize(pooled, p=2, dim=1)
        rows.append(pooled.float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def encode_texts(texts: list[str], cfg: dict, kind: str) -> np.ndarray:
    backend = cfg["embed"].get("backend", "e5")
    if backend == "hash":
        return hash_encode(texts, int(cfg["embed"].get("dim", 1024)))
    prefix = "query" if kind == "query" else "passage"
    return e5_encode(texts, cfg["embed"]["model"], int(cfg["embed"]["batch_size"]), prefix)
