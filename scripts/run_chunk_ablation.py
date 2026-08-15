#!/usr/bin/env python3
"""Build several chunk sizes and measure real retrieval recall/latency."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from doc_agent import config, pipeline
from doc_agent.retrieval.retriever import HybridRetriever, Retriever


def is_hit(rows, terms: list[str]) -> bool:
    return any(any(term.casefold() in row.text.casefold() for term in terms) for row in rows)


def main() -> None:
    labels = [json.loads(line) for line in Path("grading_kit/retrieval_labels.jsonl").read_text().splitlines()]
    variants = [(64, 16), (128, 24), (256, 32)]
    report = {"queries": len(labels), "k": 5, "variants": []}
    for size, overlap in variants:
        dense = copy.deepcopy(config.load())
        lexical = copy.deepcopy(config.load("configs/config_hash.yaml"))
        for cfg, name in ((dense, "e5"), (lexical, "hash")):
            cfg["index"].update(chunk_tokens=size, overlap=overlap)
            cfg["paths"]["index_dir"] = f"artifacts/ablations/chunk_{size}/{name}"
        started = time.perf_counter()
        pipeline.build_knowledge_base(dense)
        dense_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pipeline.build_knowledge_base(lexical)
        hash_seconds = time.perf_counter() - started
        systems = {
            "e5": Retriever(dense),
            "hash": Retriever(lexical),
            "hybrid": HybridRetriever(dense, lexical),
        }
        scores = {}
        latency = {}
        for name, retriever in systems.items():
            hits = 0
            elapsed = 0.0
            for item in labels:
                began = time.perf_counter()
                rows = retriever.retrieve(item["query"], 5)
                elapsed += time.perf_counter() - began
                hits += is_hit(rows, item["required_terms"])
            scores[name] = hits / len(labels)
            latency[name] = 1000 * elapsed / len(labels)
        stats = json.loads((Path(dense["paths"]["index_dir"]) / "index_stats.json").read_text())
        report["variants"].append(
            {
                "chunk_tokens": size,
                "overlap": overlap,
                "chunks": stats["chunks"],
                "e5_build_seconds": dense_seconds,
                "hash_build_seconds": hash_seconds,
                "recall_at_5": scores,
                "mean_query_ms": latency,
            }
        )
    output = Path("reports/evidence/chunk_ablation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
