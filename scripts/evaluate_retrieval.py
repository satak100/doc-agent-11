#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc_agent import config
from doc_agent.retrieval.retriever import HybridRetriever, Retriever


def hit(rows, terms: list[str]) -> bool:
    return any(any(term.casefold() in row.text.casefold() for term in terms) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("grading_kit/retrieval_labels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/evidence/retrieval_eval.json"))
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    dense_cfg, hash_cfg = config.load(), config.load("configs/config_hash.yaml")
    systems = {
        "multilingual_e5_small": Retriever(dense_cfg),
        "feature_hashing": Retriever(hash_cfg),
        "hybrid_rrf": HybridRetriever(dense_cfg, hash_cfg),
    }
    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines()]
    report = {"queries": len(labels), "k": args.k, "systems": {}, "details": []}
    for name, retriever in systems.items():
        hits = 0
        for item in labels:
            rows = retriever.retrieve(item["query"], args.k)
            found = hit(rows, item["required_terms"])
            hits += found
            report["details"].append(
                {"system": name, "query": item["query"], "hit": found, "top_doc": rows[0].doc_id}
            )
        report["systems"][name] = {f"recall@{args.k}": hits / len(labels), "hits": hits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
