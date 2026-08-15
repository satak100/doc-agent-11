#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from doc_agent import config
from doc_agent.retrieval.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--dense-config", default="configs/config.yaml")
    parser.add_argument("--lexical-config", default="configs/config_hash.yaml")
    args = parser.parse_args()
    retriever = HybridRetriever(config.load(args.dense_config), config.load(args.lexical_config))
    for rank, row in enumerate(retriever.retrieve(args.query, args.k), 1):
        print(json.dumps({"rank": rank, **row.model_dump()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
