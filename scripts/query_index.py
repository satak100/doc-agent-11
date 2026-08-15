#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from doc_agent import config
from doc_agent.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    rows = Retriever(config.load(args.config)).retrieve(args.query, args.k)
    for rank, row in enumerate(rows, 1):
        print(json.dumps({"rank": rank, **row.model_dump()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
