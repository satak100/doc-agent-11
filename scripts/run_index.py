#!/usr/bin/env python3
"""Build the cached-OCR -> chunk -> embedding -> vector-index path."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from doc_agent import config, pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = config.load(args.config)
    started = time.perf_counter()
    pipeline.build_knowledge_base(cfg)
    stats_path = Path(cfg["paths"]["index_dir"]) / "index_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["build_seconds"] = round(time.perf_counter() - started, 3)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
