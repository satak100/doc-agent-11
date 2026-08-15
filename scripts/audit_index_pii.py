#!/usr/bin/env python3
"""Audit detector-visible PII before and after pre-index redaction.

This measures residual matches, not recall against manually annotated page PII.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from doc_agent.governance.pii import detect


def iter_text(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield str(json.loads(line).get("text", ""))


def counts(path: Path, ignore_redacted: bool) -> tuple[int, Counter[str]]:
    records = 0
    result: Counter[str] = Counter()
    for text in iter_text(path):
        records += 1
        for start, end, kind in detect(text):
            matched = text[start:end]
            if ignore_redacted and "REDACTED" in matched:
                continue
            result[kind] += 1
    return records, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ocr", type=Path, default=Path("artifacts/ocr/final_results.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/index_e5/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/privacy_audit.json"))
    args = parser.parse_args()

    raw_records, raw_hits = counts(args.raw_ocr, ignore_redacted=False)
    chunk_records, residual_hits = counts(args.chunks, ignore_redacted=True)
    result = {
        "scope": "detector-visible text leakage scan; not manually labelled PII recall",
        "raw_ocr": {"records": raw_records, "detector_hits": dict(sorted(raw_hits.items()))},
        "indexed_chunks": {
            "records": chunk_records,
            "unresolved_detector_hits": dict(sorted(residual_hits.items())),
        },
        "nfr_target": {"metric": "PII entity recall", "target": 0.98, "status": "not_evaluated"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
