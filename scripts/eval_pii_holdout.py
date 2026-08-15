#!/usr/bin/env python3
"""Score the manually reviewed image-to-index privacy holdout without printing PII."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("grading_kit/pii_holdout_labels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pii_holdout_metrics.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line]
    total = sum(sum(row["visually_present"].values()) for row in rows)
    redacted = sum(row["explicitly_redacted"] for row in rows)
    leaked = sum(row["leaked"] for row in rows)
    outcomes = Counter(row["indexed_outcome"] for row in rows)
    extracted = redacted + leaked
    result = {
        "sample": {"pages": len(rows), "visual_pii_entities": total, "entity_types": ["patient_name"]},
        "outcomes": dict(sorted(outcomes.items())),
        "metrics": {
            "explicit_detection_and_redaction_recall": redacted / total if total else 0.0,
            "redaction_recall_conditional_on_ocr_extraction": redacted / extracted if extracted else 0.0,
            "end_to_end_non_disclosure_rate": (total - leaked) / total if total else 0.0,
            "end_to_end_leak_rate": leaked / total if total else 0.0,
        },
        "target": {"metric": "PII entity recall", "minimum": 0.98, "passed": False},
        "notes": [
            "Manual visual labels contain counts and outcomes only; identifier values are not stored.",
            "OCR omission prevents text-index disclosure but is not credited as explicit PII detection/redaction.",
            "Single-reviewer pilot; expand entity types and add a second reviewer for a final NFR claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
