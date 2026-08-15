#!/usr/bin/env python3
"""Evaluate medication-line OCR with CER, WER, and fuzzy entity F1."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LOC_RE = re.compile(r"<\|LOC_\d+\|>")
MED_LINE_RE = re.compile(
    r"^\s*(?:[-•.]\s*)?(?:tab|tob|tcob|cap|cop|gap|syp|sup|suppo|e6|bb)\.?\s*[:,-]?\s*(.+)$",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE)
    return " ".join(text.split())


def distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def candidates(text: str) -> list[str]:
    output = []
    for line in LOC_RE.sub("", text).splitlines():
        match = MED_LINE_RE.match(line)
        if not match:
            continue
        value = re.split(r"(?:\s{2,}|\s[-—]\s|\\\(|\d\s*\+)", match.group(1), maxsplit=1)[0]
        value = normalize(value)
        if len(value) >= 2:
            output.append(value)
    return output


def score(labels_path: Path, results_path: Path) -> dict:
    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    results = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "ok":
            results[row["relative_path"]] = row.get("text", "")
    char_errors = word_errors = ref_chars = ref_words = tp = fp = fn = 0
    examples = []
    for row in labels:
        truth = [normalize(value) for value in row["medications"]]
        pred = candidates(results.get(row["relative_path"], ""))
        truth_text, pred_text = "\n".join(truth), "\n".join(pred)
        char_errors += distance(truth_text, pred_text)
        ref_chars += len(truth_text)
        truth_words, pred_words = truth_text.split(), pred_text.split()
        word_errors += distance(truth_words, pred_words)
        ref_words += len(truth_words)
        unused = set(range(len(pred)))
        matched = 0
        for reference in truth:
            if not unused:
                break
            best = max(unused, key=lambda i: 1 - distance(reference, pred[i]) / max(len(reference), len(pred[i]), 1))
            similarity = 1 - distance(reference, pred[best]) / max(len(reference), len(pred[best]), 1)
            if similarity >= 0.75:
                matched += 1
                unused.remove(best)
        tp += matched
        fp += len(pred) - matched
        fn += len(truth) - matched
        examples.append({"relative_path": row["relative_path"], "reference": truth, "prediction": pred})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "scope": "medicine-name/strength lines only",
        "pages": len(labels),
        "reference_medications": sum(len(row["medications"]) for row in labels),
        "cer": char_errors / max(ref_chars, 1),
        "wer": word_errors / max(ref_words, 1),
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score(args.labels, args.results)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
