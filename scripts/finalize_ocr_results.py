#!/usr/bin/env python3
"""Merge OCR passes and trim obvious autoregressive runaway tails."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SAME_CHAR_RUN = re.compile(r"(.)\1{19,}", re.DOTALL)
REPEATED_CHUNK = re.compile(r"(.{2,80}?)\1{4,}", re.DOTALL)


def load_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["relative_path"]] = record
    return records


def trim_repetition(text: str) -> tuple[str, bool, str | None]:
    cut = len(text)
    reason = None

    same_char = SAME_CHAR_RUN.search(text)
    if same_char:
        cut = same_char.start()
        reason = "same_character_repeated_20_times"

    repeated_chunk = REPEATED_CHUNK.search(text)
    if repeated_chunk and repeated_chunk.start() < cut:
        cut = repeated_chunk.start()
        reason = "short_sequence_repeated_5_times"

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    run_start = 0
    run_value = None
    run_length = 0
    for index, line in enumerate(lines):
        normalized = line.strip()
        if normalized and normalized == run_value:
            run_length += 1
        else:
            run_start = index
            run_value = normalized
            run_length = 1
        if normalized and run_length >= 5 and offsets[run_start] < cut:
            cut = offsets[run_start]
            reason = "same_line_repeated_5_times"
            break

    cleaned = text[:cut].rstrip()
    return cleaned, cut < len(text), reason


def candidate(record: dict) -> tuple[tuple[int, int], dict]:
    cleaned, trimmed, reason = trim_repetition(record.get("text", ""))
    result = dict(record)
    result["text"] = cleaned
    result["postprocess"] = {
        "runaway_tail_trimmed": trimmed,
        "reason": reason,
        "raw_characters": len(record.get("text", "")),
        "clean_characters": len(cleaned),
    }
    # Prefer more clean content, then a normally terminated generation.
    score = (len(cleaned), int(not record.get("hit_max_new_tokens", False)))
    return score, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_jsonl(args.base)
    retry = load_jsonl(args.retry)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "records": 0,
        "selected_retry": 0,
        "runaway_tail_trimmed": 0,
        "empty_after_cleanup": 0,
        "bengali_detected": 0,
        "requires_review": 0,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        for relative_path in sorted(base):
            choices = [("base_768", base[relative_path])]
            if relative_path in retry:
                choices.append(("retry_1536", retry[relative_path]))
            scored = [(candidate(record), source) for source, record in choices]
            (score, selected), source = max(scored, key=lambda item: item[0][0])
            selected["selected_source"] = source
            selected["candidate_clean_character_counts"] = {
                source_name: candidate(record)[0][0]
                for source_name, record in choices
            }
            review_reasons = []
            if selected.get("hit_max_new_tokens"):
                review_reasons.append("generation_hit_token_ceiling")
            if selected["postprocess"]["runaway_tail_trimmed"]:
                review_reasons.append("runaway_tail_was_trimmed")
            if len(selected["text"].strip()) < 100:
                review_reasons.append("very_short_transcription")
            selected["quality_flags"] = {
                "requires_review": bool(review_reasons),
                "reasons": review_reasons,
            }
            handle.write(json.dumps(selected, ensure_ascii=False) + "\n")

            summary["records"] += 1
            summary["selected_retry"] += int(source == "retry_1536")
            summary["runaway_tail_trimmed"] += int(
                selected["postprocess"]["runaway_tail_trimmed"]
            )
            summary["empty_after_cleanup"] += int(not selected["text"].strip())
            summary["bengali_detected"] += int(
                bool(re.search(r"[\u0980-\u09ff]", selected["text"]))
            )
            summary["requires_review"] += int(bool(review_reasons))

    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
