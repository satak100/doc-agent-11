#!/usr/bin/env bash
# Build both indexes and reproduce evaluation evidence.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

OCR_INPUT="${DOC_AGENT_IMAGE_DIR:-data/raw}"
BASE_RESULTS="${DOC_AGENT_BASELINE_JSONL:-artifacts/ocr/baseline_results.jsonl}"
FINAL_RESULTS="${DOC_AGENT_OCR_JSONL:-artifacts/ocr/final_results.jsonl}"

# Normal reruns start here and reuse the measured cache. A fresh checkout can
# reproduce the expensive OCR first; the tested 4090 environment uses
# Transformers 5.14 / PyTorch 2.13 CUDA with BF16 SDPA.
if [[ ! -f "$FINAL_RESULTS" ]]; then
  mkdir -p artifacts/ocr
  python scripts/run_paddleocr_vl.py --input "$OCR_INPUT" --output artifacts/ocr/base_run --batch-size 16 --max-new-tokens 768
  BASE_RESULTS="artifacts/ocr/base_run/results.jsonl"
  python scripts/run_paddleocr_vl.py --input "$OCR_INPUT" --output artifacts/ocr/retry_run --batch-size 16 --max-new-tokens 1536 --only-truncated-from "$BASE_RESULTS"
  python scripts/finalize_ocr_results.py --base "$BASE_RESULTS" --retry artifacts/ocr/retry_run/results.jsonl --output "$FINAL_RESULTS"
fi

export DOC_AGENT_OCR_JSONL="$FINAL_RESULTS"
if [[ -f "$BASE_RESULTS" ]]; then
  python scripts/evaluate_ocr.py --labels grading_kit/labels.jsonl --results "$BASE_RESULTS" --output reports/evidence/ocr_baseline.json
fi
python scripts/evaluate_ocr.py --labels grading_kit/labels.jsonl --results "$FINAL_RESULTS" --output reports/evidence/ocr_selected_retry.json
python scripts/run_index.py --config configs/config.yaml
python scripts/run_index.py --config configs/config_hash.yaml
python scripts/evaluate_retrieval.py -k 5
