#!/usr/bin/env python3
"""Fill the A2 form with evidence-backed results and A1 carry-forward fields."""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document


def put(table, row: int, column: int, text: str) -> None:
    table.cell(row, column).text = text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("forms/A2_form.docx"))
    parser.add_argument("--output", type=Path, default=Path("forms/A2_form.docx"))
    args = parser.parse_args()
    doc = Document(args.input)
    tables = doc.tables

    put(tables[1], 0, 1, "Pharmacological Query from Prescription")
    members = [
        ("Satak Kumar Dey", "2105100", "Sections 2, 3, 4 and 8"),
        ("Jannatul Ferdous Prity", "2105113", "Sections 1, 5, 6, 7 and 8"),
        ("Not applicable (two-member team)", "—", "—"),
    ]
    for row, values in enumerate(members, 1):
        for column, value in enumerate(values, 1):
            put(tables[2], row, column, value)

    put(tables[4], 0, 1, "Domain: Healthcare / pharmacological information. Data speciality: handwritten medical prescriptions. Primary NFR: privacy; PII entity recall ≥0.98.")
    put(tables[4], 1, 1, "Groundedness ≥0.90 (carried forward unchanged from A1).")
    put(tables[4], 2, 1, "Permissioned Somatec Pharmaceuticals Ltd. handwritten prescriptions, supplied through the proper authorized project channel, with an auxiliary Bangladesh medicine corpus identified in A1. The tested working snapshot contains photographed Bangla/English prescriptions with physician handwriting, Bengali numerals, shadows, perspective and repeated printed templates. Raw clinical pages remain restricted; the delivery link is not an open redistribution licence.")

    put(tables[5], 0, 1, "Local pixel continuity forms pen strokes and characters; translation-tolerant visual features survive page position changes. Repeated prescription templates provide stable header/body and Rx-body structure. Paddle text spotting supplies line polygons; a heading/body heuristic supplies parent regions.")
    put(tables[5], 1, 1, "Reading order links a medicine line to its strength and nearby schedule. Medical names are short, spelling-sensitive Latin strings embedded in Bangla context; Unicode NFKC normalization and explicit Bangla/Latin script profiling preserve code-switching for multilingual embeddings.")
    put(tables[5], 2, 1, "Patient/page position, font, mild skew, illumination, JPEG noise, harmless whitespace/case, and printed-template variation should not change the retrieved medication evidence. Drug spelling, strength, frequency and duration must not be silently normalized because those changes can be clinically meaningful.")

    comparisons = [
        ("Raw image; autocontrast+median+sharpen; VAE/diffusion considered as bonus", "Keep raw for the main pass: classical enhancement reduced entity F1 from 0.490 to 0.413. Retain enhancement as an optional cached ablation.", "No"),
        ("Whole page; fixed heading/body regions; OCR-derived Rx-anchor crop", "Whole page for the durable index plus Rx-body crop for re-OCR. Cropped+trimmed CER/F1 were 0.413/0.505 versus 0.499/0.490 for the 768-token page baseline.", "Adapted"),
        ("PaddleOCR-VL 1.6 at 768 tokens; selective 1536 retry; layout-cropped re-OCR", "Pretrained PaddleOCR-VL 1.6 in BF16/SDPA, size-sorted batch 16. Selective retry improves recall slightly; crop re-OCR is used for failed pages. No unsupported fine-tuning claim.", "Yes — pretrained PaddleOCR-VL 1.6"),
        ("64/16, 128/24 and 256/32 fixed overlapping chunks", "64 words/16 overlap: E5 recall@5 was 1.00 vs 0.80 and 0.20, while hybrid query time stayed about 2.76 ms.", "No"),
        ("Signed feature hashing; pretrained multilingual E5-small; hybrid RRF", "Hybrid RRF: exact medicine matching from lexical vectors plus multilingual semantics from E5; final five-query recall@5 was 1.00.", "Yes — pretrained multilingual E5-small"),
        ("Exact flat cosine; HNSW/IVF considered", "Exact NumPy cosine. Only 943 chunks, so ANN adds complexity and possible recall loss without useful latency benefit.", "No"),
    ]
    for row, values in enumerate(comparisons, 1):
        for column, value in enumerate(values, 1):
            put(tables[6], row, column, value)

    built = [
        ("PIL orientation + optional classical enhancement", "EXIF transpose; autocontrast cutoff 1%; median 3×3; sharpness 1.35; disabled by selected main config", "Deterministic classical method"),
        ("Heading/body heuristic + Paddle spotting polygons", "heading ratio 0.22; OCR-derived Rx anchor used for crop ablation", "Heuristic + pretrained detector inside OCR"),
        ("PaddlePaddle/PaddleOCR-VL-1.6", "Spotting prompt; BF16; SDPA; batch 16; max 768; selective max 1536; repetition trim", "Pretrained, adapted; not fine-tuned"),
        ("Unicode line-aware fixed chunks", "NFKC; 64 words; overlap 16; Bangla/Latin script profile", "Deterministic"),
        ("intfloat/multilingual-e5-small + signed feature hash", "384-D E5 passage/query prefixes; 2048-D lexical; batch 64", "E5 pretrained; hash from scratch"),
        ("NumPy flat cosine + RRF", "943 chunks; exact normalized dot product; RRF depth 20", "Built from corpus"),
    ]
    for row, values in enumerate(built, 1):
        for column, value in enumerate(values, 1):
            put(tables[7], row, column, value)
    put(tables[8], 0, 1, "JPEG pages → stable ingest/EXIF → optional enhancement → heading/body + spotting layout → PaddleOCR-VL → repetition cleanup + PII redaction → NFKC/script ID → 64/16 chunks → E5 and lexical vectors → exact cosine stores → hybrid RRF retrieval. OCR is cached so rebuilding the KB does not rerun vision inference.")

    put(tables[9], 0, 1, "Held-out sample: 12 pages / 61 medication-name+strength entries. 768-token page baseline CER 0.499, WER 0.702, entity F1 0.490. Best measured layout-crop+trim method: CER 0.413, WER 0.702, precision 0.587, recall 0.443, F1 0.505. Scope is medication lines, not full-page clinical transcription.")
    put(tables[9], 1, 1, "310/310 pages → 943 chunks. 42,642 unique non-overlapped OCR word occurrences; 52,575 indexed occurrences with overlap. E5: exact flat cosine, 384-D, 1,448,448 vector bytes, 7.304 s build. Hash: 2048-D, 7,725,056 bytes, 1.373 s. Script chunks: 288 Bengali, 902 Latin, 276 mixed. Manual privacy pilot: 12 pages/12 patient-name entities; explicit detection/redaction recall 0.0833, conditional redaction recall 0.3333, non-disclosure 0.8333, two leaks. The ≥0.98 NFR fails.")
    put(tables[9], 2, 1, "Query: ‘Phoscon 210mg’. Hybrid rank 1: Screenshot_20260815-062822_Somatec__1RR0EEAz.jpg, chunk page:a66ce32fc45cfd08:3, containing ‘Tab. Phoscon (210mg)’. Correct page/evidence. Final recall@5: 5/5 for E5, hash and hybrid on the labelled query set.")
    put(tables[9], 3, 1, "Screenshot_20260815-063516_Somatec.jpg: whole-page OCR stopped after the printed header and recovered 0/5 medications. Rx-body cropping recovered five approximate lines, showing the cause was layout/token-budget competition. Even cropped OCR misspelled strength/name characters, so the page remains human-review-only.")

    put(tables[10], 0, 1, "Implement retrieval/reranking, evidence-gated re-search (increase k on weak evidence, abstain at k_max), read-page/re-OCR tools, grounded answers with citations, prompt-injection controls, PII/HITL and evaluation. RL/RLVR remains optional unless the team explicitly chooses the bonus.")
    put(tables[10], 1, 1, "Author prescription-specific tasks with exact drug/strength queries, Bangla paraphrases, template-confounders, multi-page aggregation and deliberately absent medicines. Include verifiable expected page/chunk IDs and abstention cases; keep patient identity out of questions.")
    put(tables[10], 2, 1, "OCR is the dominant bottleneck: a fluent wrong drug name can be retrieved confidently. First mitigation: layout-crop re-OCR on weak/failed pages, lexical drug-name checks, image citation and mandatory human review for medication fields.")

    facets = [
        ("Turn permissioned handwritten prescriptions into searchable, grounded evidence; measure OCR CER/WER/F1 and retrieval recall while targeting PII entity recall ≥0.98.", "Fits the real workflow and keeps privacy as the unchanged A1 NFR. Index speed is reported as an operational result, not substituted for the NFR."),
        ("Permissioned Somatec prescriptions; 12-page/61-medication held-out OCR set; page-level split; restricted raw data supplied through the authorized project channel.", "Real code-switched handwriting is representative, but the OCR labels are small and the privacy target still requires a separately annotated PII holdout."),
        ("PaddleOCR-VL 1.6, multilingual E5-small, signed feature hashing and exact cosine stores.", "Pretraining handles mixed scripts without training from scratch; lexical vectors protect exact names. A domain-fine-tuned reader would replace OCR if enough full labels exist."),
        ("BF16/SDPA batch 16; selective retry; Rx crop; NFKC/script ID; 64/16 chunks; hybrid RRF.", "Each setting was measured. Crop improves recall; 64-word chunks improve E5 recall. Larger labels could justify retuning."),
        ("Typed Page→Region→Chunk contracts; cached OCR JSONL; PII scrub before two parallel vector indexes.", "Caching makes iteration fast and auditable. A larger corpus would justify a database/ANN store."),
        ("YAML configs, resumable runners, focused tests, JSON evidence, fixed held-out labels and notebook outputs.", "Reproducible and fast; full CI additionally needs the starter's serving dependencies installed."),
        ("RTX 4090 OCR; GPU-batched E5; CPU hashing and exact search; local restricted demo.", "OCR on the tested snapshot ran at 0.852 image/s; post-OCR index builds total 8.677 s. Public deployment is inappropriate until PII recall ≥0.98 is demonstrated."),
        ("Version Drive manifest/results/configs; monitor OCR caps, review rate, script mix, empty text, recall and corpus drift.", "These catch the observed failure modes. Add model registry and DVC when the corpus/version count grows."),
    ]
    for row, values in enumerate(facets, 1):
        put(tables[11], row, 1, values[0])
        put(tables[11], row, 2, values[1])

    put(tables[13], 7, 0, "transcripts/<student number>.txt  ×2")
    put(tables[13], 7, 1, "one genuine, full, unedited AI-chat transcript per member")

    temp = args.output.with_suffix(".tmp.docx")
    doc.save(temp)
    temp.replace(args.output)
    print(f"filled {args.output} with Project 11 identity and A1 carry-forward fields")


if __name__ == "__main__":
    main()
