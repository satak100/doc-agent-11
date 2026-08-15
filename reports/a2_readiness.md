# A2 readiness audit

Audit basis: `forms/A2_form.docx`, `SUBMISSION.md`, `handbook/03-Project-Specification.md`, and the A2 checklist in `handbook/05-Codebase-Guide.md`.

## Complete and measured

- A2 core stages: ingest, optional preprocessing, heading/body layout, pretrained PaddleOCR-VL adapter, chunking, embeddings, vector store and retrieval.
- Data-speciality E23: Unicode normalization, Bangla/Latin chunk-level script ID, multilingual E5 embeddings.
- Primary NFR carried forward from A1: privacy, with target PII entity recall ≥0.98. Pre-index redaction and restricted-data handling are implemented.
- Privacy experiments: the regex-only scan found zero unresolved known-pattern matches, but a manual 12-page/12-patient-name holdout found two leaked names. Explicit detection/redaction recall was 0.0833, conditional redaction recall 0.3333, and end-to-end non-disclosure 0.8333. The NFR target fails honestly.
- Operational performance (not the declared NFR): final post-OCR E5 + hash builds total 8.677 seconds for the tested snapshot.
- Published/pretrained adaptation: PaddleOCR-VL 1.6 for OCR and multilingual E5-small for embeddings.
- Experiments: four OCR/preprocessing/layout variants, three chunk configurations, two embedding backends and hybrid fusion.
- Fixed held-out set: 12 real images and 61 medication labels in the official grading paths.
- Evidence: CER/WER/F1, index statistics, retrieval recall@5, rank-1 success and a real failure.
- A1 carry-forward technical artifacts: reproducible downloader, provenance, executed EDA notebook, manifest and held-out data.
- A2 artifacts: built indexes, executable build script, executed KB notebook, diagram, design table and technically filled form.
- Reproducibility: generated 126-package `requirements.lock`.

## Must be completed by the team before submission

1. **Privacy remediation and expanded evaluation:** handle clipped and Bangla/OCR-corrupted patient-name labels, rebuild both indexes, then expand the manual holdout beyond names and rerun. Current pilot fails the ≥0.98 target.
2. **Transcripts:** add exactly two `transcripts/<student-number>.txt` files, each containing that member's actual full, unedited AI conversation and reflection. Templates are supplied separately; fabricated conversations are not valid evidence.
3. **Fine-tuning decision:** A1 proposed fine-tuning TrOCR, while the measured A2 baseline adapts pretrained PaddleOCR-VL without weight updates. Either run a small honest fine-tuning experiment with a train/heldout split or obtain instructor confirmation that the reproduced/adapted pretrained baseline satisfies A2.
4. **GitHub submission:** this folder is not a Git repository. As requested, repository creation, member-attributed commits, push and the `a2-submit` tag are deferred until later.

## Non-blocking cautions

- The OCR label scope is medication name/strength, not a full-page transcription. The form and notebook state this explicitly.
- The best crop result is still only F1 0.505; it is a recovery/search aid, not clinically reliable OCR.
- Enhancement is optional and was correctly rejected by evidence rather than included for appearance.
