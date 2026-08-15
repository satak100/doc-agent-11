# Design choices — prescription knowledge base

The project is **Pharmacological Query from Prescription** (Project ID 11), in the healthcare/pharmacological-information domain. Its speciality is handwritten medical prescriptions and its primary NFR, carried forward unchanged from A1, is privacy: PII entity recall must be at least 0.98. The safety boundary is strict: this system supports search and review; it must not prescribe, dispense, or silently correct a drug name.

| Stage | Problem statement | Data | Model | Methods | Design | Development | Deployment | MLOps |
|---|---|---|---|---|---|---|---|---|
| 0 Frame | Make handwriting searchable without treating OCR as clinical truth | Permissioned prescription images; duplicate names preserved | — | Stable IDs | Page is leakage boundary | Fixed manual audit | Local/4090 | Immutable JSONL |
| 1 Ingest | Load phone photos quickly | EXIF JPEGs, shadows | Paddle normalization + optional PIL enhancement | Stable IDs; compare raw vs contrast/median/sharpen | Raw selected because enhancement reduced F1 to 0.413 | Path overrides, caching | CPU | Fail on missing data |
| 2 Layout | Retain reading order | Printed header + handwritten Rx | Heading/body heuristic + Paddle polygons | Compare full page with OCR-anchored Rx crop | Crop+trim improved CER/F1 to 0.413/0.505; use for failed-page re-OCR | PIL dimensions | Cached | Region/review counts |
| 3 OCR | Read Bangla/English handwriting | 12 labelled pages, 61 medicines | **PaddlePaddle/PaddleOCR-VL-1.6** | BF16, SDPA, batch 16, selective retry and tail trim | Published pretrained method adapted to prescriptions | Resumable/OOM fallback | RTX 4090 | 310/310 success; review flags |
| 4 Index | Keep medicine with dose context | 42,642 source OCR words | E5-small + feature hashing | NFKC, script ID, 64 words/16 overlap | Ablation selected 64/16; exact flat search for 943 chunks | Cached vectors | GPU/CPU | JSON stats |
| 5 Retrieval | Exact drugs plus semantic queries | Five labelled queries | multilingual E5 + lexical hash | Reciprocal-rank fusion | Final E5/hash/hybrid recall@5 all 1.00 | Cached model | Local CLI | Stored eval |
| 6 Agent | Prevent unsupported claims and disclosure | Retrieved OCR chunks | Existing agent contract | Widen k then abstain | Evidence is not medical advice | PII scrub before embedding | Human confirmation; restricted access | PII recall target ≥0.98; tool traces |
| 7 RL/RLVR | Outside current A2 scope | — | — | Interface retained | No unmeasured claims | Future | Disabled | — |
| 8 Serving | Reproducible demo | Two indexes | Local CLI/notebook | Cached inference | No public patient API | `query_hybrid.py` | Local | Deterministic configs |
| 9 Eval | Expose success and failure | 12 pages/61 medicines; 5 queries | — | CER/WER/F1 and recall@5 | Explicit metric scope | Method comparison | Evidence JSON | Failure artifacts |

## Method comparison and selection

OCR A is the original 768-token PaddleOCR-VL spotting pass. OCR B retries capped pages at 1,536 tokens and trims repetitive tails. B raises entity F1 from 0.490 to 0.500 but worsens CER/WER to 0.516/0.720. Classical enhancement is rejected because F1 falls to 0.413. OCR-anchored Rx cropping plus trimming gives the best CER (0.413), recall (0.443) and F1 (0.505), so it becomes an on-demand recovery method while the full-page result preserves document metadata for indexing.

Chunk ablation compared 64/16, 128/24 and 256/32. E5 recall@5 was respectively 1.00, 0.80 and 0.20; hybrid remained 1.00 and mean hybrid query latency stayed about 2.7 ms. Therefore 64/16 is selected. The final E5 and lexical indexes build in 7.304 s and 1.373 s after OCR. Hybrid RRF is selected because it needs no cross-model score calibration and retains both multilingual and exact-name candidates.

Privacy is the declared NFR, not throughput. The implementation redacts detected patient names, Bangladesh-format phone numbers and email addresses before chunking and embedding, keeps raw images out of the vector stores, and assumes restricted local access. The broad regex-only scan found zero unresolved matches, but manual visual review exposed that result as insufficient. On 12 held-out pages containing 12 patient-name entities, one was explicitly redacted, two OCR-extracted names leaked into the index, and nine names were omitted by OCR. Explicit detection/redaction recall was 0.0833, conditional redaction recall was 0.3333, and end-to-end non-disclosure was 0.8333. The A1 target ≥0.98 therefore **fails** on this pilot. Index-build time remains an operational measurement only.

## Limitations

- OCR metrics cover medicine-name/strength lines, not full-page transcription.
- Twelve pages and five queries are small evidence sets.
- Corpus-size reconciliation is intentionally deferred per the team's current instruction; measured snapshot counts remain in machine-readable evidence.
- Names, phones and emails are redacted; the index still requires access control.
- Every medication, strength, schedule and duration requires image-based human verification.
