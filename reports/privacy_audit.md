# Privacy leakage audit

The A1 primary NFR is privacy with an entity-level PII recall target of at least 0.98. The pipeline redacts detector-visible patient-name fields, Bangladesh-format telephone numbers and email addresses before chunking and embedding.

Run:

```bash
PYTHONPATH=src python scripts/audit_index_pii.py
```

The machine-readable result is `artifacts/privacy_audit.json`. This scan compares final raw OCR text with the E5 index's chunk text and reports unresolved detector matches without printing sensitive values.

Measured on the current tested snapshot, the detector found 20 email, 226 patient-name-field and 330 telephone occurrences in 310 raw OCR records. The 943 final E5 chunks contained zero unresolved matches for those same detector patterns. Because overlapping chunks can repeat content, these are occurrence counts, not unique-person counts.

This is a leakage regression test, **not a recall experiment**. A detector cannot count PII it failed to recognize, and OCR may itself omit or corrupt an identifier. Therefore passing this scan does not establish the ≥0.98 NFR. The remaining experiment is to annotate all relevant entity types—patient names, phone/email, patient or registration IDs, full addresses, and signatures—on a restricted page-level holdout and score end-to-end redaction against those labels.

## Manual held-out pilot

All 12 fixed held-out images were visually inspected without copying identifier values into the labels. Scope was patient direct identifiers visible on these pages; each page contained one patient-name entity, for 12 entities total. Public clinician/chamber contact information, age/date and clinical content were excluded from this narrowly defined pilot.

Comparison against the final E5 chunk text produced these outcomes: one name was explicitly redacted, two OCR-extracted names remained in the index, and nine names were omitted by OCR. Therefore:

- explicit detection-and-redaction recall: **1/12 = 0.0833**;
- redaction recall conditional on OCR having extracted the name: **1/3 = 0.3333**;
- end-to-end non-disclosure rate: **10/12 = 0.8333**;
- end-to-end leak rate: **2/12 = 0.1667**.

The declared ≥0.98 PII-recall target **fails** on this pilot. OCR omission is counted as non-disclosure but not as successful PII detection. The main failures are a left-cropped English `Name` label and a Bengali/OCR-corrupted name label that the English-only line rule did not recognize. Labels and reproducible scoring are in `grading_kit/pii_holdout_labels.jsonl` and `scripts/eval_pii_holdout.py`. This is a single-reviewer, patient-name-only pilot; a final privacy claim still requires more entity types and preferably independent label review.
