# Corpus provenance

- Source: handwritten prescription images supplied with permission from Somatec Pharmaceuticals Ltd.; the project owner delivered the working snapshot through Google Drive folder `1YmYjw444rIYjriDqzZ-fcrHe2Io-F28S`.
- Permission and channel: the data was obtained with proper authorization and through the proper organizational/project channel for academic research. The Drive link is a delivery mechanism, not an open-content licence. Raw prescriptions must remain access-controlled and must not be redistributed unless the permission holder gives written authorization.
- Snapshot: 310 valid JPEG pages, approximately 82 MiB; 285 original names and 25 duplicate-name pairs preserved using Drive-ID suffixes.
- Extracted/indexed coverage: 310/310 pages, 42,642 non-overlapped OCR words; the selected 64/16 configuration produces 943 chunks and 52,575 indexed word occurrences including overlap. This is below the course's 60,000 usable-word floor.
- Difficulty: photographed outpatient prescriptions; Bangla/English code-switching, physician handwriting, printed templates, shadows, perspective, Bengali numerals and medical abbreviations.
- Sensitive data: prescriptions contain patient identifiers and health information. Names, telephone numbers and email addresses are redacted before indexing; raw images/OCR require restricted access.
- Split policy: the 12 files in `grading_kit/heldout_pages/` are the fixed evaluation split and were not used for training or prompt selection. The remaining 298 pages are corpus pages. Duplicate Drive IDs remain in the same corpus side; future training must group perceptual duplicates before splitting.
- Reproduction: `bash scripts/get_data.sh`; downloader preserves a manifest and checks that every output is a valid image.
