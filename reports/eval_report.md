# Evaluation report

## OCR quality

The manually labelled sample has 12 random pages and 61 medicine-name/strength entries. Scores cover only medicine lines.

| Method | CER ↓ | WER ↓ | Precision | Recall | Entity F1 ↑ |
|---|---:|---:|---:|---:|---:|
| PaddleOCR-VL, 768 tokens | 0.499 | 0.702 | 0.649 | 0.393 | 0.490 |
| Selective retry + trimming | 0.516 | 0.720 | 0.641 | 0.410 | 0.500 |
| Classical enhancement | 0.607 | 0.789 | 0.613 | 0.311 | 0.413 |
| Rx-layout crop + trimming | **0.413** | 0.702 | 0.587 | **0.443** | **0.505** |

Classical enhancement hurts faint handwriting. Cropping reduces header competition and recovered all five lines on the worst baseline failure; tail trimming removed its runaway generation. Crop OCR remains an on-demand recovery path because it omits page metadata.

## Index statistics

| Statistic | E5 | Hash |
|---|---:|---:|
| Documents / chunks | 310 / 943 | 310 / 943 |
| Words (source / with overlap) | 42,642 / 52,575 | 42,642 / 52,575 |
| Dimension / vector bytes | 384 / 1,448,448 | 2,048 / 7,725,056 |
| Build time after OCR | 7.304 s | 1.373 s |
| Bengali / Latin / mixed chunks | 288 / 902 / 276 | 288 / 902 / 276 |

## Retrieval and evidence

Chunk ablation gave E5 recall@5 of 1.00 at 64/16, 0.80 at 128/24 and 0.20 at 256/32. Final five-query recall@5 is 1.00 for E5, hashing and hybrid RRF.

- Success: `Phoscon 210mg` returns a chunk containing `Tab. Phoscon (210mg)` at hybrid rank 1.
- Failure: full-page OCR stopped after the header on `Screenshot_20260815-063516_Somatec.jpg`, extracting 0/5 visible medicines. Rx cropping recovered five approximate lines, but spelling/strength errors still require image-based review.

Raw evidence is in `reports/evidence/` and `artifacts/index_*/index_stats.json`.
