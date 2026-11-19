# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.80: (d) full fusion — weighted, precision 1.00 at recall 0.98 (F1 0.99, FPR 0.00, accuracy 0.99, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.8); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.8); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `results/best_model.pt`; untrained fallback: False
- Models requested: roberta

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 0.8605 | 0.1250 | 0.7708 | 0.8132 | 0.8229 | 0.9019 | 0.24 | 96 |
| (b) text + OCR | 0.8776 | 0.1250 | 0.8958 | 0.8866 | 0.8854 | 0.9748 | 0.24 | 96 |
| (c) text + OCR + QR + PDF | 0.9792 | 0.0208 | 0.9792 | 0.9792 | 0.9792 | 0.9983 | 0.81 | 96 |
| (d) full fusion — weighted | 1.0000 | 0.0000 | 0.9792 | 0.9895 | 0.9896 | 0.9991 | 0.60 | 96 |
| (d) full fusion — max | 0.9130 | 0.0833 | 0.8750 | 0.8936 | 0.8958 | 0.9516 | 1.00 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 1.0000 | 0.0000 | 0.7083 | 0.8293 | 0.8542 |
| (b) text + OCR | 1.0000 | 0.0000 | 0.8542 | 0.9213 | 0.9271 |
| (c) text + OCR + QR + PDF | 0.8571 | 0.1667 | 1.0000 | 0.9231 | 0.9167 |
| (d) full fusion — weighted | 0.9057 | 0.1042 | 1.0000 | 0.9505 | 0.9479 |
| (d) full fusion — max | 0.6000 | 0.6667 | 1.0000 | 0.7500 | 0.6667 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[48, 0], [1, 47]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[44, 4], [6, 42]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 47 | 0.9792 |
| ocr | 19 | 18 | 0.9474 |
| pdf | 14 | 14 | 1.0000 |
| qr | 14 | 14 | 1.0000 |

## Failures (1 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `scanned_pdf_001.eml` | scanned_pdf | 1 | 0 | 0.00/0.73/0.73/0.54/1.00 | missed phishing; fused score reached threshold but lacked corroboration; Highest-scoring part 'pdf_image:scanned_notice.pdf#img6#p1' (kind pdf_image) scored P(phishing)=1.00.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.73. |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
