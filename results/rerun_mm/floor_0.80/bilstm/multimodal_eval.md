# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.80: (d) full fusion — weighted, precision 1.00 at recall 1.00 (F1 1.00, FPR 0.00, accuracy 1.00, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.8); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.8); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `results/bilstm/best_bilstm.pt`; untrained fallback: False
- Models requested: bilstm

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 0.9796 | 0.0208 | 1.0000 | 0.9897 | 0.9896 | 1.0000 | 0.55 | 96 |
| (b) text + OCR | 0.8889 | 0.1250 | 1.0000 | 0.9412 | 0.9375 | 1.0000 | 0.95 | 96 |
| (c) text + OCR + QR + PDF | 0.8136 | 0.2292 | 1.0000 | 0.8972 | 0.8854 | 1.0000 | 0.99 | 96 |
| (d) full fusion — weighted | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.66 | 96 |
| (d) full fusion — max | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.4032 | 0.05 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 0.8727 | 0.1458 | 1.0000 | 0.9320 | 0.9271 |
| (b) text + OCR | 0.7500 | 0.3333 | 1.0000 | 0.8571 | 0.8333 |
| (c) text + OCR + QR + PDF | 0.5926 | 0.6875 | 1.0000 | 0.7442 | 0.6562 |
| (d) full fusion — weighted | 0.5926 | 0.6875 | 1.0000 | 0.7442 | 0.6562 |
| (d) full fusion — max | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[48, 0], [0, 48]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[0, 48], [0, 48]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 48 | 1.0000 |
| ocr | 19 | 19 | 1.0000 |
| pdf | 14 | 14 | 1.0000 |
| qr | 14 | 14 | 1.0000 |

## Failures (0 misclassified on the full set)

No misclassifications on the evaluated set.

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
