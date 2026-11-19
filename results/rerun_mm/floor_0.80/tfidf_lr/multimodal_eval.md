# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.80: (d) full fusion — max, precision 1.00 at recall 1.00 (F1 1.00, FPR 0.00, accuracy 1.00, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.8); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.8); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `/kaggle/working/results/best_model.pt`; untrained fallback: False
- Models requested: tfidf_lr

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 1.0000 | 0.0000 | 0.7292 | 0.8434 | 0.8646 | 0.8138 | 0.50 | 96 |
| (b) text + OCR | 0.9750 | 0.0208 | 0.8125 | 0.8864 | 0.8958 | 0.8911 | 0.48 | 96 |
| (c) text + OCR + QR + PDF | 1.0000 | 0.0000 | 0.8125 | 0.8966 | 0.9062 | 0.9722 | 0.55 | 96 |
| (d) full fusion — weighted | 0.9412 | 0.0625 | 1.0000 | 0.9697 | 0.9688 | 1.0000 | 0.35 | 96 |
| (d) full fusion — max | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.60 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 1.0000 | 0.0000 | 0.7292 | 0.8434 | 0.8646 |
| (b) text + OCR | 1.0000 | 0.0000 | 0.7917 | 0.8837 | 0.8958 |
| (c) text + OCR + QR + PDF | 0.9574 | 0.0417 | 0.9375 | 0.9474 | 0.9479 |
| (d) full fusion — weighted | 1.0000 | 0.0000 | 0.7292 | 0.8434 | 0.8646 |
| (d) full fusion — max | 0.6857 | 0.4583 | 1.0000 | 0.8136 | 0.7708 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[45, 3], [0, 48]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[48, 0], [0, 48]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 48 | 1.0000 |
| ocr | 19 | 19 | 1.0000 |
| pdf | 14 | 14 | 1.0000 |
| qr | 14 | 14 | 1.0000 |

## Failures (3 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `newsletter_logo_qr_002.eml` | newsletter_logo_qr | 0 | 1 | 0.46/0.46/0.47/0.35/0.60 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.47; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.60.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.47. |
| `newsletter_logo_qr_013.eml` | newsletter_logo_qr | 0 | 1 | 0.46/0.46/0.47/0.35/0.60 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.47; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.60.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.47. |
| `newsletter_logo_qr_016.eml` | newsletter_logo_qr | 0 | 1 | 0.47/0.48/0.48/0.36/0.60 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.48; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.60.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.48. |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
