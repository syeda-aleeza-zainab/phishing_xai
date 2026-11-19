# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.80: (d) full fusion — max, precision 1.00 at recall 0.83 (F1 0.91, FPR 0.00, accuracy 0.92, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.8); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.8); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `/kaggle/working/results/best_model.pt`; untrained fallback: False
- Models requested: tfidf_svm

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 0.9333 | 0.0417 | 0.5833 | 0.7179 | 0.7708 | 0.8585 | 0.85 | 96 |
| (b) text + OCR | 0.9444 | 0.0417 | 0.7083 | 0.8095 | 0.8333 | 0.8945 | 0.85 | 96 |
| (c) text + OCR + QR + PDF | 0.8837 | 0.1042 | 0.7917 | 0.8352 | 0.8438 | 0.9214 | 0.71 | 96 |
| (d) full fusion — weighted | 0.9556 | 0.0417 | 0.8958 | 0.9247 | 0.9271 | 0.9683 | 0.50 | 96 |
| (d) full fusion — max | 1.0000 | 0.0000 | 0.8333 | 0.9091 | 0.9167 | 0.9484 | 0.95 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 0.7069 | 0.3542 | 0.8542 | 0.7736 | 0.7500 |
| (b) text + OCR | 0.7167 | 0.3542 | 0.8958 | 0.7963 | 0.7708 |
| (c) text + OCR + QR + PDF | 0.7458 | 0.3125 | 0.9167 | 0.8224 | 0.8021 |
| (d) full fusion — weighted | 0.9556 | 0.0417 | 0.8958 | 0.9247 | 0.9271 |
| (d) full fusion — max | 0.6154 | 0.6250 | 1.0000 | 0.7619 | 0.6875 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[46, 2], [5, 43]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[48, 0], [8, 40]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 43 | 0.8958 |
| ocr | 19 | 19 | 1.0000 |
| pdf | 14 | 14 | 1.0000 |
| qr | 14 | 9 | 0.6429 |

## Failures (8 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `qr_body_image_002.eml` | qr_body_image | 1 | 0 | 0.69/0.17/0.09/0.28/0.71 | missed phishing; context P(phish)=0.09; Highest-scoring part 'body_text' (kind body_text) scored P(phishing)=0.71.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.09. |
| `qr_body_image_003.eml` | qr_body_image | 1 | 0 | 0.69/0.30/0.20/0.33/0.71 | missed phishing; context P(phish)=0.20; Highest-scoring part 'body_text' (kind body_text) scored P(phishing)=0.71.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.20. |
| `qr_body_image_004.eml` | qr_body_image | 1 | 0 | 0.69/0.43/0.35/0.40/0.71 | missed phishing; context P(phish)=0.35; Highest-scoring part 'body_text' (kind body_text) scored P(phishing)=0.71.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.35. |
| `qr_body_image_006.eml` | qr_body_image | 1 | 0 | 0.69/0.30/0.20/0.33/0.71 | missed phishing; context P(phish)=0.20; Highest-scoring part 'body_text' (kind body_text) scored P(phishing)=0.71.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.20. |
| `qr_body_image_007.eml` | qr_body_image | 1 | 0 | 0.69/0.57/0.52/0.47/0.71 | missed phishing; context P(phish)=0.52; Highest-scoring part 'body_text' (kind body_text) scored P(phishing)=0.71.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.52. |
| `qr_body_image_008.eml` | qr_body_image | 1 | 0 | 0.69/0.30/0.20/0.33/0.71 | missed phishing; context P(phish)=0.20; Highest-scoring part 'body_text' (kind body_text) scored P(phishing)=0.71.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.20. |
| `real_invoice_006.eml` | real_invoice | 0 | 1 | 0.81/0.81/0.78/0.50/0.76 | false alarm on benign mail; context P(phish)=0.78; Highest-scoring part 'pdf_text:invoice.pdf#p1' (kind pdf_text) scored P(phishing)=0.76.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.78. |
| `real_invoice_017.eml` | real_invoice | 0 | 1 | 0.88/0.88/0.77/0.50/0.77 | false alarm on benign mail; context P(phish)=0.77; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.77.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.77. |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
