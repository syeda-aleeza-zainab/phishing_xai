# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.85: (d) full fusion — weighted, precision 1.00 at recall 0.94 (F1 0.97, FPR 0.00, accuracy 0.97, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.85); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.85); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `results/distilbert/seed_42/best_distilbert.pt`; untrained fallback: False
- Models requested: distilbert

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 1.0000 | 0.0000 | 0.7500 | 0.8571 | 0.8750 | 0.9748 | 0.95 | 96 |
| (b) text + OCR | 1.0000 | 0.0000 | 0.8333 | 0.9091 | 0.9167 | 0.9848 | 0.95 | 96 |
| (c) text + OCR + QR + PDF | 0.9714 | 0.0208 | 0.7083 | 0.8193 | 0.8438 | 0.9809 | 0.96 | 96 |
| (d) full fusion — weighted | 1.0000 | 0.0000 | 0.9375 | 0.9677 | 0.9688 | 0.9996 | 0.65 | 96 |
| (d) full fusion — max | 0.6316 | 0.5833 | 1.0000 | 0.7742 | 0.7083 | 0.7274 | 0.65 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 0.8136 | 0.2292 | 1.0000 | 0.8972 | 0.8854 |
| (b) text + OCR | 0.8136 | 0.2292 | 1.0000 | 0.8972 | 0.8854 |
| (c) text + OCR + QR + PDF | 0.8421 | 0.1875 | 1.0000 | 0.9143 | 0.9062 |
| (d) full fusion — weighted | 0.8727 | 0.1458 | 1.0000 | 0.9320 | 0.9271 |
| (d) full fusion — max | 0.6316 | 0.5833 | 1.0000 | 0.7742 | 0.7083 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[48, 0], [3, 45]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[20, 28], [0, 48]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 45 | 0.9375 |
| ocr | 19 | 17 | 0.8947 |
| pdf | 14 | 13 | 0.9286 |
| qr | 14 | 13 | 0.9286 |

## Failures (3 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `text_button_image_004.eml` | text_button_image | 1 | 0 | 0.97/0.99/0.99/0.65/0.95 | missed phishing; fused score reached threshold but lacked corroboration; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.95.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.99. |
| `text_button_image_008.eml` | text_button_image | 1 | 0 | 0.97/0.99/0.99/0.65/0.95 | missed phishing; fused score reached threshold but lacked corroboration; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.95.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.99. |
| `qr_in_text_pdf_002.eml` | qr_in_text_pdf | 1 | 0 | 0.84/0.84/0.85/0.63/0.98 | missed phishing; fused score reached threshold but lacked corroboration; Highest-scoring part 'pdf_text:verification.pdf#p1' (kind pdf_text) scored P(phishing)=0.98.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.85. |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
