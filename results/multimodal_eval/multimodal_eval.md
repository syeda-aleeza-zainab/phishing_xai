# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

> **UNTRAINED MODEL — results are NOT detection evidence: the fine-tuned checkpoint was missing, so every score below comes from untrained base roberta-base (pipeline-demonstration output only).**

Best precision at recall floor 0.80: (d) full fusion — weighted, precision 1.00 at recall 0.75 (F1 0.86, FPR 0.00, accuracy 0.88, n=96).

## Setup

- Eval set: `./data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.8); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.8); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `./results/best_model.pt`; untrained fallback: True
- Models requested: roberta, tfidf_lr

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.4245 | 0.05 | 96 |
| (b) text + OCR | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.4679 | 0.05 | 96 |
| (c) text + OCR + QR + PDF | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.2188 | 0.05 | 96 |
| (d) full fusion — weighted | 1.0000 | 0.0000 | 0.7500 | 0.8571 | 0.8750 | 0.9722 | 0.34 | 96 |
| (d) full fusion — max | 0.5614 | 0.5208 | 0.6667 | 0.6095 | 0.5729 | 0.5794 | 0.44 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (b) text + OCR | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (c) text + OCR + QR + PDF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (d) full fusion — weighted | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (d) full fusion — max | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[48, 0], [12, 36]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[23, 25], [16, 32]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 36 | 0.7500 |
| ocr | 19 | 11 | 0.5789 |
| pdf | 14 | 11 | 0.7857 |
| qr | 14 | 12 | 0.8571 |

## Cheap baselines on the same extractor output (Section H hook)

Fitted on the real-data train split; evaluated on the synthetic eval set's combined extractor text (subject+body+OCR+PDF+QR).

| model | precision | FPR | recall | F1 | accuracy | ROC-AUC | status |
|---|---:|---:|---:|---:|---:|---:|---|
| tfidf_lr | 0.9500 | 0.0500 | 0.9500 | 0.9500 | 0.9500 | 0.9875 | ok |

Per-model modality ablation (F1 on that modality's text only):

| model | modality | F1 | accuracy | status |
|---|---|---:|---:|---|
| tfidf_lr | all | 0.9500 | 0.9500 | ok |
| tfidf_lr | body_text | 0.7879 | 0.8250 | ok |
| tfidf_lr | ocr_text | 0.8276 | 0.7917 | ok |
| tfidf_lr | pdf_text | 0.7895 | 0.7333 | ok |
| tfidf_lr | qr_text | 0.7317 | 0.6333 | ok |

## Failures (14 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `image_only_phishing_001.eml` | image_only_phishing | 1 | 0 | 0.44/0.43/0.43/0.34/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `image_only_phishing_006.eml` | image_only_phishing | 1 | 0 | 0.44/0.43/0.43/0.32/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `image_only_phishing_008.eml` | image_only_phishing | 1 | 0 | 0.44/0.44/0.44/0.30/0.44 | missed phishing; context P(phish)=0.44; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.44. |
| `text_button_image_002.eml` | text_button_image | 1 | 0 | 0.43/0.43/0.43/0.34/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `text_button_image_004.eml` | text_button_image | 1 | 0 | 0.43/0.43/0.43/0.30/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `text_button_image_005.eml` | text_button_image | 1 | 0 | 0.43/0.43/0.43/0.30/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `text_button_image_006.eml` | text_button_image | 1 | 0 | 0.43/0.44/0.44/0.30/0.44 | missed phishing; context P(phish)=0.44; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.44. |
| `text_button_image_007.eml` | text_button_image | 1 | 0 | 0.43/0.43/0.43/0.30/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `text_button_image_008.eml` | text_button_image | 1 | 0 | 0.43/0.43/0.43/0.30/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'image:verify_button.png' (kind image) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `qr_in_text_pdf_002.eml` | qr_in_text_pdf | 1 | 0 | 0.44/0.44/0.44/0.34/0.44 | missed phishing; context P(phish)=0.44; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.44. |
| `qr_in_text_pdf_003.eml` | qr_in_text_pdf | 1 | 0 | 0.44/0.44/0.44/0.34/0.44 | missed phishing; context P(phish)=0.44; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.44. |
| `scanned_pdf_001.eml` | scanned_pdf | 1 | 0 | 0.43/0.43/0.43/0.32/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `html_link_mismatch_005.eml` | html_link_mismatch | 1 | 0 | 0.43/0.43/0.43/0.34/0.44 | missed phishing; context P(phish)=0.43; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.43. |
| `fake_invoice_pdf_004.eml` | fake_invoice_pdf | 1 | 0 | 0.44/0.44/0.44/0.32/0.44 | missed phishing; context P(phish)=0.44; Highest-scoring part 'subject' (kind subject) scored P(phishing)=0.44.; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.44. |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
