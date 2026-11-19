# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.80: (d) full fusion — weighted, precision 1.00 at recall 0.83 (F1 0.91, FPR 0.00, accuracy 0.92, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.8); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.8); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `/kaggle/working/results/best_model.pt`; untrained fallback: False
- Models requested: rule_based

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.5000 | 0.00 | 96 |
| (b) text + OCR | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.5729 | 0.00 | 96 |
| (c) text + OCR + QR + PDF | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.7188 | 0.00 | 96 |
| (d) full fusion — weighted | 1.0000 | 0.0000 | 0.8333 | 0.9091 | 0.9167 | 0.9444 | 0.05 | 96 |
| (d) full fusion — max | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.7188 | 0.00 | 96 |

Same table at the placeholder threshold 0.5 (for reference only):

| ablation | precision | FPR | recall | F1 | accuracy |
|---|---:|---:|---:|---:|---:|
| (a) text-body only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (b) text + OCR | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (c) text + OCR + QR + PDF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (d) full fusion — weighted | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| (d) full fusion — max | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## Confusion matrices (test slice, fitted thresholds)

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[48, 0], [8, 40]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[0, 48], [0, 48]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 40 | 0.8333 |
| ocr | 19 | 11 | 0.5789 |
| pdf | 14 | 14 | 1.0000 |
| qr | 14 | 14 | 1.0000 |

## Failures (10 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `image_only_phishing_005.eml` | image_only_phishing | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (collect-example.net) differs from From domain (secure-notice-example.com) (+0.10) |
| `image_only_phishing_006.eml` | image_only_phishing | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (login-verify-example.xyz) differs from From domain (alerts-example.com) (+0.10) |
| `image_only_phishing_008.eml` | image_only_phishing | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (account-check-example.tk) differs from From domain (corp-auth-example.com) (+0.10) |
| `text_button_image_004.eml` | text_button_image | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (collect-example.net) differs from From domain (portal-example.com) (+0.10) |
| `text_button_image_005.eml` | text_button_image | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (account-check-example.tk) differs from From domain (portal-example.com) (+0.10) |
| `text_button_image_006.eml` | text_button_image | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (login-verify-example.xyz) differs from From domain (billing-example.com) (+0.10) |
| `text_button_image_007.eml` | text_button_image | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (collect-example.net) differs from From domain (alerts-example.com) (+0.10) |
| `text_button_image_008.eml` | text_button_image | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (login-verify-example.xyz) differs from From domain (billing-example.com) (+0.10) |
| `scanned_pdf_001.eml` | scanned_pdf | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (collect-example.net) differs from From domain (portal-example.com) (+0.10) |
| `scanned_pdf_003.eml` | scanned_pdf | 1 | 0 | 0.00/0.00/0.00/0.01/0.00 | missed phishing; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Reply-To domain (account-check-example.tk) differs from From domain (secure-notice-example.com) (+0.10) |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
