# Multimodal evaluation (Section E)

> SYNTHETIC evaluation set: generated images/PDFs from a small set of templates with simulated OCR stressors (font/size variation, low contrast, rotation, noise, JPEG compression). No real-world adversary variation and no real emails. Results measure pipeline behaviour on these templates only and do NOT generalise to real phishing.

Best precision at recall floor 0.85: (d) full fusion — weighted, precision 0.75 at recall 1.00 (F1 0.86, FPR 0.33, accuracy 0.83, n=96).

## Setup

- Eval set: `/kaggle/working/repo/data/multimodal_eval` (seed 42, 120 emails: 60 phishing / 60 benign; validation 24 / test 96)
- Thresholds: fitted on the validation slice only (precision-first, recall floor 0.85); placeholder 0.5 metrics are shown alongside. Thresholds were never fitted on test. Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted on the validation slice only (precision-first, recall floor 0.85); never fitted on test.
- Primary fusion: `weighted`; checkpoint: `/kaggle/working/results/best_model.pt`; untrained fallback: False
- Models requested: rule_based

## Ablation table (test slice, fitted thresholds) — precision first

| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) text-body only | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.5000 | 0.00 | 96 |
| (b) text + OCR | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.5729 | 0.00 | 96 |
| (c) text + OCR + QR + PDF | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 0.7188 | 0.00 | 96 |
| (d) full fusion — weighted | 0.7500 | 0.3333 | 1.0000 | 0.8571 | 0.8333 | 0.9444 | 0.01 | 96 |
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

- (d) full fusion — weighted: `[[tn, fp], [fn, tp]]` = [[32, 16], [0, 48]]
- (d) full fusion — max: `[[tn, fp], [fn, tp]]` = [[0, 48], [0, 48]]

## Per-modality recall (phishing emails carrying that modality, weighted fusion)

| modality | phishing emails with it | caught | recall |
|---|---:|---:|---:|
| body | 48 | 48 | 1.0000 |
| ocr | 19 | 19 | 1.0000 |
| pdf | 14 | 14 | 1.0000 |
| qr | 14 | 14 | 1.0000 |

## Failures (20 misclassified on the full set)

| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |
|---|---|---|---|---|---|
| `newsletter_logo_qr_001.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_002.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://billing.company.org/pay' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_003.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_004.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://portal.company.org/invoices' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_005.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_006.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_007.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://portal.company.org/invoices' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_008.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://portal.company.org/invoices' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_009.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://events.company.org/spring-fair' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_010.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://events.company.org/spring-fair' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_011.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://events.company.org/spring-fair' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_012.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_013.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://billing.company.org/pay' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_014.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://events.company.org/spring-fair' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_015.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://events.company.org/spring-fair' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_016.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://billing.company.org/pay' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_017.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_018.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://billing.company.org/pay' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_019.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://portal.company.org/invoices' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |
| `newsletter_logo_qr_020.eml` | newsletter_logo_qr | 0 | 1 | 0.00/0.00/0.00/0.02/0.00 | false alarm on benign mail; benign QR/URL present; context P(phish)=0.00; Concatenated context (subject+body+OCR+PDF+QR, scored) scored P(phishing)=0.00.; Riskiest URL 'https://intranet.company.org/agenda' scored 0.10: URL appeared only inside a QR code, hiding the link from the reader. |

## Figures

- `confusion_matrix_weighted.png`
- `ablation_precision_recall_fpr.png`

## What this does and does not show

- The set is synthetic (see the warning at the top): template images/PDFs, simulated OCR stressors, no real adversary variation. Treat every number as a pipeline check on these templates, not as real-world detection performance.
- Thresholds were fitted on 24 validation emails only; with so few points the fitted threshold is coarse and should be re-fitted on a larger real validation set before any deployment claim.
- The 0.5 threshold is a placeholder; fitted-threshold rows are the reported ones.
