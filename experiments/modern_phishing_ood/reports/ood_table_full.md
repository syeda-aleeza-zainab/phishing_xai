# Task 2a — Full OOD table: 5 models x 6 sets (threshold 0.5)

Recomputed locally from the 30 saved per-email prediction CSVs in `experiments/modern_phishing_ood/predictions/` with the repo's own `phishing_xai.baselines.compute_metrics_full` (threshold 0.5). Precision@recall 0.90 is the repo `precision_at_recall` (max precision over thresholds with recall >= 0.90). Machine-readable version: `../results/ood_table_full.json`.

Cross-check: all 270 cells (9 metrics x 30 model/set pairs) recomputed against the saved per-set metrics JSONs in `experiments/modern_phishing_ood/results/` — **max absolute difference 0.0, zero mismatches**. Every number the task brief supplied for verification (primary 4-model P/R/F1/ROC-AUC/PR-AUC; Frontiers LLM recalls; Frontiers human LR vs RoBERTa recall; BEC SVC vs RoBERTa recall; PhishFuzzer LR best R/F1) matches the recomputation exactly. DistilBERT rows were computed the same way from its 6 saved prediction CSVs (Kaggle Kernel 4, seed-42 checkpoint).

Column key: P = precision, R = recall, F1, ROC = ROC-AUC, PR = PR-AUC (average precision), P@R.90 = precision at recall 0.90, FPR = false-positive rate. n_phish / n_legit are the set's class counts.

## PhishNChips (primary) — n=2000, n_phish=1000, n_legit=1000

| Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.9333 | 0.0280 | 0.0544 | 0.7787 | 0.7947 | 0.5636 | 0.0020 |
| BiLSTM | 0.8594 | 0.0550 | 0.1034 | 0.6801 | 0.6768 | 0.5263 | 0.0090 |
| TF-IDF+SVC | 0.9000 | 0.0810 | 0.1486 | 0.6604 | 0.6724 | 0.5441 | 0.0090 |
| TF-IDF+LR | 0.9788 | 0.1850 | 0.3112 | 0.7528 | 0.7771 | 0.5714 | 0.0040 |
| DistilBERT | 1.0000 | 0.0140 | 0.0276 | 0.8222 | 0.8250 | 0.6348 | 0.0000 |

## Frontiers human — n=6000, n_phish=5000, n_legit=1000

| Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.9995 | 0.8402 | 0.9130 | 0.9902 | 0.9981 | 0.9987 | 0.0020 |
| BiLSTM | 0.9975 | 0.7226 | 0.8381 | 0.9501 | 0.9901 | 0.9734 | 0.0090 |
| TF-IDF+SVC | 0.9978 | 0.8214 | 0.9011 | 0.9699 | 0.9940 | 0.9802 | 0.0090 |
| TF-IDF+LR | 0.9990 | 0.8400 | 0.9126 | 0.9823 | 0.9965 | 0.9901 | 0.0040 |
| DistilBERT | 1.0000 | 0.8572 | 0.9231 | 0.9971 | 0.9994 | 1.0000 | 0.0000 |

## Frontiers LLM — n=5986, n_phish=4986, n_legit=1000

| Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.9996 | 0.9394 | 0.9686 | 0.9994 | 0.9998 | 0.9998 | 0.0020 |
| BiLSTM | 0.9962 | 0.4787 | 0.6467 | 0.9542 | 0.9883 | 0.9720 | 0.0090 |
| TF-IDF+SVC | 0.9971 | 0.6195 | 0.7642 | 0.9379 | 0.9872 | 0.9584 | 0.0090 |
| TF-IDF+LR | 0.9990 | 0.8187 | 0.8999 | 0.9748 | 0.9951 | 0.9881 | 0.0040 |
| DistilBERT | 1.0000 | 0.7669 | 0.8681 | 0.9988 | 0.9998 | 0.9996 | 0.0000 |

## BEC — n=5181, n_phish=4181, n_legit=1000

| Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.9987 | 0.3772 | 0.5476 | 0.9513 | 0.9880 | 0.9719 | 0.0020 |
| BiLSTM | 0.9899 | 0.2102 | 0.3468 | 0.8320 | 0.9518 | 0.8774 | 0.0090 |
| TF-IDF+SVC | 0.9945 | 0.3882 | 0.5584 | 0.9001 | 0.9738 | 0.9203 | 0.0090 |
| TF-IDF+LR | 0.9970 | 0.3164 | 0.4804 | 0.9067 | 0.9747 | 0.9285 | 0.0040 |
| DistilBERT | 1.0000 | 0.1502 | 0.2612 | 0.9560 | 0.9891 | 0.9706 | 0.0000 |

## BEC-poisoned — n=5181, n_phish=4181, n_legit=1000

| Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.9989 | 0.4303 | 0.6015 | 0.9565 | 0.9894 | 0.9759 | 0.0020 |
| BiLSTM | 0.9905 | 0.2243 | 0.3658 | 0.8436 | 0.9553 | 0.8877 | 0.0090 |
| TF-IDF+SVC | 0.9942 | 0.3662 | 0.5352 | 0.8939 | 0.9719 | 0.9163 | 0.0090 |
| TF-IDF+LR | 0.9968 | 0.3004 | 0.4617 | 0.8975 | 0.9720 | 0.9225 | 0.0040 |
| DistilBERT | 1.0000 | 0.2344 | 0.3798 | 0.9614 | 0.9906 | 0.9732 | 0.0000 |

## PhishFuzzer — n=205, n_phish=103, n_legit=102

| Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.5109 | 0.6796 | 0.5833 | 0.5711 | 0.6145 | 0.5078 | 0.6569 |
| BiLSTM | 0.5312 | 0.4951 | 0.5126 | 0.5149 | 0.5770 | 0.5099 | 0.4412 |
| TF-IDF+SVC | 0.5263 | 0.6796 | 0.5932 | 0.5819 | 0.6379 | 0.5027 | 0.6176 |
| TF-IDF+LR | 0.5238 | 0.7476 | 0.6160 | 0.5603 | 0.5872 | 0.5102 | 0.6863 |
| DistilBERT | 0.5299 | 0.6019 | 0.5636 | 0.5478 | 0.5897 | 0.5108 | 0.5392 |

## DistilBERT — run 2026-11-20 (Kaggle Kernel 4, seed-42 checkpoint)

DistilBERT has now run on all six sets (rows above). Same collapse pattern as RoBERTa: precision 1.0000 with recall collapsing on the hard sets (PhishNChips 0.0140, BEC 0.1502, BEC-poisoned 0.2344), while ROC-AUC stays high (0.82–0.999) — the ranking retains signal that threshold 0.5 throws away. On PhishNChips its recall (0.0140) is worse than RoBERTa's (0.0280); on Frontiers human it is the best of the five models on recall (0.8572) and F1 (0.9231).

## Notes

- The legit side is the SAME shared 1,000 PhishNChips legit emails in five of the six sets (verified: identical `email_id` sets for the legit class in PhishNChips, Frontiers human, Frontiers LLM, BEC and BEC-poisoned). That is why mean P(legit) and FPR are identical across those sets within a model. PhishFuzzer is the exception: its own 103 phishing / 102 legit emails, and its legit side is far harder (FPR 0.44–0.69 at threshold 0.5, vs 0.000–0.009 on the shared legit).
- PhishNChips is the hard set for every model (recall 0.014–0.185 at 0.5); on Frontiers and BEC the same models rank well (ROC-AUC 0.83–0.999) and recall at 0.5 is 0.15–0.94. The collapse is set-specific, not a general OOD failure.
- P@R.90 on PhishNChips (0.53–0.63) shows that recall 0.90 is reachable at precision ~0.55–0.63 for every model — the ranking retains signal that threshold 0.5 throws away. See the calibration and threshold-transfer reports.
