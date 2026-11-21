# Task 2c — Threshold transfer on PhishNChips

**Label: target-domain recalibration (uses labelled OOD data) — NOT pure OOD.** The threshold is picked using labelled PhishNChips examples, so this measures what recalibration on the target domain buys; it does not measure zero-shot OOD generalisation.

Method: stratified 20% slice of the PhishNChips predictions (seed 42; slice n=400, 200 phishing / 200 legit; remaining 80% n=1,600, 800 / 800). On the slice, the threshold maximising precision subject to recall >= 0.85 is picked (ties broken toward the higher threshold), then evaluated on the remaining 80%. Computed from the saved PhishNChips prediction CSVs — same email_ids, labels and probabilities as the original evaluation. No retraining, no new scoring.

| Model | Threshold (from 20% slice) | Slice P / R / FPR | Holdout 80%: P | Holdout 80%: R | Holdout 80%: F1 | Holdout 80%: FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa | 0.000103913 | 0.6273 / 0.8500 / 0.5050 | 0.6290 | 0.8075 | 0.7072 | 0.4763 |
| BiLSTM | 3.29813e-09 | 0.5463 / 0.8550 / 0.7100 | 0.5467 | 0.8413 | 0.6627 | 0.6975 |
| TF-IDF+SVC | 0.00392017 | 0.5686 / 0.8500 / 0.6450 | 0.5613 | 0.8187 | 0.6660 | 0.6400 |
| TF-IDF+LR | 0.0798964 | 0.6028 / 0.8500 / 0.5600 | 0.5954 | 0.8462 | 0.6990 | 0.5750 |
| DistilBERT | 7.74991e-05 | 0.6827 / 0.8500 / 0.3950 | 0.7076 | 0.8350 | 0.7661 | 0.3450 |

Machine-readable version: `../results/threshold_transfer.json`.

## Reading

- Every model recovers most of the recall that threshold 0.5 throws away: holdout recall 0.81–0.85 (vs 0.014–0.185 at 0.5), at holdout precision 0.55–0.71 and FPR 0.35–0.70. The price of that recall is a very high false-positive rate — roughly half of legit emails would be flagged at these thresholds (DistilBERT least, FPR 0.345).
- The fitted thresholds are extremely low (RoBERTa 0.00010, DistilBERT 7.7e-05, BiLSTM 3.3e-09, SVC 0.0039, LR 0.0799) because the predicted probabilities are squashed near zero on this set (see calibration report). A threshold of 3e-09 is not a usable operating point in practice; it is a symptom of the miscalibration, not a recommendation.
- DistilBERT transfers best (holdout P 0.708 / R 0.835, F1 0.766) — consistent with its best ranking on this set (ROC-AUC 0.8222, P@R.90 0.6348); BiLSTM worst (P 0.547 / R 0.841, F1 0.663) — consistent with its weakest ranking. TF-IDF+LR holds its recall floor best (holdout R 0.846 vs slice 0.850).
- The slice-to-holdout recall drop is small (RoBERTa 0.850 -> 0.808, SVC 0.850 -> 0.819, LR 0.850 -> 0.846, DistilBERT 0.850 -> 0.835; BiLSTM 0.855 -> 0.841), so the fitted threshold is not badly overfit to the 400-email slice — but it is still fitted on labelled target-domain data, which is exactly what pure OOD evaluation does not allow.
