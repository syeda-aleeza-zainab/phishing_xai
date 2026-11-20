# Task 2b — Calibration report (OOD + in-distribution)

Coverage: mean predicted P(phishing) on phishing vs legit emails, and ECE with 10 equal-width bins, for all 5 models on all 6 OOD sets (recomputed from the saved prediction CSVs). In-distribution ECE: TF-IDF+LR, TF-IDF+SVC and DistilBERT — see below.

## Mean predicted P(phishing) by true class, and ECE (10 bins)

| Set | Model | mean P(phish) | mean P(legit) | ECE-10 |
|---|---|---:|---:|---:|
| PhishNChips (primary) | RoBERTa | 0.0331 | 0.0024 | 0.4831 |
| PhishNChips (primary) | BiLSTM | 0.0544 | 0.0107 | 0.4738 |
| PhishNChips (primary) | TF-IDF+SVC | 0.1246 | 0.0417 | 0.4169 |
| PhishNChips (primary) | TF-IDF+LR | 0.2762 | 0.1214 | 0.3012 |
| PhishNChips (primary) | DistilBERT | 0.0179 | 0.0001 | 0.4910 |
| Frontiers human | RoBERTa | 0.8369 | 0.0024 | 0.1355 |
| Frontiers human | BiLSTM | 0.7220 | 0.0107 | 0.2299 |
| Frontiers human | TF-IDF+SVC | 0.7940 | 0.0417 | 0.1647 |
| Frontiers human | TF-IDF+LR | 0.7909 | 0.1214 | 0.1540 |
| Frontiers human | DistilBERT | 0.8453 | 0.0001 | 0.1289 |
| Frontiers LLM | RoBERTa | 0.9283 | 0.0024 | 0.0594 |
| Frontiers LLM | BiLSTM | 0.4790 | 0.0107 | 0.4322 |
| Frontiers LLM | TF-IDF+SVC | 0.5943 | 0.0417 | 0.3310 |
| Frontiers LLM | TF-IDF+LR | 0.7313 | 0.1214 | 0.2035 |
| Frontiers LLM | DistilBERT | 0.7424 | 0.0001 | 0.2146 |
| BEC | RoBERTa | 0.3770 | 0.0024 | 0.5023 |
| BEC | BiLSTM | 0.2152 | 0.0107 | 0.6312 |
| BEC | TF-IDF+SVC | 0.4073 | 0.0417 | 0.4702 |
| BEC | TF-IDF+LR | 0.4074 | 0.1214 | 0.4548 |
| BEC | DistilBERT | 0.1558 | 0.0001 | 0.6812 |
| BEC-poisoned | RoBERTa | 0.4269 | 0.0024 | 0.4620 |
| BEC-poisoned | BiLSTM | 0.2307 | 0.0107 | 0.6188 |
| BEC-poisoned | TF-IDF+SVC | 0.3898 | 0.0417 | 0.4844 |
| BEC-poisoned | TF-IDF+LR | 0.3949 | 0.1214 | 0.4649 |
| BEC-poisoned | DistilBERT | 0.2385 | 0.0001 | 0.6145 |
| PhishFuzzer | RoBERTa | 0.6893 | 0.6520 | 0.4316 |
| PhishFuzzer | BiLSTM | 0.4842 | 0.4411 | 0.4370 |
| PhishFuzzer | TF-IDF+SVC | 0.6243 | 0.5575 | 0.2913 |
| PhishFuzzer | TF-IDF+LR | 0.6535 | 0.6073 | 0.2108 |
| PhishFuzzer | DistilBERT | 0.5746 | 0.5260 | 0.3953 |

## In-distribution ECE (test set, n=3,495)

Per-email in-distribution test probabilities exist locally for TF-IDF+LR and TF-IDF+SVC, as saved `y_prob` (seed-42 refit) in `results/robustness_rerun/baseline_results.json`, and for DistilBERT as saved `test_prob.npy` (Kernel 2, seed 42) in `results/distilbert/seed_42/`. The true labels were rebuilt from `data/Phishing_Email.csv` with the repo's `load_data` + `deduplicate_data`, taking `test_indices` from `results/splits.json` (n=3,495; 1,322 phishing). Verification: thresholding the saved `y_prob` at 0.5 reproduces the saved `y_pred` exactly for all three models, and recomputed precision/recall/F1/ROC-AUC/PR-AUC match the saved metrics exactly.

| Model | In-distribution ECE-10 | mean P(phish) | mean P(legit) |
|---|---:|---:|---:|
| TF-IDF+LR | 0.0316 | 0.9339 | 0.0473 |
| TF-IDF+SVC | 0.0060 | 0.9744 | 0.0149 |
| DistilBERT | 0.0151 | 0.9578 | 0.0079 |

**RoBERTa / BiLSTM in-distribution ECE: NOT RUN YET.** No saved per-email in-distribution test probabilities exist locally for either model: `results/robustness_rerun/baseline_results.json` records RoBERTa as NOT RUN YET (`results/metrics.json` not found), no `results_v7` `*.npy` prediction files exist locally, and no other `results/*.json` carries their `y_prob`. Re-scoring the test set would be a new model run, which is out of scope for this task.

## Verdict: ranking failure or calibration / threshold failure?

Judged on the primary set (PhishNChips), where recall at 0.5 collapses for every model. Two facts decide it: (1) ROC-AUC / PR-AUC and P@R.90 show how much signal the ranking retains; (2) the mean predicted probability on phishing emails shows how far the scores sit below the 0.5 threshold. In-distribution ECE is the reference for how calibrated each model is when it is NOT shifted: LR 0.032, SVC 0.006 — both well calibrated in-distribution, so their OOD ECE of 0.30 / 0.42 is shift-induced, not an intrinsic property.

### RoBERTa — calibration / threshold failure (ranking secondarily weak)

Mean P(phishing) on PhishNChips phishing emails is 0.033 (legit 0.002): the model is systematically under-confident on this set, so almost nothing crosses 0.5 and recall is 0.028. But the ranking is not dead: ROC-AUC 0.7787, PR-AUC 0.7947, and P@R.90 = 0.5636 — at a lower threshold recall 0.90 is reachable at precision 0.56. ECE-10 is 0.483, the second-worst. The threshold-transfer experiment (2c) confirms it: a recalibrated threshold recovers recall 0.81 on the 80% holdout at precision 0.63. So the primary failure is calibration/threshold, with a genuine but secondary ranking weakness (AUC 0.78 vs 0.9986 in-distribution). On Frontiers LLM and BEC the same model ranks almost perfectly (ROC-AUC 0.95–0.999), so the ranking weakness is specific to PhishNChips.

### BiLSTM — both: ranking failure AND calibration / threshold failure

Mean P(phishing) on PhishNChips phishing is 0.054, ECE-10 0.474 — the scores are squashed near zero just like RoBERTa's, so threshold 0.5 gives recall 0.055. But BiLSTM also has the second-worst ranking on this set (ROC-AUC 0.6801, PR-AUC 0.6768; P@R.90 = 0.5263, the lowest of the five). Even after recalibration (2c) it only reaches recall 0.84 at precision 0.55 on the holdout, the lowest precision of the five. Both failures are real here; the ranking failure is what separates BiLSTM from RoBERTa.

### TF-IDF+SVC — ranking failure first, threshold failure second

SVC has the worst ranking on PhishNChips (ROC-AUC 0.6604, PR-AUC 0.6724 — the lowest of the four), so its low recall is partly a genuine ranking failure: it separates the classes worst. It is also miscalibrated (mean P(phishing) 0.125, ECE-10 0.417; in-distribution ECE 0.006, so this is shift-induced), and recalibration (2c) recovers recall 0.82 at precision 0.56. Ranking first, threshold second.

### TF-IDF+LR — calibration / threshold failure (third-best ranking of the five on this set)

LR ranks third on PhishNChips by ROC-AUC (0.7528, behind DistilBERT 0.8222 and RoBERTa 0.7787) and by PR-AUC (0.7771), second by P@R.90 (0.5714, behind DistilBERT 0.6348), and it has the least-bad calibration (mean P(phishing) 0.276, ECE-10 0.301; in-distribution ECE 0.032). Its recall of 0.185 at 0.5 — the best of the four original models, but still low — is therefore mostly a threshold failure: the scores are shifted down, not uninformative. Recalibration (2c) recovers recall 0.85 on the holdout at precision 0.60.

### DistilBERT — calibration / threshold failure (best ranking of the five on this set, worst calibration)

DistilBERT has the best ranking on PhishNChips (ROC-AUC 0.8222, PR-AUC 0.8250, P@R.90 = 0.6348 — the highest of the five on all three) and the worst calibration (mean P(phishing) 0.0179, the most squashed scores of any model; ECE-10 0.491, the worst of the five; in-distribution ECE 0.015, so this is shift-induced). At threshold 0.5 almost nothing crosses, so recall is 0.014 — the worst of the five — even though the ranking is the best. Recalibration (2c) confirms it: a recalibrated threshold recovers recall 0.835 on the holdout at precision 0.708, the best holdout F1 (0.766) of the five. On BEC its ECE is 0.681, also the worst of the five. The failure is pure calibration/threshold: the scores carry the signal, but they sit far below 0.5.

### One-line summary

On PhishNChips, low recall at 0.5 is a calibration/threshold failure for RoBERTa, TF-IDF+LR and DistilBERT (usable ranking, squashed scores — DistilBERT's ranking is the best of the five and its calibration the worst), and a ranking failure as well for BiLSTM and TF-IDF+SVC (weakest AUCs on this set). No model is well calibrated OOD: ECE-10 on PhishNChips is 0.30–0.49, against 0.006–0.032 in-distribution for the three models where that comparison exists.
