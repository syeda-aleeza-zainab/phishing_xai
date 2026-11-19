# Statistical report — modern-phishing OOD experiment

Method: `experiments/modern_phishing_ood/scripts/compute_ood_stats.py`,
run locally from the SAVED per-email predictions (never re-scoring).
Paired McNemar (RoBERTa vs BiLSTM) on identical examples, paired
bootstrap 95% CIs for precision / recall / F1 and for ROC-AUC / PR-AUC
(1,000 resamples, seed 42), and paired-difference CIs (RoBERTa −
BiLSTM). Full numbers: `results/ood_stats.json`.

**McNemar is an errors-only test. It is NOT a precision test.** It
compares the two models' overall error counts on the same emails
(b = RoBERTa wrong / BiLSTM right; c = RoBERTa right / BiLSTM wrong).
A significant McNemar p says the models make different numbers of
errors; it says nothing on its own about which model has higher
precision. Precision/recall/F1 claims use the paired-difference CIs.
Verdict rule (repo, fixed 2026-11-19): a precision gain is
"significant" only if the gap is positive AND McNemar p < 0.05.

**Helpers:** the repo helpers (`baselines.mcnemar_test`,
`bootstrap_precision_ci`, `bootstrap_f1_ci`) were used for the metrics
on Kaggle, but the stats step ran locally where `baselines` cannot be
imported (it needs torch), so `compute_ood_stats.py` used local
mirrors of the same formulas (McNemar chi2 with continuity correction,
exact binomial p when b+c < 25; percentile bootstrap, resamples with a
single class skipped) — labelled `"helpers": "mirror"` in
`ood_stats.json`. The mirror formulas are unit-tested
(`tests/test_ood_stats.py`: chi2 statistic, exact-binomial path,
no-discordant-pairs p=1.0). Bootstrap CIs for recall and for the AUCs
do not exist in the repo; they follow the same percentile scheme and
are labelled extensions.

**Single-seed caveat:** both checkpoints are single (one RoBERTa
checkpoint, one BiLSTM artefact). No multi-seed variability is claimed
anywhere; `n_seeds` = 1 is recorded in `ood_stats.json`.

## McNemar (paired, identical examples)

| Set | Paired n | b (RoBERTa wrong, BiLSTM right) | c (RoBERTa right, BiLSTM wrong) | χ² (cont. corr.) | p | Reading |
|---|---|---|---|---|---|---|
| PhishNChips | 2,000 | 51 | 31 | 4.40 | 0.0359 | Significant at 0.05 — BiLSTM makes fewer errors (954 vs 974) |
| PhishFuzzer | 205 | 47 | 44 | 0.04 | 0.8339 | No difference in errors |
| Frontiers LLM | 5,986 | 11 | 2,315 | 2,280.23 | 0.0 | RoBERTa makes vastly fewer errors |
| Frontiers human | 6,000 | 269 | 864 | 311.42 | 1.1e-69 | RoBERTa makes far fewer errors |
| BEC | 5,181 | 375 | 1,080 | 340.63 | 4.7e-76 | RoBERTa makes far fewer errors |
| BEC poisoned | 5,181 | 345 | 1,213 | 482.47 | 6.2e-107 | RoBERTa makes far fewer errors |

Note the direction on PhishNChips: the significant McNemar favours
BiLSTM on errors, while RoBERTa has the higher precision there
(0.9333 vs 0.8594) — the two statements are compatible because
McNemar counts all errors and RoBERTa's errors are almost all false
negatives (972 FN vs 2 FP), while BiLSTM trades 7 extra FPs for 27
fewer FNs. Do not quote p=0.0359 as "RoBERTa is significantly better"
on PhishNChips; it is the opposite direction on errors.

## Bootstrap 95% CIs (threshold 0.5; 1,000 resamples, seed 42)

### PhishNChips (n=2,000)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RoBERTa | 0.9333 [0.8387, 1.0000] | 0.0280 [0.0185, 0.0392] | 0.0544 [0.0363, 0.0751] | 0.7787 [0.7584, 0.7987] | 0.7947 [0.7702, 0.8201] |
| BiLSTM | 0.8594 [0.7758, 0.9375] | 0.0550 [0.0417, 0.0694] | 0.1034 [0.0795, 0.1287] | 0.6801 [0.6563, 0.7036] | 0.6768 [0.6457, 0.7098] |

The ROC-AUC CIs do not overlap (RoBERTa ranks better); the precision
CIs overlap heavily; the recall and F1 CIs do not overlap (BiLSTM
higher at 0.5). Paired differences (RoBERTa − BiLSTM): precision
+0.074 [−0.033, +0.180] (includes 0 — NOT a significant precision
gain under the verdict rule, and McNemar p=0.0359 is about errors, in
BiLSTM's favour); recall −0.027 [−0.043, −0.010]; F1 −0.049
[−0.080, −0.018] (both exclude 0 — BiLSTM ahead on recall/F1 at 0.5).

### PhishFuzzer binary (n=205, low-n — wide CIs)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RoBERTa | 0.5109 [0.4253, 0.5942] | 0.6796 [0.5882, 0.7619] | 0.5833 [0.5065, 0.6512] | 0.5711 [0.4904, 0.6500] | 0.6145 [0.5228, 0.7095] |
| BiLSTM | 0.5312 [0.4352, 0.6344] | 0.4951 [0.3956, 0.6019] | 0.5126 [0.4242, 0.5954] | 0.5149 [0.4327, 0.6000] | 0.5770 [0.4854, 0.6694] |

All CIs overlap except the paired recall difference (+0.184
[+0.064, +0.308], RoBERTa higher). With n=205 and McNemar p=0.83,
no overall claim is supported on this slice beyond "both near chance".

### Frontiers LLM (n=5,986)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RoBERTa | 0.9996 [0.9989, 1.0000] | 0.9394 [0.9326, 0.9460] | 0.9686 [0.9648, 0.9722] | 0.9994 [0.9984, 0.9999] | 0.9998 [0.9996, 1.0000] |
| BiLSTM | 0.9962 [0.9937, 0.9987] | 0.4787 [0.4656, 0.4910] | 0.6467 [0.6346, 0.6579] | 0.9542 [0.9474, 0.9611] | 0.9883 [0.9852, 0.9911] |

Paired differences: precision +0.003 [+0.001, +0.006]; recall +0.461
[+0.446, +0.475]; F1 +0.322 [+0.310, +0.334]. All exclude 0, and
McNemar p≈0 — the largest, cleanest RoBERTa advantage in the study.

### Frontiers human (n=6,000)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RoBERTa | 0.9995 [0.9988, 1.0000] | 0.8402 [0.8307, 0.8501] | 0.9130 [0.9074, 0.9189] | 0.9902 [0.9885, 0.9920] | 0.9981 [0.9977, 0.9984] |
| BiLSTM | 0.9975 [0.9959, 0.9989] | 0.7226 [0.7095, 0.7345] | 0.8381 [0.8291, 0.8458] | 0.9501 [0.9443, 0.9554] | 0.9901 [0.9886, 0.9915] |

Paired differences: precision +0.002 [+0.001, +0.004]; recall +0.118
[+0.105, +0.131]; F1 +0.075 [+0.067, +0.083]. All exclude 0.

### BEC (n=5,181)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RoBERTa | 0.9987 [0.9968, 1.0000] | 0.3772 [0.3635, 0.3911] | 0.5476 [0.5330, 0.5621] | 0.9513 [0.9454, 0.9571] | 0.9880 [0.9858, 0.9899] |
| BiLSTM | 0.9899 [0.9834, 0.9956] | 0.2102 [0.1982, 0.2227] | 0.3468 [0.3300, 0.3637] | 0.8320 [0.8191, 0.8438] | 0.9518 [0.9460, 0.9574] |

Paired differences: precision +0.009 [+0.003, +0.015]; recall +0.167
[+0.149, +0.184]; F1 +0.201 [+0.182, +0.222]. All exclude 0.

### BEC poisoned (n=5,181)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RoBERTa | 0.9989 [0.9972, 1.0000] | 0.4303 [0.4160, 0.4448] | 0.6015 [0.5872, 0.6155] | 0.9565 [0.9511, 0.9618] | 0.9894 [0.9875, 0.9911] |
| BiLSTM | 0.9905 [0.9844, 0.9959] | 0.2243 [0.2118, 0.2370] | 0.3658 [0.3491, 0.3825] | 0.8436 [0.8312, 0.8554] | 0.9553 [0.9498, 0.9605] |

Paired differences: precision +0.008 [+0.003, +0.014]; recall +0.206
[+0.189, +0.224]; F1 +0.236 [+0.217, +0.255]. All exclude 0.

## ROC/PR-AUC comparison

RoBERTa's ROC-AUC CI sits above BiLSTM's on every set except
PhishFuzzer, where they overlap (RoBERTa 0.5711 [0.4904, 0.6500] vs
BiLSTM 0.5149 [0.4327, 0.6000]). Largest ranking gap: Frontiers LLM
(0.9994 vs 0.9542). Smallest real gap: PhishNChips (0.7787 vs 0.6801,
CIs non-overlapping but both far below in-distribution 0.9986).
PR-AUC follows the same ordering on every set.

## What is NOT claimed

* No 3-seed variability: single checkpoints, so no mean ± std across
  seeds. GPU nondeterminism at the tenth-of-a-point level was observed
  in earlier phases; the McNemar p=0.0359 on PhishNChips (b=51, c=31)
  is close enough to 0.05 that a re-run could flip it — treat it as
  borderline, not as a firm "BiLSTM is better on errors" verdict.
* No precision-significance claim is made from McNemar alone anywhere
  in this report; precision claims use the paired-difference CIs.
* PhishFuzzer (n=205) cannot support a model ranking: every CI
  overlaps except paired recall.
