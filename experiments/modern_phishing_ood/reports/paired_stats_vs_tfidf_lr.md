# Task 2d — Paired statistics on PhishNChips vs TF-IDF+LR

Primary set only, n=2,000, threshold 0.5, reference TF-IDF+LR. McNemar uses the repo's `phishing_xai.baselines.mcnemar_test` (imported directly — the venv has torch, so no mirror was needed). **McNemar compares paired classification errors only; it is explicitly NOT a precision test.** b = model wrong / LR right, c = model right / LR wrong; chi-square with continuity correction, exact binomial p when b+c < 25 (not needed here — all discordant counts are large). Bootstrap: paired percentile 95% CIs, 1,000 resamples, seed 42. Precision/recall/F1 differences via `compute_ood_stats.paired_diff_ci`; ROC-AUC/PR-AUC differences via the same percentile scheme (extension — the repo has no AUC-difference bootstrap).

Rule applied throughout: the word 'significant' is used for a difference only when it is positive AND its test supports it (McNemar p < 0.05 for error counts; bootstrap 95% CI excluding 0 for a metric difference). DistilBERT is the exception to the earlier pattern: against LR it shows a positive precision difference (+0.0212, CI excludes 0) and positive ROC-AUC / PR-AUC differences — but its precision is 1.0000 on 14 true positives and 0 false positives, so the precision gain is a small-denominator artefact of the collapsed recall, not a usable gain. Every other model's P/R/F1 difference against LR is negative.

## RoBERTa vs TF-IDF+LR

- McNemar (paired errors): b=177 (model wrong, LR right), c=22 (model right, LR wrong), discordant n=199, chi2 statistic=119.18, p=9.58e-28. LR commits significantly fewer paired errors than RoBERTa (p < 0.05). This is a statement about error counts, not about precision.
- Paired bootstrap 95% CIs for the difference (model minus LR):

  | Metric | Difference | 95% CI | CI excludes 0? |
  |---|---:|---:|---|
  | Precision | -0.0455 | [-0.1446, 0.0326] | no |
  | Recall | -0.1570 | [-0.1830, -0.1331] | yes |
  | F1 | -0.2568 | [-0.2918, -0.2160] | yes |
  | ROC-AUC | 0.0259 | [0.0027, 0.0498] | yes |
  | PR-AUC | 0.0175 | [-0.0085, 0.0429] | no |

## BiLSTM vs TF-IDF+LR

- McNemar (paired errors): b=162 (model wrong, LR right), c=27 (model right, LR wrong), discordant n=189, chi2 statistic=95.01, p=1.9e-22. LR commits significantly fewer paired errors than BiLSTM (p < 0.05). This is a statement about error counts, not about precision.
- Paired bootstrap 95% CIs for the difference (model minus LR):

  | Metric | Difference | 95% CI | CI excludes 0? |
  |---|---:|---:|---|
  | Precision | -0.1195 | [-0.2043, -0.0390] | yes |
  | Recall | -0.1300 | [-0.1551, -0.1061] | yes |
  | F1 | -0.2078 | [-0.2446, -0.1711] | yes |
  | ROC-AUC | -0.0727 | [-0.0968, -0.0493] | yes |
  | PR-AUC | -0.1004 | [-0.1267, -0.0739] | yes |

## TF-IDF+SVC vs TF-IDF+LR

- McNemar (paired errors): b=129 (model wrong, LR right), c=20 (model right, LR wrong), discordant n=149, chi2 statistic=78.28, p=8.93e-19. LR commits significantly fewer paired errors than TF-IDF+SVC (p < 0.05). This is a statement about error counts, not about precision.
- Paired bootstrap 95% CIs for the difference (model minus LR):

  | Metric | Difference | 95% CI | CI excludes 0? |
  |---|---:|---:|---|
  | Precision | -0.0788 | [-0.1430, -0.0249] | yes |
  | Recall | -0.1040 | [-0.1257, -0.0826] | yes |
  | F1 | -0.1626 | [-0.1952, -0.1292] | yes |
  | ROC-AUC | -0.0924 | [-0.1105, -0.0744] | yes |
  | PR-AUC | -0.1047 | [-0.1226, -0.0870] | yes |

## DistilBERT vs TF-IDF+LR

- McNemar (paired errors): b=178 (model wrong, LR right), c=11 (model right, LR wrong), discordant n=189, chi2 statistic=145.80, p=1.4e-33. LR commits significantly fewer paired errors than DistilBERT (p < 0.05). This is a statement about error counts, not about precision.
- Paired bootstrap 95% CIs for the difference (model minus LR):

  | Metric | Difference | 95% CI | CI excludes 0? |
  |---|---:|---:|---|
  | Precision | +0.0212 | [0.0050, 0.0445] | yes |
  | Recall | -0.1710 | [-0.1963, -0.1479] | yes |
  | F1 | -0.2836 | [-0.3164, -0.2476] | yes |
  | ROC-AUC | +0.0695 | [0.0472, 0.0905] | yes |
  | PR-AUC | +0.0478 | [0.0273, 0.0692] | yes |

## DistilBERT vs RoBERTa (same set, n=2,000)

- McNemar (paired errors): b=19 (DistilBERT wrong, RoBERTa right), c=7 (DistilBERT right, RoBERTa wrong), discordant n=26, chi2 statistic=4.65, p=0.0310. RoBERTa commits significantly fewer paired errors than DistilBERT (p < 0.05).
- Paired bootstrap 95% CIs for the difference (DistilBERT minus RoBERTa):

  | Metric | Difference | 95% CI | CI excludes 0? |
  |---|---:|---:|---|
  | Precision | +0.0667 | [0.0000, 0.1613] | no (lower bound is 0) |
  | Recall | -0.0140 | [-0.0238, -0.0049] | yes |
  | F1 | -0.0268 | [-0.0454, -0.0097] | yes |
  | ROC-AUC | +0.0435 | [0.0257, 0.0622] | yes |
  | PR-AUC | +0.0303 | [0.0089, 0.0514] | yes |

## Reading

- **DistilBERT vs LR:** McNemar p=1.4e-33 — LR makes significantly fewer paired errors. The precision difference is positive (+0.0212 [0.0050, 0.0445], CI excludes 0): DistilBERT's precision on this set is 1.0000 against LR's 0.9788 — but that is 14 true positives and 0 false positives, a degenerate precision at recall 0.014, not a usable gain. Recall -0.1710 and F1 -0.2836 both exclude 0: LR is far better on recall and F1 at threshold 0.5. The ranking differences are positive and exclude 0: ROC-AUC +0.0695 [0.0472, 0.0905] and PR-AUC +0.0478 [0.0273, 0.0692] — DistilBERT ranks the classes better than LR even though it classifies far worse at 0.5, consistent with the calibration verdict (2b): best ranking, worst calibration.
- **DistilBERT vs RoBERTa:** McNemar p=0.0310 — RoBERTa makes significantly fewer paired errors (19 vs 7 discordant). Precision +0.0667 [0.0000, 0.1613] includes 0 at its lower bound: no significant precision difference. Recall -0.0140 and F1 -0.0268 both exclude 0: RoBERTa is better on recall and F1 at 0.5. ROC-AUC +0.0435 [0.0257, 0.0622] and PR-AUC +0.0303 [0.0089, 0.0514] exclude 0: DistilBERT ranks better than RoBERTa on this set.
- **RoBERTa vs LR:** McNemar p=9.6e-28 — LR makes significantly fewer paired errors. Precision difference -0.0455, CI [-0.1446, 0.0326] includes 0: no significant precision difference. Recall -0.1570 [-0.1830, -0.1331] and F1 -0.2568 [-0.2918, -0.2160] both exclude 0: LR is better on recall and F1 at threshold 0.5. The one positive difference is ROC-AUC, +0.0259 [0.0027, 0.0498], CI excludes 0: RoBERTa ranks the classes slightly better than LR even though it classifies worse at 0.5 — consistent with the calibration verdict (2b). PR-AUC +0.0175 [-0.0085, 0.0429] includes 0.
- **BiLSTM vs LR:** McNemar p=1.9e-22 — LR makes significantly fewer paired errors. Every difference is negative with CI excluding 0: precision -0.1195, recall -0.1300, F1 -0.2078, ROC-AUC -0.0727, PR-AUC -0.1004. LR is better on every metric.
- **TF-IDF+SVC vs LR:** McNemar p=8.9e-19 — LR makes significantly fewer paired errors. Every difference is negative with CI excluding 0: precision -0.0788, recall -0.1040, F1 -0.1626, ROC-AUC -0.0924, PR-AUC -0.1047. LR is better on every metric.

Machine-readable version: `../results/paired_stats_vs_tfidf_lr.json`.
