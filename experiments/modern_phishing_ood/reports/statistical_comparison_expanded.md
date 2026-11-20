# OOD Statistical Comparison — Expanded (RoBERTa vs BiLSTM vs TF-IDF+SVC vs TF-IDF+LR)

Paired comparisons on IDENTICAL OOD examples (per-email predictions aligned on `email_id`). McNemar compares overall errors ONLY — it is explicitly NOT a precision test. A difference is called significant only when its paired bootstrap 95% CI excludes 0 (metric differences) or its Holm-adjusted McNemar p < 0.05; a bare numerical gap is reported as a numerical difference only, never as significant.

Method: paired McNemar (errors only) + paired bootstrap 95% CIs (n_resamples=1000, seed=42) for precision/recall/F1 and ROC-AUC/PR-AUC differences, on identical examples aligned on email_id

Multiple-comparison correction: Holm-Bonferroni across the McNemar family WITHIN each set (primary); global Holm across all pairs x sets reported as a sensitivity check; Bonferroni-adjusted p for reference. Significant = Holm-adjusted p < 0.05.

Original RoBERTa-vs-BiLSTM findings are RETAINED (see `statistical_report.md` / `ood_stats.json`); this report ADDS the TF-IDF pairs and the correction. Nothing in the original evaluation was changed.

Models present in the predictions dir: bilstm, roberta, tfidf_lr, tfidf_svm. TF-IDF OOD: COMPLETED.

## phishnchips (n=2000)

### roberta vs bilstm (n paired = 2000)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | bilstm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [972, 28]] | [[991, 9], [945, 55]] |
| prediction agreement [[both phish, roberta only], [bilstm only, both safe]] | [[6, 24], [58, 1912]] | |
| errors: both correct / roberta wrong only / bilstm wrong only / both wrong | 995 / 51 / 31 / 923 | |

McNemar (errors only): b=51 (A wrong, B right), c=31 (A right, B wrong), raw p=0.0359, Holm-adjusted p=0.0359, Bonferroni-adjusted p=0.2153 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0740 | [-0.0329, 0.1803] | no — numerical difference only |
| recall | -0.0270 | [-0.0434, -0.0102] | yes (CI excludes 0) |
| f1 | -0.0490 | [-0.0796, -0.0179] | yes (CI excludes 0) |
| roc_auc | 0.0986 | [0.0760, 0.1247] | yes (CI excludes 0) |
| pr_auc | 0.1179 | [0.0878, 0.1461] | yes (CI excludes 0) |

### roberta vs tfidf_svm (n paired = 2000)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [972, 28]] | [[991, 9], [919, 81]] |
| prediction agreement [[both phish, roberta only], [tfidf_svm only, both safe]] | [[7, 23], [83, 1887]] | |
| errors: both correct / roberta wrong only / tfidf_svm wrong only / both wrong | 996 / 76 / 30 / 898 | |

McNemar (errors only): b=76 (A wrong, B right), c=30 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0001 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0333 | [-0.0771, 0.1395] | no — numerical difference only |
| recall | -0.0530 | [-0.0704, -0.0347] | yes (CI excludes 0) |
| f1 | -0.0943 | [-0.1258, -0.0613] | yes (CI excludes 0) |
| roc_auc | 0.1183 | [0.0927, 0.1438] | yes (CI excludes 0) |
| pr_auc | 0.1223 | [0.0919, 0.1518] | yes (CI excludes 0) |

### roberta vs tfidf_lr (n paired = 2000)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [972, 28]] | [[996, 4], [815, 185]] |
| prediction agreement [[both phish, roberta only], [tfidf_lr only, both safe]] | [[10, 20], [179, 1791]] | |
| errors: both correct / roberta wrong only / tfidf_lr wrong only / both wrong | 1004 / 177 / 22 / 797 | |

McNemar (errors only): b=177 (A wrong, B right), c=22 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0455 | [-0.1446, 0.0326] | no — numerical difference only |
| recall | -0.1570 | [-0.1830, -0.1331] | yes (CI excludes 0) |
| f1 | -0.2568 | [-0.2918, -0.2160] | yes (CI excludes 0) |
| roc_auc | 0.0259 | [0.0027, 0.0498] | yes (CI excludes 0) |
| pr_auc | 0.0175 | [-0.0085, 0.0429] | no — numerical difference only |

### bilstm vs tfidf_svm (n paired = 2000)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [945, 55]] | [[991, 9], [919, 81]] |
| prediction agreement [[both phish, bilstm only], [tfidf_svm only, both safe]] | [[29, 35], [61, 1875]] | |
| errors: both correct / bilstm wrong only / tfidf_svm wrong only / both wrong | 1011 / 61 / 35 / 893 | |

McNemar (errors only): b=61 (A wrong, B right), c=35 (A right, B wrong), raw p=0.0107, Holm-adjusted p=0.0214, Bonferroni-adjusted p=0.0643 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0406 | [-0.1363, 0.0563] | no — numerical difference only |
| recall | -0.0260 | [-0.0427, -0.0089] | yes (CI excludes 0) |
| f1 | -0.0452 | [-0.0758, -0.0181] | yes (CI excludes 0) |
| roc_auc | 0.0197 | [-0.0028, 0.0415] | no — numerical difference only |
| pr_auc | 0.0043 | [-0.0198, 0.0314] | no — numerical difference only |

### bilstm vs tfidf_lr (n paired = 2000)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [945, 55]] | [[996, 4], [815, 185]] |
| prediction agreement [[both phish, bilstm only], [tfidf_lr only, both safe]] | [[32, 32], [157, 1779]] | |
| errors: both correct / bilstm wrong only / tfidf_lr wrong only / both wrong | 1019 / 162 / 27 / 792 | |

McNemar (errors only): b=162 (A wrong, B right), c=27 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.1195 | [-0.2043, -0.0390] | yes (CI excludes 0) |
| recall | -0.1300 | [-0.1551, -0.1061] | yes (CI excludes 0) |
| f1 | -0.2078 | [-0.2446, -0.1711] | yes (CI excludes 0) |
| roc_auc | -0.0727 | [-0.0968, -0.0493] | yes (CI excludes 0) |
| pr_auc | -0.1004 | [-0.1267, -0.0739] | yes (CI excludes 0) |

### tfidf_svm vs tfidf_lr (n paired = 2000)

| confusion (rows = truth, cols = predicted) | tfidf_svm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [919, 81]] | [[996, 4], [815, 185]] |
| prediction agreement [[both phish, tfidf_svm only], [tfidf_lr only, both safe]] | [[65, 25], [124, 1786]] | |
| errors: both correct / tfidf_svm wrong only / tfidf_lr wrong only / both wrong | 1052 / 129 / 20 / 799 | |

McNemar (errors only): b=129 (A wrong, B right), c=20 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0788 | [-0.1430, -0.0249] | yes (CI excludes 0) |
| recall | -0.1040 | [-0.1257, -0.0826] | yes (CI excludes 0) |
| f1 | -0.1626 | [-0.1952, -0.1292] | yes (CI excludes 0) |
| roc_auc | -0.0924 | [-0.1105, -0.0744] | yes (CI excludes 0) |
| pr_auc | -0.1047 | [-0.1226, -0.0870] | yes (CI excludes 0) |

## phishfuzzer (n=205)

### roberta vs bilstm (n paired = 205)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | bilstm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[35, 67], [33, 70]] | [[57, 45], [52, 51]] |
| prediction agreement [[both phish, roberta only], [bilstm only, both safe]] | [[71, 66], [25, 43]] | |
| errors: both correct / roberta wrong only / bilstm wrong only / both wrong | 61 / 47 / 44 / 53 | |

McNemar (errors only): b=47 (A wrong, B right), c=44 (A right, B wrong), raw p=0.8339, Holm-adjusted p=1.0000, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0203 | [-0.1084, 0.0563] | no — numerical difference only |
| recall | 0.1845 | [0.0638, 0.3078] | yes (CI excludes 0) |
| f1 | 0.0708 | [-0.0245, 0.1655] | no — numerical difference only |
| roc_auc | 0.0562 | [-0.0425, 0.1514] | no — numerical difference only |
| pr_auc | 0.0375 | [-0.0484, 0.1248] | no — numerical difference only |

### roberta vs tfidf_svm (n paired = 205)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[35, 67], [33, 70]] | [[39, 63], [33, 70]] |
| prediction agreement [[both phish, roberta only], [tfidf_svm only, both safe]] | [[106, 31], [27, 41]] | |
| errors: both correct / roberta wrong only / tfidf_svm wrong only / both wrong | 78 / 31 / 27 / 69 | |

McNemar (errors only): b=31 (A wrong, B right), c=27 (A right, B wrong), raw p=0.6936, Holm-adjusted p=1.0000, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0154 | [-0.0690, 0.0394] | no — numerical difference only |
| recall | 0.0000 | [-0.0926, 0.1000] | no — numerical difference only |
| f1 | -0.0099 | [-0.0774, 0.0524] | no — numerical difference only |
| roc_auc | -0.0108 | [-0.0826, 0.0694] | no — numerical difference only |
| pr_auc | -0.0234 | [-0.0850, 0.0476] | no — numerical difference only |

### roberta vs tfidf_lr (n paired = 205)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[35, 67], [33, 70]] | [[32, 70], [26, 77]] |
| prediction agreement [[both phish, roberta only], [tfidf_lr only, both safe]] | [[110, 27], [37, 31]] | |
| errors: both correct / roberta wrong only / tfidf_lr wrong only / both wrong | 75 / 34 / 30 / 66 | |

McNemar (errors only): b=34 (A wrong, B right), c=30 (A right, B wrong), raw p=0.7077, Holm-adjusted p=1.0000, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0129 | [-0.0695, 0.0406] | no — numerical difference only |
| recall | -0.0680 | [-0.1667, 0.0303] | no — numerical difference only |
| f1 | -0.0327 | [-0.1016, 0.0295] | no — numerical difference only |
| roc_auc | 0.0108 | [-0.0636, 0.0904] | no — numerical difference only |
| pr_auc | 0.0273 | [-0.0370, 0.0959] | no — numerical difference only |

### bilstm vs tfidf_svm (n paired = 205)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[57, 45], [52, 51]] | [[39, 63], [33, 70]] |
| prediction agreement [[both phish, bilstm only], [tfidf_svm only, both safe]] | [[84, 12], [49, 60]] | |
| errors: both correct / bilstm wrong only / tfidf_svm wrong only / both wrong | 78 / 31 / 30 / 66 | |

McNemar (errors only): b=31 (A wrong, B right), c=30 (A right, B wrong), raw p=1.0000, Holm-adjusted p=1.0000, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0049 | [-0.0607, 0.0743] | no — numerical difference only |
| recall | -0.1845 | [-0.2832, -0.0769] | yes (CI excludes 0) |
| f1 | -0.0807 | [-0.1626, -0.0010] | yes (CI excludes 0) |
| roc_auc | -0.0669 | [-0.1421, 0.0082] | no — numerical difference only |
| pr_auc | -0.0610 | [-0.1331, 0.0163] | no — numerical difference only |

### bilstm vs tfidf_lr (n paired = 205)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[57, 45], [52, 51]] | [[32, 70], [26, 77]] |
| prediction agreement [[both phish, bilstm only], [tfidf_lr only, both safe]] | [[86, 10], [61, 48]] | |
| errors: both correct / bilstm wrong only / tfidf_lr wrong only / both wrong | 73 / 36 / 35 / 61 | |

McNemar (errors only): b=36 (A wrong, B right), c=35 (A right, B wrong), raw p=1.0000, Holm-adjusted p=1.0000, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0074 | [-0.0648, 0.0750] | no — numerical difference only |
| recall | -0.2524 | [-0.3431, -0.1613] | yes (CI excludes 0) |
| f1 | -0.1034 | [-0.1799, -0.0308] | yes (CI excludes 0) |
| roc_auc | -0.0454 | [-0.1225, 0.0343] | no — numerical difference only |
| pr_auc | -0.0103 | [-0.0815, 0.0610] | no — numerical difference only |

### tfidf_svm vs tfidf_lr (n paired = 205)

| confusion (rows = truth, cols = predicted) | tfidf_svm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[39, 63], [33, 70]] | [[32, 70], [26, 77]] |
| prediction agreement [[both phish, tfidf_svm only], [tfidf_lr only, both safe]] | [[121, 12], [26, 46]] | |
| errors: both correct / tfidf_svm wrong only / tfidf_lr wrong only / both wrong | 90 / 19 / 19 / 77 | |

McNemar (errors only): b=19 (A wrong, B right), c=19 (A right, B wrong), raw p=0.8711, Holm-adjusted p=1.0000, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0025 | [-0.0431, 0.0441] | no — numerical difference only |
| recall | -0.0680 | [-0.1346, -0.0103] | yes (CI excludes 0) |
| f1 | -0.0228 | [-0.0688, 0.0227] | no — numerical difference only |
| roc_auc | 0.0215 | [-0.0350, 0.0750] | no — numerical difference only |
| pr_auc | 0.0507 | [-0.0058, 0.1104] | no — numerical difference only |

## frontiers_llm (n=5986)

### roberta vs bilstm (n paired = 5986)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | bilstm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [302, 4684]] | [[991, 9], [2599, 2387]] |
| prediction agreement [[both phish, roberta only], [bilstm only, both safe]] | [[2378, 2308], [18, 1282]] | |
| errors: both correct / roberta wrong only / bilstm wrong only / both wrong | 3367 / 11 / 2315 / 293 | |

McNemar (errors only): b=11 (A wrong, B right), c=2315 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0033 | [0.0010, 0.0059] | yes (CI excludes 0) |
| recall | 0.4607 | [0.4459, 0.4747] | yes (CI excludes 0) |
| f1 | 0.3219 | [0.3102, 0.3344] | yes (CI excludes 0) |
| roc_auc | 0.0452 | [0.0384, 0.0517] | yes (CI excludes 0) |
| pr_auc | 0.0115 | [0.0091, 0.0148] | yes (CI excludes 0) |

### roberta vs tfidf_svm (n paired = 5986)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [302, 4684]] | [[991, 9], [1897, 3089]] |
| prediction agreement [[both phish, roberta only], [tfidf_svm only, both safe]] | [[3083, 1603], [15, 1285]] | |
| errors: both correct / roberta wrong only / tfidf_svm wrong only / both wrong | 4072 / 8 / 1610 / 296 | |

McNemar (errors only): b=8 (A wrong, B right), c=1610 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0025 | [0.0008, 0.0046] | yes (CI excludes 0) |
| recall | 0.3199 | [0.3066, 0.3328] | yes (CI excludes 0) |
| f1 | 0.2043 | [0.1948, 0.2147] | yes (CI excludes 0) |
| roc_auc | 0.0615 | [0.0546, 0.0680] | yes (CI excludes 0) |
| pr_auc | 0.0126 | [0.0112, 0.0143] | yes (CI excludes 0) |

### roberta vs tfidf_lr (n paired = 5986)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [302, 4684]] | [[996, 4], [904, 4082]] |
| prediction agreement [[both phish, roberta only], [tfidf_lr only, both safe]] | [[4052, 634], [34, 1266]] | |
| errors: both correct / roberta wrong only / tfidf_lr wrong only / both wrong | 5046 / 32 / 636 / 272 | |

McNemar (errors only): b=32 (A wrong, B right), c=636 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0006 | [-0.0004, 0.0018] | no — numerical difference only |
| recall | 0.1207 | [0.1112, 0.1296] | yes (CI excludes 0) |
| f1 | 0.0687 | [0.0627, 0.0746] | yes (CI excludes 0) |
| roc_auc | 0.0245 | [0.0212, 0.0280] | yes (CI excludes 0) |
| pr_auc | 0.0048 | [0.0041, 0.0055] | yes (CI excludes 0) |

### bilstm vs tfidf_svm (n paired = 5986)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [2599, 2387]] | [[991, 9], [1897, 3089]] |
| prediction agreement [[both phish, bilstm only], [tfidf_svm only, both safe]] | [[2061, 335], [1037, 2553]] | |
| errors: both correct / bilstm wrong only / tfidf_svm wrong only / both wrong | 3043 / 1037 / 335 / 1571 | |

McNemar (errors only): b=1037 (A wrong, B right), c=335 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0009 | [-0.0036, 0.0020] | no — numerical difference only |
| recall | -0.1408 | [-0.1541, -0.1266] | yes (CI excludes 0) |
| f1 | -0.1175 | [-0.1293, -0.1059] | yes (CI excludes 0) |
| roc_auc | 0.0163 | [0.0094, 0.0229] | yes (CI excludes 0) |
| pr_auc | 0.0011 | [-0.0018, 0.0036] | no — numerical difference only |

### bilstm vs tfidf_lr (n paired = 5986)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [2599, 2387]] | [[996, 4], [904, 4082]] |
| prediction agreement [[both phish, bilstm only], [tfidf_lr only, both safe]] | [[2347, 49], [1739, 1851]] | |
| errors: both correct / bilstm wrong only / tfidf_lr wrong only / both wrong | 3334 / 1744 / 44 / 864 | |

McNemar (errors only): b=1744 (A wrong, B right), c=44 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0028 | [-0.0053, -0.0003] | yes (CI excludes 0) |
| recall | -0.3400 | [-0.3530, -0.3269] | yes (CI excludes 0) |
| f1 | -0.2532 | [-0.2646, -0.2420] | yes (CI excludes 0) |
| roc_auc | -0.0206 | [-0.0272, -0.0142] | yes (CI excludes 0) |
| pr_auc | -0.0068 | [-0.0097, -0.0046] | yes (CI excludes 0) |

### tfidf_svm vs tfidf_lr (n paired = 5986)

| confusion (rows = truth, cols = predicted) | tfidf_svm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [1897, 3089]] | [[996, 4], [904, 4082]] |
| prediction agreement [[both phish, tfidf_svm only], [tfidf_lr only, both safe]] | [[3067, 31], [1019, 1869]] | |
| errors: both correct / tfidf_svm wrong only / tfidf_lr wrong only / both wrong | 4054 / 1024 / 26 / 882 | |

McNemar (errors only): b=1024 (A wrong, B right), c=26 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0019 | [-0.0038, -0.0002] | yes (CI excludes 0) |
| recall | -0.1992 | [-0.2104, -0.1882] | yes (CI excludes 0) |
| f1 | -0.1357 | [-0.1449, -0.1272] | yes (CI excludes 0) |
| roc_auc | -0.0369 | [-0.0416, -0.0320] | yes (CI excludes 0) |
| pr_auc | -0.0079 | [-0.0091, -0.0068] | yes (CI excludes 0) |

## frontiers_human (n=6000)

### roberta vs bilstm (n paired = 6000)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | bilstm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [799, 4201]] | [[991, 9], [1387, 3613]] |
| prediction agreement [[both phish, roberta only], [bilstm only, both safe]] | [[3346, 857], [276, 1521]] | |
| errors: both correct / roberta wrong only / bilstm wrong only / both wrong | 4335 / 269 / 864 / 532 | |

McNemar (errors only): b=269 (A wrong, B right), c=864 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0020 | [0.0006, 0.0036] | yes (CI excludes 0) |
| recall | 0.1176 | [0.1047, 0.1308] | yes (CI excludes 0) |
| f1 | 0.0749 | [0.0669, 0.0834] | yes (CI excludes 0) |
| roc_auc | 0.0401 | [0.0354, 0.0453] | yes (CI excludes 0) |
| pr_auc | 0.0079 | [0.0068, 0.0091] | yes (CI excludes 0) |

### roberta vs tfidf_svm (n paired = 6000)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [799, 4201]] | [[991, 9], [893, 4107]] |
| prediction agreement [[both phish, roberta only], [tfidf_svm only, both safe]] | [[3786, 417], [330, 1467]] | |
| errors: both correct / roberta wrong only / tfidf_svm wrong only / both wrong | 4775 / 323 / 424 / 478 | |

McNemar (errors only): b=323 (A wrong, B right), c=424 (A right, B wrong), raw p=0.0003, Holm-adjusted p=0.0005, Bonferroni-adjusted p=0.0015 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0017 | [0.0000, 0.0034] | yes (CI excludes 0) |
| recall | 0.0188 | [0.0075, 0.0294] | yes (CI excludes 0) |
| f1 | 0.0119 | [0.0056, 0.0180] | yes (CI excludes 0) |
| roc_auc | 0.0203 | [0.0170, 0.0238] | yes (CI excludes 0) |
| pr_auc | 0.0040 | [0.0033, 0.0048] | yes (CI excludes 0) |

### roberta vs tfidf_lr (n paired = 6000)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [799, 4201]] | [[996, 4], [800, 4200]] |
| prediction agreement [[both phish, roberta only], [tfidf_lr only, both safe]] | [[3871, 332], [333, 1464]] | |
| errors: both correct / roberta wrong only / tfidf_lr wrong only / both wrong | 4865 / 331 / 334 / 470 | |

McNemar (errors only): b=331 (A wrong, B right), c=334 (A right, B wrong), raw p=0.9382, Holm-adjusted p=0.9382, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0005 | [-0.0007, 0.0017] | no — numerical difference only |
| recall | 0.0002 | [-0.0100, 0.0095] | no — numerical difference only |
| f1 | 0.0003 | [-0.0055, 0.0058] | no — numerical difference only |
| roc_auc | 0.0080 | [0.0054, 0.0110] | yes (CI excludes 0) |
| pr_auc | 0.0015 | [0.0010, 0.0021] | yes (CI excludes 0) |

### bilstm vs tfidf_svm (n paired = 6000)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [1387, 3613]] | [[991, 9], [893, 4107]] |
| prediction agreement [[both phish, bilstm only], [tfidf_svm only, both safe]] | [[3441, 181], [675, 1703]] | |
| errors: both correct / bilstm wrong only / tfidf_svm wrong only / both wrong | 4423 / 675 / 181 / 721 | |

McNemar (errors only): b=675 (A wrong, B right), c=181 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0003 | [-0.0022, 0.0017] | no — numerical difference only |
| recall | -0.0988 | [-0.1098, -0.0871] | yes (CI excludes 0) |
| f1 | -0.0630 | [-0.0701, -0.0566] | yes (CI excludes 0) |
| roc_auc | -0.0198 | [-0.0244, -0.0152] | yes (CI excludes 0) |
| pr_auc | -0.0039 | [-0.0050, -0.0029] | yes (CI excludes 0) |

### bilstm vs tfidf_lr (n paired = 6000)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [1387, 3613]] | [[996, 4], [800, 4200]] |
| prediction agreement [[both phish, bilstm only], [tfidf_lr only, both safe]] | [[3497, 125], [707, 1671]] | |
| errors: both correct / bilstm wrong only / tfidf_lr wrong only / both wrong | 4484 / 712 / 120 / 684 | |

McNemar (errors only): b=712 (A wrong, B right), c=120 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0015 | [-0.0034, 0.0002] | no — numerical difference only |
| recall | -0.1174 | [-0.1279, -0.1067] | yes (CI excludes 0) |
| f1 | -0.0746 | [-0.0821, -0.0686] | yes (CI excludes 0) |
| roc_auc | -0.0321 | [-0.0372, -0.0270] | yes (CI excludes 0) |
| pr_auc | -0.0064 | [-0.0076, -0.0053] | yes (CI excludes 0) |

### tfidf_svm vs tfidf_lr (n paired = 6000)

| confusion (rows = truth, cols = predicted) | tfidf_svm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [893, 4107]] | [[996, 4], [800, 4200]] |
| prediction agreement [[both phish, tfidf_svm only], [tfidf_lr only, both safe]] | [[3994, 122], [210, 1674]] | |
| errors: both correct / tfidf_svm wrong only / tfidf_lr wrong only / both wrong | 4981 / 215 / 117 / 687 | |

McNemar (errors only): b=215 (A wrong, B right), c=117 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0012 | [-0.0027, -0.0000] | yes (CI excludes 0) |
| recall | -0.0186 | [-0.0258, -0.0116] | yes (CI excludes 0) |
| f1 | -0.0116 | [-0.0159, -0.0074] | yes (CI excludes 0) |
| roc_auc | -0.0123 | [-0.0150, -0.0097] | yes (CI excludes 0) |
| pr_auc | -0.0025 | [-0.0031, -0.0019] | yes (CI excludes 0) |

## bec (n=5181)

### roberta vs bilstm (n paired = 5181)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | bilstm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [2604, 1577]] | [[991, 9], [3302, 879]] |
| prediction agreement [[both phish, roberta only], [bilstm only, both safe]] | [[506, 1073], [382, 3220]] | |
| errors: both correct / roberta wrong only / bilstm wrong only / both wrong | 1495 / 375 / 1080 / 2231 | |

McNemar (errors only): b=375 (A wrong, B right), c=1080 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0089 | [0.0033, 0.0153] | yes (CI excludes 0) |
| recall | 0.1669 | [0.1494, 0.1844] | yes (CI excludes 0) |
| f1 | 0.2008 | [0.1815, 0.2217] | yes (CI excludes 0) |
| roc_auc | 0.1194 | [0.1072, 0.1317] | yes (CI excludes 0) |
| pr_auc | 0.0361 | [0.0310, 0.0420] | yes (CI excludes 0) |

### roberta vs tfidf_svm (n paired = 5181)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [2604, 1577]] | [[991, 9], [2558, 1623]] |
| prediction agreement [[both phish, roberta only], [tfidf_svm only, both safe]] | [[1024, 555], [608, 2994]] | |
| errors: both correct / roberta wrong only / tfidf_svm wrong only / both wrong | 2013 / 601 / 562 / 2005 | |

McNemar (errors only): b=601 (A wrong, B right), c=562 (A right, B wrong), raw p=0.2652, Holm-adjusted p=0.2652, Bonferroni-adjusted p=1.0000 — not significant after Holm correction. McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0042 | [0.0005, 0.0083] | yes (CI excludes 0) |
| recall | -0.0110 | [-0.0267, 0.0066] | no — numerical difference only |
| f1 | -0.0108 | [-0.0277, 0.0070] | no — numerical difference only |
| roc_auc | 0.0513 | [0.0419, 0.0615] | yes (CI excludes 0) |
| pr_auc | 0.0142 | [0.0109, 0.0176] | yes (CI excludes 0) |

### roberta vs tfidf_lr (n paired = 5181)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [2604, 1577]] | [[996, 4], [2858, 1323]] |
| prediction agreement [[both phish, roberta only], [tfidf_lr only, both safe]] | [[911, 668], [416, 3186]] | |
| errors: both correct / roberta wrong only / tfidf_lr wrong only / both wrong | 1905 / 414 / 670 / 2192 | |

McNemar (errors only): b=414 (A wrong, B right), c=670 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0017 | [-0.0017, 0.0054] | no — numerical difference only |
| recall | 0.0608 | [0.0453, 0.0761] | yes (CI excludes 0) |
| f1 | 0.0672 | [0.0498, 0.0837] | yes (CI excludes 0) |
| roc_auc | 0.0447 | [0.0348, 0.0551] | yes (CI excludes 0) |
| pr_auc | 0.0133 | [0.0101, 0.0167] | yes (CI excludes 0) |

### bilstm vs tfidf_svm (n paired = 5181)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [3302, 879]] | [[991, 9], [2558, 1623]] |
| prediction agreement [[both phish, bilstm only], [tfidf_svm only, both safe]] | [[619, 269], [1013, 3280]] | |
| errors: both correct / bilstm wrong only / tfidf_svm wrong only / both wrong | 1601 / 1013 / 269 / 2298 | |

McNemar (errors only): b=1013 (A wrong, B right), c=269 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0046 | [-0.0115, 0.0023] | no — numerical difference only |
| recall | -0.1779 | [-0.1944, -0.1616] | yes (CI excludes 0) |
| f1 | -0.2116 | [-0.2287, -0.1932] | yes (CI excludes 0) |
| roc_auc | -0.0681 | [-0.0795, -0.0564] | yes (CI excludes 0) |
| pr_auc | -0.0220 | [-0.0272, -0.0174] | yes (CI excludes 0) |

### bilstm vs tfidf_lr (n paired = 5181)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [3302, 879]] | [[996, 4], [2858, 1323]] |
| prediction agreement [[both phish, bilstm only], [tfidf_lr only, both safe]] | [[547, 341], [780, 3513]] | |
| errors: both correct / bilstm wrong only / tfidf_lr wrong only / both wrong | 1534 / 785 / 336 / 2526 | |

McNemar (errors only): b=785 (A wrong, B right), c=336 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0071 | [-0.0141, -0.0004] | yes (CI excludes 0) |
| recall | -0.1062 | [-0.1219, -0.0905] | yes (CI excludes 0) |
| f1 | -0.1336 | [-0.1516, -0.1155] | yes (CI excludes 0) |
| roc_auc | -0.0747 | [-0.0873, -0.0638] | yes (CI excludes 0) |
| pr_auc | -0.0228 | [-0.0276, -0.0184] | yes (CI excludes 0) |

### tfidf_svm vs tfidf_lr (n paired = 5181)

| confusion (rows = truth, cols = predicted) | tfidf_svm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [2558, 1623]] | [[996, 4], [2858, 1323]] |
| prediction agreement [[both phish, tfidf_svm only], [tfidf_lr only, both safe]] | [[1016, 616], [311, 3238]] | |
| errors: both correct / tfidf_svm wrong only / tfidf_lr wrong only / both wrong | 2003 / 316 / 611 / 2251 | |

McNemar (errors only): b=316 (A wrong, B right), c=611 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0025 | [-0.0063, 0.0012] | no — numerical difference only |
| recall | 0.0718 | [0.0575, 0.0855] | yes (CI excludes 0) |
| f1 | 0.0780 | [0.0625, 0.0936] | yes (CI excludes 0) |
| roc_auc | -0.0066 | [-0.0132, 0.0007] | no — numerical difference only |
| pr_auc | -0.0009 | [-0.0030, 0.0012] | no — numerical difference only |

## bec_poisoned (n=5181)

### roberta vs bilstm (n paired = 5181)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | bilstm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [2382, 1799]] | [[991, 9], [3243, 938]] |
| prediction agreement [[both phish, roberta only], [bilstm only, both safe]] | [[595, 1206], [352, 3028]] | |
| errors: both correct / roberta wrong only / bilstm wrong only / both wrong | 1584 / 345 / 1213 / 2039 | |

McNemar (errors only): b=345 (A wrong, B right), c=1213 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0084 | [0.0032, 0.0144] | yes (CI excludes 0) |
| recall | 0.2059 | [0.1887, 0.2240] | yes (CI excludes 0) |
| f1 | 0.2356 | [0.2173, 0.2546] | yes (CI excludes 0) |
| roc_auc | 0.1129 | [0.1015, 0.1250] | yes (CI excludes 0) |
| pr_auc | 0.0341 | [0.0294, 0.0399] | yes (CI excludes 0) |

### roberta vs tfidf_svm (n paired = 5181)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [2382, 1799]] | [[991, 9], [2650, 1531]] |
| prediction agreement [[both phish, roberta only], [tfidf_svm only, both safe]] | [[1047, 754], [493, 2887]] | |
| errors: both correct / roberta wrong only / tfidf_svm wrong only / both wrong | 2036 / 486 / 761 / 1898 | |

McNemar (errors only): b=486 (A wrong, B right), c=761 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0047 | [0.0009, 0.0088] | yes (CI excludes 0) |
| recall | 0.0641 | [0.0477, 0.0812] | yes (CI excludes 0) |
| f1 | 0.0662 | [0.0486, 0.0829] | yes (CI excludes 0) |
| roc_auc | 0.0626 | [0.0527, 0.0730] | yes (CI excludes 0) |
| pr_auc | 0.0175 | [0.0141, 0.0210] | yes (CI excludes 0) |

### roberta vs tfidf_lr (n paired = 5181)

| confusion (rows = truth, cols = predicted) | roberta [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[998, 2], [2382, 1799]] | [[996, 4], [2925, 1256]] |
| prediction agreement [[both phish, roberta only], [tfidf_lr only, both safe]] | [[922, 879], [338, 3042]] | |
| errors: both correct / roberta wrong only / tfidf_lr wrong only / both wrong | 1916 / 336 / 881 / 2048 | |

McNemar (errors only): b=336 (A wrong, B right), c=881 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | 0.0021 | [-0.0014, 0.0058] | no — numerical difference only |
| recall | 0.1299 | [0.1142, 0.1454] | yes (CI excludes 0) |
| f1 | 0.1398 | [0.1220, 0.1562] | yes (CI excludes 0) |
| roc_auc | 0.0590 | [0.0486, 0.0700] | yes (CI excludes 0) |
| pr_auc | 0.0174 | [0.0141, 0.0210] | yes (CI excludes 0) |

### bilstm vs tfidf_svm (n paired = 5181)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_svm [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [3243, 938]] | [[991, 9], [2650, 1531]] |
| prediction agreement [[both phish, bilstm only], [tfidf_svm only, both safe]] | [[629, 318], [911, 3323]] | |
| errors: both correct / bilstm wrong only / tfidf_svm wrong only / both wrong | 1611 / 911 / 318 / 2341 | |

McNemar (errors only): b=911 (A wrong, B right), c=318 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0037 | [-0.0103, 0.0032] | no — numerical difference only |
| recall | -0.1418 | [-0.1581, -0.1254] | yes (CI excludes 0) |
| f1 | -0.1694 | [-0.1868, -0.1511] | yes (CI excludes 0) |
| roc_auc | -0.0502 | [-0.0617, -0.0391] | yes (CI excludes 0) |
| pr_auc | -0.0166 | [-0.0217, -0.0122] | yes (CI excludes 0) |

### bilstm vs tfidf_lr (n paired = 5181)

| confusion (rows = truth, cols = predicted) | bilstm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [3243, 938]] | [[996, 4], [2925, 1256]] |
| prediction agreement [[both phish, bilstm only], [tfidf_lr only, both safe]] | [[557, 390], [703, 3531]] | |
| errors: both correct / bilstm wrong only / tfidf_lr wrong only / both wrong | 1544 / 708 / 385 / 2544 | |

McNemar (errors only): b=708 (A wrong, B right), c=385 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0063 | [-0.0131, 0.0001] | no — numerical difference only |
| recall | -0.0761 | [-0.0918, -0.0609] | yes (CI excludes 0) |
| f1 | -0.0958 | [-0.1138, -0.0780] | yes (CI excludes 0) |
| roc_auc | -0.0538 | [-0.0660, -0.0426] | yes (CI excludes 0) |
| pr_auc | -0.0167 | [-0.0215, -0.0123] | yes (CI excludes 0) |

### tfidf_svm vs tfidf_lr (n paired = 5181)

| confusion (rows = truth, cols = predicted) | tfidf_svm [[TN, FP], [FN, TP]] | tfidf_lr [[TN, FP], [FN, TP]] |
|---|---|---|
| confusion matrix | [[991, 9], [2650, 1531]] | [[996, 4], [2925, 1256]] |
| prediction agreement [[both phish, tfidf_svm only], [tfidf_lr only, both safe]] | [[937, 603], [323, 3318]] | |
| errors: both correct / tfidf_svm wrong only / tfidf_lr wrong only / both wrong | 1924 / 328 / 598 / 2331 | |

McNemar (errors only): b=328 (A wrong, B right), c=598 (A right, B wrong), raw p=0.0000, Holm-adjusted p=0.0000, Bonferroni-adjusted p=0.0000 — SIGNIFICANT (Holm p < 0.05). McNemar is NOT a precision test.

| paired difference (A − B) | value | 95% CI | significant? |
|---|---|---|---|
| precision | -0.0027 | [-0.0067, 0.0012] | no — numerical difference only |
| recall | 0.0658 | [0.0515, 0.0791] | yes (CI excludes 0) |
| f1 | 0.0735 | [0.0569, 0.0884] | yes (CI excludes 0) |
| roc_auc | -0.0036 | [-0.0107, 0.0042] | no — numerical difference only |
| pr_auc | -0.0001 | [-0.0023, 0.0021] | no — numerical difference only |

## Hypotheses (primary OOD, PhishNChips v5.2)

### H1a — RoBERTa better RANKING than BiLSTM: **SUPPORTED**

- roc_auc_diff_roberta_minus_bilstm: diff 0.0986, 95% CI [0.0760, 0.1247], significant: True
- pr_auc_diff_roberta_minus_bilstm: diff 0.1179, 95% CI [0.0878, 0.1461], significant: True

### H1b — RoBERTa better OPERATIONAL detection (recall floor 0.85): **REJECTED — NEITHER model meets the predefined recall floor 0.85 on the primary OOD set, at threshold 0.5 or at any validation-fitted threshold**

- roberta: recall at 0.5 = 0.0280, at fitted 0.85 = 0.0000, at fitted 0.90 = 0.0000, at fitted 0.95 = 0.0070; floor met at 0.5: False, at any fitted threshold: False
- bilstm: recall at 0.5 = 0.0550, at fitted 0.85 = 0.0030, at fitted 0.90 = 0.0060, at fitted 0.95 = 0.0260; floor met at 0.5: False, at any fitted threshold: False

H1a and H1b are independent: a ranking win is not a deployment win, and a deployment failure is not a ranking failure.

## Status

TF-IDF pairs: COMPLETED

Generated by `compute_ood_stats_expanded.py` from saved prediction CSVs only (never re-scored). Bootstrap: 1000 resamples, seed 42.
