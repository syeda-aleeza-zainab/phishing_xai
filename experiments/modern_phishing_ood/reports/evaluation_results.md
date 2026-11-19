# Evaluation results — modern-phishing OOD experiment

All numbers below are executed numbers from Kaggle Kernels 1–3
(GPU T4) unless marked NOT RUN YET. Threshold 0.5 unless stated.
Every table cell traces to a metrics JSON path under
`experiments/modern_phishing_ood/results/` (or `results/RESULTS.md`
for the in-distribution reference). Machine-readable versions:
`results/ood_tables.json`, `results/ood_stats.json`.

**Caveat on every result here:** the primary set (PhishNChips v5.2) is
a modern URL-grounded synthetic-stimuli OOD set, not real-world inbox
mail. The synthetic-track sets are labelled synthetic throughout.

## Headline (threshold 0.5)

| Set (n) | Model | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | CM [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|---|---|
| PhishNChips (2,000) | RoBERTa | 0.5130 | 0.9333 | 0.0280 | 0.0544 | 0.7787 | 0.7947 | 0.0020 | [[998,2],[972,28]] |
| PhishNChips (2,000) | BiLSTM | 0.5230 | 0.8594 | 0.0550 | 0.1034 | 0.6801 | 0.6768 | 0.0090 | [[991,9],[945,55]] |
| PhishFuzzer binary (205) | RoBERTa | 0.5122 | 0.5109 | 0.6796 | 0.5833 | 0.5711 | 0.6145 | 0.6569 | [[35,67],[33,70]] |
| PhishFuzzer binary (205) | BiLSTM | 0.5268 | 0.5312 | 0.4951 | 0.5126 | 0.5149 | 0.5770 | 0.4412 | [[57,45],[52,51]] |
| Frontiers LLM + legit (5,986) | RoBERTa | 0.9492 | 0.9996 | 0.9394 | 0.9686 | 0.9994 | 0.9998 | 0.0020 | [[998,2],[302,4684]] |
| Frontiers LLM + legit (5,986) | BiLSTM | 0.5643 | 0.9962 | 0.4787 | 0.6467 | 0.9542 | 0.9883 | 0.0090 | [[991,9],[2599,2387]] |
| Frontiers human + legit (6,000) | RoBERTa | 0.8665 | 0.9995 | 0.8402 | 0.9130 | 0.9902 | 0.9981 | 0.0020 | [[998,2],[799,4201]] |
| Frontiers human + legit (6,000) | BiLSTM | 0.7673 | 0.9975 | 0.7226 | 0.8381 | 0.9501 | 0.9901 | 0.0090 | [[991,9],[1387,3613]] |
| BEC + legit (5,181) | RoBERTa | 0.4970 | 0.9987 | 0.3772 | 0.5476 | 0.9513 | 0.9880 | 0.0020 | [[998,2],[2604,1577]] |
| BEC + legit (5,181) | BiLSTM | 0.3609 | 0.9899 | 0.2102 | 0.3468 | 0.8320 | 0.9518 | 0.0090 | [[991,9],[3302,879]] |
| BEC poisoned + legit (5,181) | RoBERTa | 0.5399 | 0.9989 | 0.4303 | 0.6015 | 0.9565 | 0.9894 | 0.0020 | [[998,2],[2382,1799]] |
| BEC poisoned + legit (5,181) | BiLSTM | 0.3723 | 0.9905 | 0.2243 | 0.3658 | 0.8436 | 0.9553 | 0.0090 | [[991,9],[3243,938]] |

Sources: `results/phishnchips_{roberta,bilstm}_metrics.json`,
`results/phishfuzzer_{roberta,bilstm}_metrics.json`,
`results/frontiers_{llm,human}_{roberta,bilstm}_metrics.json`,
`results/bec{,_poisoned}_{roberta,bilstm}_metrics.json`
(`metrics_at_0_5`).

Reading: on the primary set both models collapse to near-zero recall
at 0.5 — the phishing prose is benign-sounding and the signal is in
the URL, which text-only models cannot use. Ranking survives better
(RoBERTa ROC-AUC 0.779 vs BiLSTM 0.680). On the synthetic track both
models detect LLM phishing well (RoBERTa far better), BEC is hard for
both, and poisoning makes BEC *more* detectable (see below). On the
small real-human PhishFuzzer slice both models are near chance
(ROC-AUC 0.57 / 0.51) with very high false-positive rates.

## Table A — original vs modern

In-distribution test (n=3,495, threshold 0.5, `results/RESULTS.md`):
RoBERTa P 0.9690 / R 0.9932 / F1 0.9809 / acc 0.9854 / ROC-AUC 0.9986;
BiLSTM P 0.9719 ± 0.0015 / R 0.9428 / F1 0.9571 (3 seeds).

| Model | Metric | In-distribution test | Modern OOD (PhishNChips, n=2,000) | Change |
|---|---|---|---|---|
| RoBERTa | Precision | 0.9690 | 0.9333 | −0.036 |
| RoBERTa | Recall | 0.9932 | 0.0280 | −0.965 |
| RoBERTa | F1 | 0.9809 | 0.0544 | −0.927 |
| RoBERTa | Accuracy | 0.9854 | 0.5130 | −0.472 |
| RoBERTa | ROC-AUC | 0.9986 | 0.7787 | −0.220 |
| BiLSTM | Precision | 0.9719 | 0.8594 | −0.113 |
| BiLSTM | Recall | 0.9428 | 0.0550 | −0.888 |
| BiLSTM | F1 | 0.9571 | 0.1034 | −0.854 |
| BiLSTM | Accuracy | — (not recorded) | 0.5230 | — |
| BiLSTM | ROC-AUC | — (not recorded) | 0.6801 | — |

The collapse is a recall collapse, not a precision collapse: at 0.5
both models still almost never flag legitimate mail (FPR 0.002 /
0.009), but they miss 97% / 94% of the phishing. The in-distribution
numbers describe a different task (classic phishing prose) than the
synthetic-stimuli set (benign prose + malicious URL).

## Table B — human vs AI (synthetic track only)

Recall at 0.5, each phishing set paired with the same PhishNChips
legit 1,000 (FPR on that shared legit set: RoBERTa 0.0020, BiLSTM
0.0090 in both pairings).

| Model | Human-authored phishing (old corpora, n=5,000) | LLM-generated phishing (2026, n=4,986) | Δ (LLM − human) |
|---|---|---|---|
| RoBERTa recall | 0.8402 | 0.9394 | +0.0992 |
| RoBERTa F1 | 0.9130 | 0.9686 | +0.0556 |
| RoBERTa ROC-AUC | 0.9902 | 0.9994 | +0.0092 |
| BiLSTM recall | 0.7226 | 0.4787 | −0.2439 |
| BiLSTM F1 | 0.8381 | 0.6467 | −0.1914 |
| BiLSTM ROC-AUC | 0.9501 | 0.9542 | +0.0041 |

RoBERTa detects LLM phishing *better* than human phishing; BiLSTM
detects it *worse* (recall 0.48 vs 0.72) — the two models diverge on
LLM prose. The human side is 2007–2008 corpora and is NOT a modern
claim; it is the comparison slice only.

Per-LLM-generator recall (from `per_category_at_0_5` in
`results/frontiers_llm_{roberta,bilstm}_metrics.json`, aggregating the
five topics per generator):

| Generator | RoBERTa recall | BiLSTM recall |
|---|---|---|
| GPT-4.1 (n=1,665) | 0.9207 | 0.4162 |
| DeepSeek 3.2 (n=1,665) | 0.9508 | 0.5225 |
| Llama 3.3 70B (n=1,656) | 0.9469 | 0.4976 |

By topic (both models struggle most on HR): RoBERTa HR recall 0.7028
vs banking 1.0000 / parcel_delivery 1.0000 / tax_irs 0.9990 /
it_support 0.9950; BiLSTM HR recall 0.0653 vs banking 0.7715 /
parcel_delivery 0.5495 / tax_irs 0.5721 / it_support 0.4342. The HR
topic — internal, low-urgency prose — is the hardest LLM-phishing
slice for both models, and by far the hardest for BiLSTM.

## Table C — temporal

**NOT RUN YET.** No per-email timestamps exist in any OOD set
(PhishNChips has only the April 2026 release date; PhishFuzzer has a
Year field only, 2024–2026). Temporal degradation cannot be measured
without dated emails. No temporal plot was produced.

## Table D — robustness (PhishNChips, F1 at 0.5; Kernel 3)

| Model | Clean | Truncated (first 128 tokens) | URLs removed | Typo/synonym (20%) |
|---|---|---|---|---|
| RoBERTa | 0.0544 | 0.0544 | 0.0619 | 0.1519 |
| BiLSTM | 0.1034 | 0.1034 | 0.0926 | 0.1342 |

Sources: `results/robustness_{roberta,bilstm}.json`. Full analysis in
`robustness_report.md`. Short version: truncation changes nothing
(bodies are already short); URL removal barely moves F1; typo/synonym
*raises* F1 for both models (ranking/ROC-AUC is essentially unchanged:
RoBERTa 0.779 → 0.767–0.782; BiLSTM 0.680 → 0.672–0.688), which reads
as a threshold artifact on a tiny-TP base (RoBERTa clean TP = 28),
not as robustness — see the robustness report before quoting this.

## Table E — statistical comparison

Full numbers in `statistical_report.md` and
`results/ood_stats.json`. Summary (McNemar is errors-only, NOT a
precision test; paired CIs are RoBERTa − BiLSTM, 1,000 resamples,
seed 42):

| Set | McNemar (RoBERTa vs BiLSTM) | Precision diff [95% CI] | Recall diff [95% CI] | F1 diff [95% CI] |
|---|---|---|---|---|
| PhishNChips (2,000) | b=51, c=31, p=0.0359 | +0.074 [−0.033, +0.180] | −0.027 [−0.043, −0.010] | −0.049 [−0.080, −0.018] |
| PhishFuzzer (205) | b=47, c=44, p=0.8339 | −0.020 [−0.108, +0.056] | +0.184 [+0.064, +0.308] | +0.071 [−0.024, +0.165] |
| Frontiers LLM (5,986) | b=11, c=2315, p≈0.0 | +0.003 [+0.001, +0.006] | +0.461 [+0.446, +0.475] | +0.322 [+0.310, +0.334] |
| Frontiers human (6,000) | b=269, c=864, p=1.1e-69 | +0.002 [+0.001, +0.004] | +0.118 [+0.105, +0.131] | +0.075 [+0.067, +0.083] |
| BEC (5,181) | b=375, c=1080, p=4.7e-76 | +0.009 [+0.003, +0.015] | +0.167 [+0.149, +0.184] | +0.201 [+0.182, +0.222] |
| BEC poisoned (5,181) | b=345, c=1213, p=6.2e-107 | +0.008 [+0.003, +0.014] | +0.206 [+0.189, +0.224] | +0.236 [+0.217, +0.255] |

## Table F — efficiency

Full numbers in `efficiency_report.md`. Median ms/email, batch 1
(three kernel runs; T4 instances vary — ranges, not single numbers):

| Model | GPU median (K1 / K2 / K3) | CPU median (K1 / K2 / K3) | Params | Checkpoint size |
|---|---|---|---|---|
| RoBERTa | 9.54 / 16.84 / 10.01 ms | 129.73 / 254.19 / 133.11 ms | 124,647,170 | 498.7 MB |
| BiLSTM | 6.23 / 8.33 / 6.42 ms | 11.09 / 18.92 / 12.09 ms | 6,236,034 | 25.6 MB |

RoBERTa is ~20× the parameters and ~1.5–2.0× the GPU latency /
~12–14× the CPU latency of BiLSTM. Latency n: GPU 200, CPU 100 per
run (RoBERTa CPU n=100 by design).

## Validation-fitted thresholds (fit on original validation only)

| Model | Floor 0.85 | Floor 0.90 | Floor 0.95 |
|---|---|---|---|
| RoBERTa | 0.9989 | 0.9981 | 0.9913 |
| BiLSTM | 0.9985 | 0.9910 | 0.8769 |

Sources: `ood_metrics_combined_kernel{1,2}.json` (`thresholds_used`).
Validation precision/recall at those thresholds: RoBERTa — 0.85 →
val P 0.9978 / R 0.8911; 0.90 → val P 0.9969 / R 0.9171; 0.95 → val
P 0.9920 / R 0.9557. BiLSTM (also in `results/thresholds.json`, the
last-written model): 0.85 → val P 0.9879 / R 0.8642; 0.90 → val
P 0.9853 / R 0.9066; 0.95 → val P 0.9802 / R 0.9518. Cross-check vs the
pure-numpy mirror: max abs threshold difference 0.0.

Applied to OOD, these thresholds transfer catastrophically badly:

| Set / Model | Recall @0.5 | Recall @floor-0.85 thr | Recall @floor-0.90 thr | Recall @floor-0.95 thr |
|---|---|---|---|---|
| PhishNChips / RoBERTa | 0.0280 | 0.0000 | 0.0000 | 0.0070 |
| PhishNChips / BiLSTM | 0.0550 | 0.0030 | 0.0060 | 0.0260 |
| PhishFuzzer / RoBERTa | 0.6796 | 0.2330 | 0.3010 | 0.5146 |
| PhishFuzzer / BiLSTM | 0.4951 | 0.2913 | 0.3495 | 0.4175 |
| Frontiers LLM / RoBERTa | 0.9394 | 0.3181 | 0.5223 | 0.7808 |
| Frontiers LLM / BiLSTM | 0.4787 | 0.1410 | 0.2302 | 0.3751 |
| Frontiers human / RoBERTa | 0.8402 | 0.6110 | 0.6580 | 0.7368 |
| Frontiers human / BiLSTM | 0.7226 | 0.5150 | 0.5914 | 0.6776 |
| BEC / RoBERTa | 0.3772 | 0.0356 | 0.0608 | 0.1509 |
| BEC / BiLSTM | 0.2102 | 0.0828 | 0.1100 | 0.1641 |
| BEC poisoned / RoBERTa | 0.4303 | 0.0531 | 0.0875 | 0.1913 |
| BEC poisoned / BiLSTM | 0.2243 | 0.0892 | 0.1232 | 0.1787 |

In-distribution validation probabilities are extremely confident
(thresholds ≈ 0.99), so a threshold fitted for recall 0.85 on
validation admits almost nothing OOD. Any deployment that carried
these thresholds to modern mail would detect ~nothing on the primary
set. This is a finding about threshold transfer, not about the
models' ranking (ROC-AUC is threshold-free and is reported alongside).

## PhishNChips per-category (threshold 0.5)

Recall per phishing URL category / FPR per legit category
(`per_category_at_0_5` in `results/phishnchips_{roberta,bilstm}_metrics.json`):

| Category | n | RoBERTa recall | BiLSTM recall | Note |
|---|---|---|---|---|
| github_pages (phish) | 227 | 0.0000 | 0.0617 | RoBERTa catches none |
| google_docs (phish) | 196 | 0.0000 | 0.1224 | BiLSTM best slice |
| short_clean_https (phish) | 190 | 0.1000 | 0.0211 | RoBERTa's only real slice (19/190) |
| url_shortener (phish) | 154 | 0.0584 | 0.0260 | |
| ipfs (phish) | 102 | 0.0000 | 0.0098 | |
| firebase (phish) | 101 | 0.0000 | 0.0594 | |
| hosting_platform (phish) | 17 | 0.0000 | 0.0000 | |
| other_https (phish) | 13 | 0.0000 | 0.1538 | tiny n |
| legitimate (legit) | 667 | FPR 0.0015 | FPR 0.0000 | |
| cross_domain_legitimate (legit) | 333 | FPR 0.0030 | FPR 0.0270 | BiLSTM's 9 FPs are all here |

By datasource (phishing): PhishTank (n=700) recall RoBERTa 0.0300 /
BiLSTM 0.0543; GitHub DB live (172) 0.0291 / 0.0581; GitHub DB (62)
0.0161 / 0.0968; OpenPhish (62) 0.0161 / 0.0161; OpenPhish-live (4)
0.0000 / 0.0000. By datasource (legit): Tranco (662) FPR RoBERTa
0.0015 / BiLSTM 0.0000; cross-domain expansion (333) 0.0030 / 0.0270;
adversarial legit (5) 0.0000 / 0.0000.

## BEC clean-vs-poisoned contrast (paired, 1,187 actually-changed rows)

| Model | Recall clean (changed rows) | Recall poisoned (changed rows) | Δ | Median prob change |
|---|---|---|---|---|
| RoBERTa | 0.3555 | 0.5324 | +0.1769 | +0.0040 |
| BiLSTM | 0.1963 | 0.2367 | +0.0404 | +0.0000 |

Source: `ood_metrics_combined_kernel2.json` (`bec_poisoned_contrast`).
The homoglyph/invisible-Unicode poisoning made BEC emails MORE
detectable, not less — the obfuscation itself is a signal (unusual
characters are rare in the models' legitimate training text).
Unchanged rows (2,994): RoBERTa predictions exactly identical
(max abs diff 0.0); BiLSTM max abs diff 4.4e-08 — GPU float
nondeterminism, not a real difference.

## PhishFuzzer spam rows (reported separately, never in binary metrics)

| Model | Spam flagged as phishing @0.5 | Median prob |
|---|---|---|
| RoBERTa | 58 / 95 (0.6105) | 0.7890 |
| BiLSTM | 37 / 95 (0.3895) | 0.0127 |

RoBERTa flags most spam as phishing (spam prose resembles phishing
prose to it); BiLSTM's median spam probability is near zero. Neither
number enters the binary metrics.
