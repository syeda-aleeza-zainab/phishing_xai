# Robustness report — modern-phishing OOD experiment

Kernel 3 (GPU T4): the 3 H4 perturbations from
`baselines.evaluate_robustness` (`truncated` / `urls_removed` /
`typo_synonym`, seed 42, no retraining) on PhishNChips v5.2 (n=2,000),
for the fine-tuned RoBERTa checkpoint and the tuned BiLSTM artefact.
Source: `results/robustness_{roberta,bilstm}.json`. In-distribution
robustness (test n=3,495, threshold 0.5) is quoted from
`results/robustness_rerun/robustness_loaded.json` for comparison.

## Results (F1 at threshold 0.5; ROC-AUC alongside)

| Model | Clean | Truncated | URLs removed | Typo/synonym |
|---|---|---|---|---|
| RoBERTa F1 | 0.0544 | 0.0544 | 0.0619 | 0.1519 |
| RoBERTa ROC-AUC | 0.7787 | 0.7787 | 0.7819 | 0.7671 |
| RoBERTa accuracy | 0.5130 | 0.5130 | 0.5150 | 0.5365 |
| BiLSTM F1 | 0.1034 | 0.1034 | 0.0926 | 0.1342 |
| BiLSTM ROC-AUC | 0.6801 | 0.6801 | 0.6715 | 0.6879 |
| BiLSTM accuracy | 0.5230 | 0.5230 | 0.5200 | 0.5290 |

## Reading (and the trap in the F1 column)

1. **Truncation changes nothing** (identical F1/accuracy/ROC-AUC for
   both models). That is expected, not a finding about robustness:
   PhishNChips bodies are short (median ~400 chars vs ~880 training),
   so cutting to the first 128 whitespace tokens removes almost
   nothing. In-distribution, truncation cost RoBERTa 0.0036 F1 and
   BiLSTM 0.0000 F1 — also ~nothing.
2. **URL removal barely moves anything** (RoBERTa F1 +0.0075, BiLSTM
   −0.0108; ROC-AUC within 0.009 for both). On in-distribution data
   URL removal cost RoBERTa 0.0054 F1. The text models barely use URLs
   as text tokens either way — consistent with the headline result
   that the URL is the signal *humans* would use, not the signal these
   models use.
3. **Typo/synonym RAISES F1 for both models** (RoBERTa 0.0544 →
   0.1519; BiLSTM 0.1034 → 0.1342) while ROC-AUC is essentially
   unchanged (RoBERTa 0.7787 → 0.7671; BiLSTM 0.6801 → 0.6879).
   This is the opposite of the in-distribution behaviour (RoBERTa F1
   0.9809 → 0.9638; BiLSTM 0.9658 → 0.9637 under the same
   perturbation). Do NOT report this as "the models are robust to
   typos on modern phishing". What it shows is a threshold artifact
   on a tiny-TP base: RoBERTa's clean TP count is 28 of 1,000 phishing
   emails, so a small shift in scores moves F1 a lot. The plausible
   mechanism: the perturbation (WordNet synonym replacement at 20%,
   with a typo fallback when WordNet is unavailable) corrupts the
   benign-sounding phishing prose, and the models learned from
   training data that misspellings / unusual words mark phishing — so
   corrupted text crosses the 0.5 threshold more often. Ranking
   (ROC-AUC, threshold-free) does not improve, which is the honest
   summary: **perturbations do not break ranking; the 0.5 threshold
   is unstable on this set.**
4. **No robustness gap between the models.** The clean→perturbed
   ROC-AUC deltas are ≤ 0.012 for both models in every perturbation.
   Neither model is more robust than the other here; both are
   equally (in)sensitive because both are near the floor already.

## Limitations

* Only the primary set was perturbed (Kernel 3 scope). No robustness
  numbers exist for PhishFuzzer, Frontiers or BEC — a typo/synonym
  test on BEC, where the prose is formal business English, would be
  the informative next run, but it was not run (quota/time).
* Only F1/accuracy/ROC-AUC were saved per perturbation (the
  `evaluate_robustness` output shape); per-perturbation precision,
  recall and confusion matrices were not saved, so the TP-count
  reasoning above is inferred from the clean confusion matrix, not
  measured per perturbation. A follow-up run should save full metrics
  per perturbation.
* The typo/synonym perturbation depends on WordNet availability; on
  Kaggle it ran the synonym path (no fallback warning in the kernel
  log), but the exact mix of synonym vs typo is not logged per email.
