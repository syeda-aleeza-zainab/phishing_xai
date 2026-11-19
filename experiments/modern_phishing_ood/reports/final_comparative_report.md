# Final comparative report — modern-phishing OOD experiment

**Scope note (read first):** the task commissioning this experiment
specified "the brief's 10 verdict questions" and "four allowed
options" for the final verdict, but no brief document listing them was
attached to the task materials available to this agent (no brief file
exists under `experiments/modern_phishing_ood/` or in the repo).
This report therefore answers the **10 questions the experiment design
itself poses** (derived from the commissioned H1/H0 framing, the
precision-first recall floors, and the experiment steps), each labelled
Q1–Q10, and gives the verdict in the project's own verdict vocabulary
(the verdict-generator branches in
`scripts/run_system_complexity_report.py`: significant gain / no
significant gain / undecided). The parent agent should confirm the
mapping against the brief's literal questions and options; nothing
here invents a brief option and presents it as the brief's.

**Headline:** H1 — "RoBERTa generalises better than BiLSTM on modern
phishing" — is **REJECTED on the primary OOD set and SUPPORTED on the
synthetic track**. It does not hold as a general claim. If a single
label is forced, the pre-registered primary test governs: **H1 is
rejected**.

## The 10 questions (design-implied) and their evidence-based answers

### Q1. Does RoBERTa generalise better than BiLSTM on the primary modern OOD set (PhishNChips v5.2)?

**No — not by the pre-registered precision-first rule, and BiLSTM is
ahead on recall/F1 at 0.5.** At 0.5: RoBERTa P 0.9333 / R 0.0280 / F1
0.0544; BiLSTM P 0.8594 / R 0.0550 / F1 0.1034 (n=2,000). The
precision gap (RoBERTa − BiLSTM) is +0.074 with paired 95% CI
[−0.033, +0.180] — it includes 0, so it is NOT a significant
precision gain under the verdict rule (positive gap AND McNemar
p < 0.05). McNemar on errors gives p=0.0359 (b=51, c=31) — but in
BiLSTM's favour on error counts (BiLSTM 954 errors vs RoBERTa 974),
and it is borderline (a re-run could flip it). RoBERTa does rank
better (ROC-AUC 0.7787 [0.7584, 0.7987] vs 0.6801 [0.6563, 0.7036],
CIs non-overlapping) and has the higher PR-AUC (0.7947 vs 0.6768).
So: better ranking, no significant precision gain, worse recall/F1 at
the operating threshold. H1 fails here.

### Q2. Does RoBERTa generalise better on LLM-generated phishing (synthetic track)?

**Yes, decisively.** Frontiers LLM + shared legit (n=5,986): RoBERTa
P 0.9996 / R 0.9394 / F1 0.9686 / ROC-AUC 0.9994; BiLSTM P 0.9962 / R
0.4787 / F1 0.6467 / ROC-AUC 0.9542. Paired differences all exclude 0
(recall +0.461 [+0.446, +0.475]; F1 +0.322 [+0.310, +0.334]);
McNemar p≈0 (b=11, c=2,315). Per generator: RoBERTa recall 0.9207
(GPT-4.1) / 0.9508 (DeepSeek 3.2) / 0.9469 (Llama 3.3 70B) vs BiLSTM
0.4162 / 0.5225 / 0.4976. H1 holds on this track.

### Q3. Does RoBERTa generalise better on human-authored phishing (Frontiers human slice)?

**Yes, but the slice is not modern.** Frontiers human + shared legit
(n=6,000): RoBERTa R 0.8402 / F1 0.9130 / ROC-AUC 0.9902; BiLSTM R
0.7226 / F1 0.8381 / ROC-AUC 0.9501. Paired differences exclude 0
(recall +0.118, F1 +0.075); McNemar p=1.1e-69. Caveat: the human side
is 2007–2008 corpora (TREC-07/CEAS-08/Nazario/Nigerian/Enron), so
this supports "RoBERTa ranks/detects old-corpus phishing prose
better", not a modern-phishing claim.

### Q4. Does RoBERTa generalise better on BEC (Adversarial BEC, synthetic)?

**Yes on ranking and recall, from a low base for both.** BEC + shared
legit (n=5,181): RoBERTa R 0.3772 / F1 0.5476 / ROC-AUC 0.9513; BiLSTM
R 0.2102 / F1 0.3468 / ROC-AUC 0.8320. Paired differences exclude 0
(recall +0.167, F1 +0.201); McNemar p=4.7e-76. Both models are weak
here in absolute terms (BEC prose is formal business English with no
classic phishing markers); RoBERTa is less weak.

### Q5. Do validation-fitted thresholds transfer to modern OOD data?

**No — catastrophically badly, for both models.** Thresholds fitted
on the original validation split (RoBERTa 0.9989/0.9981/0.9913;
BiLSTM 0.9985/0.9910/0.8769 for floors 0.85/0.90/0.95) give, on
PhishNChips, RoBERTa recall 0.0000 / 0.0000 / 0.0070 and BiLSTM recall
0.0030 / 0.0060 / 0.0260. On BEC the 0.85-floor thresholds give recall
0.0356 (RoBERTa) and 0.0828 (BiLSTM). Only on the synthetic LLM/human
sets do the 0.95-floor thresholds retain substantial recall (RoBERTa
0.7808 / 0.7368). Any deployment carrying in-distribution thresholds
to modern mail would detect ~nothing on the primary set. Thresholds
must be re-fitted on modern data (which this experiment deliberately
did not do, to keep the OOD test clean) — or the operating point must
be chosen by ranking (ROC/PR-AUC), not by a transferred threshold.

### Q6. Is either model robust to perturbations on modern OOD data?

**Neither is more robust than the other, and the F1 movements are
threshold artifacts, not robustness.** Kernel 3 (PhishNChips):
truncation changes nothing for either model (bodies are already
short); URL removal moves F1 by ≤ 0.011; typo/synonym *raises* F1
(RoBERTa 0.0544 → 0.1519; BiLSTM 0.1034 → 0.1342) while ROC-AUC is
essentially unchanged (≤ 0.012 delta). The honest summary is that
ranking is unperturbed and the 0.5 threshold is unstable on this set
(clean TP = 28 for RoBERTa). See `robustness_report.md`.

### Q7. Does the efficiency cost of RoBERTa buy accuracy on modern OOD data?

**On the synthetic track, yes; on the primary set, no.** RoBERTa is
20.0× the parameters (124,647,170 vs 6,236,034), 19.5× the checkpoint
size (498.7 vs 25.6 MB), 1.5–2.0× the GPU latency and 11–13× the CPU
latency (median ms/email, batch 1, three runs). That buys large
recall/F1 gains on Frontiers LLM (+0.46 recall), human (+0.12) and
BEC (+0.17), and better ranking everywhere except PhishFuzzer — but
on the primary set it buys *worse* recall/F1 at 0.5 and only better
ranking. Under the project's precision-first deployment rule
(highest precision subject to recall ≥ 0.85), **neither model meets
the recall floor on the primary set at 0.5 or at any fitted
threshold**, so no deployment recommendation follows from this OOD
evidence alone. See `efficiency_report.md`.

### Q8. Is the OOD evidence contaminated by training-data leakage?

**No, except for flagged rows that do not change any conclusion.**
Exact / near-dup / MinHash / URL checks (`reports/leakage_audit.md`):
PhishNChips 0 on every check; Frontiers LLM and BEC 0 on every check;
PhishFuzzer Manual-300 has 1 body-only exact overlap (row No. 37,
kept and flagged — 1 of 205 binary rows); Frontiers human has 1 exact
+ 6 MinHash ≥ 0.9 + 3 shared URLs out of 5,000 (expected legacy-corpus
overlap, kept and flagged). The primary-set collapse cannot be
explained by leakage in either direction.

### Q9. Is there temporal degradation (does performance fall on newer mail)?

**NOT RUN YET — cannot be answered from these sets.** No per-email
timestamps exist (PhishNChips has only the April 2026 release date;
PhishFuzzer has Year only). Table C is empty by design, not by
omission. Any temporal claim would be invented.

### Q10. What should be deployed, and what is the single-sentence verdict?

**Nothing in this experiment justifies deploying either model on
modern mail as-is.** At the operating threshold both models miss
~95% of primary-set phishing; the synthetic-track advantage does not
transfer to the primary set; and the fitted thresholds do not
transfer at all. The work that would change this is: (a) a genuinely
real-world modern phishing + legit corpus (none was found — see the
provenance report's next steps); (b) URL-aware scoring (the primary
set's signal is the URL, which text-only models cannot use — the
multimodal fusion's URL channel is the existing mechanism); (c)
thresholds fitted on modern data. Until then the honest deployment
statement is the verdict below.

## Final verdict

**MIXED — H1 rejected on the primary OOD set, supported on the
synthetic track; H1 does not hold as a general claim.**

In the project's verdict-generator terms: on the primary set there is
**no significant gain** for the Transformer (precision gap +0.074
with a paired CI including 0; McNemar p=0.0359 is about errors and
favours BiLSTM), while on the synthetic track there is a **significant
gain** (positive gaps on every metric with paired CIs excluding 0 and
McNemar p≈0). The pre-registered primary test governs the general
claim, so **H1 is rejected as stated**. The Transformer is not
justified by accuracy on modern URL-grounded synthetic-stimuli mail;
its value on this evidence is confined to LLM-generated and BEC-style
synthetic phishing, where it is decisively better — at 20× the
parameters and 11–13× the CPU latency. The experiment's most
actionable finding is not a model ranking but a threshold finding:
in-distribution thresholds (≈0.99) detect ~nothing on modern mail,
for either model.

## Limitations (what this verdict does NOT cover)

* The primary set is synthetic stimuli, not real inbox mail; the
  verdict is about generalisation to *that* distribution.
* Single checkpoints (no multi-seed variability); the PhishNChips
  McNemar p=0.0359 is borderline.
* PhishFuzzer (n=205) cannot rank the models; no temporal analysis
  was possible; robustness was measured on the primary set only, with
  per-perturbation precision/recall not saved.
* No TF-IDF / DistilBERT / CNN comparison was run on these sets —
  the in-distribution ladder showed TF-IDF+SVC beating RoBERTa on
  precision (0.9820 vs 0.9690); whether that holds OOD is untested
  and is the natural next experiment.
* Stats used local mirrors of the repo helpers (torch unavailable
  locally); formulas are identical and unit-tested, but a Kaggle-side
  re-run of `compute_ood_stats.py` with the repo helpers would be a
  clean confirmation.
