# Leakage audit — modern-phishing OOD sets vs training data

**Method (executed locally, CPU, 2026-11-20; Kaggle Kernel 1 re-runs the
same code with the repo `load_data` + `deduplicate_data` path):**
`experiments/modern_phishing_ood/scripts/run_ood_eval.py::run_leakage_audit`
via `audit_leakage_local.py`. Training texts: all non-null rows of
`data/Phishing_Email.csv` (n = 18,634 after dropping 16 null rows;
17,486 unique normalized hashes). Local run used the pandas fallback
for training texts (`phishing_xai.data` needs torch, not installed
locally); the Kaggle run uses the repo path (deduplicated frame, 17,513
rows). Overlap counts are set intersections, so duplicates do not
change them.

**Checks per set:**
1. **Exact overlap** — MD5 of normalized text (lowercase, strip
   non-alphanumerics, collapse whitespace), subject+body combined.
2. **Body-only exact overlap** — same hash on the body alone. Catches
   rows whose subject differs from training while the body matches.
3. **Near-dup-hash overlap** — MD5 of the first 200 alphanumeric chars
   (the repo's `data.compute_near_dup_hash` rule).
4. **MinHash near-duplicates** — word 5-shingles, 64 permutations,
   seed 42; per-OOD-text maximum estimated Jaccard vs ANY training text.
   Full pairwise comparison (no sampling); training signatures computed
   once and reused. Threshold 0.9 reported, with 0.8 / 0.5 counts and
   the median/max of the per-OOD maximum Jaccard.
5. **URL overlap** — URLs extracted from OOD phishing texts vs URLs in
   training phishing texts (986 unique training phishing URLs).

**Full results:** `results/leakage_audit_local.json` (local run) and
`results/leakage_audit.json` (Kaggle Kernel 1 run, when available).

## Results (local run, 214.9 s)

| Set | n | Exact (subj+body) | Body-only exact | Near-dup-hash | MinHash ≥ 0.9 | MinHash ≥ 0.5 | Median max Jaccard | Shared URLs |
|---|---|---|---|---|---|---|---|---|
| phishnchips | 2,000 | 0 | 0 | 0 | 0 | 0 | 0.016 | 0 / 943 |
| phishfuzzer (Manual-300) | 300 | 0 | **1** | 0 | 0 | 0 | 0.000 | 0 / 9 |
| frontiers_llm | 4,986 | 0 | 0 | 0 | 0 | 0 | 0.016 | 0 / 2,241 |
| frontiers_human | 5,000 | **1** | 0 | **1** | **6** | 54 | 0.000 | **3** / 1,998 |
| bec (clean) | 4,181 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0 / 92 |
| bec_poisoned | 4,181 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0 / 89 |

## Findings

1. **PhishNChips (primary): clean.** Zero overlap on every check.
   Median per-email maximum Jaccard 0.016 — the OOD texts are genuinely
   distant from training text, not paraphrases of it.
2. **PhishFuzzer Manual-300: one body-only overlap.** Manual row
   No. 37 (Phishing, 2025) has a body that exactly matches a training
   email body; its subject differs, so the combined subject+body check
   is 0/300. The row is KEPT in the eval (it is 1 of 205 binary rows;
   removing it cannot move any metric materially) and flagged here.
   This matches the provenance hunt's body-only finding (1/300).
3. **Frontiers human (old corpora): expected legacy overlap, tiny.**
   1 exact + 1 near-dup-hash + 6 MinHash ≥ 0.9 (0.12%) + 3 shared URLs
   out of 5,000. These are the known TREC-07 / CEAS-08 / Nazario /
   Nigerian / Enron rows that also fed the training corpora. The human
   slice is a comparison slice only (Table B), not a modern claim; the
   6 near-dup rows are kept and flagged, not silently dropped.
4. **Frontiers LLM and BEC: clean.** Zero on every check, including
   URLs. The synthetic track is uncontaminated by training text.
5. **No URL leakage anywhere except the 3 Frontiers-human URLs**,
   which are the legacy-corpus URLs noted above.

## Limitations of these checks

* The repo's near-dup hash covers only the first 200 alphanumeric
  characters; template variants differing inside that window are not
  caught by it — which is why the MinHash check (full-text shingles)
  is reported alongside. MinHash at threshold 0.9 is an estimate, not
  an exact Jaccard; the 0.5 counts are given so the tail is visible.
* The local run's training texts came from the pandas fallback (18,634
  non-null raw rows); the Kaggle run uses the deduplicated frame
  (17,513 rows). Counts are set intersections and agree to within the
  one-hash difference in unique-hash totals (17,486 local vs 17,487 in
  the provenance hunt, from the 16 null rows); the Kaggle JSON is
  authoritative when it lands.
* The original paper-protocol leakage figure (288/3,727 = 7.7%) is
  brief-supplied and NOT verified in this repo (see
  `logs/inspection.md` §9.2); it is not used as a leakage claim here.
