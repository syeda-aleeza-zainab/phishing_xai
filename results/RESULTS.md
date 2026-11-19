# Consolidated Experimental Results & Evaluation Report

*Generated on: 2026-10-19 00:00:40; updated with the executed Kaggle run on 2026-11-19 (see §1). Earlier smoke/debug numbers are never quoted as results.*

## 1. Run Metadata

Executed on Kaggle, Tesla T4 GPU, Linux, Internet ON, 2026-11-19 (kernels `phishing-xai-stage-a` through `phishing-xai-stage-6b-2`; repo dataset `syedaaleezazainab/phishing-xai-repo-ah-6b1` v8; checkpoint dataset `syedaaleezazainab/phishing-xai-checkpoint` v1; results dataset `syedaaleezazainab/phishing-xai-results` v7). Environment: torch 2.11, transformers 5.20.2. Primary metric: phishing-class **precision** (user decision 2026-11-19); thresholds fitted on **validation only, never test**. Full per-stage logs and outputs are in the results dataset v7 (`stage_a/`, `stage_b/`, `stage_c/`, `stage_e2/`, `stage_6b2/`, `baselines/`); the run is also logged in `decision.md`.

## 2. Dataset Partitioning & Hygiene
### Preprocessing & Deduplication Hygiene
- **Raw Rows Loaded**: 18650
- **Null Rows Dropped**: 16
- **Conflicting-Label Rows Dropped**: 533 (all copies removed; texts: `['empty']`)
- **Exact Duplicate Rows Dropped**: 585
- **Total Unique Cleaned Emails Retained**: 17513
- **WordNet Synonym Augmentations Added (Train Only)**: 2863
- **Final Balanced Training Set Size**: 14078

### Split Partitions (Double Stratified, Zero Leakage)

| Split | Total Samples | Safe Emails (Label 0) | Phishing Emails (Label 1) | Positive Ratio |
|---|---|---|---|---|
| **Train (pre-augmentation)** | 11215 | 7039| 4176 | 37.24% |
| **Validation**               | 2803 | 1765 | 1038 | 37.03% |
| **Test**                     | 3495 | 2173 | 1322 | 37.83% |

- **Split Manifest**: Saved row IDs to [`results\splits.json`](splits.json). Canonical splits verified identical in every stage kernel (11,215 / 2,803 / 3,495).

## 3. Hyperparameter Tuning (Random Search)
Random-search tuning of RoBERTa did **NOT** run in the production Kaggle run (the Stage B kernel evaluated the fixed configuration instead: LR 1e-5, batch 8, best epoch 3). `stage_b/tuning/best_params.json` in the results dataset comes from an earlier run and is not a result of this run. BiLSTM tuning DID run (6B-1, 10 trials x 3 epochs, seed 42): best config emb_dim 200, hidden 64, layers 2, dropout 0.5, lr 0.002, max_len 128, batch_size 16; validation precision **0.9798**, recall 0.9364, F1 0.9576, recall floor 0.85 met.

## 4. Final Model Training & Test Evaluation
Single test evaluation, seed 42, test n=3,495 (Stage B, best epoch 3, val F1 0.9834): **precision 0.9690, recall 0.9932, F1 0.9809, accuracy 0.9854, ROC-AUC 0.9986, FPR 0.0193**. Classification report support: Safe 2,173, Phishing 1,322. Checkpoint saved as `results/best_model.pt` (124,647,170 params, 201 tensors) and published as the private Kaggle dataset `syedaaleezazainab/phishing-xai-checkpoint` v1 immediately after evaluation.

## 5. Explainability & Faithfulness Evaluation
### 5.1 Faithfulness Benchmarks (Comprehensiveness & Sufficiency)
RoBERTa (Stage C, n=97 of 100 requested; 3 skipped as too short after truncation to 512 tokens), k=5: LITA-magnitude comp 0.1055 / suff 0.0809; LIME comp 0.1115 / suff 0.1640; Transformers-Interpret comp 0.0337 / suff 0.2058; random comp 0.0165 / suff 0.4258. All explainers beat random on comprehensiveness; LITA has the best sufficiency at every k (1, 3, 5). BiLSTM (6B.3, n=97; truncation to 128 tokens): LITA-magnitude @5 comp 0.0950 / suff -0.0027; LIME @5 comp 0.0814 / suff 0.0002; LITA-IG (Captum) @5 comp 0.0752 / suff -0.0044; random @5 comp 0.0145 / suff 0.1826. Multimodal faithfulness (Stage E, 6 samples, LITA k=5): comprehensiveness 0.004-0.065 on phishing samples (model saturated near P 0.999), sufficiency about 0.0, clearly beating random sufficiency (0.18-0.23) on the PDF samples; benign sample comp 0.9418.

### 5.2 Alpha Parameter Ablation Study (LITA Weighting)
RoBERTa (Stage C, n=49 of 50, k=3): alpha 0.0 (pure TI) comp 0.0703 / suff 0.1743; 0.25 comp 0.0947 / suff 0.1074; **0.5 comp 0.1296 / suff 0.0665 (best on both)**; 0.75 comp 0.1424 / suff 0.1169; 1.0 (pure LIME) comp 0.1272 / suff 0.1900. BiLSTM (6B.3, n=49, k=3): minmax variant alpha 0.5 best on both (comp 0.0873 / suff 0.0126); magnitude variant alpha 0.5 best comprehensiveness (0.0829), alpha 0.25 best sufficiency (0.0150). Alpha 0.5 is the best balance on both models.

### 5.3 Global Explanations (Top Attributed Words)
Top-5 word agreement between RoBERTa and BiLSTM LITA explanations on the same 10 emails: mean Jaccard **0.1004**, mean Spearman rho **-0.4974** (n=10). The two models point at largely different words; per-email top-5 lists are in `stage_6b2/` (`system_complexity_report.json`, `agreement.per_email`). BiLSTM LIME top-word lists contain duplicate tokens (subword pieces mapping to the same word), which depresses Jaccard slightly.

## 6. Baseline Ladder & Complexity Verdict (Stage D + 6B.1 + 6B.5)
Test set n=3,495, 3 seeds unless noted. TF-IDF+SVC precision **0.9820** +/- 0.0007 / recall 0.9892 / F1 0.9856 / AUC 0.9993; TF-IDF+LR 0.9702 / 0.9834 / 0.9767; DistilBERT (1 seed) 0.9731 / 0.9849 / 0.9789; CNN (1 seed) 0.9732 / 0.9614 / 0.9673; tuned BiLSTM 0.9719 +/- 0.0015 / 0.9428 / 0.9571 (bigger hidden-256 variant, 1 seed: 0.9711 / 0.9652 / 0.9681); RoBERTa 0.9690 / 0.9932 / 0.9809; dummy accuracy 0.6217. McNemar vs RoBERTa: TF-IDF+SVC p=0.253, TF-IDF+LR p=0.349, DistilBERT p=0.603 (all not significant); CNN p=0.0006, tuned BiLSTM p=0.0002, bigger BiLSTM p=0.0009 (RoBERTa significantly better on overall errors). Bootstrap 95% precision CIs: TF-IDF+SVC [0.9746, 0.9889], TF-IDF+LR [0.9607, 0.9793], tuned BiLSTM [0.9613, 0.9789] — all contain RoBERTa's 0.9690. Deployment pick by the ladder rule (highest precision, recall >= 0.85): TF-IDF+SVC. The system report's pick among S1-S4 only is DistilBERT (0.9731 / 0.9849). Code-generated verdict (6B.5, `system_complexity_report.md` in the results dataset): "Transformer gives a significant precision gain of 5.88 points on the multimodal held-out set (RoBERTa 1.0000 vs best simple baseline tfidf_lr 0.9412; McNemar p = 0.6250) at unknown latency and 3116.1x parameters." Text-only verdict: -1.30 points vs TF-IDF+SVC (p = 0.2530) — RoBERTa's precision is *below* the best simple baseline on text; its advantage is recall. No simple system passed all three verdict conditions (gap <= 1.0pt, CI overlap, McNemar p > 0.05) on BOTH sets, so the "Transformer is not statistically justified" verdict did not trigger. Robustness (truncation / URL / typo) is NOT RUN YET; OOD test is NOT RUN YET (no OOD set available).

## 7. Multimodal Evaluation (Stage E + 6B.4)
Synthetic eval set, 120 emails (60 phishing / 60 benign; 24 validation / 96 test), thresholds fitted on validation only (recall floor 0.80). Trained-checkpoint RoBERTa, test n=96: (a) body only P 0.8605 / R 0.7708 / F1 0.8132 / FPR 0.125; (b) +OCR P 0.8776 / R 0.8958 / F1 0.8866; (c) +QR+PDF P 0.9792 / R 0.9792 / F1 0.9792 / FPR 0.0208; (d) full fusion weighted **P 1.00 / R 0.9792 / F1 0.9895 / FPR 0.00** (threshold 0.6); (d) max rule P 0.9130 / R 0.875 / F1 0.8936. One failure: `scanned_pdf_001.eml` (corroboration gate; max rule would have caught it). Per-backbone full fusion (d) weighted, test n=96: BiLSTM **P 1.00 / R 1.00 / F1 1.00** (0 failures); TF-IDF+LR P 0.9412 / R 1.00 / F1 0.9697 (3 failures); rule-based P 1.00 / R 0.8333 / F1 0.9091 (10 failures). Multimodal McNemar vs RoBERTa: BiLSTM p=1.0, TF-IDF+LR p=0.625, rule-based p=0.0156. OCR (Tesseract) and QR decode (zbar) verified working on all 6 samples. The synthetic-set caveat applies throughout: these numbers measure the pipeline on generated templates, not real multimodal phishing.

## 8. Limitations & Methodological Caveats
- Evaluations conducted with deterministic seed; variance across random seeds may exist.
- Deduplication removes duplicate texts and conflicting labels (e.g. 'empty' placeholders) to ensure clean train/test boundary.
- Synonym-replacement augmentation balances class frequencies on the training split only.
- The training corpus is an old Enron-style dataset, highly separable; high simple-model scores do not imply the same performance on modern, AI-written or multimodal phishing. A harder out-of-distribution test is needed before any deployment claim.
- The multimodal eval set is synthetic (generated templates); its numbers measure the pipeline, not wild phishing.
- RoBERTa random-search tuning did NOT run in this Kaggle run; robustness (truncation / URL / typo) did NOT run; DistilBERT and CNN ran 1 seed only; explanation time per email was never recorded.
- Stage D's own `comparisons_vs_roberta`, error analysis and the top-level `roberta` block were completed/corrected locally from the saved test predictions with the repo's own stats functions (seed 42, 1,000 resamples) — see `decision.md` 2026-11-19; no metric was invented and nothing was re-fit. The multimodal McNemar/CI values in §6-7 were computed the same way from saved per-row predictions.
