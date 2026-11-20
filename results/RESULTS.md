# Consolidated Experimental Results & Evaluation Report

*Generated on: 2026-10-19 00:00:40; updated with the executed Kaggle run on 2026-11-19 (see §1). Earlier smoke/debug numbers are never quoted as results.*

## 1. Run Metadata

Executed on Kaggle, Tesla T4 GPU, Linux, Internet ON, 2026-11-19 (kernels `phishing-xai-stage-a` through `phishing-xai-stage-6b-2`; repo dataset `syedaaleezazainab/phishing-xai-repo-ah-6b1` v8; checkpoint dataset `syedaaleezazainab/phishing-xai-checkpoint` v1; results dataset `syedaaleezazainab/phishing-xai-results` v7). Environment: torch 2.11, transformers 5.20.2. Primary metric: phishing-class **precision** (user decision 2026-11-19); thresholds fitted on **validation only, never test**. Full per-stage logs and outputs are in the results dataset v7 (`stage_a/`, `stage_b/`, `stage_c/`, `stage_e2/`, `stage_6b2/`, `baselines/`); the run is also logged in `decision.md`. Rerun round, also 2026-11-19 (kernels `phishing-xai-rerun-multimodal-floors`, `phishing-xai-rerun-robustness`, `phishing-xai-rerun-latency`; repo dataset v9/v10 with the fixed verdict generator): TF-IDF multimodal runs at floors 0.80 + 0.85 (§7), robustness (§8), latency + explanation time (§10). Rerun outputs are stored in this repo under `results/rerun_mm/`, `results/robustness_rerun/` and `results/rerun_latency/`; the results Kaggle dataset was NOT re-uploaded with them (v7 remains the published results version).

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
RoBERTa random-search tuning DID run and is recorded in the results dataset v7 (`stage_b/tuning/results.json`): 10 iterations, max_length 256, subsample_train 4000, seed 42, search space batch [8,16,32] x lr [1e-5,2e-5,3e-5] x epochs [3,4,5]; best config batch 8 / lr 1e-5 / epochs 3, best val F1 **0.9807** (val P 0.9722). The Stage B retrain used exactly those params (summary.json config lr 1e-5, epochs 3, best epoch 3; test P 0.9690 / R 0.9932 / F1 0.9809), so the "fixed config" IS the tuned config — no re-run of item 4 was needed (verified 2026-11-19 from the dataset file; an earlier RESULTS.md line saying tuning did not run was wrong and is corrected here). Note: the same trial params (16/1e-5/3) scored val F1 0.9796 in iteration 3 but 0.9661 in iteration 6 — GPU nondeterminism at the tenth-of-a-point level, so the 0.9807 best is within noise of the other trials. BiLSTM tuning DID run (6B-1, 10 trials x 3 epochs, seed 42): best config emb_dim 200, hidden 64, layers 2, dropout 0.5, lr 0.002, max_len 128, batch_size 16; validation precision **0.9798**, recall 0.9364, F1 0.9576, recall floor 0.85 met.

## 4. Final Model Training & Test Evaluation
Single test evaluation, seed 42, test n=3,495 (Stage B, best epoch 3, val F1 0.9834): **precision 0.9690, recall 0.9932, F1 0.9809, accuracy 0.9854, ROC-AUC 0.9986, FPR 0.0193**. Classification report support: Safe 2,173, Phishing 1,322. Checkpoint saved as `results/best_model.pt` (124,647,170 params, 201 tensors) and published as the private Kaggle dataset `syedaaleezazainab/phishing-xai-checkpoint` v1 immediately after evaluation.

## 5. Explainability & Faithfulness Evaluation
### 5.1 Faithfulness Benchmarks (Comprehensiveness & Sufficiency)
RoBERTa (Stage C, n=97 of 100 requested; 3 skipped as too short after truncation to 512 tokens), k=5: LITA-magnitude comp 0.1055 / suff 0.0809; LIME comp 0.1115 / suff 0.1640; Transformers-Interpret comp 0.0337 / suff 0.2058; random comp 0.0165 / suff 0.4258. All explainers beat random on comprehensiveness; LITA has the best sufficiency at every k (1, 3, 5). BiLSTM (6B.3, n=97; truncation to 128 tokens): LITA-magnitude @5 comp 0.0950 / suff -0.0027; LIME @5 comp 0.0814 / suff 0.0002; LITA-IG (Captum) @5 comp 0.0752 / suff -0.0044; random @5 comp 0.0145 / suff 0.1826. DistilBERT (2026-11-20, n=97 of 100 requested — 3 emails failed on CUDA OOM during explanation; same 97 email ids as RoBERTa Stage C, verified 97/97 overlap; truncation to 128 tokens): LITA-magnitude @5 comp 0.0498 / suff 0.0363; LIME @5 comp 0.0469 / suff 0.0538; LITA-minmax @5 comp 0.0494 / suff 0.0146; random @5 comp 0.0004 / suff 0.2843. All three LITA variants beat random on comprehensiveness, but DistilBERT's faithfulness is well below RoBERTa's and BiLSTM's on comprehensiveness (0.0498 vs 0.1055 / 0.0950). Multimodal faithfulness (Stage E, 6 samples, LITA k=5): comprehensiveness 0.004-0.065 on phishing samples (model saturated near P 0.999), sufficiency about 0.0, clearly beating random sufficiency (0.18-0.23) on the PDF samples; benign sample comp 0.9418.

### 5.2 Alpha Parameter Ablation Study (LITA Weighting)
RoBERTa (Stage C, n=49 of 50, k=3): alpha 0.0 (pure TI) comp 0.0703 / suff 0.1743; 0.25 comp 0.0947 / suff 0.1074; **0.5 comp 0.1296 / suff 0.0665 (best on both)**; 0.75 comp 0.1424 / suff 0.1169; 1.0 (pure LIME) comp 0.1272 / suff 0.1900. BiLSTM (6B.3, n=49, k=3): minmax variant alpha 0.5 best on both (comp 0.0873 / suff 0.0126); magnitude variant alpha 0.5 best comprehensiveness (0.0829), alpha 0.25 best sufficiency (0.0150). DistilBERT (2026-11-20, n=49 of 50, k=3): magnitude variant alpha 0.5 best on both (comp 0.0423 / suff 0.0345); minmax variant alpha 0.5 best on both (comp 0.0434 / suff 0.0356). Alpha 0.5 is the best balance on all three models.

### 5.3 Global Explanations (Top Attributed Words)
Top-5 word agreement between RoBERTa and BiLSTM LITA explanations on the same 10 emails: mean Jaccard **0.1004**, mean Spearman rho **-0.4974** (n=10). The two models point at largely different words; per-email top-5 lists are in `stage_6b2/` (`system_complexity_report.json`, `agreement.per_email`). BiLSTM LIME top-word lists contain duplicate tokens (subword pieces mapping to the same word), which depresses Jaccard slightly.

## 6. Baseline Ladder & Complexity Verdict (Stage D + 6B.1 + 6B.5)
Test set n=3,495, 3 seeds unless noted. TF-IDF+SVC precision **0.9820** +/- 0.0007 / recall 0.9892 / F1 0.9856 / AUC 0.9993; TF-IDF+LR 0.9702 / 0.9834 / 0.9767; DistilBERT (3 seeds, 2026-11-20) 0.9890 +/- 0.0022 / 0.9476 +/- 0.0122 / 0.9678 +/- 0.0054; CNN (1 seed) 0.9732 / 0.9614 / 0.9673; tuned BiLSTM 0.9719 +/- 0.0015 / 0.9428 / 0.9571 (bigger hidden-256 variant, 1 seed: 0.9711 / 0.9652 / 0.9681); RoBERTa 0.9690 / 0.9932 / 0.9809; dummy accuracy 0.6217. McNemar vs RoBERTa: TF-IDF+SVC p=0.253, TF-IDF+LR p=0.349, DistilBERT p=0.0754 (all not significant); CNN p=0.0006, tuned BiLSTM p=0.0002, bigger BiLSTM p=0.0009 (RoBERTa significantly better on overall errors). New paired McNemar, computed locally 2026-11-20 from the saved seed-42 test predictions (`results/paired_mcnemar_seed42.json`, verified against the known seed-42 metrics first): TF-IDF+SVC vs BiLSTM b=16, c=65, exact two-sided p=3.64e-08 — SVC is significantly better than BiLSTM on test errors; TF-IDF+SVC vs TF-IDF+LR b=13, c=39, p=0.00041. Bootstrap 95% precision CIs: TF-IDF+SVC [0.9746, 0.9889], TF-IDF+LR [0.9607, 0.9793], tuned BiLSTM [0.9613, 0.9789] — all contain RoBERTa's 0.9690; DistilBERT [0.9799, 0.9930] sits entirely above it. Deployment pick by the ladder rule (highest precision, recall >= 0.85): DistilBERT (S3, 0.9890 / 0.9476). The regenerated system report (2026-11-20, `results/system_complexity_report.md/.json`) now recommends DistilBERT (S3) as the deployment model; the previous regeneration recommended TF-IDF+SVC (S0) when DistilBERT was still a single seed at 0.9731. Corrected code-generated verdict (fixed generator, regenerated 2026-11-20 from the measured numbers, `results/system_complexity_report.json` `verdict`): "RoBERTa is below the best simple baseline by 1.30 precision points on the text test set (RoBERTa 0.9690 vs SVC 0.9820; McNemar p = 0.2530) at 28.6x CPU latency and 31.4x parameters. GPU latency multiple: NOT MEASURED (TF-IDF has no GPU implementation). Its complexity is NOT justified by a precision gain on this dataset. BiLSTM is also below SVC, by 1.00 precision points (0.9719 vs 0.9820; paired seed-42 McNemar p = 3.64e-08), at 2.35x CPU latency and 1.57x parameters." The older LR-based verdict is kept in the same file under `verdict_vs_tfidf_lr`: "below the best simple baseline by 0.11 precision points (RoBERTa 0.9690 vs tfidf_lr 0.9701; McNemar p = 0.3491) at 222.6x latency and 3116.1x parameters" — the latency multiple changed from 223.7x (2026-11-19) to 222.6x because the LR CPU latency source changed from the 3-seed aggregate 1.084 ms to the seed-42 refit 1.089 ms. The 222.6x multiple uses the now-measured RoBERTa CPU latency (242.50 ms/email, §10) against TF-IDF+LR's seed-42 refit 1.089 ms/email; the earlier "unknown latency" wording is superseded. Text-only verdict (from the same fixed generator, now comparing against DistilBERT as the highest-precision baseline): -2.00 points vs DistilBERT (p = 0.0754) — RoBERTa's precision is *below* the best baseline on text; its advantage is recall (F1 gap +1.32 points in RoBERTa's favour, 0.9809 vs 0.9678). No simple system passed all three verdict conditions (gap <= 1.0pt, CI overlap, McNemar p > 0.05) on BOTH sets, so the "Transformer is not statistically justified" verdict did not trigger. Robustness (truncation / URL / typo) HAS now run — see §8. OOD test HAS now run for all five models — see §11.

**Generalized assessment (all metrics, not precision alone).** The code-generated verdict above is the precision-first view, one view among several. Across the full picture no single model wins everywhere. On the text test set DistilBERT has the highest precision (0.9890), TF-IDF+SVC has the highest F1 (0.9856), accuracy (0.9889), ROC-AUC (0.9993) and PR-AUC (0.9987), RoBERTa has the highest recall (0.9932), and TF-IDF+LR has the lowest latency (1.09 ms CPU) with near-top accuracy. On robustness DistilBERT loses the least (worst F1 drop 0.51 points) while TF-IDF+SVC keeps the highest F1 on clean, URL-removed and typo sets. On OOD, TF-IDF+LR is best operationally on the primary set (F1 0.3112) while DistilBERT ranks best (ROC-AUC 0.8222, P@R.90 0.6348) but has the worst recall (0.0140) and the worst calibration (ECE 0.491); after target-domain recalibration DistilBERT transfers best (holdout F1 0.766). On explanations RoBERTa is most faithful (LITA-magnitude comp@5 0.1055) and DistilBERT least (0.0498), while DistilBERT explains about twice as fast as RoBERTa (15.69 vs 30.87 s/email) and BiLSTM is fastest (1.87 s/email). On the synthetic multimodal set BiLSTM is perfect (P 1.00 / R 1.00), RoBERTa and DistilBERT tie on precision (1.00) with RoBERTa ahead on recall (0.9792 vs 0.9375). The precision-first ladder picks DistilBERT; the generalization picture says the choice depends on which metric the deployment can least afford to lose — precision (DistilBERT), recall (RoBERTa), F1 (TF-IDF+SVC), latency (TF-IDF+LR), or OOD robustness after recalibration (DistilBERT).

## 7. Multimodal Evaluation (Stage E + 6B.4)
Synthetic eval set, 120 emails (60 phishing / 60 benign; 24 validation / 96 test), thresholds fitted on validation only. Rerun round 2026-11-19 (kernel `phishing-xai-rerun-multimodal-floors`, COMPLETE): all five backbones run at BOTH recall floors 0.80 and 0.85, same split and protocol. Weighted fusion (d), test n=96, floor 0.80: RoBERTa **P 1.00 / R 0.9792 / F1 0.9895 / FPR 0.00** (threshold 0.6, 1 failure: `scanned_pdf_001.eml`, corroboration gate); BiLSTM **P 1.00 / R 1.00 / F1 1.00** (threshold 0.6617, 0 failures); TF-IDF+LR P 0.9412 / R 1.00 / F1 0.9697 (threshold 0.35, 3 failures, all benign QR newsletters); TF-IDF+SVC P 0.9556 / R 0.8958 / F1 0.9247 / FPR 0.0417 (threshold 0.5, 8 failures); rule-based P 1.00 / R 0.8333 / F1 0.9091 (threshold 0.05, 10 failures). DistilBERT (2026-11-20, kernel `phishing-xai-distilbert-multimodal-eval`, seed-42 checkpoint, floor 0.85): weighted fusion (d) **P 1.00 / R 0.9375 / F1 0.9677 / FPR 0.00** (threshold 0.65, 3 failures — all missed phishing where the fused score reached the threshold but lacked corroboration: `text_button_image_004.eml`, `text_button_image_008.eml`, `qr_in_text_pdf_002.eml`); 4-ablation at fitted thresholds — (a) text-body only P 1.00 / R 0.7500 / F1 0.8571; (b) text + OCR P 1.00 / R 0.8333 / F1 0.9091; (c) text + OCR + QR + PDF P 0.9714 / R 0.7083 / F1 0.8193 / FPR 0.0208; (d) max fusion P 0.6316 / R 1.00 / F1 0.7742 / FPR 0.5833. Per-modality recall (weighted fusion): body 0.9375 (45/48), OCR 0.8947 (17/19), PDF 0.9286 (13/14), QR 0.9286 (13/14). Paired McNemar vs RoBERTa on the multimodal test set: NOT COMPUTED for DistilBERT — the only stored RoBERTa multimodal rows are from the original floor-0.80 run, while DistilBERT ran at floor 0.85, and the audited rerun stored no per-email RoBERTa predictions, so no protocol-matched paired test is possible from stored data. At floor 0.85 the learned backbones are IDENTICAL (same thresholds, same metrics) — their precision-first thresholds already gave validation recall >= 0.85. Only the rule-based scorer moved: to meet the 0.85 floor its threshold dropped 0.05 to 0.015, buying test recall 0.8333 to 1.00 at the cost of precision 1.00 to 0.75 (FPR 0.3333, 20 failures). TF-IDF+SVC's body-only threshold also moved (0.85 to 0.6895) but its fusion threshold stayed 0.5, so its fusion metrics did not change. What the floor costs: nothing for any learned model on this set, because all four already cleared 0.85 recall on validation at their precision-first thresholds; the floor only binds on the weak rule-based scorer, where it trades 25 precision points for 16.7 recall points. The old Stage E numbers (floor 0.80 only) are unchanged and reproduced exactly by the rerun for RoBERTa, BiLSTM, TF-IDF+LR and rule-based. Per-backbone McNemar vs RoBERTa on the multimodal test set (recomputed from the rerun per-row predictions, repo's own `mcnemar_test`, n=96): BiLSTM p=1.0, TF-IDF+LR p=0.625, TF-IDF+SVC p=0.0703, rule-based p=0.0156 at floor 0.80 (p=0.0003 at floor 0.85, after its precision collapse). Bootstrap 95% precision CIs (1,000 resamples, seed 42): TF-IDF+LR [0.8718, 1.0], TF-IDF+SVC [0.8864, 1.0], BiLSTM [1.0, 1.0]. Best simple baseline on the multimodal set (weighted fusion, highest precision): BiLSTM at 1.00, tied on precision with RoBERTa — gap 0.00 points, McNemar p=1.0, CI overlaps: the fixed generator's multimodal check gives *no statistically significant difference*, never a "significant gain". The old verdict's 5.88-point "significant gain" (p=0.625) was the bug item 1 fixed; the corrected system verdict is in §6. Trained-checkpoint RoBERTa 4-ablation (Stage E, floor 0.80): (a) body only P 0.8605 / R 0.7708 / F1 0.8132 / FPR 0.125; (b) +OCR P 0.8776 / R 0.8958 / F1 0.8866; (c) +QR+PDF P 0.9792 / R 0.9792 / F1 0.9792 / FPR 0.0208; (d) full fusion weighted **P 1.00 / R 0.9792 / F1 0.9895 / FPR 0.00** (threshold 0.6); (d) max rule P 0.9130 / R 0.875 / F1 0.8936. OCR (Tesseract) and QR decode (zbar) verified working on all 6 samples. The synthetic-set caveat applies throughout: these numbers measure the pipeline on generated templates, not real multimodal phishing.

## 8. Robustness (rerun round, 2026-11-19)
Perturbations truncated / urls_removed / typo_synonym applied to the canonical test split (n=3,495), threshold fixed at 0.5, no retraining, no threshold fitting. Kernel `phishing-xai-rerun-robustness` (COMPLETE 2026-11-19). F1 per model (clean / truncated / urls_removed / typo_synonym), seed 42:

| Model | Clean | Truncated | URLs removed | Typo/synonym |
|---|---|---|---|---|
| RoBERTa (loaded checkpoint) | 0.9809 | 0.9773 | 0.9755 | 0.9638 |
| Tuned BiLSTM (loaded artefact) | 0.9658 | 0.9658 | 0.9654 | 0.9637 |
| TF-IDF+SVC (refit, seed 42) | 0.9845 | 0.9726 | 0.9849 | 0.9835 |
| TF-IDF+LR (refit, seed 42) | 0.9748 | 0.9668 | 0.9730 | 0.9739 |
| CNN (refit, seed 42) | 0.9673 | 0.9591 | 0.9665 | 0.9660 |
| DistilBERT (loaded checkpoint, seed 42) | 0.9739 | 0.9688 | 0.9739 | 0.9745 |

All six models lose at most ~1.7 F1 points on any perturbation; the typo/synonym perturbation is the hardest for every model except DistilBERT (its hardest is truncation, 0.51 points). DistilBERT loses the least overall (worst drop 0.51 points, on truncation) and TF-IDF+SVC keeps the highest F1 on clean, URL-removed and typo sets; RoBERTa keeps the highest F1 on truncation (0.9773). RoBERTa's clean F1 here (0.9809 at threshold 0.5) matches its Stage B test F1. Sources: `results/robustness_rerun/robustness_loaded.json` (RoBERTa, BiLSTM) and `results/robustness_rerun/baseline_results.json` (TF-IDF+LR, TF-IDF+SVC, CNN) in the rerun download `~/workspace/kaggle_downloads/rerun_robustness/`; DistilBERT from `results/distilbert/robustness/robustness_loaded.json` (Kernel 3 text-evals v2, seed-42 checkpoint, same protocol). Not run: bigger BiLSTM hidden-256 (deliberately not saved into the results dataset), CNN via the artefact route (no saved artefact; the refit route above covers it). Latency measured in the same kernel for the refit models only (median of 200, batch 1, CPU): TF-IDF+LR 1.09 ms/email, TF-IDF+SVC 8.47 ms/email, CNN 1.65 ms/email CPU / 1.33 ms/email GPU. RoBERTa latency and explanation time are NOT in this kernel — see §10.

## 9. Limitations & Methodological Caveats
- Evaluations conducted with deterministic seed; variance across random seeds may exist.
- Deduplication removes duplicate texts and conflicting labels (e.g. 'empty' placeholders) to ensure clean train/test boundary.
- Synonym-replacement augmentation balances class frequencies on the training split only.
- The training corpus is an old Enron-style dataset, highly separable; high simple-model scores do not imply the same performance on modern, AI-written or multimodal phishing. The out-of-distribution test HAS now run (2026-11-20, see §11) and confirms the gap: in-distribution F1 ~0.98 does not transfer to the primary modern set (best F1 0.3112). No model meets the deployment recall floor on modern phishing.
- The multimodal eval set is synthetic (generated templates); its numbers measure the pipeline, not wild phishing.
- Robustness (truncation / URL / typo) HAS now run (see §8); OOD test HAS now run for all five models (RoBERTa, BiLSTM, TF-IDF+LR, TF-IDF+SVC, DistilBERT — see §11). DistilBERT ran 3 seeds on text (2026-11-20); CNN ran 1 seed only on text.
- Stage D's own `comparisons_vs_roberta`, error analysis and the top-level `roberta` block were completed/corrected locally from the saved test predictions with the repo's own stats functions (seed 42, 1,000 resamples) — see `decision.md` 2026-11-19; no metric was invented and nothing was re-fit. The multimodal McNemar/CI values in §6-7 were computed the same way from saved per-row predictions.

## 10. Latency & Explanation Time (rerun round, 2026-11-19; DistilBERT 2026-11-20)
Kernel `phishing-xai-rerun-latency` (script printed `RERUN_LATENCY DONE` and saved `results/rerun_latency/latency_explain_time.json` at 2026-11-19 18:30 UTC; the kernel status read RUNNING when the outputs were first retrieved and COMPLETE on a later check — post-run phase, same pattern as Stage B). Inference latency, median of 200 emails at batch size 1, Tesla T4, seed 42: RoBERTa **GPU 15.50 ms/email** (mean 18.58), **CPU 242.50 ms/email** (mean 282.53); tuned BiLSTM GPU 8.37 ms/email (mean 7.76), CPU 19.90 ms/email (mean 16.92). DistilBERT (2026-11-20, latency kernel, seed-42 checkpoint, same protocol, device switched properly): GPU **9.07 ms/email** (mean 9.05), CPU **118.62 ms/email** (mean 101.47). Explanation time per email, same 10 test emails Stage C explained (lime_samples=1000): RoBERTa LITA **median 30.87 s/email** (mean 36.65, std 22.85; range 14.62–75.92 s); BiLSTM LITA-IG median 1.87 s/email (mean 1.71, std 0.49; range 0.66–2.30 s); DistilBERT LITA median 15.69 s/email (mean 14.32, std 5.20; range 7.55–21.37 s). RoBERTa LITA explanations are ~16x slower than BiLSTM LITA-IG and ~2,000x slower than a single RoBERTa GPU inference — the cost is the LIME perturbation sampling, not the model. For reference, the refit baselines in the robustness kernel (CPU, median of 200): TF-IDF+LR 1.09 ms/email, TF-IDF+SVC 8.47 ms/email, CNN 1.65 ms/email CPU / 1.33 ms/email GPU. The old "unknown latency" wording in the system-complexity verdict is now superseded: RoBERTa GPU latency is 15.50 ms/email. Correction: the DistilBERT final-training driver reported a "CPU median 9.55 ms" — that timed the model without moving it off CUDA, so it measured GPU time again; the true CPU latency is 118.62 ms from the latency kernel above.

## 11. Out-of-distribution test (2026-11-20, COMPLETED for all five models)

No verified real-world 2023+ phishing+legit email-body corpus exists. The primary set is PhishNChips v5.2 (Hugging Face, 2026-04, 2,000 rows, synthetic stimuli around real verified URLs, NOT real inbox mail). Kernels 1–3 executed on Kaggle; full analysis in `experiments/modern_phishing_ood/reports/final_comparative_report.md`, expanded paired statistics in `experiments/modern_phishing_ood/reports/statistical_comparison_expanded.md` (`ood_stats_expanded.json`).

**Primary OOD — PhishNChips v5.2 (n=2,000, threshold 0.5):**

| Model | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR | FNR | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---|---|---|---|---|
| RoBERTa | 0.513 | 0.9333 | 0.0280 | 0.0544 | 0.7787 | 0.7947 | 0.5636 | 0.002 | 0.972 | [[998, 2], [972, 28]] |
| BiLSTM | 0.523 | 0.8594 | 0.0550 | 0.1034 | 0.6801 | 0.6768 | 0.5263 | 0.009 | 0.945 | [[991, 9], [945, 55]] |
| TF-IDF+SVC | 0.536 | 0.9000 | 0.0810 | 0.1486 | 0.6604 | 0.6724 | 0.5441 | 0.009 | 0.919 | [[991, 9], [919, 81]] |
| TF-IDF+LR | 0.591 | 0.9788 | 0.1850 | 0.3112 | 0.7528 | 0.7771 | 0.5714 | 0.004 | 0.815 | [[996, 4], [815, 185]] |
| DistilBERT | 0.507 | 1.0000 | 0.0140 | 0.0276 | 0.8222 | 0.8250 | 0.6348 | 0.000 | 0.986 | [[1000, 0], [986, 14]] |

TF-IDF+LR is the best model on the primary set by precision, recall and F1; DistilBERT keeps the best ranking (ROC-AUC 0.8222, PR-AUC 0.8250, P@R.90 0.6348) but the worst recall (0.0140). All five models miss most phishing on this set. Paired differences (RoBERTa − BiLSTM, 1,000 bootstrap resamples, seed 42): precision +0.074 [−0.033, +0.180] (not significant); recall −0.027 [−0.043, −0.010] and F1 −0.049 [−0.080, −0.018] (both exclude 0, BiLSTM ahead); ROC-AUC +0.099 [0.076, 0.125] and PR-AUC +0.118 [0.088, 0.146] (both exclude 0, RoBERTa ahead). McNemar on errors p=0.0359 (b=51, c=31, BiLSTM fewer errors) — significant under the within-set Holm correction (adjusted p=0.0359 — it is the largest p in the 6-pair family, so the multiplier is 1) but NOT under the global Holm correction across all 36 pairs × sets (adjusted p=0.323), so it is borderline and not a firm finding. McNemar is an errors-only test, not a precision test. Full paired statistics for all 6 model pairs × 6 sets are in `experiments/modern_phishing_ood/reports/statistical_comparison_expanded.md` (`ood_stats_expanded.json`). Primary-set TF-IDF pairs: TF-IDF+LR makes significantly fewer errors than RoBERTa (McNemar p=9.6e-28, b=177, c=22) and BiLSTM (p=1.9e-22); paired recall/F1 differences are significant in LR's favour vs RoBERTa (recall −0.157 [−0.183, −0.133]; F1 −0.257 [−0.292, −0.216]) but the paired precision difference vs RoBERTa is NOT significant (−0.046 [−0.145, +0.033]). TF-IDF+SVC also makes significantly fewer errors than RoBERTa (p=1.2e-05) and BiLSTM (p=0.0107; within-set Holm 0.0214, global Holm 0.107 not significant). LR beats SVC on every paired metric (all CIs exclude 0; McNemar p=8.9e-19). The fitted thresholds detect ~nothing for TF-IDF too (SVC 0.85-floor recall 0.0000; LR 0.0070). OOD CPU latency (n=100, phishnchips texts): TF-IDF+LR 1.02 ms/email, TF-IDF+SVC 7.58 ms/email; GPU NOT MEASURED for TF-IDF (no GPU implementation).

**Secondary OOD (threshold 0.5; all five models):** Frontiers LLM n=5,986 — RoBERTa R 0.9394 / F1 0.9686, TF-IDF+LR R 0.8187 / F1 0.8999, DistilBERT R 0.7669 / F1 0.8681, TF-IDF+SVC R 0.6195 / F1 0.7642, BiLSTM R 0.4787 / F1 0.6467 (paired recall difference RoBERTa − BiLSTM +0.461 [+0.446, +0.475]; McNemar p≈0); Frontiers human n=6,000 (2007–2008 corpora, not modern) — DistilBERT R 0.8572 / F1 0.9231 (best), RoBERTa R 0.8402 / F1 0.9130, TF-IDF+LR R 0.8400 / F1 0.9126 (essentially tied), TF-IDF+SVC R 0.8214 / F1 0.9011, BiLSTM R 0.7226 / F1 0.8381; BEC n=5,181 — TF-IDF+SVC R 0.3882 / F1 0.5584 (ahead of RoBERTa R 0.3772 / F1 0.5476), BiLSTM R 0.2102 / F1 0.3468, TF-IDF+LR R 0.3164 / F1 0.4804, DistilBERT R 0.1502 / F1 0.2612; BEC poisoned n=5,181 — RoBERTa R 0.4303, TF-IDF+SVC R 0.3662, TF-IDF+LR R 0.3004, DistilBERT R 0.2344, BiLSTM R 0.2243; PhishFuzzer n=205 — TF-IDF+LR R 0.7476 / F1 0.6160, TF-IDF+SVC R 0.6796 / F1 0.5932, RoBERTa R 0.6796 / F1 0.5833, DistilBERT R 0.6019 / F1 0.5636, BiLSTM R 0.4951 / F1 0.5126 (too small to rank the models). Validation-fitted thresholds (~0.99) detect ~nothing on the primary set for either model. Temporal analysis NOT RUN — no timestamps in any OOD set.

**Hypotheses (Task 8):** H1a (RoBERTa better RANKING than BiLSTM on primary OOD) **SUPPORTED** — both paired ROC-AUC and PR-AUC difference CIs exclude 0 in RoBERTa's favour. H1b (RoBERTa better OPERATIONAL detection under the predefined recall floor 0.85) **REJECTED** — neither model meets the floor at threshold 0.5 or at any validation-fitted threshold. **Final decision:** best on primary OOD by precision, recall and F1 is TF-IDF+LR (0.9788 / 0.1850 / 0.3112); best ranking is DistilBERT (ROC-AUC 0.8222, PR-AUC 0.8250); most efficient is TF-IDF+LR (CPU 1.02 ms/email OOD); the best simple baseline IS competitive OOD — it beats both neural models on the primary set; deployment requirements are NOT met by any model (H1b rejected for all five).

### 11.1 Full OOD table — Task 2a (2026-11-20, recomputed locally from the 30 saved per-email prediction CSVs)

All 5 models × 6 sets at threshold 0.5, plus precision at recall 0.90. Recomputation cross-checked against the saved per-set metrics JSONs: 270 cells, max abs diff 0.0, zero mismatches. Machine-readable: `experiments/modern_phishing_ood/results/ood_table_full.json`; report: `experiments/modern_phishing_ood/reports/ood_table_full.md`. DistilBERT rows were computed the same way from its 6 saved prediction CSVs (Kaggle Kernel 4, seed-42 checkpoint).

| Set (n) | Model | P | R | F1 | ROC-AUC | PR-AUC | P@R.90 | FPR |
|---|---|---|---|---|---|---|---|---|
| PhishNChips (2000) | RoBERTa | 0.9333 | 0.0280 | 0.0544 | 0.7787 | 0.7947 | 0.5636 | 0.0020 |
| | BiLSTM | 0.8594 | 0.0550 | 0.1034 | 0.6801 | 0.6768 | 0.5263 | 0.0090 |
| | TF-IDF+SVC | 0.9000 | 0.0810 | 0.1486 | 0.6604 | 0.6724 | 0.5441 | 0.0090 |
| | TF-IDF+LR | 0.9788 | 0.1850 | 0.3112 | 0.7528 | 0.7771 | 0.5714 | 0.0040 |
| | DistilBERT | 1.0000 | 0.0140 | 0.0276 | 0.8222 | 0.8250 | 0.6348 | 0.0000 |
| Frontiers human (6000) | RoBERTa | 0.9995 | 0.8402 | 0.9130 | 0.9902 | 0.9981 | 0.9987 | 0.0020 |
| | BiLSTM | 0.9975 | 0.7226 | 0.8381 | 0.9501 | 0.9901 | 0.9734 | 0.0090 |
| | TF-IDF+SVC | 0.9978 | 0.8214 | 0.9011 | 0.9699 | 0.9940 | 0.9802 | 0.0090 |
| | TF-IDF+LR | 0.9990 | 0.8400 | 0.9126 | 0.9823 | 0.9965 | 0.9901 | 0.0040 |
| | DistilBERT | 1.0000 | 0.8572 | 0.9231 | 0.9971 | 0.9994 | 1.0000 | 0.0000 |
| Frontiers LLM (5986) | RoBERTa | 0.9996 | 0.9394 | 0.9686 | 0.9994 | 0.9998 | 0.9998 | 0.0020 |
| | BiLSTM | 0.9962 | 0.4787 | 0.6467 | 0.9542 | 0.9883 | 0.9720 | 0.0090 |
| | TF-IDF+SVC | 0.9971 | 0.6195 | 0.7642 | 0.9379 | 0.9872 | 0.9584 | 0.0090 |
| | TF-IDF+LR | 0.9990 | 0.8187 | 0.8999 | 0.9748 | 0.9951 | 0.9881 | 0.0040 |
| | DistilBERT | 1.0000 | 0.7669 | 0.8681 | 0.9988 | 0.9998 | 0.9996 | 0.0000 |
| BEC (5181) | RoBERTa | 0.9987 | 0.3772 | 0.5476 | 0.9513 | 0.9880 | 0.9719 | 0.0020 |
| | BiLSTM | 0.9899 | 0.2102 | 0.3468 | 0.8320 | 0.9518 | 0.8774 | 0.0090 |
| | TF-IDF+SVC | 0.9945 | 0.3882 | 0.5584 | 0.9001 | 0.9738 | 0.9203 | 0.0090 |
| | TF-IDF+LR | 0.9970 | 0.3164 | 0.4804 | 0.9067 | 0.9747 | 0.9285 | 0.0040 |
| | DistilBERT | 1.0000 | 0.1502 | 0.2612 | 0.9560 | 0.9891 | 0.9706 | 0.0000 |
| BEC-poisoned (5181) | RoBERTa | 0.9989 | 0.4303 | 0.6015 | 0.9565 | 0.9894 | 0.9759 | 0.0020 |
| | BiLSTM | 0.9905 | 0.2243 | 0.3658 | 0.8436 | 0.9553 | 0.8877 | 0.0090 |
| | TF-IDF+SVC | 0.9942 | 0.3662 | 0.5352 | 0.8939 | 0.9719 | 0.9163 | 0.0090 |
| | TF-IDF+LR | 0.9968 | 0.3004 | 0.4617 | 0.8975 | 0.9720 | 0.9225 | 0.0040 |
| | DistilBERT | 1.0000 | 0.2344 | 0.3798 | 0.9614 | 0.9906 | 0.9732 | 0.0000 |
| PhishFuzzer (205) | RoBERTa | 0.5109 | 0.6796 | 0.5833 | 0.5711 | 0.6145 | 0.5078 | 0.6569 |
| | BiLSTM | 0.5312 | 0.4951 | 0.5126 | 0.5149 | 0.5770 | 0.5099 | 0.4412 |
| | TF-IDF+SVC | 0.5263 | 0.6796 | 0.5932 | 0.5819 | 0.6379 | 0.5027 | 0.6176 |
| | TF-IDF+LR | 0.5238 | 0.7476 | 0.6160 | 0.5603 | 0.5872 | 0.5102 | 0.6863 |
| | DistilBERT | 0.5299 | 0.6019 | 0.5636 | 0.5478 | 0.5897 | 0.5108 | 0.5392 |

### 11.2 Calibration — Task 2b (2026-11-20)

Mean predicted P(phishing) on phishing vs legit emails, ECE with 10 bins, recomputed from the saved prediction CSVs. Full report: `experiments/modern_phishing_ood/reports/calibration_report.md`.

| Set | Model | mean P(phish) | mean P(legit) | ECE-10 |
|---|---|---|---|---|
| PhishNChips | RoBERTa | 0.033 | 0.002 | 0.483 |
| | BiLSTM | 0.054 | 0.011 | 0.474 |
| | TF-IDF+SVC | 0.125 | 0.042 | 0.417 |
| | TF-IDF+LR | 0.276 | 0.121 | 0.301 |
| | DistilBERT | 0.018 | 0.000 | 0.491 |

In-distribution ECE-10 (test n=3,495, from saved seed-42-refit `y_prob`): TF-IDF+LR 0.032, TF-IDF+SVC 0.006, DistilBERT 0.015 (from saved Kernel-2 seed-42 `test_prob.npy`) — all three well calibrated in-distribution, so their OOD ECE of 0.30 / 0.42 / 0.49 is shift-induced. RoBERTa/BiLSTM in-distribution ECE: NOT RUN YET — no saved per-email in-distribution test probabilities exist locally for either model (re-scoring would be a new model run).

**Verdict — ranking failure or calibration/threshold failure (primary set):** RoBERTa — calibration/threshold failure (ranking usable: ROC-AUC 0.7787, P@R.90 0.5636; scores squashed near zero, mean P(phishing) 0.033). TF-IDF+LR — calibration/threshold failure (third-best ranking on this set: ROC-AUC 0.7528, P@R.90 0.5714; least miscalibrated, ECE 0.301). BiLSTM — both (weakest P@R.90 0.5263 and second-worst ROC-AUC 0.6801, plus squashed scores). TF-IDF+SVC — ranking failure first (worst ROC-AUC 0.6604), threshold second. DistilBERT — calibration/threshold failure (best ranking of the five: ROC-AUC 0.8222, PR-AUC 0.8250, P@R.90 0.6348; worst calibration: mean P(phishing) 0.018, ECE 0.491). No model is well calibrated OOD (ECE-10 0.30–0.49).

### 11.3 Threshold transfer — Task 2c (2026-11-20)

**Label: target-domain recalibration (uses labelled OOD data) — NOT pure OOD.** Threshold picked on a stratified 20% slice of PhishNChips (seed 42, n=400) at recall floor 0.85 (maximising precision), evaluated on the remaining 80% (n=1,600). Computed from the saved PhishNChips prediction CSVs; no retraining, no new scoring. Full report: `experiments/modern_phishing_ood/reports/threshold_transfer.md`.

| Model | Threshold (20% slice) | Holdout P | Holdout R | Holdout F1 | Holdout FPR |
|---|---:|---|---|---|---|
| RoBERTa | 0.000104 | 0.629 | 0.808 | 0.707 | 0.476 |
| BiLSTM | 3.3e-09 | 0.547 | 0.841 | 0.663 | 0.698 |
| TF-IDF+SVC | 0.00392 | 0.561 | 0.819 | 0.666 | 0.640 |
| TF-IDF+LR | 0.0799 | 0.595 | 0.846 | 0.699 | 0.575 |
| DistilBERT | 7.7e-05 | 0.708 | 0.835 | 0.766 | 0.345 |

Recall recovers to 0.81–0.85 at precision 0.55–0.71 — but at FPR 0.35–0.70, and the fitted thresholds (notably BiLSTM 3.3e-09) are symptoms of miscalibration, not usable operating points. DistilBERT transfers best (holdout F1 0.766), consistent with its best ranking on this set.

### 11.4 Paired statistics vs TF-IDF+LR — Task 2d (2026-11-20)

Primary set only, n=2,000, threshold 0.5, repo's own `mcnemar_test` / bootstrap functions (1,000 resamples, seed 42). McNemar compares paired classification errors only; it is NOT a precision test. Full report: `experiments/modern_phishing_ood/reports/paired_stats_vs_tfidf_lr.md`.

| Comparison | McNemar b/c | p | P diff [95% CI] | R diff [95% CI] | F1 diff [95% CI] | ROC-AUC diff [95% CI] |
|---|---|---|---|---|---|---|
| RoBERTa vs LR | 177 / 22 | 9.6e-28 | −0.046 [−0.145, 0.033] | −0.157 [−0.183, −0.133] | −0.257 [−0.292, −0.216] | +0.026 [0.003, 0.050] |
| BiLSTM vs LR | 162 / 27 | 1.9e-22 | −0.120 [−0.204, −0.039] | −0.130 [−0.155, −0.106] | −0.208 [−0.245, −0.171] | −0.073 [−0.097, −0.049] |
| SVC vs LR | 129 / 20 | 8.9e-19 | −0.079 [−0.143, −0.025] | −0.104 [−0.126, −0.083] | −0.163 [−0.195, −0.129] | −0.092 [−0.111, −0.074] |
| DistilBERT vs LR | 178 / 11 | 1.4e-33 | +0.021 [0.005, 0.044] | −0.171 [−0.196, −0.148] | −0.284 [−0.316, −0.248] | +0.069 [0.047, 0.090] |
| DistilBERT vs RoBERTa | 19 / 7 | 0.0310 | +0.067 [0.000, 0.161] | −0.014 [−0.024, −0.005] | −0.027 [−0.045, −0.010] | +0.044 [0.026, 0.062] |

LR commits significantly fewer paired errors than each of the other four. DistilBERT is the one exception to the no-gain pattern: its precision difference vs LR is positive (+0.021 [0.005, 0.044], CI excludes 0) and so are its ROC-AUC (+0.069) and PR-AUC (+0.048 [0.027, 0.069]) differences — but its precision is 1.0000 on 14 true positives and 0 false positives, a degenerate precision at recall 0.014, not a usable gain. Against RoBERTa, DistilBERT makes significantly more paired errors (p=0.0310) but ranks better (ROC-AUC +0.044, PR-AUC +0.030, both CIs exclude 0). LR's precision lead over RoBERTa is NOT paired-significant (CI includes 0).

### 11.5 Latency — Task 2e (verified 2026-11-20)

Verified against `results/rerun_latency/latency_explain_time.json` (median of 200, batch 1, Tesla T4) and `results/robustness_rerun/baseline_results.json` (seed-42 refit, n_timed=200). RoBERTa GPU 15.50 / CPU 242.50 ms/email; BiLSTM GPU 8.37 / CPU 19.90; TF-IDF+SVC CPU 8.469, TF-IDF+LR CPU 1.089; DistilBERT GPU 9.07 / CPU 118.62 ms/email (from `results/distilbert/latency/latency_explain_time.json`, same protocol, seed-42 checkpoint). CPU multiples vs SVC: RoBERTa 28.63x, BiLSTM 2.35x, LR 0.13x, DistilBERT 14.00x. **GPU multiples vs TF-IDF+SVC: NOT COMPUTABLE** — TF-IDF is CPU-only (sklearn, GPU latency never measured). Correction: the DistilBERT final-training driver reported a "CPU median 9.55 ms" — that timed the model without moving it off CUDA, so it measured GPU time again; the true CPU latency is 118.62 ms from the latency kernel, which switches device properly.

### 11.6 Threats to validity

1. The OOD "legit" class is a shared 1,000-email PhishNChips legit sample (legit bodies derived from Tranco top domains, plus cross-domain expansion) used in 5 of the 6 sets — FPR estimates depend on it; only PhishFuzzer has its own legit emails (102), and there FPR is 0.44–0.69.
2. Synthetic/LLM-generated sets may carry generator artifacts (Frontiers LLM, BEC).
3. The Frontiers human slice is 2007–08-era corpora, not modern.
4. The multimodal set is synthetic with n=96 test.
5. Single seeds for some models (DistilBERT 1 seed, CNN 1 seed, RoBERTa single test evaluation; BiLSTM and TF-IDF 3 seeds on text).
6. No verified real-world 2023+ phishing+legit email-body corpus exists; the primary set is synthetic stimuli around real verified URLs, not real inbox mail.

### 11.7 Test verification (Task 4, Kaggle 2026-11-20)

Full suite on Kaggle (kernel `phishing-xai-pytest-ocr-qr-env-check` v3, CPU-only, tesseract 5.3.4 + libzbar0 installed, Python 3.13.18): **315 passed / 1 failed** (316 collected). The only failure is `tests/test_sanity.py::test_environment_sanity` (asserts Python 3.10 exactly; Kaggle runs 3.13.18) — a test/environment expectation mismatch, not a code failure. All 5 OCR/QR `test_extract.py` tests pass on Kaggle; `transformers_interpret` imports fine there. Local run the same day: 309 passed / 7 failed (same sanity failure, `transformers_interpret` missing locally, 5 OCR/QR failures — tesseract/zbar not installed in this VM). Report: `experiments/modern_phishing_ood/logs/kaggle_pytest_report.md`.
