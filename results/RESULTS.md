# Consolidated Experimental Results & Evaluation Report

*Generated on: 2026-10-19 00:00:40*

## 1. Run Metadata
*NOT RUN YET* (Production training/tuning metadata not yet available).

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

- **Split Manifest**: Saved row IDs to [`results\splits.json`](splits.json)

## 3. Hyperparameter Tuning (Random Search)
*NOT RUN YET* (Random search tuning has not yet been executed in production mode).

## 4. Final Model Training & Test Evaluation
*NOT RUN YET* (Final model training and single-pass test evaluation pending).

## 5. Explainability & Faithfulness Evaluation
### 5.1 Faithfulness Benchmarks (Comprehensiveness & Sufficiency)
*NOT RUN YET*

### 5.2 Alpha Parameter Ablation Study (LITA Weighting)
*NOT RUN YET*

### 5.3 Global Explanations (Top Attributed Words)
*NOT RUN YET*

## 6. Limitations & Methodological Caveats
- Evaluations conducted with deterministic seed; variance across random seeds may exist.
- Deduplication removes duplicate texts and conflicting labels (e.g. 'empty' placeholders) to ensure clean train/test boundary.
- Synonym-replacement augmentation balances class frequencies on the training split only.
