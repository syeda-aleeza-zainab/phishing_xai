# Cross-model evaluation of phishing detectors against LLM-generated emails

This repository contains the complete data, code and results accompanying
the manuscript:

> Gutierrez, R., Villegas-Ch, W., and Govea, J. (2026).
> Cross-model evaluation of phishing detectors against LLM-generated emails.
> Submitted to Frontiers.

If you use this dataset or code, please cite both the manuscript and
this Zenodo record.

---

## 1. Repository structure

```
cross-model-phishing/
├── README.md
├── LICENSE
├── code/
│   ├── 01_unify_datasets.py        Assemble human phishing corpus from 6 sources
│   ├── 02_inspect_corpus.py        Exploratory analysis of the combined human pool
│   ├── 03_sample_corpus.py         Stratified sub-sampling to 5,000 emails
│   ├── 04_merge_llm_corpus.py      Merge raw LLM generations into one corpus
│   ├── 05_extract_features.py      Compute 17 stylometric features
│   ├── 06_train_evaluate.py        Tasks A, B, B', C, D + SHAP analysis
│   ├── 07_make_figures.py          Generate all figures used in the paper
│   ├── config.py                   Centralised paths and parameters
│   ├── prompts.py                  LLM phishing generation prompts
│   ├── generate.py                 Unified Azure AI Foundry generator
│   └── requirements.txt            Python dependencies
├── data/
│   ├── corpus_features.csv         9,986 rows × 17 stylometric features
│   └── llm_corpus_sampled.csv      4,986 LLM-generated phishing emails
└── results/
    ├── task_a_intra_model.csv          5-fold CV per LLM
    ├── task_b_cross_model_matrix.csv   Default-threshold cross-model F1 matrix
    ├── task_b_cross_model_full.csv     Full per-cell metrics for Task B
    ├── task_b_recalibrated_matrix.csv  Recalibrated F1 matrix
    ├── task_b_recalibrated_full.csv    Full per-cell metrics for Task B'
    ├── task_b_summary.json             Gap analysis summary
    ├── task_c_cross_dataset_human.csv  Cross-dataset human verification
    ├── task_d_aggregated.csv           Aggregated-pool detector results
    └── feature_importance.csv          SHAP values per LLM
```

---

## 2. Dataset description

The combined corpus contains **9,986 phishing emails**:

- **5,000 human-written phishing emails** drawn from six publicly available
  corpora (CEAS-08, TREC-07, Nazario, Nigerian Fraud, lingspam and a
  fraud-labeled Enron subset), with stratified per-source quotas. Token
  length range: 30-500 tokens. Mean length: 164.8 tokens (median 115).

- **4,986 LLM-generated phishing emails** produced through Azure AI Foundry
  under controlled prompting conditions across three modern LLMs and five
  thematic categories:

  | Model        | Emails | Categories                                        |
  |--------------|-------:|---------------------------------------------------|
  | GPT-4.1      | 1,665  | banking, parcel delivery, IT support, tax, HR    |
  | DeepSeek 3.2 | 1,665  | banking, parcel delivery, IT support, tax, HR    |
  | LLaMA 3.3 70B| 1,656  | banking, parcel delivery, IT support, tax, HR    |

Generation parameters: temperature = 0.7, top-p = 0.95, max tokens = 600,
single roleplay prompt template (see `code/prompts.py`).

---

## 3. Stylometric features

Seventeen features are extracted from each email, grouped into four
families:

- **Lexical (4):** type-token ratio (TTR), mean word length, mean
  sentence length in tokens, Yule's K.
- **Syntactic (4):** clause density, noun ratio, verb ratio, mean
  dependency-parse depth.
- **Stylistic (5):** imperative count, first-person pronoun ratio,
  second-person pronoun ratio, politeness density, urgency density.
- **Phishing-specific (4):** URL density, call-to-action density,
  authority-appeal density, time-pressure density.

Densities are reported per 100 words. Implementation uses spaCy
(`en_core_web_sm`) for tokenisation, POS tagging and dependency
parsing. Dictionaries are documented in `code/05_extract_features.py`.

---

## 4. How to reproduce the experiments

### Requirements

```bash
pip install -r code/requirements.txt
python -m spacy download en_core_web_sm
```

Tested on Python 3.11 (Windows and Linux).

### Pipeline (full reproduction from human + LLM raw data)

```bash
cd code
python 01_unify_datasets.py
python 02_inspect_corpus.py
python 03_sample_corpus.py
python generate.py --model all        # requires AZURE_API_KEY env var
python 04_merge_llm_corpus.py
python 05_extract_features.py
python 06_train_evaluate.py
python 07_make_figures.py
```

### Quick reproduction (from pre-computed features)

If you only want to reproduce the experiments using the features we
already extracted, skip steps 1-5 and run:

```bash
cd code
python 06_train_evaluate.py
python 07_make_figures.py
```

This uses `data/corpus_features.csv` directly. Expected runtime:
under 10 minutes on a modern laptop.

### Random seed

All splitting, sampling and model initialisation use seed = 42 for
reproducibility. Exact replication of LLM outputs cannot be guaranteed
because API providers may update model versions; we document the
model identifiers and API parameters used.

---

## 5. Key results

Headline numbers from the paper:

- **Intra-model F1 (XGBoost, 5-fold CV):** 0.968 (GPT-4.1), 0.966
  (DeepSeek 3.2), 0.955 (LLaMA 3.3 70B); AUC-ROC > 0.998 in all cases.
- **Cross-model F1 (default threshold = 0.5):** mean diagonal = 0.999,
  mean off-diagonal = 0.719, transferability gap = **28.1 percentage
  points**.
- **Cross-model F1 (threshold recalibrated on a 30% slice of the
  target):** mean off-diagonal = 0.957, transferability gap = **4.0
  percentage points** (86% reduction).
- **Aggregated-pool detector:** F1 = 0.997 on each individual LLM.
- **Stable top-5 features across all three LLMs:** politeness density
  and type-token ratio.

---

## 6. Human corpora sources (not included here)

The human phishing emails used in this study were taken from publicly
available corpora. Please obtain them from their original sources:

- CEAS-08 (Cormack, Goodman & Heckerman, 2008)
- TREC-07 Spam Track (Cormack, 2007)
- Nazario Phishing Corpus (Nazario, 2007;
  https://monkey.org/~jose/phishing/)
- Nigerian Fraud Email Corpus (CLAIR group, University of New Brunswick)
- lingspam (Androutsopoulos et al., 2000)
- Enron Email Dataset (Klimt & Yang, 2004)

We do not redistribute these corpora here. Our normalisation pipeline
(`code/01_unify_datasets.py`) shows the column-name mapping applied
to each source.

---

## 7. License

- Code: MIT License (see `LICENSE` file).
- Data: Creative Commons Attribution 4.0 International (CC BY 4.0).

You are free to use, modify, and redistribute the code and data with
attribution.

---

## 8. Citation

If you use this work, please cite the manuscript and the dataset:

```bibtex
@article{gutierrez2026crossmodel,
  author  = {Gutierrez, Rommel and Villegas-Ch, William and Govea, Jaime},
  title   = {Cross-model evaluation of phishing detectors against
             {LLM}-generated emails},
  journal = {[Frontiers journal name]},
  year    = {2026},
  note    = {Submitted}
}

@dataset{gutierrez2026data,
  author    = {Gutierrez, Rommel and Villegas-Ch, William and Govea, Jaime},
  title     = {Cross-model evaluation of phishing detectors against
               {LLM}-generated emails: dataset, code and results},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

---

## 9. Contact

For questions about the dataset or code, contact:

**Rommel Gutierrez**
rommeljair.gutierrez@udla.edu.ec
