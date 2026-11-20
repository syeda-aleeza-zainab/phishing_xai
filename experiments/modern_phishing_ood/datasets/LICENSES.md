# Third-party dataset licences — NOTICE

The files under `experiments/modern_phishing_ood/datasets/` are third-party
data. They are NOT tracked in Git (kept on disk locally and in the Kaggle OOD
dataset `phishing-xai-ood-modern`). Only download scripts, citations and
licence files are tracked. Full provenance (dates, sizes, overlap checks,
accept/reject reasons) is in
`../reports/dataset_provenance.md`.

## Training set (original project data)

| Dataset | Source | Licence |
|---|---|---|
| Phishing Email Detection (subhajournal, Kaggle, 2023-07-06) — `data/Phishing_Email.csv`, 18,650 rows | https://www.kaggle.com/datasets/subhajournal/phishingemails | Not stated on the Kaggle page. Hugging Face mirror `zefang-liu/phishing-email-dataset` is LGPL-3.0. |

## OOD sets used in the experiment (accepted)

| Dataset | Source | Licence |
|---|---|---|
| PhishNChips v5.2 (primary OOD, 2,000 rows) | https://huggingface.co/datasets/AreLit/PhishNChips | MIT (code + synthetic content); URL indicators redistributed for academic research |
| PhishFuzzer seeds, "Manual" 300 subset (secondary check) | https://github.com/DataPhish/PhishFuzzer | CC BY-NC 4.0 |
| Frontiers 2026 multi-LLM corpus (synthetic track, phishing-only) | https://doi.org/10.5281/zenodo.20250116 | CC-BY-4.0 |
| Adversarial BEC Email Dataset (synthetic track, phishing-only) | https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset | CC-BY-4.0 (README, cite Adjei, Y.O. 2025) |

## OOD candidates downloaded but REJECTED (kept on disk for audit, not used)

| Dataset | Source | Licence |
|---|---|---|
| Email Phishing Dataset (ethancratchley) — features only, no raw text | https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset | Apache 2.0 |
| Human-LLM Generated Phishing–Legitimate (francescogreco97) | https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails | Unknown (not recorded on the Kaggle page) |
| RSB-1000 (RevealWire) — SMS-style, wrong modality | https://huggingface.co/datasets/RevealWire/rsb-1000 | CC-BY-4.0 |

## Notes

- The isuranga 100k set (licence not stated on its Kaggle page) was downloaded
  to `~/workspace/kaggle_downloads/` only; it is not in this repository.
- The Zenodo Frontiers vendored code once contained a leaked Azure AI Services
  key (dataset authors', not ours). The key was removed from the working copy
  and from the pushed GitHub copy; it remains in local Git history (commit
  8a78ed9) pending a history-rewrite decision. Do not re-add it.
- `generate_emails.py` (BEC generation) reads its Gemini key from the
  `GEMINI_API_KEY` environment variable and refuses to run without it.
  See `.env.example` at the repository root.
