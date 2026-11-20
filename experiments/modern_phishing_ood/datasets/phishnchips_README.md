---
pretty_name: PhishNChips
language:
- en
task_categories:
- text-classification
tags:
- phishing-detection
- llm-security
- benchmark
- email-security
size_categories:
- 1K<n<10K
license: other
configs:
- config_name: emails
  data_files:
  - split: core
    path: core_emails.csv
  - split: cross_domain_legitimate_v5
    path: cross_domain_legitimate_v5.csv
  - split: infrastructure_phishing_expanded
    path: infrastructure_phishing_expanded.csv
  - split: real_phishing_validation
    path: real_phishing_validation.csv
- config_name: results
  data_files:
  - split: benchmark_results
    path: benchmark_results.csv
---

# PhishNChips: A Benchmark for LLM Email-Agent Security

PhishNChips is a large-scale benchmark for evaluating how system prompt configurations influence the security behavior of LLM-based email agents. This repository contains the canonical v5.2 release, featuring 2,000 email stimuli and 220,000 adjudicated model evaluations.

## Dataset Overview

The benchmark measures a critical deployment variable: how strongly an LLM's system prompt shapes its phishing detection capabilities and false-positive characteristics. PhishNChips provides a controlled environment to study the trade-offs between security, helpfulness, and instruction following in agentic systems.

### Core Components

- **Core Benchmark (2,000 emails):** 1,000 phishing emails (grounded in real malicious URLs) and 1,000 legitimate workplace emails (including 333 cross-domain samples).
- **Adjudicated Evaluations (220,000):** A full result grid spanning 11 frontier models and 10 distinct system prompt strategies.
- **URL Evasion Taxonomy:** Stratified samples covering zero-signal, hidden-signal, and inverted-signal (infrastructure phishing) evasion techniques.

## File Structure

| File | Description |
|---|---|
| `core_emails.csv` | The primary 2,000-email benchmark stimuli. |
| `benchmark_results.csv` | Full result grid (11 models x 10 strategies x 2,000 emails). |
| `reference_results.csv` | Summary metrics and leaderboard rankings. |
| `prompt_strategies.json` | Detailed definitions of the 10 evaluated system prompts. |
| `cross_domain_legitimate_v5.csv` | Dedicated split for cross-domain false-positive analysis. |
| `infrastructure_phishing_expanded.csv` | Auxiliary split for infrastructure-level stress testing. |
| `real_phishing_validation.csv` | Historical real-phishing validation set (Nazario). |
| `croissant.json` | Machine-readable metadata (MLCommons Croissant format). |
| `SOURCE_LICENSES.md` | Comprehensive provenance and licensing documentation. |

## Loading the Dataset

The dataset is organized into two configs because the email stimuli and the model-evaluation results have different schemas:

```python
from datasets import load_dataset

# Email stimuli (one row per email): id, url_raw, phish_label, email_content,
# strategy, url_category, datasource, model_used
emails = load_dataset("AreLit/PhishNChips", "emails", split="core")
cross_domain = load_dataset("AreLit/PhishNChips", "emails", split="cross_domain_legitimate_v5")
infra_phish = load_dataset("AreLit/PhishNChips", "emails", split="infrastructure_phishing_expanded")
nazario = load_dataset("AreLit/PhishNChips", "emails", split="real_phishing_validation")

# Adjudicated model evaluations (one row per model x strategy x sample):
# sample_id, model, strategy, true_label, prediction, correct, raw_response, error
results = load_dataset("AreLit/PhishNChips", "results", split="benchmark_results")
```

`reference_results.csv`, `prompt_strategies.json`, `croissant.json`, and `SOURCE_LICENSES.md` are not exposed as `datasets` splits; download them directly from the file list.

## Datasource Composition

The benchmark leverages high-quality, verified data from several major security feeds and research utilities:

| Datasource | Count | Category |
|---|---:|---:|
| `phishtank` | 700 | Phishing |
| `tranco` | 662 | Legitimate |
| `cross_domain_expansion_v1` | 333 | Legitimate |
| `github_phishing_db_live` | 172 | Phishing |
| `github_phishing_db` | 62 | Phishing |
| `openphish` | 66 | Phishing |
| `adversarial_legit` | 5 | Legitimate |

## Responsible Use

This dataset contains real malicious URL indicators. Treat all URLs as **offline text strings only**. Do not visit, crawl, or execute URLs from this repository. This resource is intended strictly for security research and defensive evaluation.

## License and Attribution

The PhishNChips benchmark project code and synthetic content are released under an **MIT License**. 

However, the dataset incorporates third-party-derived malicious URL indicators and benign domain seeds. These components are redistributed for academic research with attribution. **Review [SOURCE_LICENSES.md](https://huggingface.co/datasets/AreLit/PhishNChips/blob/main/SOURCE_LICENSES.md) for full citations and provenance details.**

- **Nazario Phishing Corpus:** CC-BY-4.0.
- **OpenPhish:** Approved for academic research use (Apr 6, 2026).
- **PhishTank:** Cleared via Cisco/PhishTank terms.
- **Tranco:** Sourced via academic use norms (Le Pochat et al. 2019).

## Citation

During active conference review, this public mirror minimizes direct
author-identifying citation metadata. A neutral provisional citation is:

```bibtex
@misc{phishnchips2026benchmark,
  title={PhishNChips: A Benchmark for Evaluating LLM Email-Agent Security Under Deployment Configuration Variation},
  year={2026},
  note={Public benchmark release; full bibliographic metadata restored after review}
}
```
