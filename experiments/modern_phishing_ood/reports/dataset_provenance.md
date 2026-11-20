# OOD Dataset Provenance — Modern Phishing Email Test Sets

**Date of hunt:** 2026-11-19/20 (PKT). **Hunt agent:** dataset-hunt subagent.
**Task:** find MODERN (collection 2023–2026, verified dates) phishing email datasets usable as an
out-of-distribution (OOD) test set for RoBERTa vs BiLSTM phishing detection.

## 0. What the training set actually is (leakage baseline)

The training file `~/workspace/phishing_xai_impl/data/Phishing_Email.csv` has **18,650 rows:
11,322 Safe Email / 7,328 Phishing Email**, columns `Email Text, Email Type`, 1,112 exact-duplicate
texts. It is the Kaggle **"Phishing Email Detection"** set by `subhajournal` (uploaded
2023-07-06, file `Phishing_Email.csv`, 52 MB), mirrored on Hugging Face as
`zefang-liu/phishing-email-dataset` (created 2024-01-17, same filename, LGPL-3.0, card says
"direct copy of the 'Phishing Email Detection' dataset from Kaggle"). Any OOD candidate built
from the same legacy corpora (Enron, Ling, CEAS-08, Nazario, Nigerian Fraud, SpamAssassin,
TREC) is therefore a leakage risk, not an OOD set — several candidates below were rejected
on exactly this ground.

**Overlap method used throughout:** normalized-text MD5 — lowercase, strip non-alphanumerics,
collapse whitespace — computed on email body text, compared against all 17,487 unique
normalized hashes of the 18,650 training rows. Exact-duplicate detection only; near-duplicates
(paraphrases, re-wrapped bodies) were NOT checked. "0 overlap" below means zero exact
normalized-text matches, not zero near-duplicates.

## Headline finding

**No verified real-world 2023+ phishing + legitimate email-body corpus was found.**
Every candidate that is both (a) dated 2023–2026 and (b) downloadable is either synthetic /
simulated email text, or synthetic stimuli built around real (verified) malicious URLs, or
phishing-only. The recommendation below therefore picks the best available modern set and
labels its nature honestly: it is a **modern URL-grounded synthetic-stimuli OOD test**, not
real-world inbox mail. Do not present its results as "real-world 2025 phishing performance".

## Summary table

| # | Name | Source URL | Collection / publication dates + evidence | Size | Phishing / Legit | Licence | Real vs synthetic | Verdict |
|---|------|-----------|-------------------------------------------|------|------------------|---------|-------------------|---------|
| 1 | PhishNChips v5.2 | https://huggingface.co/datasets/AreLit/PhishNChips | HF created 2026-04-04, modified 2026-04-18; card cites OpenPhish approval "Apr 6, 2026"; citation year 2026 | 2,000 | 1,000 / 1,000 | MIT (code + synthetic content); URL indicators redistributed for academic research | Synthetic email bodies built around REAL verified malicious URLs (PhishTank/OpenPhish/GitHub phishing DB); legit bodies derived from Tranco domains | **ACCEPTED — primary OOD set** (with caveats, §1) |
| 2 | PhishFuzzer seeds — "Manual" 300 subset | https://github.com/DataPhish/PhishFuzzer | Repo created 2026-03-12, updated 2026-11-07; per-row `Year` field: Manual rows 2024: 22, 2025: 210, 2026: 65 | 300 (of 3,300 seeds) | 103 phish / 102 valid / 95 spam | CC BY-NC 4.0 | Real human-collected ("privately collected") per README; collection method not yet verified from creation docs | **NEEDS WORK — small secondary check** (§2) |
| 3 | Frontiers 2026 multi-LLM corpus | https://doi.org/10.5281/zenodo.20250116 | Zenodo published 2026-05-16; paper received 2026-05-17, published 2026-07-07 (Front. Big Data 9:1883452) | 9,986 | 9,986 phish / 0 legit | CC-BY-4.0 | 4,986 LLM-generated (GPT-4.1, DeepSeek 3.2, Llama 3.3 70B); 5,000 human phish from OLD corpora (TREC-07, CEAS-08, Nazario, Nigerian, Enron) | **ACCEPTED — synthetic track only, phishing-only** (§3) |
| 4 | Adversarial BEC Email Dataset | https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset | Kaggle updated 2025-11-21; README: generated with Gemini Flash 2.5 | 4,211 | 4,181 phish / 0 legit (30 NaN labels) | CC-BY-4.0 (README) | Fully synthetic (LLM-generated BEC) | **ACCEPTED — synthetic track only, phishing-only** (§4) |
| 5 | SPAM + Genuine Mail Contents (isuranga 100k) | https://www.kaggle.com/datasets/isuranga/spam-genuine-mail-contents-dataset | Kaggle updated 2025-09-27; in-data `date` 2024-03-31 → 2025-09-23, ~5,000/month (suspiciously uniform) | 100,000 | 50,000 spam / 50,000 ham | Not stated on Kaggle page (checked 2026-11-19) | Template-simulated, NOT real mail (§5) | **REJECTED** |
| 6 | Email Phishing Dataset (ethancratchley) | https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset | Kaggle updated 2025-04-01 | >500,000 rows claimed | ~20k phish / ~500k safe (features only) | Apache 2.0 | Real sources, but NO raw text | **REJECTED** (§6) |
| 7 | Human-LLM Generated Phishing–Legitimate (francescogreco97) | https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails | Kaggle updated 2024-04-10; in-data `date`: legit 2007–2019, human phish 2019–2022 | 4,000 (4 × 1,000) | 2,000 / 2,000 claimed | Not recorded | Human side pre-2023; LLM side broken files | **REJECTED** (§7) |
| 8 | Opara et al. 2025 GPT-4o set (63 + 63) | Paper doi 10.1016/j.eswa.2025.127044 | Paper: generated Aug–Sep 2024; received 2024-10-04, online 2025-03-10 | 126 | 63 / 63 | CC BY-NC (paper) | LLM-generated (GPT-4o) | **NOT LOCATED** (§8) |
| 9 | Phishing Email Curated Datasets (Zenodo) | https://doi.org/10.5281/zenodo.8339691 | Zenodo updated 2024-09-25; source: "11 datasets spanning from 1995 to 2022" | 11 corpora | varies | Not recorded | Real, but pre-2023 by construction | **REJECTED** (§9) |
| 10 | naserabdullahalam / zefang-liu / subhajournal / puyang "seven-phishing" | Kaggle / HF (see §10) | Uploads 2023–2026, but underlying corpora 1995–2022 | up to ~82,500 / 203k | varies | LGPL-3.0 (zefang-liu) | Real legacy corpora = training sources | **REJECTED — leakage** (§10) |
| 11 | PhreshPhish | https://huggingface.co/datasets/phreshphish/phreshphish | v1.0.0 2025-05-14; v1.0.1 2026-02-07 (samples Mar–Dec 2025) | 666,315 (train 498,255 + test 168,060) | 298,326 phish / 367,989 benign | CC BY 4.0 | Real webpages, NOT emails | **REJECTED** (§11) |
| 12 | PhishTank / OpenPhish / APWG | https://www.phishtank.com/ , https://openphish.com/ | Live feeds (hourly / 12-hourly) | ~74k verified-online URLs (PhishTank, Aug 2026 check) | URLs only | Feed terms; OpenPhish community free | URL indicators only, no email bodies | **REJECTED as email source** (§12) |
| 13 | RSB-1000 (RevealWire) | https://huggingface.co/datasets/RevealWire/rsb-1000 | HF created 2026-10-22; "Frozen October 2026" | 1,000 | 780 scam / 220 legit | CC-BY-4.0 | Fully synthetic, SMS-style short messages | **REJECTED for email task** (§13) |
| 14 | Zero-Day Phishing Emails Corpus (mutahirshaukat) | https://www.kaggle.com/datasets/mutahirshaukat/zero-day-phishing-emails-corpus | Kaggle updated 2026-05-11; 31.4 MB CSV | Unknown (not inspected) | Unknown | Not recorded | Unknown | **NOT INSPECTED** (§14) |

---

## 1. PhishNChips v5.2 — ACCEPTED (primary OOD set, with caveats)

- **Source:** https://huggingface.co/datasets/AreLit/PhishNChips
- **Dates (verified):** HF API `createdAt` 2026-04-04T17:40:58Z, `lastModified` 2026-04-18T23:31:05Z;
  README cites "OpenPhish: Approved for academic research use (Apr 6, 2026)"; provisional
  citation `@misc{phishnchips2026benchmark, year={2026}}`. No per-email collection dates exist
  in the data — the 2026 date is the benchmark release date, and the URLs come from feeds
  current at build time.
- **Downloaded:** `datasets/phishnchips_core_emails.csv` (1,560,717 bytes) + `phishnchips_README.md`.
- **Inspected (measured, not claimed):** 2,000 rows; `phish_label` 1,000 phishing / 1,000 legitimate;
  0 duplicate bodies; columns `id, url_raw, phish_label, email_content (JSON: sender/from/subject/body),
  strategy, url_category, datasource, model_used`. Datasource × label: PhishTank 700 phish,
  GitHub phishing DB live 172 + DB 62, OpenPhish 62 + OpenPhish-live 4, Tranco 662 legit,
  cross-domain expansion 333 legit, adversarial legit 5. Median body length 381 chars (phish) /
  427 chars (legit) — much shorter than training median 880 chars. `model_used`: primary 1,610,
  template_repair_v1 333, v5.2_release 57. English; 13.5% of bodies contain non-ASCII.
- **Overlap vs training:** **0 / 2,000** exact normalized-text matches against all 17,487 unique
  training hashes (and 0 / 200 against the first-200 training rows).
- **Labelling method (documented):** phishing labels are grounded in the URL, not the prose —
  URLs verified via PhishTank (community-verified), OpenPhish, GitHub phishing DB; legitimate
  URLs from Tranco top domains. This is a real, documented verification chain for the *URL*;
  the *email prose* is synthetic stimulus text written around the URL.
- **Licence / provenance:** "benchmark project code and synthetic content are released under an
  MIT License"; third-party URL indicators redistributed for academic research with attribution;
  see `SOURCE_LICENSES.md` in the repo (Nazario CC-BY-4.0; PhishTank via Cisco terms; Tranco via
  Le Pochat et al. 2019). Note: README datasource table says OpenPhish 66, but the file has
  OpenPhish 62 + OpenPhish-live 4 (= 66) — table is stale by split, totals agree.
- **Why it is the pick:** the only candidate that is simultaneously (a) released 2026, (b) both
  classes, (c) labels verified by an external feed, (d) zero exact overlap with training,
  (e) downloadable and small (1.6 MB).
- **Caveats that must travel with any result:**
  1. Email bodies are **synthetic stimuli for LLM-agent evaluation, not real inbox mail**.
     Phishing bodies are often benign-sounding ("I just finished the demo… check it out live
     here: <malicious URL>") — the discriminative signal is largely the URL, not classic
     phishing prose. A text-only RoBERTa/BiLSTM may score very differently here than on real
     phishing prose, and that difference is a property of the benchmark, not necessarily of
     the models. Report it as "modern URL-grounded synthetic-stimuli OOD", never as
     "real-world 2026 phishing".
  2. Bodies are short (median ~400 chars vs ~880 training); consider reporting a
     length-stratified breakdown.
  3. No per-email dates: temporal claims rest on the April 2026 release, not on dated emails.
  4. Contains real malicious URL indicators — treat as offline text only; do not visit/crawl
     (README "Responsible Use" says the same).

## 2. PhishFuzzer seeds — NEEDS WORK (small secondary check only)

- **Source:** https://github.com/DataPhish/PhishFuzzer ; paper "The Phish, The Spam, and The
  Valid" (Toth, Gruschka & Bisztray, CYBERSEC 2026, accepted/forthcoming).
- **Dates (verified):** repo created 2026-03-12, pushed 2026-11-07; per-row `Year` field in the
  seed file. Year distribution of all 3,300 seeds: 2023: 868, 2025: 307, 2026: 65, 2024: 22,
  plus large old blocks (2008: 839 = CEAS-08, 2002: 183, 2015–2022: ~1,016). Only **1,262
  seeds are dated 2023–2026**, and of those **1,058 are Spam, 103 Phishing, 101 Valid** —
  the 2023 block is 866 Spam + 0 Phishing. The balanced modern slice is the
  **"Manual" (privately collected) 300**: Year 2024: 22, 2025: 210, 2026: 65, 2023: 2,
  2022: 1; Type Phishing 103 / Valid 102 / Spam 95.
- **Downloaded:** `datasets/phishfuzzer_seeds.json` (8,218,997 bytes; 3,300 rows). The 19,800
  rephrased variants file (48 MB) was NOT downloaded.
- **Inspected:** fields `No., Subject, Body, Sender, URL, File, Type (Phishing/Spam/Valid),
  Created by (Human 3,176 / SpamAssassin 122), Source, Year, Motivation, Language`
  (en 3,194; 106 non-English). Full-seed body-only overlap vs training: **85 / 3,300 (2.6%)**;
  Manual-300 body-only overlap: **1 / 300**; modern-subset (Year ≥ 2023) overlap 0 by
  subject+body hash, 71 duplicate bodies within the 1,262. Median body length (modern subset):
  1,313 chars.
- **Licence:** CC BY-NC 4.0 (repo LICENSE.md); "Emails taken from third-party sources (Kaggle
  Phishing Dataset, SpamAssassin) remain under their original terms."
- **Verdict:** usable ONLY as a small secondary check on the Manual-300 (n = 300, 103 phish /
  102 valid if Spam is excluded, or 3-class with Spam kept). Before using it, verify from the
  `DataSet_Creation` docs / paper how the "Manual" emails were collected and labelled — that
  provenance was not verified in this hunt. Do NOT use the full 3,300 seeds as OOD: most are
  pre-2023 and 85 overlap training exactly. Do NOT mix the 19,800 LLM-rephrased variants into
  real-world results; they belong in the synthetic track if used at all.

## 3. Frontiers 2026 multi-LLM corpus — ACCEPTED, synthetic track only (phishing-only)

- **Source:** Zenodo https://doi.org/10.5281/zenodo.20250116 (data availability statement in
  Gutierrez, Villegas-Ch & Govea, "Cross-model evaluation of phishing detectors against
  LLM-generated emails", Front. Big Data 9:1883452, received 2026-05-17, published 2026-07-07).
- **Dates (verified):** Zenodo record published 2026-05-16, licence CC-BY-4.0, files
  `cross-model-phishing.zip` (7,283,980 bytes) + `results.zip`.
- **Downloaded + extracted:** `datasets/zenodo_frontiers/` — `data/human_corpus_sampled.csv`
  (5,000 rows), `data/llm_corpus_sampled.csv` (4,986 rows), `data/corpus_features.csv`, code, README.
- **Inspected:** LLM side: GPT-4.1 1,665 / DeepSeek 3.2 1,665 / Llama 3.3 70B 1,656, categories
  banking / parcel_delivery / tax_irs / hr / it_support (~995–999 each), median text 1,304 chars,
  **label = 1 for all 4,986**. Human side: TREC-07 1,500 / CEAS-08 1,500 / Nazario 1,000 /
  Nigerian_Fraud 750 / enron_data_fraud_labeled 250, median text 726 chars, **label = 1 for all
  5,000**. **There is no legitimate class at all** — the paper's task is human-vs-LLM *phishing
  authorship*, not phishing-vs-legitimate detection.
- **Overlap vs training:** human body-only 0 / 5,000; LLM body-only 0 / 4,986 (human full-text
  hash overlap 1 / 5,000 — one Nigerian_Fraud row).
- **Verdict:** accepted ONLY as a synthetic-track phishing-side set. To use it for RoBERTa vs
  BiLSTM OOD, pair the 4,986 LLM phishing emails with a legitimate set (e.g. PhishNChips legit
  1,000, or PhishFuzzer Valid) and report it as a **paired synthetic-track** result, never
  mixed with real-world results. The human 5,000 must not be presented as "modern": they are
  2007–2008 corpora.

## 4. Adversarial BEC Email Dataset — ACCEPTED, synthetic track only (phishing-only)

- **Source:** https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset
- **Dates (verified):** Kaggle updated 2025-11-21T12:27:17Z; README: "4,211 synthetic Business
  Email Compromise (BEC) phishing emails I generated using Gemini Flash 2.5"; generation
  scripts included (`generate_emails.py` uses `genai.GenerativeModel('gemini-2.5-flash')`).
- **Downloaded:** `datasets/yoadjei_bec/` — `synthetic_emails.csv` (1,746,243 bytes) +
  `synthetic_emails_poisoned.csv` (1,811,843 bytes) + README + both scripts.
- **Inspected:** columns `subject, body, label`. Clean file: label 1.0 for 4,181 rows, **30 rows
  NaN label**; 4,129 unique bodies; median body 305 chars. **No legitimate emails.** Poisoned
  file: same label distribution; **3,024 / 4,211 bodies byte-identical to clean** — only 1,187
  rows actually changed by the homoglyph/invisible-Unicode obfuscation the dataset advertises
  (30% poisoned claim vs 28.2% rows changed — close, but the identical 71.8% means clean-vs-
  poisoned comparisons on most rows are vacuous).
- **Licence:** CC-BY-4.0 (README, cite Adjei, Y.O. 2025).
- **Verdict:** accepted as a **synthetic-track, phishing-only BEC stress set** (pair with a legit
  set; drop or fix the 30 NaN-label rows; use only the 1,187 actually-changed rows for the
  clean-vs-poisoned contrast). Never a standalone OOD test set.

## 5. SPAM + Genuine Mail Contents (isuranga, 100k) — REJECTED

- **Source:** https://www.kaggle.com/datasets/isuranga/spam-genuine-mail-contents-dataset
- **Dates:** Kaggle updated 2025-09-27; in-data `date` 2024-03-31T11:04:46 → 2025-09-23T10:48:47,
  ~4,843–5,211 rows per month — suspiciously uniform for real mail.
- **Downloaded:** `~/workspace/kaggle_downloads/isuranga_100k/email_dataset_100k.csv` (63,717,161 bytes).
- **Inspected:** 100,000 rows × 30 cols (`raw_text, subject, body_plain, body_html, from/to/cc,
  date, …, spf/dkim/dmarc, label`); label 50,000 ham (0) / 50,000 spam (1); median body 115
  chars; **19,482 duplicate bodies**; top body "Hello,\n\nAre you free for lunch tomorrow near
  the office?\n\n-Jack" appears 184× (Eve 175×, Dan 173×); subject counts are near-identical
  across ~5 templates (3,544–3,587 each); sender domains are Microsoft sample domains
  (contoso.com, fabrikam.io, adatum.net, northwind.com, cohosu.org). Page description, quoted:
  "This dataset contains 100,000 unique email records **designed to simulate real-world** ham
  (legitimate) and spam (malicious/phishing) messages." No real source, no labelling method
  beyond simulation; licence not stated on the page. Overlap vs training first-200 hashes: 0.
- **Why rejected:** template-simulated data with heavy repetition and fabricated metadata is not
  a valid OOD proxy for real phishing; a model scoring well here proves it can spot the
  generator's templates, nothing more. At most a smoke-test, and even that would be misleading
  to report alongside real results.

## 6. Email Phishing Dataset (ethancratchley) — REJECTED

- **Source:** https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset
- **Dates:** Kaggle updated 2025-04-01; file `email_phishing_data.csv` 11,812,045 bytes.
- **Downloaded:** `datasets/ethancratchley/email_phishing_data.csv`.
- **Why rejected (two independent reasons):** (1) page states, quoted: "This dataset **does not
  contain raw text or headers, only engineered features**" — 9 numeric columns (`num_words,
  num_unique_words, num_stopwords, num_links, num_unique_domains, num_email_addresses,
  num_spelling_errors, num_urgent_keywords, label`); a text classifier cannot run on it.
  (2) Its sources are "~500,000 non-phishing emails from the Enron Email Dataset" and "~20,000
  phishing and safe emails from the Phishing Email Dataset" — i.e. the training set's own
  sources. Licence Apache 2.0.

## 7. Human-LLM Generated Phishing–Legitimate (francescogreco97) — REJECTED

- **Source:** https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails
- **Dates:** Kaggle updated 2024-04-10. In-data `date` ranges (measured): human legit
  2007-07-05 → 2019-10-29; human phishing 2019-08-19 → 2022-12-27. Both pre-2023.
- **Downloaded:** `datasets/human_llm_kaggle/` (4 files).
- **Inspected:** human legit 1,000 (label 0), human phishing 1,000 (label 1, overlap vs training
  2 / 1,000); `llm-generated/legit.csv` has **label = 1 for all 1,000 rows** (labels unreliable /
  mislabelled — a "legit" file labelled entirely phishing); `llm-generated/phishing.csv` is
  **malformed CSV** (only 595 / 1,000 rows parse with `on_bad_lines='skip'`, 391 NaN labels).
- **Why rejected:** human side fails the 2023+ date requirement; LLM side fails basic label/file
  integrity. (Cited by the MDPI Algorithms 2025 paper as its LLM-email source; that does not
  repair the files.)

## 8. Opara et al. 2025 GPT-4o set (63 + 63) — NOT LOCATED

- **Paper:** Opara, Modesti & Golightly, "Evaluating spam filters and Stylometric Detection of
  AI-generated phishing emails", Expert Systems with Applications 276 (2025) 127044,
  doi 10.1016/j.eswa.2025.127044; received 2024-10-04, online 2025-03-10, CC BY-NC. Paper
  states (quoted): "The dataset of AI-generated phishing emails is publicly available on
  Kaggle" and "Using GPT-4o, we generated a total of 63 phishing emails and 63 legitimate
  emails between August and September 2024."
- **Searches run:** Kaggle CLI `datasets list --search` for "Opara phishing", "styloai",
  "ai-generated phishing emails", "AI generated phishing" — no match; web searches for the
  Kaggle link — no slug found in any result. No dataset URL appears in the paper snippets
  retrieved.
- **Verdict:** not located; and at n = 126 it would be a tiny supplementary set at best, not a
  primary OOD set. Next step: email the authors (c.opara@tees.ac.uk, from the paper) or check
  the paper's data-availability section in the full text.

## 9. Zenodo Phishing Email Curated Datasets — REJECTED

- **Source:** https://doi.org/10.5281/zenodo.8339691 (Islam 2024), updated 2024-09-25.
- **Why rejected:** the record's own description (quoted from Google Dataset Search):
  "We have curated **11 datasets spanning from 1995 to 2022**." Pre-2023 by construction, and
  the same legacy corpora as the training set. Upload date (2024) is not collection date.

## 10. Same-source legacy corpora — REJECTED (leakage)

- **naserabdullahalam/phishing-email-dataset** (Kaggle, updated 2024-05-24, 80.9 MB, 50,084
  downloads): "combines … Enron and Ling … CEAS, Nazario, Nigerian Fraud, and SpamAssassin …
  Approximately 82,500 emails — 42,891 spam / 39,595 legitimate". Same sources as training.
- **zefang-liu/phishing-email-dataset** (HF, created 2024-01-17): literally the training file
  (`Phishing_Email.csv`, 160 MB storage); card says direct copy of the Kaggle set.
- **subhajournal/phishingemails** (Kaggle, 2023-07-06): the training file's source (52 MB,
  `Phishing_Email.csv`).
- **puyang2025/seven-phishing-email-datasets** (HF, modified 2026-01-15, 203k downloads):
  "unified, row-level email corpus built from seven commonly used public email datasets"
  (Enron, SpamAssassin, Nazario, CEAS, etc., per card); **afnanmohamed99** mirror of the same.
  `locuoco/the-biggest-spam-ham-phish-email-dataset-300000` / `akshatsharma2/...-300000`
  (Kaggle, updated 2026-02-03, 190 MB / 630 MB `df.csv`) are the same kind of legacy
  compilation; not downloaded (190–630 MB, same sources, no modern collection claim found).
- **Why rejected:** using any of these as "OOD" would measure train-test overlap, not
  generalization. Their 2023–2026 upload/modified dates are republication dates, not
  collection dates.

## 11. PhreshPhish — REJECTED (not email)

- **Source:** https://huggingface.co/datasets/phreshphish/phreshphish ; paper arXiv:2507.10854
  (Dalton et al. 2025).
- **Dates:** v1.0.0 2025-05-14; v1.0.1 2026-02-07 ("Added ~200k new samples collected between
  March and December 2025"). Train 498,255 + test 168,060 rows, CC BY 4.0.
- **Why rejected:** phishing **webpage** (HTML + URL) detection, ~145 GB train / 48 GB test.
  No email bodies. Genuinely modern and well-labelled — for URL/webpage models, not this task.

## 12. PhishTank / OpenPhish / APWG — REJECTED as email sources

- **Evidence:** PhishTank bulk export fields are `phish_id, url, phish_detail_url,
  submission_time, verified, verification_time, online, target` — URLs with verification
  metadata, no message text. OpenPhish community feed (`https://openphish.com/feed.txt`) is
  "Plain text (URLs, one per line)"; "Community feed = URLs only. Premium adds 15+ attributes
  (brand, industry, language, country)" — still no email bodies. APWG eCrime Exchange is a
  members-only URL/indicator repository, not a public email corpus.
- **Why recorded:** these feeds are the label-verification layer *behind* PhishNChips (§1),
  which is the correct way to use them for this task. Pulling them directly cannot supply an
  email-body test set.

## 13. RSB-1000 — REJECTED for the email task

- **Source:** https://huggingface.co/datasets/RevealWire/rsb-1000 (created 2026-10-22,
  CC-BY-4.0, "Frozen October 2026", SHA256-pinned).
- **Downloaded + inspected:** `datasets/rsb-1000.jsonl` (1,000 rows): 780 scam / 220 legitimate;
  categories legitimate 200, phishing 200, fake_job 150, invoice 150, impersonation 100,
  investment 100, romance 50, ambiguous 50; **median text length 205 chars**; samples are
  SMS/bank-alert style ("YourBank: Rs 4,250 debited from a/c XX4821…", UPI messages), not
  email bodies (no headers, no subject/body structure).
- **Why rejected:** wrong modality (short SMS-style messages, not emails); fully synthetic.
  Could serve only as an out-of-modality robustness probe, not an email OOD set.

## 14. Not inspected (time-boxed; listed honestly)

- **mutahirshaukat/zero-day-phishing-emails-corpus** (Kaggle, updated 2026-05-11,
  `Zero-Day_Phishing_Emails_Corpus.csv`, 31.4 MB) — name suggests modern phishing emails, but
  provenance, labels and date coverage were not checked. First next step if a second
  real-world candidate is needed.
- **tommyf1/phishing-email-dataset** (Kaggle, updated 2026-04-15, 25 KB) and
  **prince7489/phishing-email-detection-2026-dataset** (Kaggle, updated 2026-05-13, 23 KB) —
  tiny (tens of KB), no provenance checked; size alone rules them out as primary sets.
- **Dizzzy0x00/LLMGen-Phishing-Email-Dataset** (HF, created 2025-12-13): LLM-generated phishing
  + legitimate, DeepSeek (Chinese) + OpenAI (English); not downloaded/inspected — synthetic
  track only, and mixed-language.
- **luongnv89/phishing-email** (HF, 832 rows), **nosadaniel/phishing-email-training-dataset**
  (HF), **cybersectony/PhishingEmailDetectionv2.0** (HF) — small, no dates/provenance checked.
- **DataPhish "Constructing and Benchmarking: a Labeled Email Dataset…"** (arXiv:2511.21448):
  "integrates recent real-world emails, curated public datasets, and controlled synthetic
  samples… distinguishing human- and LLM-generated content", resources at
  github.com/DataPhish/PhishingSp… — the GitHub org `DataPhish` lists only the PhishFuzzer
  repo via API (2026-11-19); the companion repo was not located in this hunt. Worth one more
  lookup before the experiment is finalized.

---

## Ranked recommendation

### Primary OOD test set: PhishNChips v5.2 core (2,000 emails)

**Use exactly:** `datasets/phishnchips_core_emails.csv`, `email_content` JSON → `subject + body`
as the model input text, `phish_label` as the label. Keep the `datasource`, `strategy` and
`url_category` columns for stratified reporting (e.g. PhishTank vs GitHub-pages vs short-URL
slices; the 333 cross-domain legit rows as a false-positive slice).

**Why:** only candidate meeting all hard gates — modern (Apr 2026 release), both classes
balanced 1,000/1,000, externally verified labels, zero exact overlap with training, small and
clean (0 duplicates), permissive licence (MIT for the synthetic content).

**How to report it (mandatory wording):** "modern URL-grounded synthetic-stimuli OOD set
(PhishNChips v5.2, Apr 2026): phishing labels verified via PhishTank/OpenPhish/GitHub phishing
DB URLs; email prose is synthetic stimulus text, not real inbox mail; median body ~400 chars."
Do not call it real-world phishing. Report precision/recall/F1/ROC-AUC for RoBERTa vs BiLSTM
on it, plus the datasource-stratified breakdown, and note the short-body distribution shift
vs training (median 880 chars).

### Secondary check (small, real-human): PhishFuzzer "Manual" 300

Filter seeds to `Source == 'Manual'` (n = 300; 103 Phishing / 102 Valid / 95 Spam; years
2024–2026). Report as a secondary, low-n real-human check — and only after verifying the
Manual collection/labelling method from the `DataSet_Creation` docs or paper. Exclude the
1,058 Spam rows dated 2023 and all pre-2023 seeds from any "modern" claim; drop or
separately report the 85 training-overlapping seed rows (body-only hash) if the full seeds
are ever used.

### Separate synthetic track (never mixed with the above)

- Frontiers LLM phishing 4,986 (GPT-4.1 / DeepSeek 3.2 / Llama 3.3 70B) paired with PhishNChips
  legit 1,000 (or PhishFuzzer Valid) — tests LLM-generated phishing detection.
- Adversarial BEC clean 4,181 + poisoned subset (1,187 actually-changed rows) paired with a
  legit set — tests BEC + homoglyph/invisible-Unicode robustness.
- Keep synthetic-track numbers in their own table/section, labelled synthetic.

## Exact next steps to get a genuinely real-world modern set

1. **Check the uninspected lead first:** download + inspect
   `mutahirshaukat/zero-day-phishing-emails-corpus` (Kaggle, 2026-05-11, 31 MB) with the same
   checklist (dates inside data, label method, overlap vs training).
2. **Locate the DataPhish companion dataset** ("Constructing and Benchmarking…", arXiv:2511.21448)
   — it claims "recent real-world emails" with human-vs-LLM labels; find its repo (the
   arXiv HTML links github.com/DataPhish/PhishingSp…).
3. **Email Opara et al.** (c.opara@tees.ac.uk) for the Kaggle link to the 63+63 GPT-4o set
   (small, but real peer-reviewed AI-phishing ground truth).
4. **If none of 1–3 yields a real-world set:** build one by sampling *current* PhishTank /
   OpenPhish verified URLs and pairing them with *recent* legitimate mail the user controls
   (e.g. the user's own inbox exports, Enron is NOT recent) — or accept PhishNChips as the
   primary set and state the synthetic-stimuli limitation in the report's limitations section.
   Do not present any template-simulated set (isuranga) as real-world evidence.
5. **Before running the experiment:** re-run the overlap check with a near-duplicate method
   (e.g. MinHash / TF-IDF cosine > 0.9) on the chosen set, since the checks above are
   exact-hash only; and confirm the PhishNChips licence file (`SOURCE_LICENSES.md`) permits
   the intended use (benchmarking a classifier is within its stated academic-research scope).

## Files downloaded in this hunt

- `experiments/modern_phishing_ood/datasets/phishnchips_core_emails.csv` (1.6 MB) + README
- `experiments/modern_phishing_ood/datasets/phishfuzzer_seeds.json` (8.2 MB)
- `experiments/modern_phishing_ood/datasets/zenodo_frontiers/cross-model-phishing.zip` (7.3 MB, extracted)
- `experiments/modern_phishing_ood/datasets/yoadjei_bec/` (clean + poisoned + scripts + README)
- `experiments/modern_phishing_ood/datasets/ethancratchley/email_phishing_data.csv` (11.8 MB)
- `experiments/modern_phishing_ood/datasets/human_llm_kaggle/` (4 files)
- `experiments/modern_phishing_ood/datasets/rsb-1000.jsonl`
- `~/workspace/kaggle_downloads/isuranga_100k/email_dataset_100k.csv` (63.7 MB)

## Methods note

Searches: `browser.search` (HF/Kaggle/web, 2026-11-19/20), Hugging Face API
(`huggingface.co/api/datasets?search=…` and per-dataset API), Kaggle CLI
(`kaggle datasets list --search …`, `datasets files`, `datasets download`), GitHub API
(DataPhish org/repos), Zenodo API (record 20250116), Kaggle dataset pages via `browser.open`
(quoted where cited). No file larger than 500 MB was downloaded; the largest was 63.7 MB.
No dates, sources or labels in this report are invented: every date is quoted from the
source itself (dataset card, Kaggle page, paper, or a field inside the downloaded data), and
every size/balance/overlap figure was measured from the downloaded files with the scripts
described in §0. Gaps are stated as gaps (§8, §14).
