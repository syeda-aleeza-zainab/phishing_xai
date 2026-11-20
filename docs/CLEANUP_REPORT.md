# Repository cleanup report — 2026-11-20

Task 5 of the corrections + OOD-baselines brief. All changes are
UNCOMMITTED (staged deletions + modified files), nothing pushed.
Branch: `feature/multimodal-baselines`, local HEAD `8a78ed9`.

## 1. Absolute paths removed

A grep for the absolute home-directory path over the whole tree (excluding `.venv/` and `.git/`) returned 4 files; all were fixed to relative paths and the grep now returns **0 occurrences**:

- `results/multimodal_eval/multimodal_eval.json` — `eval_dir`,
  `labels` and `checkpoint` fields now use `./data/...` and
  `./results/best_model.pt`.
- `results/multimodal_eval/multimodal_eval.md` — same fields, same fix.
- `.impl_logs/section_G.md` — paths made relative.
- `.impl_logs/full_suite_6b5.log` — paths made relative.

## 2. Third-party dataset files removed from Git tracking

`git rm --cached` (34 files). **No local file was deleted** — every
file below was verified still on disk with `ls` after the command.
`git diff --cached --stat`: 34 files changed, 1,200,249 deletions.

- `data/Phishing_Email.csv`
- `experiments/modern_phishing_ood/datasets/ethancratchley/email_phishing_data.csv`
- `experiments/modern_phishing_ood/datasets/human_llm_kaggle/human-generated/legit.csv`
- `experiments/modern_phishing_ood/datasets/human_llm_kaggle/human-generated/phishing.csv`
- `experiments/modern_phishing_ood/datasets/human_llm_kaggle/llm-generated/legit.csv`
- `experiments/modern_phishing_ood/datasets/human_llm_kaggle/llm-generated/phishing.csv`
- `experiments/modern_phishing_ood/datasets/phishfuzzer_seeds.json`
- `experiments/modern_phishing_ood/datasets/phishnchips_core_emails.csv`
- `experiments/modern_phishing_ood/datasets/rsb-1000.jsonl`
- `experiments/modern_phishing_ood/datasets/yoadjei_bec/synthetic_emails.csv`
- `experiments/modern_phishing_ood/datasets/yoadjei_bec/synthetic_emails_poisoned.csv`
- `experiments/modern_phishing_ood/datasets/zenodo_frontiers/code/` (11 files,
  including `config.py` — see §4)
- `experiments/modern_phishing_ood/datasets/zenodo_frontiers/data/` (3 CSVs)
- `experiments/modern_phishing_ood/datasets/zenodo_frontiers/results/` (9 files)

Deliberately KEPT tracked (project-authored, not third-party data):
`phishnchips_README.md`, `yoadjei_bec/README.md`,
`yoadjei_bec/generate_emails.py`, `yoadjei_bec/poison_emails.py`
(generation scripts), `zenodo_frontiers/LICENSE`,
`zenodo_frontiers/README.md`. Project-generated artifacts were NOT
untracked: the 24 OOD prediction CSVs, `data/multimodal_eval/*.eml` +
`labels.csv`, and all results JSONs.

## 3. `.gitignore` changes

`data/*.csv` was ALREADY present — no change needed there. Added
explicit ignore rules for the OOD dataset paths listed in §2
(including `cross-model-phishing.zip`, which was never tracked —
`*.zip` was already ignored). `git check-ignore` confirms the
untracked files are now ignored and do not show as untracked.

## 4. History-presence issue — separate report, no action taken

Third-party data, including the leaked-key
`experiments/modern_phishing_ood/datasets/zenodo_frontiers/code/config.py`
(line 21, a dataset-authors' Azure AI Services key, not the user's),
is already in published history: local commit `8a78ed9` was pushed to
the remote branch on 2026-11-20 in sanitised form for `config.py`
(the pushed copy is redacted), but the key is still present in the
local commit. **History was NOT rewritten** — that needs the user's
explicit approval. Untracking the files (this report) stops them
appearing in future commits; it does not remove them from history.

## 5. Additional secret flag — found during this round, NOT in the brief

`experiments/modern_phishing_ood/datasets/yoadjei_bec/generate_emails.py`
line 9 contains a hardcoded `API_KEY` assignment (value not reproduced
here). Left untouched — out of scope for this task. If the key is live,
it should be moved to an environment variable and rotated; if it is a
placeholder from the dataset authors, it should still not be committed
in executable form.

## 6. Verification

- Grep for the absolute home-directory path (excluding `.venv/`, `.git/`): 0 hits.
- `ls` on all 34 untracked files: all present on disk.
- `git check-ignore` on the untracked paths: ignored.
- `git status --short`: 89 lines at time of writing (34 staged
  deletions + modified docs/code + new untracked files listed in the
  final summary). Nothing committed, nothing pushed.
