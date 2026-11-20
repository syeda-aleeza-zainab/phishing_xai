# Section F — Kaggle notebook + packaging: Change Log

Author: Section F subagent. Base commit: `5d86e95`. Work done 2026-11-19.
Do NOT edit `decision.md` or `docs/PROJECT_LOG.md` directly — the coordinator merges this log.
Nothing was pushed. Nothing was retrained. `src/phishing_xai/model.py` and
`src/phishing_xai/faithfulness.py` were NOT modified (verified: `git diff --stat`
on both is empty). No metrics are claimed anywhere below — anything not executed
on Kaggle is **NOT RUN YET**. No `.venv` content is included in any output
(final `git status --short` shows no `.venv` paths).

## User decisions recorded

- **Precision is the primary metric** (user, 2026-11-19). Phishing-class precision,
  minimise FPR. Applied in the notebook as: a precision-first note in the intro
  markdown cell, threshold tuning on validation only (multimodal-predict cell
  deliberately omits `--threshold`, leaving the 0.5 placeholder default, with a
  comment that any tuned value must come from validation), and the H8 cells use
  the precision-first verdict from the complexity report.

## Files created / changed by this section

| File | Change |
|---|---|
| `notebooks/kaggle_run.ipynb` (new) | 23 cells, nbformat 4, Kaggle GPU runner (F1) |
| `requirements-kaggle.txt` (new) | Unpinned minimums, no torch line (F2) |
| `requirements.txt` (rewritten) | Minimal direct deps only, mirrors pyproject (F3) |
| `pyproject.toml` (edited) | Added `scipy`, `transformers-interpret`, `Pillow` to main deps + new `[multimodal]` extra (F3) |
| `.gitignore` (edited) | Added `data/*.csv` + explicit `results/best_model.pt` (F4) |
| `.impl_logs/section_F.md` (this file) | Change log |

`notebooks/colab_run.ipynb` was read as the template and NOT deleted or modified.
All other `M` / `??` entries in `git status` belong to other sections/agents
(A–E, H and the coordinator's working-tree state), not to Section F.

## Decisions and reasons (per change)

### F1 — `notebooks/kaggle_run.ipynb` (23 cells)
- **Template:** mirrors `notebooks/colab_run.ipynb` cell order (GPU -> install ->
  data -> splits -> tune -> train -> explain -> faithfulness -> ablation ->
  report) with Colab specifics replaced by Kaggle ones (no Drive mount, no
  `/content` paths). Every CLI flag was verified against `--help` output or the
  argparse code of the actual script; no flag was guessed. Verified flags:
  `download_data.py --output_path`; `make_splits.py --data_path/--output_dir/--seed`;
  `run_tune.py --data_path/--output_dir/--n_iterations/--max_length/--subsample_train/--seed`;
  `run_train.py --data_path/--output_dir/--params/--ckpt_dir/--resume/--max_length/--seed`;
  `run_explain.py --checkpoint_path/--output_dir/--num_samples/--seed`;
  `run_faithfulness.py --model_path/--output_dir/--sample_size/--max_length/--seed`;
  `run_ablation.py --model_path/--output_dir/--sample_size/--k/--max_length/--seed`;
  `run_multimodal_predict.py inputs --checkpoint/--output_dir/--fusion/--max_samples`
  (+ `--threshold` documented, intentionally not passed);
  `run_multimodal_faithfulness.py --samples_dir/--checkpoint/--output_json`;
  `run_baselines.py --models/--seeds/--epochs/--emb_dim/--hidden/--layers/--max_len/--with_cnn/--with_distilbert/--checkpoint/--analysis/--robustness/--faithfulness/--sample_size/--output_dir`;
  `run_complexity_report.py --results_dir/--output_dir`;
  `make_report.py --results_dir`;
  `make_multimodal_eval_set.py --out_dir/--seed` (verified `--help` 2026-11-19);
  `run_multimodal_eval.py --eval_dir/--checkpoint/--output_dir/--fusion/--models`
  (verified `--help` 2026-11-19).
- **Path setup (Cell 3):** the repo dataset is copied from `/kaggle/input` to
  writable `/kaggle/working` (input is read-only), then `%cd /kaggle/working`,
  so every later relative path resolves to `/kaggle/working/results/...` and
  training writes `/kaggle/working/results/best_model.pt` as specified. A
  `best_model.pt` found in any attached dataset is copied to
  `/kaggle/working/results/best_model.pt`. The repo lookup checks each dataset
  root and one level down (dataset may wrap the repo in a folder).
- **Install (Cell 2) before copy (Cell 3) as specified:** Cell 2 locates
  `requirements-kaggle.txt` under `/kaggle/input` by glob and installs it, plus
  `apt-get install -y tesseract-ocr libzbar0`, with a comment that Internet must
  be ON in Kaggle notebook settings. The editable install is done in Cell 3
  after the copy, as `pip install -q -e . --no-deps` — `--no-deps` is deliberate:
  plain `-e .` would reinstall torch/transformers from pyproject.toml, which
  Kaggle already provides.
- **Data (Cell 4):** uses the shipped `data/Phishing_Email.csv` if present, else
  a CSV attached as a dataset, else `download_data.py` (Hugging Face mirror,
  strict schema/label/row-count verification; it re-verifies an existing file).
  Then `make_splits.py` (train 11215 / val 2803 / test 3495 per the manifest).
- **Tuning (Cell 5) is optional and self-skipping** when
  `results/tuning/best_params.json` exists. **Training (Cell 6)** passes
  `--params` only when that file exists (two branches), always with `--resume`
  and `--ckpt_dir /kaggle/working/results/checkpoints`; the cell notes how to
  skip training when a checkpoint was attached.
- **Multimodal (Cell 8):** regenerates the 6 samples, runs
  `run_multimodal_predict.py --fusion weighted` (precision-first default; `max`
  is the recall-heavy alternative), then `run_multimodal_faithfulness.py`,
  then the Section E eval set + evaluator with explicit Kaggle paths.
- **H8 baseline cells (Cells 9–16), in the specified order:** cheap
  (`--models dummy tfidf_lr tfidf_svm --seeds 42 1 7`); BiLSTM
  (`--models bilstm --seeds 42 1 7 --epochs 8 --emb_dim 200 --hidden 128
  --layers 2 --max_len 256`); optional CNN/DistilBERT
  (`--models cnn distilbert --seeds 42 --with_cnn --with_distilbert`);
  `roberta_eval` (`--checkpoint /kaggle/working/results/best_model.pt`);
  `--analysis --robustness` (on `--models tfidf_lr bilstm --seeds 42` — the
  fitted first-seed models those flags analyse; the choice is stated in the
  cell comment); `--faithfulness --sample_size 100` (same two models);
  Section E multimodal eval (`--models tfidf_lr bilstm roberta`);
  `run_complexity_report.py --results_dir /kaggle/working/results`. All save
  under `/kaggle/working/results/baselines/` (explicit `--output_dir`; the
  scripts also auto-redirect there on Kaggle).
- **Finding recorded in the notebook (not hidden): each `run_baselines.py`
  invocation rewrites `results/baselines/baseline_results.json`** (verified in
  `scripts/run_baselines.py` `main()`: `all_results` is built fresh and saved
  with `save_json`, no merge). The complexity report therefore only sees the
  most recent baseline invocation, not the accumulated ladder. The notebook
  states this in the H8 intro markdown and adds an optional commented-out
  consolidated single-invocation cell (Cell 15) to run before the complexity
  report if one combined table is wanted. This is a real caveat for the
  coordinator, not a notebook bug.
- **Checkpoint-as-dataset note:** final markdown cell gives Option A (Save
  Version with output saved, then attach that output as an input dataset in the
  next notebook) and Option B (download `best_model.pt`, upload via Datasets ->
  New Dataset). Final code cell lists every file under
  `/kaggle/working/results/` with sizes.
- **Section E status changed during this task:** at the start of Section F,
  `scripts/run_multimodal_eval.py` and `scripts/make_multimodal_eval_set.py`
  did not exist; both landed (untracked, 06:57 and 06:59) while this section
  was being written. The notebook was updated to their verified CLIs (see flag
  list above) and the intro note corrected to "Section E has landed". The
  evaluator calls are still guarded by `os.path.exists` so an older dataset
  snapshot prints `PENDING` instead of failing. No filenames were invented or
  renamed.
- **Precision-first note** is in the intro markdown cell and the multimodal
  threshold note, as specified.

### F2 — `requirements-kaggle.txt` (new, 21 requirement lines)
- Unpinned minimums matching Kaggle's preinstalled stack. **No torch line**,
  with a comment explaining why (Kaggle ships a CUDA torch build; reinstalling
  it wastes time and can break the stack). `transformers>=4.36.0` minimum only,
  no upper bound: `transformers-interpret` needs transformers 4.x, but if the
  Kaggle image ships 5.x the code falls back to Captum automatically
  (`src/phishing_xai/explain.py` lazy import + warning), so no downgrade is
  forced here.
- Multimodal deps the code actually imports: `pytesseract`, `pdfplumber`,
  `PyMuPDF`, `pyzbar`, `opencv-python-headless`, `qrcode`, `python-docx`
  (marked optional inline: only for `.docx` attachments in `extract.py`).
  Plus the baseline/core deps the code imports: `lime`, `captum`,
  `transformers-interpret`, `scikit-learn`, `pandas`, `numpy`, `nltk`,
  `beautifulsoup4`, `matplotlib`, `seaborn`, `tqdm`, `scipy`, `Pillow`.
  `tldextract` is explicitly NOT included (optional in `url_risk.py`, offline
  fallback when absent). No transitive noise (no IPython/backcall/colorama etc.).

### F3 — `requirements.txt` cleaned + `pyproject.toml` aligned
- Old file was a frozen `pip freeze` dump (78 lines incl. `IPython`,
  `backcall`, `colorama`, `pickleshare`, `traitlets`, `annotated-doc`, etc.).
  Replaced with 23 direct-dependency lines (minimums): the 15 core deps +
  7 multimodal deps + `pytest` (dev). Header comment states the canonical
  source: **pyproject.toml is canonical**; `requirements.txt` mirrors
  `[project] dependencies` + `[multimodal]` + `[dev]`, so
  `pip install -r requirements.txt` == `pip install -e ".[multimodal,dev]"`.
- `pyproject.toml` changes: added the three deps the code actually imports but
  pyproject was missing — `transformers-interpret>=0.10.0` (lazy import in
  `explain.py`, used by `run_faithfulness.py`), `scipy>=1.11.0` (used by
  `src/phishing_xai/baselines.py`), `Pillow>=10.0.0` (used by `extract.py`,
  `multimodal.py`, `make_multimodal_samples.py`). Added new optional group
  `[multimodal]` with the 7 multimodal deps. Existing `[dev]` unchanged.
- Cross-check (scripted, exact): every pyproject dep name appears in
  `requirements.txt` with the identical version floor, and vice versa
  (excluding pytest, which is `[dev]`-only). Result: **no mismatches**.
- `pip install -r requirements.txt --dry-run` (venv pip): **exit 0** — resolves;
  would install `transformers-interpret-0.10.0` (not currently in the venv;
  expected, it is a lazy import with Captum fallback) plus IPython-related
  transitive packages pulled in by other deps' own requirements. Same dry-run
  for `requirements-kaggle.txt`: **exit 0**.
- Editable install not broken: `.venv/bin/python -c "import phishing_xai"` OK.

### F4 — `.gitignore` + `results/debug/*.json`
- Added `data/*.csv` and an explicit `results/best_model.pt` line. Everything
  else listed in the task was already present and was kept, not duplicated:
  `.venv` (line 29), `__pycache__/` (line 2), `.pytest_cache/` (line 52),
  `*.egg-info/` (line 23), `checkpoints/` (line 61); `*.pt` already covered
  `best_model.pt`, the explicit line is belt-and-braces as specified.
- **Caveat:** `data/Phishing_Email.csv` is already tracked in git
  (`git ls-files data/`), so the new `data/*.csv` rule does not untrack it;
  it only stops future CSVs from being added. Untracking the committed CSV was
  not done (out of scope for this section; coordinator decision).
- **`results/debug/` does not exist in the repo** (verified `ls` + a find for
  `*debug*` JSON files: none found). There were no absolute paths to replace
  and nothing to regenerate; noted here instead of inventing files.

## Status per item (F1–F4)

- **F1 Done (code) / NOT RUN YET (execution)** — notebook created, valid, all
  15 referenced script paths exist in the repo (verified programmatically;
  none missing). Kaggle execution itself has not happened and cannot be done
  from here.
- **F2 Done** — `requirements-kaggle.txt` created, 21 lines, parses, dry-run
  exit 0, no torch line.
- **F3 Done** — `requirements.txt` rewritten (78 frozen lines -> 23 direct
  lines), pyproject aligned with new `[multimodal]` extra, cross-check exact,
  both dry-runs exit 0, editable install still imports.
- **F4 Done** — `.gitignore` additions made (only the two missing entries);
  `results/debug/*.json`: verified absent, nothing to fix (see above).

## Validation results

- `python3 -c "import json; json.load(...)"` on `notebooks/kaggle_run.ipynb`: OK.
- `nbformat.read` + `nbformat.validate` (system python3, nbformat 5.9.1;
  the `.venv` does not have nbformat): **OK**, 23 cells (5 markdown, 18 code).
- Script-path check over notebook source: 15 `scripts/*.py` references,
  **0 missing** (at final check; Section E scripts landed mid-task, see above).
- No hard-coded local/Windows paths in the notebook (regex check for
  `C:\`, `/home/`, `/content/`, `MyDrive`, `google.colab`: none); only
  `/kaggle/input/...` and `/kaggle/working/...` paths.
- `packaging.requirements.Requirement` parse: `requirements.txt` 23 lines,
  `requirements-kaggle.txt` 21 lines, all valid; `torch` absent from the
  Kaggle file (checked programmatically).
- `pip install -r requirements.txt --dry-run`: exit 0.
  `pip install -r requirements-kaggle.txt --dry-run`: exit 0.
- `git status --short` (final): Section F files are
  `?? notebooks/kaggle_run.ipynb`, `?? requirements-kaggle.txt`,
  `M requirements.txt`, `M pyproject.toml`, `M .gitignore`; **no `.venv` paths**.

## Could only be verified by actually running on Kaggle

- That `apt-get install -y tesseract-ocr libzbar0` succeeds in the Kaggle
  notebook environment and that `pytesseract`/`pyzbar` then import.
- That Kaggle's preinstalled torch/transformers satisfy the minimums (in
  particular which transformers major version the image ships, 4.x vs 5.x —
  decides whether the TI explainer or the Captum fallback runs).
- That `pip install -r requirements-kaggle.txt` installs nothing conflicting
  on the real Kaggle image (dry-run here used the local venv, not Kaggle).
- That the repo-dataset copy in Cell 3 finds the repo (depends on how the user
  uploads the dataset: flat vs one folder deep — both are handled, deeper
  nesting is not), and that a checkpoint dataset is found by the glob.
- That training fits in Kaggle GPU memory/time at `--max_length 512` and that
  `--resume` picks up `results/checkpoints/latest_checkpoint.pt` after an
  interruption (code path mirrors the Colab notebook, unexecuted here).
- That `run_multimodal_eval.py`'s `--recall_floor` default and the
  `--models tfidf_lr bilstm roberta` combination behave as specified on the
  real eval set (only `--help` was verified here).
- The `baseline_results.json` overwrite behaviour (Cell 15 consolidated run)
  is verified from code reading, not from an executed run.

## Kaggle setup instructions (as written in the notebook, for the final report)

1. **Accelerator: GPU** — Settings -> Accelerator -> GPU (T4 or P100).
2. **Internet: ON** — Settings -> Internet -> ON (needed for apt-get, pip, and
   the Cell 4 dataset download).
3. **Attach the repo as a dataset** — Add Input -> upload a zip/folder of this
   repo (must contain `scripts/run_train.py` and `requirements-kaggle.txt`).
   It appears read-only under `/kaggle/input/<dataset-name>/`.
4. **(Optional) Attach the data + checkpoint** — if `data/Phishing_Email.csv`
   is inside the repo dataset Cell 4 uses it; otherwise attach the CSV as a
   second dataset or let Cell 4 download it from the Hugging Face mirror.
   Attach a dataset containing `best_model.pt` to skip training (Cell 3 copies
   it to `/kaggle/working/results/best_model.pt`).
5. **Run cells top to bottom, in order** (Cells 1–18; Cell 5 tuning and Cell 11
   CNN/DistilBERT and Cell 15 consolidated run are optional; Cell 6 training
   can be skipped only if a checkpoint was attached).
6. **Save output** — Save Version (output saved) so `/kaggle/working/results/`
   persists. To reuse the checkpoint: Option A — attach that saved run output
   as an input dataset in the next notebook; Option B — download
   `results/best_model.pt` and upload it via Datasets -> New Dataset.
7. **Download checklist** (Cell 18 lists every file): at minimum
   `best_model.pt`, `metrics.json`, `RESULTS.md`, `splits.json`, and the
   `baselines/`, `explanations/` and `multimodal/` folders.
