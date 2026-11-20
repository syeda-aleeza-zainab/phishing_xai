# Integration notes (coordinator, 2026-11-19)

## Full pytest run (after A+B+C landed, H+D still running)
- `.venv/bin/python -m pytest -q` -> **132 passed, 2 failed** (81s).
- FAILED tests/test_sanity.py::test_environment_sanity — asserts Python 3.10; local env is Python 3.12.3. PRE-EXISTING environment mismatch, not caused by any section change. (Kaggle env Python version to be confirmed in notebook; test asserts 3.10.)
- FAILED tests/test_explain_and_resumability.py::test_captum_vs_transformers_interpret_tiny_model — `transformers_interpret` package is NOT installed in the local venv (ModuleNotFoundError); ti_explainer is None. PRE-EXISTING environment gap, not caused by any section change. Not installed locally because it would force a transformers downgrade from 5.23 while other agents are using the venv; Kaggle requirements pin transformers 4.x where transformers-interpret is expected to work — verify on Kaggle.
- All other tests pass, including A (17), B (20), C (39) subsets.

## Venv status
- .venv complete: torch 2.15.0+cpu, transformers 5.23.0, pandas 3.1.1, sklearn 1.9.1, lime, captum 0.9.0, nltk, seaborn, bs4, pdfplumber, PyMuPDF, pytesseract, python-docx, qrcode, matplotlib, pyzbar. tesseract-ocr + libzbar0 apt-installed by Section B agent. Note: pip TMPDIR workaround needed (parent torch install failed: /tmp tmpfs 512MB, No space left on device) — Section A agent used TMPDIR=~/pip_tmp.
