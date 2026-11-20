# Task 2e — Latency verification note

## Verification against source files

- `results/rerun_latency/latency_explain_time.json` exists. Protocol recorded in the file: median of 200 single-email `predict_proba` calls after warmup, batch size 1, Tesla T4, CPU and GPU measured separately on the same model. Values: RoBERTa GPU 15.4957 ms, RoBERTa CPU 242.4985 ms, BiLSTM GPU 8.3735 ms, BiLSTM CPU 19.8976 ms — all match the task-brief numbers (15.50 / 242.50 / 8.37 / 19.90) to the stated precision.
- TF-IDF+SVC / TF-IDF+LR medians come from `results/robustness_rerun/baseline_results.json` (seed-42 refit, `latency_cpu_ms`, n_timed=200): SVC CPU 8.4692 ms, LR CPU 1.0893 ms — both match the task-brief numbers (8.469 / 1.089). Note the provenance differs from the RoBERTa/BiLSTM file (different run, seed-42 refit rather than the rerun-latency run), though the protocol is the same shape (median, n=200, batch 1).
- DistilBERT medians come from `results/distilbert/latency/latency_explain_time.json` (latency kernel, seed-42 checkpoint, same protocol: median of 200 single-email `predict_proba` calls after warmup, batch 1, Tesla T4, CPU and GPU measured separately with the device switched properly): CPU 118.6152 ms, GPU 9.0738 ms. LITA explanation time on the same 10 Stage C emails (lime_samples=1000): median 15.6930 s/email.

## Latency table (median ms per email, batch 1)

| Model | CPU | GPU |
|---|---:|---:|
| RoBERTa | 242.50 | 15.50 |
| BiLSTM | 19.90 | 8.37 |
| TF-IDF+SVC | 8.469 | NOT MEASURED (CPU-only) |
| TF-IDF+LR | 1.089 | NOT MEASURED (CPU-only) |
| DistilBERT | 118.62 | 9.07 |

CPU multiples vs TF-IDF+SVC (medians): RoBERTa CPU is 28.63x SVC CPU (242.4985 / 8.4692); BiLSTM CPU is 2.35x SVC CPU (19.8976 / 8.4692); TF-IDF+LR CPU is 0.13x SVC CPU (1.0893 / 8.4692); **DistilBERT CPU is 14.00x SVC CPU (118.6152 / 8.4692)**. (These match the multiples already recorded in `results/system_complexity_report.json`: 28.633 and 2.349.)

**GPU multiples vs TF-IDF+SVC: NOT COMPUTABLE.** TF-IDF+SVC and TF-IDF+LR are CPU-only (sklearn has no GPU implementation); their GPU latency was never measured (`latency_gpu_ms` is None in both source files). No GPU multiple is fabricated here. For reference only, RoBERTa GPU (15.50 ms) is 1.85x BiLSTM GPU (8.37 ms) — a GPU-to-GPU comparison between two models that both have GPU measurements. DistilBERT GPU (9.07 ms) sits between them: 1.08x BiLSTM GPU, 0.59x RoBERTa GPU.

**Correction to an earlier figure:** the final-training driver reported a DistilBERT "CPU median 9.55 ms". That figure is wrong — the driver timed the "CPU" case without moving the model off CUDA, so it measured GPU time again. The true CPU latency is 118.62 ms from the latency kernel above, which switches device properly. Do not use 9.55 ms as a CPU figure. (The driver's GPU median 9.57 ms is a genuine GPU measurement and agrees with the latency kernel's 9.07 ms to within 5%.)

Machine-readable version: `../results/latency_verification.json`.
