# Efficiency report — modern-phishing OOD experiment

Latency: median ms/email at batch 1 via `baselines.measure_latency`
(warmup 5), params from the loaded model / artefact, checkpoint size
from the file on disk. Sources: `ood_metrics_combined_kernel{1,2,3}.json`
(`latency_ms_per_email`, `model_info`). The rerun-round latencies are
quoted from `results/rerun_latency/latency_explain_time.json`.

## Latency (median ms/email, batch 1)

Three independent kernel runs measured the same protocol; T4 instances
vary, so ranges are reported, not a single number.

| Model | GPU median (K1 / K2 / K3) | CPU median (K1 / K2 / K3) | Rerun round (GPU / CPU) |
|---|---|---|---|
| RoBERTa | 9.54 / 16.84 / 10.01 ms | 129.73 / 254.19 / 133.11 ms | 15.50 / 242.50 ms |
| BiLSTM | 6.23 / 8.33 / 6.42 ms | 11.09 / 18.92 / 12.09 ms | 8.37 / 19.90 ms |

* n timed: GPU 200, CPU 100 per run (RoBERTa CPU n=100 by design).
  Latency texts: K1/K3 used PhishNChips texts; K2 used Frontiers LLM
  texts (longer: median 1,304 chars vs ~400) — that is the likely
  reason K2's numbers are the highest of the three runs for both
  models. Do not compare K2's latency against K1's as if the texts
  were the same.
* Ratios (same run): RoBERTa/BiLSTM GPU 1.53× (K1), 2.02× (K2),
  1.56× (K3); CPU 11.7× (K1), 13.4× (K2), 11.0× (K3). The CPU ratio
  is the deployment-relevant one if inference runs without a GPU:
  RoBERTa costs an order of magnitude more per email on CPU.
* The BiLSTM GPU device field is recorded as `"gpu?"` in the JSONs
  (`BaselineModel` exposes no `.device` attribute for the label).
  The scorer timed as "GPU" was CUDA-loaded (the main scorer loads on
  `get_device()` = cuda on a GPU machine, and a separate CUDA copy
  was loaded without error); the numbers are consistent with the
  rerun round's BiLSTM GPU figure (8.37 ms), so they are treated as
  genuine GPU timings with that label caveat.

## Size and parameters

| Model | Parameters | Checkpoint file | Params ratio (RoBERTa/BiLSTM) | Size ratio |
|---|---|---|---|---|
| RoBERTa | 124,647,170 | 498.7 MB | 20.0× | 19.5× |
| BiLSTM | 6,236,034 | 25.6 MB | — | — |

RoBERTa is 20× the parameters and ~19.5× the disk size of BiLSTM,
for 1.5–2.0× the GPU latency and 11–13× the CPU latency.

## What the efficiency numbers do and do not decide

* On this OOD evidence, RoBERTa's extra cost buys better ranking
  everywhere except PhishFuzzer (ROC-AUC gaps: +0.099 PhishNChips,
  +0.045 Frontiers LLM, +0.040 Frontiers human, +0.119 BEC) and far
  better recall on the synthetic track — but on the primary set it
  buys *worse* recall/F1 at 0.5 than BiLSTM. Cost alone does not
  settle the deployment question; the verdict in
  `final_comparative_report.md` weighs accuracy against cost per set.
* No explanation-time measurement was run in this experiment (LITA
  time is in the rerun round: RoBERTa LITA median 30.87 s/email vs
  BiLSTM LITA-IG 1.87 s/email). If explanations are required at
  serving time, that 16× gap matters more than the inference gap.
* No throughput (batch 32/64) measurement was run; batch-1 latency
  is the deployment-relevant number for a per-email API and the
  comparable number to the rerun round.
