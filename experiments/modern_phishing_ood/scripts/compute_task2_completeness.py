"""Task 2 (OOD completeness) — local recomputation from SAVED predictions only.

Reads the 30 saved per-email prediction CSVs in
experiments/modern_phishing_ood/predictions/ (24 original + 6 DistilBERT)
and computes:

  (2a) full 4-model x 6-set table at threshold 0.5 + ROC-AUC + PR-AUC +
       precision@recall 0.90 + FPR (repo compute_metrics_full);
  (2b) calibration: mean P(phishing) by true class per set/model, ECE
       (10 equal-width bins) per model per set; in-distribution ECE for
       TF-IDF+LR / TF-IDF+SVC only, from the saved seed-42-refit y_prob in
       results/robustness_rerun/baseline_results.json with y_true rebuilt
       from data/Phishing_Email.csv + results/splits.json test_indices
       (repo load_data + deduplicate_data);
  (2c) threshold transfer on PhishNChips: stratified 20% slice (seed 42),
       threshold maximising precision s.t. recall >= 0.85 on the slice,
       evaluated on the remaining 80%;
  (2d) paired stats on PhishNChips vs TF-IDF+LR: repo mcnemar_test
       (errors only, NOT a precision test) + paired bootstrap 95% CIs
       (1000 resamples, seed 42) for precision/recall/F1 differences
       (compute_ood_stats.paired_diff_ci) and for ROC-AUC / PR-AUC
       differences (same percentile scheme, extension);
  (2e) latency verification against results/rerun_latency/
       latency_explain_time.json and the SVC/LR medians in
       results/robustness_rerun/baseline_results.json.

No Kaggle, no retraining, no new model runs, no checkpoint loading.
DistilBERT: checkpoint published (Kaggle dataset
phishing-xai-distilbert-checkpoint v2) and the 6 saved OOD prediction
CSVs + the saved Kernel-2 seed-42 test probabilities
(results/distilbert/seed_42/test_prob.npy) exist locally, so the
DistilBERT cells are computed here like the other models.

Run with the repo venv python (torch present, so the repo stats helpers
in phishing_xai.baselines are used directly, not the mirrors):
  .venv/bin/python experiments/modern_phishing_ood/scripts/compute_task2_completeness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OOD_DIR = REPO_ROOT / "experiments" / "modern_phishing_ood"
PRED_DIR = OOD_DIR / "predictions"
RESULTS_DIR = OOD_DIR / "results"
REPORTS_DIR = OOD_DIR / "reports"
SCRIPTS_DIR = OOD_DIR / "scripts"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from phishing_xai.baselines import (  # noqa: E402
    bootstrap_f1_ci,
    bootstrap_precision_ci,
    compute_metrics_full,
    mcnemar_test,
)
import compute_ood_stats as cs  # noqa: E402  (paired_diff_ci, mirrors)

SETS = ["phishnchips", "frontiers_human", "frontiers_llm", "bec",
        "bec_poisoned", "phishfuzzer"]
MODELS = ["roberta", "bilstm", "tfidf_svm", "tfidf_lr", "distilbert"]
MODEL_LABEL = {"roberta": "RoBERTa", "bilstm": "BiLSTM",
               "tfidf_svm": "TF-IDF+SVC", "tfidf_lr": "TF-IDF+LR",
               "distilbert": "DistilBERT"}
SET_LABEL = {"phishnchips": "PhishNChips (primary)", "frontiers_human": "Frontiers human",
             "frontiers_llm": "Frontiers LLM", "bec": "BEC",
             "bec_poisoned": "BEC-poisoned", "phishfuzzer": "PhishFuzzer"}


def load_all():
    data = {}
    checks = []
    for s in SETS:
        for m in MODELS:
            path = PRED_DIR / f"{s}_{m}_predictions.csv"
            assert path.exists(), f"missing {path}"
            df = pd.read_csv(path)
            assert list(df.columns) == ["email_id", "true_label",
                                        "predicted_label_at_0_5",
                                        "prob_phishing", "source",
                                        "category", "model", "set"], \
                f"{path.name}: unexpected columns {list(df.columns)}"
            assert (df["model"] == m).all() and (df["set"] == s).all(), \
                f"{path.name}: model/set column mismatch"
            assert df["prob_phishing"].between(0, 1).all()
            pred_check = (df["prob_phishing"] >= 0.5).astype(int)
            assert (pred_check == df["predicted_label_at_0_5"]).all(), \
                f"{path.name}: predicted_label_at_0_5 != prob>=0.5"
            data[(s, m)] = df
            checks.append(f"{path.name}: n={len(df)} "
                          f"phish={int((df.true_label == 1).sum())} "
                          f"legit={int((df.true_label == 0).sum())} OK")
    # email_id alignment across models within each set
    for s in SETS:
        ids = [set(data[(s, m)]["email_id"]) for m in MODELS]
        assert all(x == ids[0] for x in ids), f"{s}: email_id sets differ across models"
    return data, checks


def ece_10(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    ece = 0.0
    bins = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = (p >= lo) & (p < hi if b < 9 else p <= hi)
        nb = int(mask.sum())
        if nb == 0:
            bins.append({"bin": f"[{lo:.1f},{hi:.1f})" if b < 9 else "[0.9,1.0]",
                         "n": 0, "mean_prob": None, "frac_phishing": None})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (nb / n) * abs(acc - conf)
        bins.append({"bin": f"[{lo:.1f},{hi:.1f})" if b < 9 else "[0.9,1.0]",
                     "n": nb, "mean_prob": round(conf, 4),
                     "frac_phishing": round(acc, 4)})
    return float(ece), bins


def main():
    data, checks = load_all()
    print("\n".join(checks))

    # ---- (2a) full table + cross-check vs saved metrics JSONs ----
    table = {}
    cross = {"max_abs_diff": 0.0, "mismatches": [], "checked": 0}
    for s in SETS:
        table[s] = {}
        for m in MODELS:
            df = data[(s, m)]
            y = df["true_label"].to_numpy(dtype=int)
            p = df["prob_phishing"].to_numpy(dtype=float)
            met = compute_metrics_full(y, p, threshold=0.5)
            table[s][m] = {
                "n": met["n"],
                "n_phish": int((y == 1).sum()),
                "n_legit": int((y == 0).sum()),
                "precision": met["precision"], "recall": met["recall"],
                "f1": met["f1"], "roc_auc": met["roc_auc"],
                "pr_auc": met["pr_auc"],
                "precision_at_recall_0.90": met["precision_at_recall_0.90"],
                "precision_at_recall_0.90_threshold": met["precision_at_recall_0.90_threshold"],
                "fpr": met["false_positive_rate"],
                "confusion_matrix": met["confusion_matrix"],
                "accuracy": met["accuracy"],
                "source": f"experiments/modern_phishing_ood/predictions/{s}_{m}_predictions.csv",
            }
            saved = json.loads((RESULTS_DIR / f"{s}_{m}_metrics.json").read_text())
            sm = saved["metrics_at_0_5"]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc",
                      "pr_auc", "precision_at_recall_0.90",
                      "precision_at_recall_0.90_threshold", "false_positive_rate"):
                cross["checked"] += 1
                d = abs(float(met[k if k != "false_positive_rate" else "false_positive_rate"])
                        - float(sm[k]))
                # compute_metrics_full key for FPR is false_positive_rate
                cross["max_abs_diff"] = max(cross["max_abs_diff"], d)
                if d > 1e-9:
                    cross["mismatches"].append(f"{s}/{m}/{k}: recomputed "
                                                f"{met[k]} vs saved {sm[k]}")
    print(f"cross-check: {cross['checked']} cells, max abs diff "
          f"{cross['max_abs_diff']:.2e}, mismatches: {cross['mismatches']}")

    # ---- (2b) calibration ----
    calib = {"coverage": "all 6 OOD sets x 5 models (mean prob by class + ECE-10)",
             "per_set_model": {}, "ece_phishnchips": {}}
    for s in SETS:
        for m in MODELS:
            df = data[(s, m)]
            y = df["true_label"].to_numpy(dtype=int)
            p = df["prob_phishing"].to_numpy(dtype=float)
            e, bins = ece_10(y, p)
            calib["per_set_model"][f"{s}__{m}"] = {
                "set": s, "model": m, "n": len(y),
                "mean_prob_phishing_class": float(p[y == 1].mean()),
                "mean_prob_legit_class": float(p[y == 0].mean()),
                "ece_10bin": e, "ece_bins": bins,
            }
            if s == "phishnchips":
                calib["ece_phishnchips"][m] = e

    # in-distribution ECE: LR/SVC y_prob exist; RoBERTa/BiLSTM do not.
    indist = {"status": {}, "ece_10bin": {}, "verification": {}}
    base = json.loads((REPO_ROOT / "results" / "robustness_rerun"
                       / "baseline_results.json").read_text())
    from phishing_xai.data import deduplicate_data, load_data
    df_raw = load_data(REPO_ROOT / "data" / "Phishing_Email.csv")
    df_clean = deduplicate_data(df_raw)
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())
    test_idx = splits["test_indices"]
    assert len(test_idx) == 3495, len(test_idx)
    assert len(df_clean) == 17513, len(df_clean)
    y_test = df_clean.loc[test_idx, "label"].to_numpy(dtype=int)
    assert int((y_test == 1).sum()) == 1322, int((y_test == 1).sum())  # 1294 TP + 28 FN
    for m in ("tfidf_lr", "tfidf_svm"):
        ps = base["models"][m]["per_seed"][0]
        y_prob = np.asarray(ps["y_prob"], dtype=float)
        y_pred_saved = np.asarray(ps["y_pred"], dtype=int)
        assert len(y_prob) == 3495
        assert ((y_prob >= 0.5).astype(int) == y_pred_saved).all(), \
            f"{m}: y_prob@0.5 != saved y_pred"
        met = compute_metrics_full(y_test, y_prob, threshold=0.5)
        sm = ps["metrics"]
        for k in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert abs(met[k] - sm[k]) < 1e-9, f"{m}/{k}: {met[k]} vs {sm[k]}"
        e, bins = ece_10(y_test, y_prob)
        indist["ece_10bin"][m] = e
        indist["status"][m] = ("computed from saved seed-42-refit y_prob "
                               "(results/robustness_rerun/baseline_results.json, "
                               "n=3495) with y_true rebuilt from "
                               "data/Phishing_Email.csv + results/splits.json; "
                               "recomputed P/R/F1/ROC-AUC/PR-AUC match the saved "
                               "metrics exactly")
        indist["verification"][m] = {
            "n": 3495, "n_phish": int((y_test == 1).sum()),
            "mean_prob_phishing_class": float(y_prob[y_test == 1].mean()),
            "mean_prob_legit_class": float(y_prob[y_test == 0].mean()),
            "ece_bins": bins}
    # In-distribution ECE for DistilBERT: saved Kernel-2 seed-42 test
    # probabilities (results/distilbert/seed_42/test_prob.npy), same
    # verification pattern as TF-IDF above.
    d_prob = np.load(REPO_ROOT / "results" / "distilbert" / "seed_42"
                     / "test_prob.npy").astype(float)
    d_pred_saved = np.load(REPO_ROOT / "results" / "distilbert" / "seed_42"
                           / "test_pred.npy").astype(int)
    assert len(d_prob) == 3495 and len(d_pred_saved) == 3495
    assert ((d_prob >= 0.5).astype(int) == d_pred_saved).all(), \
        "distilbert: test_prob@0.5 != saved test_pred"
    d_met = compute_metrics_full(y_test, d_prob, threshold=0.5)
    d_saved = json.loads((REPO_ROOT / "results" / "distilbert" / "seed_42"
                          / "metrics.json").read_text())["test_metrics"]
    for k in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert abs(d_met[k] - d_saved[k]) < 1e-9, f"distilbert/{k}: {d_met[k]} vs {d_saved[k]}"
    e, bins = ece_10(y_test, d_prob)
    indist["ece_10bin"]["distilbert"] = e
    indist["status"]["distilbert"] = (
        "computed from saved Kernel-2 seed-42 test_prob.npy "
        "(results/distilbert/seed_42/test_prob.npy, n=3495) with y_true "
        "rebuilt from data/Phishing_Email.csv + results/splits.json; "
        "recomputed P/R/F1/ROC-AUC/PR-AUC match the saved metrics exactly")
    indist["verification"]["distilbert"] = {
        "n": 3495, "n_phish": int((y_test == 1).sum()),
        "mean_prob_phishing_class": float(d_prob[y_test == 1].mean()),
        "mean_prob_legit_class": float(d_prob[y_test == 0].mean()),
        "ece_bins": bins}
    for m in ("roberta", "bilstm"):
        indist["status"][m] = ("NOT RUN YET — no saved per-email in-distribution "
                               "test probabilities exist locally for this model: "
                               "results/robustness_rerun/baseline_results.json "
                               "records RoBERTa as NOT RUN YET (results/metrics.json "
                               "not found), no results_v7 *.npy files exist locally, "
                               "and no other results/*.json carries its y_prob.")
        indist["ece_10bin"][m] = None
    calib["in_distribution"] = indist

    # ---- (2c) threshold transfer on PhishNChips ----
    transfer = {}
    for m in MODELS:
        df = data[("phishnchips", m)]
        y = df["true_label"].to_numpy(dtype=int)
        p = df["prob_phishing"].to_numpy(dtype=float)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
        slice_idx, rest_idx = next(sss.split(np.zeros(len(y)), y))
        ys, ps = y[slice_idx], p[slice_idx]
        yr, pr = y[rest_idx], p[rest_idx]
        # candidate thresholds: unique slice probs, descending; pick max
        # precision s.t. recall >= 0.85 on the slice (tie -> highest threshold).
        best = None
        for thr in np.unique(ps)[::-1]:
            pred = (ps >= thr).astype(int)
            tp = int(((pred == 1) & (ys == 1)).sum())
            fp = int(((pred == 1) & (ys == 0)).sum())
            rec = tp / int((ys == 1).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            if rec >= 0.85:
                if best is None or prec > best["precision"] or (
                        prec == best["precision"] and thr > best["threshold"]):
                    best = {"threshold": float(thr), "precision": prec,
                            "recall": rec}
        assert best is not None
        met_slice = compute_metrics_full(ys, ps, threshold=best["threshold"])
        met_rest = compute_metrics_full(yr, pr, threshold=best["threshold"])
        transfer[m] = {
            "model": m, "set": "phishnchips",
            "method": ("target-domain recalibration (uses labelled OOD data) — "
                       "NOT pure OOD"),
            "slice_n": int(len(ys)), "slice_n_phish": int((ys == 1).sum()),
            "rest_n": int(len(yr)), "rest_n_phish": int((yr == 1).sum()),
            "seed": 42, "recall_floor": 0.85,
            "threshold": best["threshold"],
            "slice_precision_at_threshold": met_slice["precision"],
            "slice_recall_at_threshold": met_slice["recall"],
            "slice_fpr_at_threshold": met_slice["false_positive_rate"],
            "rest_precision": met_rest["precision"],
            "rest_recall": met_rest["recall"],
            "rest_f1": met_rest["f1"],
            "rest_fpr": met_rest["false_positive_rate"],
            "rest_accuracy": met_rest["accuracy"],
        }

    # ---- (2d) paired stats on PhishNChips vs TF-IDF+LR ----
    y = data[("phishnchips", "tfidf_lr")]["true_label"].to_numpy(dtype=int)
    p_lr = data[("phishnchips", "tfidf_lr")]["prob_phishing"].to_numpy(dtype=float)
    pred_lr = (p_lr >= 0.5).astype(int)
    paired = {"set": "phishnchips", "n": 2000, "reference": "tfidf_lr",
              "method": ("McNemar: repo phishing_xai.baselines.mcnemar_test "
                         "(paired classification ERRORS only — explicitly NOT "
                         "a precision test; b = model wrong / LR right, "
                         "c = model right / LR wrong; chi2 with continuity "
                         "correction, exact binomial p when b+c<25). "
                         "Bootstrap: paired percentile CIs, 1000 resamples, "
                         "seed 42; P/R/F1 diffs via "
                         "compute_ood_stats.paired_diff_ci; ROC-AUC/PR-AUC "
                         "diffs via the same percentile scheme (extension, "
                         "repo has no AUC-diff bootstrap)."),
              "comparisons": {}}
    rng_auc = np.random.RandomState(42)

    def paired_auc_diff_ci(y_true, pa, pb, n_resamples=1000, seed=42):
        rng = np.random.RandomState(seed)
        out = {}
        for name, fn in (("roc_auc", roc_auc_score),
                         ("pr_auc", average_precision_score)):
            diffs = []
            for _ in range(n_resamples):
                idx = rng.randint(0, len(y_true), size=len(y_true))
                if len(np.unique(y_true[idx])) < 2:
                    continue
                try:
                    diffs.append(float(fn(y_true[idx], pa[idx])
                                     - fn(y_true[idx], pb[idx])))
                except ValueError:
                    continue
            diffs = np.asarray(diffs, dtype=float)
            out[name] = {
                "diff_A_minus_B": float(fn(y_true, pa) - fn(y_true, pb)),
                "ci_low": float(np.percentile(diffs, 2.5)) if len(diffs) else None,
                "ci_high": float(np.percentile(diffs, 97.5)) if len(diffs) else None,
                "n_resamples": int(len(diffs))}
        return out

    for m in ("roberta", "bilstm", "tfidf_svm", "distilbert"):
        pm = data[("phishnchips", m)]["prob_phishing"].to_numpy(dtype=float)
        pred_m = (pm >= 0.5).astype(int)
        mc = mcnemar_test(y, pred_m, pred_lr)
        diffs = cs.paired_diff_ci(y, pm, p_lr, threshold=0.5,
                                  n_resamples=1000, seed=42)
        auc_diffs = paired_auc_diff_ci(y, pm, p_lr)
        paired["comparisons"][m] = {
            "model": m, "vs": "tfidf_lr", "n": 2000,
            "mcnemar": {"b_model_wrong_lr_right": mc["b"],
                        "c_model_right_lr_wrong": mc["c"],
                        "n_discordant": mc["b"] + mc["c"],
                        "statistic": mc["statistic"],
                        "p_value": mc["p_value"]},
            "bootstrap_diff_A_minus_B": diffs,
            "bootstrap_auc_diff_A_minus_B": auc_diffs,
        }

    # DistilBERT vs RoBERTa on PhishNChips (same paired machinery).
    y_roberta = data[("phishnchips", "roberta")]["true_label"].to_numpy(dtype=int)
    assert (y_roberta == y).all()
    p_db = data[("phishnchips", "distilbert")]["prob_phishing"].to_numpy(dtype=float)
    p_rb = data[("phishnchips", "roberta")]["prob_phishing"].to_numpy(dtype=float)
    pred_db = (p_db >= 0.5).astype(int)
    pred_rb = (p_rb >= 0.5).astype(int)
    mc_dr = mcnemar_test(y, pred_db, pred_rb)
    diffs_dr = cs.paired_diff_ci(y, p_db, p_rb, threshold=0.5,
                                 n_resamples=1000, seed=42)
    auc_diffs_dr = paired_auc_diff_ci(y, p_db, p_rb)
    paired["comparisons_vs_roberta"] = {
        "distilbert": {
            "model": "distilbert", "vs": "roberta", "n": 2000,
            "mcnemar": {"b_model_wrong_roberta_right": mc_dr["b"],
                        "c_model_right_roberta_wrong": mc_dr["c"],
                        "n_discordant": mc_dr["b"] + mc_dr["c"],
                        "statistic": mc_dr["statistic"],
                        "p_value": mc_dr["p_value"]},
            "bootstrap_diff_A_minus_B": diffs_dr,
            "bootstrap_auc_diff_A_minus_B": auc_diffs_dr,
        }
    }

    # ---- (2e) latency verification ----
    lat_file = json.loads((REPO_ROOT / "results" / "rerun_latency"
                           / "latency_explain_time.json").read_text())
    lat = lat_file["latency_ms_per_email"]
    base_lat = {m: base["models"][m]["per_seed"][0]["latency_cpu_ms"]
                for m in ("tfidf_lr", "tfidf_svm")}
    latency = {
        "files": {
            "rerun_latency": "results/rerun_latency/latency_explain_time.json",
            "tfidf_latency": ("results/robustness_rerun/baseline_results.json "
                              "(seed-42 refit, latency_cpu_ms per model)")},
        "protocol_rerun": lat_file["protocol"],
        "gpu": lat_file["gpu"], "n_timed": lat_file["n_latency"],
        "batch_size": lat_file["batch_size"],
        "verified_values_median_ms": {
            "roberta_gpu": lat["roberta_gpu"]["latency_ms_per_email_median"],
            "roberta_cpu": lat["roberta_cpu"]["latency_ms_per_email_median"],
            "bilstm_gpu": lat["bilstm_gpu"]["latency_ms_per_email_median"],
            "bilstm_cpu": lat["bilstm_cpu"]["latency_ms_per_email_median"],
            "tfidf_svm_cpu": base_lat["tfidf_svm"]["latency_ms_per_email_median"],
            "tfidf_lr_cpu": base_lat["tfidf_lr"]["latency_ms_per_email_median"]},
        "task_numbers_match": {
            "roberta_gpu_15.50": abs(lat["roberta_gpu"]["latency_ms_per_email_median"] - 15.50) < 0.01,
            "roberta_cpu_242.50": abs(lat["roberta_cpu"]["latency_ms_per_email_median"] - 242.50) < 0.01,
            "bilstm_gpu_8.37": abs(lat["bilstm_gpu"]["latency_ms_per_email_median"] - 8.37) < 0.01,
            "bilstm_cpu_19.90": abs(lat["bilstm_cpu"]["latency_ms_per_email_median"] - 19.90) < 0.01,
            "tfidf_svm_cpu_8.469": abs(base_lat["tfidf_svm"]["latency_ms_per_email_median"] - 8.469) < 0.001,
            "tfidf_lr_cpu_1.089": abs(base_lat["tfidf_lr"]["latency_ms_per_email_median"] - 1.089) < 0.001},
        "tfidf_gpu": None,
        "tfidf_gpu_note": ("NOT COMPUTABLE — TF-IDF+SVC and TF-IDF+LR are "
                           "CPU-only (sklearn); no GPU latency was ever measured "
                           "for them (latency_gpu_ms is None in both source "
                           "files). Any GPU multiple vs TF-IDF+SVC is therefore "
                           "NOT COMPUTABLE and is not reported."),
        "cpu_multiples_vs_tfidf_svm_median": {
            "roberta_cpu_div_svc_cpu": float(
                lat["roberta_cpu"]["latency_ms_per_email_median"]
                / base_lat["tfidf_svm"]["latency_ms_per_email_median"]),
            "bilstm_cpu_div_svc_cpu": float(
                lat["bilstm_cpu"]["latency_ms_per_email_median"]
                / base_lat["tfidf_svm"]["latency_ms_per_email_median"]),
            "tfidf_lr_cpu_div_svc_cpu": float(
                base_lat["tfidf_lr"]["latency_ms_per_email_median"]
                / base_lat["tfidf_svm"]["latency_ms_per_email_median"])},
    }

    # ---- write outputs ----
    distil_note = ("computed here from the 6 saved DistilBERT OOD prediction "
                   "CSVs (Kernel 4) and the saved Kernel-2 seed-42 test "
                   "probabilities for in-distribution ECE.")
    table_out = {"experiment": "modern_phishing_ood task 2a",
                 "method": ("recomputed locally from the 30 saved per-email "
                            "prediction CSVs with repo "
                            "phishing_xai.baselines.compute_metrics_full "
                            "(threshold 0.5); precision@recall 0.90 is the repo "
                            "precision_at_recall (max precision over thresholds "
                            "with recall >= 0.90)."),
                 "sets": SETS, "models": MODELS,
                 "distilbert": {"status": "computed", "reason": distil_note},
                 "cross_check_vs_saved_metrics_jsons": cross,
                 "table": table}
    (RESULTS_DIR / "ood_table_full.json").write_text(
        json.dumps(table_out, indent=2))
    (RESULTS_DIR / "calibration.json").write_text(
        json.dumps(calib, indent=2))
    (RESULTS_DIR / "threshold_transfer.json").write_text(
        json.dumps({"experiment": "threshold transfer (2c)",
                    "label": ("target-domain recalibration (uses labelled OOD "
                              "data) — NOT pure OOD"),
                    "results": transfer}, indent=2))
    (RESULTS_DIR / "paired_stats_vs_tfidf_lr.json").write_text(
        json.dumps(paired, indent=2))
    (RESULTS_DIR / "latency_verification.json").write_text(
        json.dumps(latency, indent=2))
    print("wrote ood_table_full.json, calibration.json, "
          "threshold_transfer.json, paired_stats_vs_tfidf_lr.json, "
          "latency_verification.json")
    print(json.dumps({"cross_check": cross,
                      "ece_phishnchips": calib["ece_phishnchips"],
                      "in_dist_ece": indist["ece_10bin"],
                      "transfer": transfer}, indent=2, default=str))


if __name__ == "__main__":
    main()
