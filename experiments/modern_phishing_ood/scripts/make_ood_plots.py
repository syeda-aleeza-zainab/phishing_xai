"""Publication-quality plots for the modern-phishing OOD experiment.

Reads the metrics JSONs, per-email predictions and ``ood_stats.json``
written by ``run_ood_eval.py`` / ``compute_ood_stats.py`` and writes
PNGs to ``plots/``. Every plot is built only from executed numbers;
a plot whose inputs are missing is skipped with a NOT RUN YET note
instead of a fabricated stand-in.

Plots:
1. ``performance_comparison.png`` — grouped bars (P/R/F1/ROC-AUC) per
   model on PhishNChips + PhishFuzzer (+ synthetic sets when present).
2. ``roc_curves.png`` / ``pr_curves.png`` — ROC + PR curves per model
   per set, from saved per-email probabilities.
3. ``confusion_matrices.png`` — one panel per model per set (at 0.5).
4. ``per_category_phishnchips.png`` — recall per phishing URL category
   and FPR per legit category, per model.
5. ``human_vs_ai.png`` — synthetic track only: recall on Frontiers
   human vs LLM phishing, per model (labelled synthetic).
6. ``robustness_degradation.png`` — clean vs perturbed F1 (Kernel 3).
7. ``latency_comparison.png`` — median ms/email, GPU vs CPU, log scale.
8. ``ci_plot.png`` — bootstrap 95% CIs for precision/recall/F1 per
   model on the primary set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
ROBERTA = "#1f77b4"
BILSTM = "#ff7f0e"
SETS_PRIMARY = ["phishnchips", "phishfuzzer"]
SETS_SYNTHETIC = ["frontiers_llm", "frontiers_human", "bec", "bec_poisoned"]


def load_metrics(results_dir: Path, set_name: str, model: str) -> Optional[Dict]:
    p = results_dir / f"{set_name}_{model}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {name}", flush=True)


def _roc_pr(y, p):
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return (fpr, tpr), (rec, prec)


def plot_performance(results_dir: Path, out: Path):
    """Grouped metric bars per set (only sets with both models)."""
    metrics = ["precision", "recall", "f1", "roc_auc"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC"]
    rows = []
    for s in SETS_PRIMARY + SETS_SYNTHETIC:
        for m in ("roberta", "bilstm"):
            d = load_metrics(results_dir, s, m)
            if d is None:
                continue
            rows.append((s, m, d["metrics_at_0_5"]))
    if not rows:
        print("[plot] performance_comparison: NOT RUN YET (no metrics)", flush=True)
        return
    sets = sorted({r[0] for r in rows})
    n = len(sets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, sets):
        x = np.arange(len(metrics))
        w = 0.36
        for j, m in enumerate(("roberta", "bilstm")):
            d = next((r[2] for r in rows if r[0] == s and r[1] == m), None)
            vals = [d[k] if d else np.nan for k in metrics]
            ax.bar(x + (j - 0.5) * w, vals, w,
                   label="RoBERTa" if m == "roberta" else "BiLSTM",
                   color=ROBERTA if m == "roberta" else BILSTM)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(s.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score (threshold 0.5)")
    axes[0].legend()
    fig.suptitle("OOD performance at threshold 0.5 (PhishNChips = modern "
                 "URL-grounded synthetic-stimuli OOD)")
    _save(fig, out, "performance_comparison.png")


def plot_curves(pred_dir: Path, out: Path):
    """ROC + PR curves from saved per-email probabilities."""
    files = sorted(pred_dir.glob("*_predictions.csv"))
    if not files:
        print("[plot] ROC/PR curves: NOT RUN YET (no predictions)", flush=True)
        return
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for path in files:
        df = pd.read_csv(path)
        binary = df[df["true_label"] >= 0]
        y = binary["true_label"].astype(int).to_numpy()
        p = binary["prob_phishing"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        (fpr, tpr), (rec, prec) = _roc_pr(y, p)
        stem = path.name[: -len("_predictions.csv")]
        color = ROBERTA if stem.endswith("_roberta") else BILSTM
        ls = "-" if stem.endswith("_roberta") else "--"
        ax_roc.plot(fpr, tpr, color=color, ls=ls, label=stem.replace("_", " "))
        ax_pr.plot(rec, prec, color=color, ls=ls, label=stem.replace("_", " "))
    ax_roc.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves (OOD sets, per-email probabilities)")
    ax_roc.legend(fontsize=9)
    ax_roc.grid(alpha=0.3)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–recall curves (OOD sets)")
    ax_pr.legend(fontsize=9)
    ax_pr.grid(alpha=0.3)
    _save(fig_roc, out, "roc_curves.png")
    _save(fig_pr, out, "pr_curves.png")


def plot_confusion(results_dir: Path, out: Path):
    """Confusion-matrix panels (threshold 0.5)."""
    entries = []
    for s in SETS_PRIMARY + SETS_SYNTHETIC:
        for m in ("roberta", "bilstm"):
            d = load_metrics(results_dir, s, m)
            if d is not None:
                entries.append((s, m, d["metrics_at_0_5"]["confusion_matrix"]))
    if not entries:
        print("[plot] confusion matrices: NOT RUN YET", flush=True)
        return
    n = len(entries)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    axes = np.atleast_2d(axes)
    for ax, (s, m, cm) in zip(axes.flat, entries):
        cm = np.asarray(cm)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred legit", "pred phish"])
        ax.set_yticklabels(["true legit", "true phish"])
        ax.set_title(f"{s} / {m}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold")
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices at threshold 0.5")
    _save(fig, out, "confusion_matrices.png")


def plot_per_category(results_dir: Path, out: Path):
    """PhishNChips: recall per phishing URL category, FPR per legit slice."""
    d_r = load_metrics(results_dir, "phishnchips", "roberta")
    d_b = load_metrics(results_dir, "phishnchips", "bilstm")
    if d_r is None or d_b is None:
        print("[plot] per-category: NOT RUN YET", flush=True)
        return
    cats = d_r["per_category_at_0_5"]
    phish_cats, legit_cats = [], []
    for k, v in cats.items():
        if not k.startswith("category="):
            continue
        cm = v["confusion_matrix"]
        n_pos = cm[1][0] + cm[1][1]
        n_neg = cm[0][0] + cm[0][1]
        if n_pos > 0 and n_neg == 0:
            phish_cats.append(k.split("=", 1)[1])
        elif n_neg > 0 and n_pos == 0:
            legit_cats.append(k.split("=", 1)[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    x = np.arange(len(phish_cats))
    w = 0.36
    for j, (d, m) in enumerate(((d_r, "roberta"), (d_b, "bilstm"))):
        rec = []
        for c in phish_cats:
            v = d["per_category_at_0_5"][f"category={c}"]
            rec.append(v["recall"])
        axes[0].bar(x + (j - 0.5) * w, rec, w, label=m,
                    color=ROBERTA if m == "roberta" else BILSTM)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phish_cats, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Recall (phishing rows only)")
    axes[0].set_title("Recall per phishing URL category")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    x = np.arange(len(legit_cats))
    for j, (d, m) in enumerate(((d_r, "roberta"), (d_b, "bilstm"))):
        fpr = []
        for c in legit_cats:
            v = d["per_category_at_0_5"][f"category={c}"]
            fpr.append(v["false_positive_rate"])
        axes[1].bar(x + (j - 0.5) * w, fpr, w, label=m,
                    color=ROBERTA if m == "roberta" else BILSTM)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(legit_cats, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("False-positive rate (legit rows only)")
    axes[1].set_title("FPR per legitimate category")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("PhishNChips per-category performance at threshold 0.5")
    _save(fig, out, "per_category_phishnchips.png")


def plot_human_vs_ai(results_dir: Path, out: Path):
    """Synthetic track only: recall on human (old corpora) vs LLM phishing."""
    vals = {}
    for s in ("frontiers_human", "frontiers_llm"):
        for m in ("roberta", "bilstm"):
            d = load_metrics(results_dir, s, m)
            if d is not None:
                vals[(s, m)] = d["metrics_at_0_5"]["recall"]
    if len(vals) < 4:
        print("[plot] human-vs-AI: NOT RUN YET (synthetic track incomplete)",
              flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    w = 0.36
    for j, m in enumerate(("roberta", "bilstm")):
        ax.bar(x + (j - 0.5) * w,
               [vals[("frontiers_human", m)], vals[("frontiers_llm", m)]],
               w, label=m, color=ROBERTA if m == "roberta" else BILSTM)
    ax.set_xticks(x)
    ax.set_xticklabels(["Human-authored\n(2007–08 corpora)", "LLM-generated\n(2026)"])
    ax.set_ylabel("Recall at threshold 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Human-vs-AI phishing recall (SYNTHETIC TRACK ONLY;\n"
                 "both paired with the same PhishNChips legit 1,000)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "human_vs_ai.png")


def plot_robustness(results_dir: Path, out: Path):
    """Clean vs perturbed F1 on PhishNChips (Kernel 3)."""
    kinds = ["clean", "truncated", "urls_removed", "typo_synonym"]
    data = {}
    for m in ("roberta", "bilstm"):
        p = results_dir / f"robustness_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        row = {"truncated": d["robustness"]["truncated"]["f1"],
               "urls_removed": d["robustness"]["urls_removed"]["f1"],
               "typo_synonym": d["robustness"]["typo_synonym"]["f1"]}
        base = load_metrics(results_dir, "phishnchips", m)
        row["clean"] = (base["metrics_at_0_5"]["f1"]
                        if base is not None else np.nan)
        data[m] = row
    if not data:
        print("[plot] robustness: NOT RUN YET (Kernel 3 not run)", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(kinds))
    w = 0.36
    for j, m in enumerate(("roberta", "bilstm")):
        if m not in data:
            continue
        ax.bar(x + (j - 0.5) * w, [data[m][k] for k in kinds], w, label=m,
               color=ROBERTA if m == "roberta" else BILSTM)
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "truncated", "URLs removed", "typo/synonym"])
    ax.set_ylabel("F1 at threshold 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness degradation on PhishNChips")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "robustness_degradation.png")


def plot_latency(results_dir: Path, out: Path):
    """Median ms/email, GPU vs CPU (log scale)."""
    rows = []
    for m in ("roberta", "bilstm"):
        d = load_metrics(results_dir, "phishnchips", m)
        if d is None:
            d = load_metrics(results_dir, "frontiers_llm", m)
        if d is None:
            continue
        lat = d.get("latency_ms_per_email", {})
        gpu = lat.get("gpu", {}) if isinstance(lat.get("gpu"), dict) else {}
        cpu = lat.get("cpu", {}) if isinstance(lat.get("cpu"), dict) else {}
        rows.append((m, gpu.get("latency_ms_per_email_median"),
                     cpu.get("latency_ms_per_email_median")))
    if not rows:
        print("[plot] latency: NOT RUN YET", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r[1] if r[1] else np.nan for r in rows], w,
           label="GPU (T4)", color=ROBERTA)
    ax.bar(x + w / 2, [r[2] if r[2] else np.nan for r in rows], w,
           label="CPU", color=BILSTM)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Median ms / email (batch 1, log scale)")
    ax.set_yscale("log")
    ax.set_title("Inference latency (this OOD run, T4 GPU)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "latency_comparison.png")


def plot_cis(results_dir: Path, out: Path):
    """Bootstrap 95% CIs for P/R/F1 on the primary set."""
    p = results_dir / "ood_stats.json"
    if not p.exists():
        print("[plot] CI plot: NOT RUN YET (no ood_stats.json)", flush=True)
        return
    stats = json.loads(p.read_text())
    entry = stats.get("per_set", {}).get("phishnchips", {})
    if "roberta" not in entry:
        print("[plot] CI plot: NOT RUN YET (no phishnchips stats)", flush=True)
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (key, label) in zip(axes, (
            ("bootstrap_precision_ci", "Precision"),
            ("bootstrap_recall_ci", "Recall"),
            ("bootstrap_f1_ci", "F1"))):
        for j, m in enumerate(("roberta", "bilstm")):
            d = entry.get(m, {}).get(key, {})
            val = d.get("precision", d.get("recall", d.get("f1")))
            lo, hi = d.get("ci_low"), d.get("ci_high")
            if val is None or lo is None:
                continue
            ax.errorbar(j, val, yerr=[[val - lo], [hi - val]], fmt="o",
                        capsize=6, markersize=8,
                        color=ROBERTA if m == "roberta" else BILSTM,
                        label=m)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RoBERTa", "BiLSTM"])
        ax.set_title(f"{label} (95% bootstrap CI)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score at threshold 0.5")
    fig.suptitle("PhishNChips — bootstrap 95% CIs (1,000 resamples, seed 42)")
    _save(fig, out, "ci_plot.png")


def main():
    parser = argparse.ArgumentParser(description="OOD plots.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    pred_dir = Path(args.predictions_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_performance(results_dir, out)
    plot_curves(pred_dir, out)
    plot_confusion(results_dir, out)
    plot_per_category(results_dir, out)
    plot_human_vs_ai(results_dir, out)
    plot_robustness(results_dir, out)
    plot_latency(results_dir, out)
    plot_cis(results_dir, out)
    print("[done]", sorted(p.name for p in out.glob("*.png")), flush=True)


if __name__ == "__main__":
    main()
