
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import config

RESULTS_DIR = config.DATA_DIR / "results"
FEATURES_FILE = config.DATA_DIR / "features" / "corpus_features.csv"
FIGS_DIR = config.PROJECT_ROOT / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================================
# IMPORTS
# ==========================================================================
try:
    import matplotlib
    matplotlib.use("Agg")    # headless safe
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("[ERROR] matplotlib/seaborn not installed.")
    print("        Run: pip install matplotlib seaborn")
    sys.exit(1)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

PALETTE_DIVERGE = "RdYlGn"      # for cross-model F1 heatmaps
PALETTE_BARS    = "viridis"

# Friendly model names for display
MODEL_DISPLAY = {
    "gpt-4.1":          "GPT-4.1",
    "deepseek3.2":      "DeepSeek 3.2",
    "llama-3.3-70b":    "LLaMA 3.3 70B",
}

def pretty(name):
    return MODEL_DISPLAY.get(name, name)

def save_both(fig, base_name):
    """Save the same figure as PNG (review) and PDF (camera-ready)."""
    png_path = FIGS_DIR / f"{base_name}.png"
    pdf_path = FIGS_DIR / f"{base_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  [OK] {base_name}.png  +  {base_name}.pdf")

def fig_b_original():
    print("\n[FIG 1] Cross-model transferability (default threshold)")
    df = pd.read_csv(RESULTS_DIR / "task_b_cross_model_matrix.csv", index_col=0)
    df = df.rename(index=pretty, columns=pretty)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.heatmap(df, annot=True, fmt=".3f", cmap=PALETTE_DIVERGE,
                vmin=0.4, vmax=1.0, cbar_kws={"label": "F1-score"},
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Evaluation LLM")
    ax.set_ylabel("Training LLM")
    ax.set_title("Cross-model transferability matrix (default threshold = 0.5)")
    save_both(fig, "fig1_cross_model_default")

def fig_b_recal():
    print("\n[FIG 2] Cross-model transferability (recalibrated threshold)")
    df = pd.read_csv(RESULTS_DIR / "task_b_recalibrated_matrix.csv", index_col=0)
    df = df.rename(index=pretty, columns=pretty)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.heatmap(df, annot=True, fmt=".3f", cmap=PALETTE_DIVERGE,
                vmin=0.4, vmax=1.0, cbar_kws={"label": "F1-score"},
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Evaluation LLM")
    ax.set_ylabel("Training LLM")
    ax.set_title("Cross-model transferability with threshold recalibration")
    save_both(fig, "fig2_cross_model_recalibrated")

def fig_b_compare():
    print("\n[FIG 3] Side-by-side comparison default vs recalibrated")
    df_def = pd.read_csv(RESULTS_DIR / "task_b_cross_model_matrix.csv",   index_col=0)
    df_rec = pd.read_csv(RESULTS_DIR / "task_b_recalibrated_matrix.csv",  index_col=0)
    df_def = df_def.rename(index=pretty, columns=pretty)
    df_rec = df_rec.rename(index=pretty, columns=pretty)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    sns.heatmap(df_def, annot=True, fmt=".3f", cmap=PALETTE_DIVERGE,
                vmin=0.4, vmax=1.0, cbar=False, linewidths=0.5,
                ax=axes[0])
    axes[0].set_title("(a) Default threshold = 0.5\nGap = 28.1 pp")
    axes[0].set_xlabel("Evaluation LLM")
    axes[0].set_ylabel("Training LLM")

    sns.heatmap(df_rec, annot=True, fmt=".3f", cmap=PALETTE_DIVERGE,
                vmin=0.4, vmax=1.0, cbar_kws={"label": "F1-score"},
                linewidths=0.5, ax=axes[1])
    axes[1].set_title("(b) Recalibrated threshold\nGap = 4.0 pp")
    axes[1].set_xlabel("Evaluation LLM")
    axes[1].set_ylabel("")

    fig.suptitle("Effect of threshold recalibration on cross-model transferability",
                 fontsize=13, fontweight="bold", y=1.02)
    save_both(fig, "fig3_default_vs_recalibrated")

def fig_feature_importance():
    print("\n[FIG 4] Top-10 feature importance per LLM")
    fi = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    metric_col = [c for c in fi.columns if c not in ("llm_model", "feature")][0]

    models = sorted(fi["llm_model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.5),
                             sharex=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = fi[fi["llm_model"] == model].nlargest(10, metric_col)
        sub = sub.iloc[::-1]    # reverse so top is at top
        sns.barplot(data=sub, y="feature", x=metric_col,
                    palette=PALETTE_BARS, ax=ax, hue="feature", legend=False)
        ax.set_title(pretty(model))
        ax.set_xlabel(metric_col.replace("_", " "))
        ax.set_ylabel("")

    fig.suptitle("Top-10 stylometric features per LLM",
                 fontsize=13, fontweight="bold", y=1.02)
    save_both(fig, "fig4_feature_importance_per_llm")

def fig_stable_features():
    print("\n[FIG 5] Stable vs model-specific top features")
    fi = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    metric_col = [c for c in fi.columns if c not in ("llm_model", "feature")][0]
    models = sorted(fi["llm_model"].unique())

    # Build top-5 set per model
    top5 = {}
    for m in models:
        sub = fi[fi["llm_model"] == m].nlargest(5, metric_col)
        top5[m] = set(sub["feature"])

    # Stable: in all top-5; partial: in 2 of 3; specific: in 1 only
    all_features = set.union(*top5.values())
    rows = []
    for f in sorted(all_features):
        n = sum(1 for s in top5.values() if f in s)
        category = ("Stable (3/3)" if n == 3 else
                    "Partial (2/3)" if n == 2 else
                    "Model-specific (1/3)")
        for m in models:
            rows.append({
                "feature": f, "category": category,
                "llm": pretty(m), "in_top5": int(f in top5[m]),
            })
    plot_df = pd.DataFrame(rows)

    # Pivot for heatmap-style chart
    pivot = plot_df.pivot_table(index="feature", columns="llm",
                                values="in_top5", fill_value=0)
    # Order features by how many models include them (descending)
    feat_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[feat_order]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(pivot))))
    pivot = pivot.astype(int)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
                cbar_kws={"label": "In top-5 (1=yes)"},
                linewidths=0.5, ax=ax)
    ax.set_title("Top-5 feature membership across LLMs\n"
                 "(features ordered by overlap)")
    ax.set_xlabel("LLM")
    ax.set_ylabel("Stylometric feature")
    save_both(fig, "fig5_stable_vs_specific_features")

def fig_task_a():
    print("\n[FIG 6] Task A intra-model performance")
    df = pd.read_csv(RESULTS_DIR / "task_a_intra_model.csv")
    df["llm_display"] = df["llm_model"].map(pretty)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="llm_display", y="f1_mean",
                hue="classifier", palette="Set2", ax=ax,
                err_kws={"linewidth": 1.5}, capsize=0.1)

    # Add error bars manually for std
    n_models = df["llm_display"].nunique()
    n_classifiers = df["classifier"].nunique()
    width = 0.8 / n_classifiers
    classifiers = sorted(df["classifier"].unique())
    models_order = sorted(df["llm_display"].unique())
    for i, model in enumerate(models_order):
        for j, clf in enumerate(classifiers):
            row = df[(df["llm_display"] == model) & (df["classifier"] == clf)]
            if len(row) == 0:
                continue
            x_pos = i + (j - (n_classifiers - 1) / 2) * width
            ax.errorbar(x_pos, row["f1_mean"].values[0],
                        yerr=row["f1_std"].values[0],
                        fmt="none", ecolor="black", capsize=4,
                        linewidth=1.2)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("LLM")
    ax.set_ylabel("F1-score (mean ± std, 5-fold CV)")
    ax.set_title("Task A — Intra-model classification performance")
    ax.legend(title="Classifier", loc="lower right")
    save_both(fig, "fig6_task_a_intra_model")

def fig_feature_distributions():
    print("\n[FIG 7] Distribution of key stylometric features")
    if not FEATURES_FILE.exists():
        print(f"  [SKIP] {FEATURES_FILE} not found")
        return
    df = pd.read_csv(FEATURES_FILE)

    # Pick the most discriminative features for the visual
    key_features = [
        "politeness_density",
        "cta_density",
        "time_pressure_density",
        "authority_density",
        "ttr",
        "url_density",
    ]
    key_features = [f for f in key_features if f in df.columns]

    n = len(key_features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, feat in zip(axes, key_features):
        sns.violinplot(data=df, x="label_origin", y=feat,
                       hue="label_origin", palette="Set2",
                       ax=ax, inner="quartile", legend=False)
        ax.set_title(feat.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylabel("")
    # Hide unused axes
    for ax in axes[len(key_features):]:
        ax.axis("off")

    fig.suptitle("Distribution of key stylometric features: human vs LLM",
                 fontsize=13, fontweight="bold", y=1.01)
    save_both(fig, "fig7_feature_distributions")

def main():
    print("=" * 60)
    print("Generating publication-ready figures")
    print(f"Output folder: {FIGS_DIR}")
    print("=" * 60)

    fig_b_original()
    fig_b_recal()
    fig_b_compare()
    fig_feature_importance()
    fig_stable_features()
    fig_task_a()
    fig_feature_distributions()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    for f in sorted(FIGS_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name}  ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()