
import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

FEATURES_FILE = config.DATA_DIR / "features" / "corpus_features.csv"
RESULTS_DIR   = config.DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing  import StandardScaler
    from sklearn.linear_model   import LogisticRegression
    from sklearn.inspection     import permutation_importance
    from sklearn.metrics        import (accuracy_score, precision_score,
                                        recall_score, f1_score, roc_auc_score,
                                        precision_recall_curve)
    from xgboost import XGBClassifier
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("        Run: pip install scikit-learn xgboost")
    sys.exit(1)

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print("[WARN] shap not installed. Will fall back to permutation importance.")
    print("       For SHAP analysis: pip install shap")

RANDOM_SEED = config.RANDOM_SEED
N_FOLDS     = 5

FEATURE_NAMES = [
    "ttr", "mean_word_len", "mean_sentence_len_tokens", "yules_k",
    "clause_density", "noun_ratio", "verb_ratio", "mean_parse_depth",
    "imperative_count", "first_person_ratio", "second_person_ratio",
    "politeness_density", "urgency_density",
    "url_density", "cta_density", "authority_density", "time_pressure_density",
]

def metrics_dict(y_true, y_pred, y_proba=None):
    out = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out

def best_f1_threshold(y_true, y_proba):
    """Find the probability threshold that maximises F1 on the given data.
    Used for the threshold-recalibrated cross-model experiment."""
    precs, recs, ths = precision_recall_curve(y_true, y_proba)
    # precs and recs are length n+1; ths is length n
    f1s = 2 * precs * recs / np.where((precs + recs) > 0, precs + recs, 1)
    # Drop the last point which corresponds to threshold = +inf
    f1s = f1s[:-1]
    if len(f1s) == 0:
        return 0.5, float("nan")
    best_idx = int(np.argmax(f1s))
    return float(ths[best_idx]), float(f1s[best_idx])

def make_xgb():
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_SEED,
        n_jobs=-1, verbosity=0,
    )

def make_lr():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                              class_weight="balanced")

def fit_predict(X_train, y_train, X_test, classifier_name):
    if classifier_name == "logreg":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        clf = make_lr()
        clf.fit(X_train_s, y_train)
        y_pred  = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]
        return clf, y_pred, y_proba
    elif classifier_name == "xgboost":
        clf = make_xgb()
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        return clf, y_pred, y_proba
    raise ValueError(classifier_name)

def load_features():
    if not FEATURES_FILE.exists():
        print(f"[ERROR] Features file not found: {FEATURES_FILE}")
        sys.exit(1)
    df = pd.read_csv(FEATURES_FILE)
    print(f"[INFO] Loaded {len(df):,} rows from {FEATURES_FILE}")
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing feature columns: {missing}")
        sys.exit(1)
    if "label_origin" not in df.columns or "source" not in df.columns:
        print("[ERROR] Required metadata columns not found.")
        sys.exit(1)
    df["y"] = (df["label_origin"] == "llm").astype(int)
    df = df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    return df

def get_llm_models(df):
    return sorted(df.loc[df["label_origin"] == "llm", "source"].unique().tolist())

def get_human_sources(df):
    return sorted(df.loc[df["label_origin"] == "human", "source"].unique().tolist())

def run_task_a(df, llm_models):
    print("\n" + "=" * 70)
    print("TASK A — Intra-model performance (5-fold CV per model)")
    print("=" * 70)
    rows = []
    humans = df[df["label_origin"] == "human"].copy()
    for model in llm_models:
        llm_rows = df[(df["label_origin"] == "llm") & (df["source"] == model)].copy()
        sub = pd.concat([humans, llm_rows], ignore_index=True)
        X, y = sub[FEATURE_NAMES].values, sub["y"].values
        for clf_name in ["logreg", "xgboost"]:
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            fold_metrics = defaultdict(list)
            for tr_idx, te_idx in kf.split(X, y):
                _, y_pred, y_proba = fit_predict(X[tr_idx], y[tr_idx], X[te_idx], clf_name)
                m = metrics_dict(y[te_idx], y_pred, y_proba)
                for k, v in m.items():
                    fold_metrics[k].append(v)
            row = {"llm_model": model, "classifier": clf_name,
                   "n_human": len(humans), "n_llm": len(llm_rows)}
            for k, vals in fold_metrics.items():
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"]  = float(np.std(vals))
            rows.append(row)
            print(f"  {model:30s} {clf_name:8s} "
                  f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
                  f"AUC={row['auc_mean']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "task_a_intra_model.csv", index=False)
    print(f"\n[OK] Saved -> {RESULTS_DIR / 'task_a_intra_model.csv'}")
    return out

def run_task_b(df, llm_models, classifier_name="xgboost"):
    print("\n" + "=" * 70)
    print(f"TASK B — Cross-model transferability matrix ({classifier_name})")
    print("=" * 70)
    humans = df[df["label_origin"] == "human"].copy()
    h_tr, h_te = train_test_split(humans, test_size=0.2,
                                  random_state=RANDOM_SEED,
                                  stratify=humans["source"])
    matrix, full_records = [], []
    for train_model in llm_models:
        row = {"train_model": train_model}
        llm_train = df[(df["label_origin"] == "llm") & (df["source"] == train_model)]
        train_df = pd.concat([h_tr, llm_train], ignore_index=True)
        X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["y"].values
        for eval_model in llm_models:
            llm_eval = df[(df["label_origin"] == "llm") & (df["source"] == eval_model)]
            test_df = pd.concat([h_te, llm_eval], ignore_index=True)
            X_te, y_te = test_df[FEATURE_NAMES].values, test_df["y"].values
            _, y_pred, y_proba = fit_predict(X_tr, y_tr, X_te, classifier_name)
            m = metrics_dict(y_te, y_pred, y_proba)
            row[eval_model] = round(m["f1"], 4)
            full_records.append({"train_model": train_model, "eval_model": eval_model,
                                 "classifier": classifier_name,
                                 **{k: round(v, 4) for k, v in m.items()}})
            print(f"  train={train_model:25s} eval={eval_model:25s} "
                  f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
        matrix.append(row)
    matrix_df = pd.DataFrame(matrix).set_index("train_model")
    matrix_df.to_csv(RESULTS_DIR / "task_b_cross_model_matrix.csv")
    pd.DataFrame(full_records).to_csv(RESULTS_DIR / "task_b_cross_model_full.csv", index=False)

    diag = [matrix_df.loc[m, m] for m in llm_models if m in matrix_df.columns]
    off = [matrix_df.loc[a, b] for a in llm_models for b in llm_models
           if a != b and a in matrix_df.index and b in matrix_df.columns]
    summary = {
        "classifier": classifier_name,
        "diagonal_mean_f1":   float(np.mean(diag)) if diag else None,
        "off_diagonal_mean_f1": float(np.mean(off)) if off else None,
        "transferability_gap_pp": (float(np.mean(diag) - np.mean(off)) * 100
                                   if diag and off else None),
        "n_train_humans": int(len(h_tr)),
        "n_test_humans":  int(len(h_te)),
    }
    with open(RESULTS_DIR / "task_b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Cross-model matrix:")
    print(matrix_df.round(4))
    print(f"\n[OK] Mean diagonal F1     : {summary['diagonal_mean_f1']:.4f}")
    print(f"[OK] Mean off-diagonal F1 : {summary['off_diagonal_mean_f1']:.4f}")
    print(f"[OK] Transferability gap  : {summary['transferability_gap_pp']:.2f} pp")
    return matrix_df, summary, h_tr, h_te

def run_task_b_recalibrated(df, llm_models, h_tr, h_te, classifier_name="xgboost"):
    """Same as Task B but threshold is tuned on a small validation slice
    of the EVAL distribution. This isolates the calibration component
    of the transferability gap from the discrimination component.

    Practical reading: how much of the cross-model F1 drop is just
    threshold mis-calibration that an operator could fix with a small
    labeled validation set on the new model?"""
    print("\n" + "=" * 70)
    print("TASK B' — Cross-model WITH threshold recalibration")
    print("=" * 70)
    matrix, full_records = [], []
    for train_model in llm_models:
        row = {"train_model": train_model}
        llm_train = df[(df["label_origin"] == "llm") & (df["source"] == train_model)]
        train_df = pd.concat([h_tr, llm_train], ignore_index=True)
        X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["y"].values
        for eval_model in llm_models:
            llm_eval = df[(df["label_origin"] == "llm") & (df["source"] == eval_model)]
            # Split eval LLM into 30% calibration / 70% test
            calib_llm, test_llm = train_test_split(
                llm_eval, test_size=0.7, random_state=RANDOM_SEED)
            # Use a small slice of human test for calibration too
            calib_h, test_h = train_test_split(
                h_te, test_size=0.7, random_state=RANDOM_SEED, stratify=h_te["source"])
            calib_df = pd.concat([calib_h, calib_llm], ignore_index=True)
            test_df  = pd.concat([test_h,  test_llm],  ignore_index=True)
            X_cal, y_cal = calib_df[FEATURE_NAMES].values, calib_df["y"].values
            X_te,  y_te  = test_df[FEATURE_NAMES].values,  test_df["y"].values

            clf, _, _ = fit_predict(X_tr, y_tr, X_cal, classifier_name)
            # Get probabilities for both calibration and test
            if classifier_name == "logreg":
                # Need to refit scaler on training data only
                scaler = StandardScaler().fit(X_tr)
                proba_cal  = clf.predict_proba(scaler.transform(X_cal))[:, 1]
                proba_test = clf.predict_proba(scaler.transform(X_te))[:, 1]
            else:
                proba_cal  = clf.predict_proba(X_cal)[:, 1]
                proba_test = clf.predict_proba(X_te)[:, 1]

            best_thr, _ = best_f1_threshold(y_cal, proba_cal)
            y_pred_recal = (proba_test >= best_thr).astype(int)
            m = metrics_dict(y_te, y_pred_recal, proba_test)
            row[eval_model] = round(m["f1"], 4)
            full_records.append({"train_model": train_model, "eval_model": eval_model,
                                 "best_threshold": round(best_thr, 4),
                                 **{k: round(v, 4) for k, v in m.items()}})
            print(f"  train={train_model:25s} eval={eval_model:25s} "
                  f"thr={best_thr:.3f}  F1_recal={m['f1']:.4f}  AUC={m['auc']:.4f}")
        matrix.append(row)
    matrix_df = pd.DataFrame(matrix).set_index("train_model")
    matrix_df.to_csv(RESULTS_DIR / "task_b_recalibrated_matrix.csv")
    pd.DataFrame(full_records).to_csv(
        RESULTS_DIR / "task_b_recalibrated_full.csv", index=False)
    diag = [matrix_df.loc[m, m] for m in llm_models if m in matrix_df.columns]
    off  = [matrix_df.loc[a, b] for a in llm_models for b in llm_models
            if a != b and a in matrix_df.index and b in matrix_df.columns]
    print(f"\n[OK] Recalibrated matrix:")
    print(matrix_df.round(4))
    if diag and off:
        gap = (np.mean(diag) - np.mean(off)) * 100
        print(f"[OK] Mean diagonal F1 (recal)     : {np.mean(diag):.4f}")
        print(f"[OK] Mean off-diagonal F1 (recal) : {np.mean(off):.4f}")
        print(f"[OK] Recalibrated gap             : {gap:.2f} pp")
    return matrix_df

def run_task_c(df, llm_models, classifier_name="xgboost"):
    print("\n" + "=" * 70)
    print("TASK C — Cross-dataset human verification")
    print("=" * 70)
    human_sources = get_human_sources(df)
    all_llms = df[df["label_origin"] == "llm"].copy()
    rows = []
    for src_train in human_sources:
        for src_test in human_sources:
            if src_train == src_test:
                continue
            h_tr = df[(df["label_origin"] == "human") & (df["source"] == src_train)]
            h_te = df[(df["label_origin"] == "human") & (df["source"] == src_test)]
            if len(h_tr) < 50 or len(h_te) < 50:
                continue
            train_df = pd.concat([h_tr, all_llms], ignore_index=True)
            test_df  = pd.concat([h_te, all_llms], ignore_index=True)
            X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["y"].values
            X_te, y_te = test_df[FEATURE_NAMES].values,  test_df["y"].values
            _, y_pred, y_proba = fit_predict(X_tr, y_tr, X_te, classifier_name)
            m = metrics_dict(y_te, y_pred, y_proba)
            rows.append({"train_human_source": src_train, "test_human_source": src_test,
                         "classifier": classifier_name,
                         **{k: round(v, 4) for k, v in m.items()}})
            print(f"  train={src_train:25s} test={src_test:25s} F1={m['f1']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "task_c_cross_dataset_human.csv", index=False)
    return out

def run_task_d(df, llm_models, classifier_name="xgboost"):
    print("\n" + "=" * 70)
    print(f"TASK D — Aggregated-pool detector ({classifier_name})")
    print("=" * 70)
    humans = df[df["label_origin"] == "human"].copy()
    h_tr, h_te = train_test_split(humans, test_size=0.2,
                                  random_state=RANDOM_SEED,
                                  stratify=humans["source"])
    rows = []
    for held_out in llm_models:
        all_llms = df[df["label_origin"] == "llm"].copy()
        train_df = pd.concat([h_tr, all_llms], ignore_index=True)
        X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["y"].values
        llm_rows = df[(df["label_origin"] == "llm") & (df["source"] == held_out)]
        test_df  = pd.concat([h_te, llm_rows], ignore_index=True)
        X_te, y_te = test_df[FEATURE_NAMES].values, test_df["y"].values
        _, y_pred, y_proba = fit_predict(X_tr, y_tr, X_te, classifier_name)
        m = metrics_dict(y_te, y_pred, y_proba)
        rows.append({"eval_llm": held_out, "classifier": classifier_name,
                     **{k: round(v, 4) for k, v in m.items()}})
        print(f"  pooled-train, eval={held_out:25s} F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "task_d_aggregated.csv", index=False)
    return out

def run_feature_importance(df, llm_models):
    print("\n" + "=" * 70)
    if HAVE_SHAP:
        print("FEATURE IMPORTANCE — SHAP per intra-model XGBoost")
    else:
        print("FEATURE IMPORTANCE — Permutation Importance (SHAP not installed)")
    print("=" * 70)

    humans = df[df["label_origin"] == "human"].copy()
    rows = []
    for model in llm_models:
        llm_rows = df[(df["label_origin"] == "llm") & (df["source"] == model)]
        sub = pd.concat([humans, llm_rows], ignore_index=True)
        X, y = sub[FEATURE_NAMES].values, sub["y"].values
        clf = make_xgb()
        clf.fit(X, y)

        n_sample = min(1000, len(sub))
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(sub), size=n_sample, replace=False)

        if HAVE_SHAP:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X[idx])
            mean_abs = np.abs(shap_vals).mean(axis=0)
            metric_name = "mean_abs_shap"
        else:
            # Permutation importance — model-agnostic, no SHAP needed
            result = permutation_importance(
                clf, X[idx], y[idx],
                n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
            mean_abs = result.importances_mean
            metric_name = "permutation_importance"

        for fname, val in zip(FEATURE_NAMES, mean_abs):
            rows.append({"llm_model": model, "feature": fname,
                         metric_name: float(val)})
        top5 = np.array(FEATURE_NAMES)[np.argsort(-mean_abs)[:5]]
        print(f"  {model}: top-5 = {', '.join(top5)}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print(f"\n[OK] Saved -> {RESULTS_DIR / 'feature_importance.csv'}")

    # Stable vs model-specific
    metric_col = [c for c in out.columns if c not in ("llm_model", "feature")][0]
    top5_per_model = {}
    for model in llm_models:
        sub = out[out["llm_model"] == model].nlargest(5, metric_col)
        top5_per_model[model] = set(sub["feature"])
    stable = set.intersection(*top5_per_model.values()) if top5_per_model else set()
    only_one = set()
    for model, feats in top5_per_model.items():
        only_in_this = feats - set.union(*(s for m, s in top5_per_model.items() if m != model))
        only_one |= only_in_this
    print(f"\n[STABLE features in top-5 across all 3 models]: {sorted(stable)}")
    print(f"[MODEL-SPECIFIC top-5 features              ]: {sorted(only_one)}")
    return out

def main():
    df = load_features()
    llm_models    = get_llm_models(df)
    human_sources = get_human_sources(df)
    print(f"\n[INFO] LLM models found       : {llm_models}")
    print(f"[INFO] Human sources found    : {human_sources}")
    print(f"[INFO] Per-class counts:")
    print(df.groupby(["label_origin", "source"]).size().to_string())

    if len(llm_models) < 2:
        print("\n[ERROR] Need at least 2 distinct LLM sources for cross-model.")
        sys.exit(1)

    run_task_a(df, llm_models)
    matrix_b, summary_b, h_tr, h_te = run_task_b(df, llm_models, "xgboost")
    run_task_b_recalibrated(df, llm_models, h_tr, h_te, "xgboost")
    run_task_c(df, llm_models, "xgboost")
    run_task_d(df, llm_models, "xgboost")
    run_feature_importance(df, llm_models)

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETE")
    print("=" * 70)
    print(f"Results saved under: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()