#!/usr/bin/env python3
"""
Analyze PCA-based feature contributions for VR kinematics experiment
==================================================================

This script reproduces the interpretation workflow described in the ChatGPT
explanation:

1.  Loads **nested_metrics.pkl** / **final_results.pkl** produced by a previous
    training run.
2.  Re‑builds the best pipeline for every outer‑CV fold, fits it on the original
    training fold and computes **permutation importance** over the validation
    fold → importance per principal component (PC).
3.  Projects PC importance back onto the *original* feature space with the PCA
    loadings and VarianceThreshold mask.
4.  Aggregates importances across folds → mean ± SD.
5.  Saves a CSV of all feature scores and a bar plot of the Top‑k features.

Example usage
-------------

```bash
python analyze_pca_feature_importance.py \
  --results_dir /srv/STP/results_relevance_0.20_redundancy_0.70_study_1_IBT \
  --experiment_label Position_Velocity_Linear \
  --data_dir /srv/STP/experimentData \
  --config_dir /srv/STP/study_1_configs_IBT \
  --output_dir ./analysis_outputs \
  --topk 25
```
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def rel_red_filter(df_wide: pd.DataFrame,
                   feature_cols: list[str],
                   score_col: pd.Series,
                   score_thr: float = 0.20,
                   pearson_thr: float = 0.95,
                   spearman_thr: float = 0.95) -> list[str]:
    """Reproduce the relevance‑redundancy filter used during training."""
    # 1. relevance wrt score
    corr_with_score = df_wide[feature_cols].corrwith(score_col, method="pearson").abs()
    relevant_cols = corr_with_score[corr_with_score >= score_thr].index.tolist()

    if len(relevant_cols) <= 1:
        return relevant_cols

    # 2. redundancy
    corr_p = df_wide[relevant_cols].corr(method="pearson").abs()
    corr_s = df_wide[relevant_cols].corr(method="spearman").abs()

    upper = np.triu_indices_from(corr_p, k=1)
    mask_high = (corr_p.values[upper] > pearson_thr) | (corr_s.values[upper] > spearman_thr)

    to_drop, kept = set(), set()
    for (i, j, hi) in zip(upper[0], upper[1], mask_high):
        if not hi:
            continue
        col_i, col_j = corr_p.columns[i], corr_p.columns[j]
        # precedence: keep col_i if not dropped yet
        if col_j in to_drop or col_i in kept:
            to_drop.add(col_j)
        elif col_i in to_drop or col_j in kept:
            to_drop.add(col_i)
        else:
            kept.add(col_i)
            to_drop.add(col_j)

    return [c for c in relevant_cols if c not in to_drop]


def build_pipeline(best_params: dict, random_state: int = 42) -> Pipeline:
    """Re‑construct the exact pipeline that was trained."""
    var = VarianceThreshold(threshold=0.01)
    scaler = StandardScaler()

    pca_k = best_params["pca_k"]
    pca = PCA(n_components=pca_k, random_state=random_state, svd_solver="full")

    kernel = best_params["model__kernel"]
    C = best_params["model__C"]
    if kernel in {"rbf", "sigmoid"}:
        gamma = best_params["model__gamma"]
    else:
        gamma = "scale"

    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=best_params["model__class_weight"],
        max_iter=best_params["model__max_iter"],
        tol=best_params["model__tol"],
        random_state=random_state,
    )
    return Pipeline([
        ("var", var),
        ("scaler", scaler),
        ("pca", pca),
        ("svc", svc),
    ])


def pc_to_feature_importance(pipe: Pipeline, X_val: np.ndarray, y_val: np.ndarray, featnames: list[str], n_repeats: int = 25, random_state: int = 1) -> np.ndarray:
    """Return importance per original feature given a fitted pipeline."""
    # 1. Transform validation data through the pipeline up to PCA
    X_val_transformed = pipe.named_steps["pca"].transform(
        pipe.named_steps["scaler"].transform(
            pipe.named_steps["var"].transform(X_val)
        )
    )
    
    # 2. permutation importance over PCs using only the SVC classifier
    pi = permutation_importance(
        pipe.named_steps["svc"], X_val_transformed, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="balanced_accuracy",
    )
    pc_imp = pi.importances_mean              # (k,)

    # 3. Back‑project
    loadings = pipe.named_steps["pca"].components_.T  # (n_feats_after_var, k)
    feat_imp = np.abs(loadings) @ pc_imp             # (n_feats_after_var,)

    # Recover original‑feature names after VarThresh mask
    mask_var = pipe.named_steps["var"].get_support()
    full_imp = np.zeros(len(featnames))              # init with zeros for dropped feats
    full_imp[mask_var] = feat_imp

    full_imp /= full_imp.sum()                       # normalise to 1
    return full_imp


# ────────────────────────────────────────────────────────────────────────────
# Main script
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA feature importance aggregator")
    parser.add_argument("--results_dir", required=True, help="Dir with *_nested_metrics.pkl & *_final_results.pkl")
    parser.add_argument("--experiment_label", required=True, help="Basename of the experiment (without extension)")
    parser.add_argument("--data_dir", required=True, help="Directory containing the step_df pickles & score pickle")
    parser.add_argument("--config_dir", required=True, help="Directory holding the JSON feature‑set configs")
    parser.add_argument("--useIBT", default="noIBT", help="Part of the filename pattern (default IBT)")
    parser.add_argument("--topk", type=int, default=20, help="How many features to show in the bar plot")
    parser.add_argument("--output_dir", default="./analysis_outputs", help="Where to save CSV and figures")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load experiment pickles ────────────────────────────────────────────
    nested_path = Path(args.results_dir) / f"{args.experiment_label}_nested_metrics.pkl"
    if not nested_path.exists():
        raise FileNotFoundError(nested_path)

    with open(nested_path, "rb") as f:
        nested_metrics = pickle.load(f)

    outer_results = nested_metrics["outer_results"]

    # ── Load feature‑set config ───────────────────────────────────────────
    cfg_path = Path(args.config_dir) / f"{args.experiment_label}.json"
    with open(cfg_path, "r") as f:
        feature_set = json.load(f)

    # ── Load raw data (same logic as training script) ─────────────────────
    pqdf_path = Path(args.data_dir) / "pos_quat_step_df.pkl"
    with open(pqdf_path, "rb") as f:
        pqdf = pickle.load(f)

    stride_options = [1, 9, 45, 90, 225, 450]
    stride_dict = {1: 0, 9: 1, 45: 2, 90: 3, 225: 4, 450: 5}
    step_dfs = {}
    for stride in stride_options:
        step_path = Path(args.data_dir) / f"450shift{stride}_central_{args.useIBT}_step_df.pkl"
        with open(step_path, "rb") as f:
            sdf = pickle.load(f)
            sdf = sdf.merge(pqdf, on=["PID", "task_id"], how="outer")
            step_dfs[stride] = sdf

    # Classification labels
    score_path = Path(args.data_dir) / "step_score.pkl"
    with open(score_path, "rb") as f:
        score_df = pickle.load(f)

    # Build a master y Series (0 = success, 1 = failure) over *all* rows
    for stride in stride_options:
        y = step_dfs[stride].merge(score_df, on=["PID", "task_id"], how="left")["score"].apply(lambda x: 0 if x > 0 else 1)
        step_dfs[stride]["score"] = y

    # ── Apply relevance‑redundancy filter once per stride (global) ─────────
    RELEVANCE_THR = 0.15
    REDUNDANCY_THR = 0.70
    for stride in stride_options:
        filtered_cols = rel_red_filter(
            step_dfs[stride],
            feature_set,
            score_col=step_dfs[stride]["score"],
            score_thr=RELEVANCE_THR,
            pearson_thr=REDUNDANCY_THR,
            spearman_thr=REDUNDANCY_THR,
        )
        step_dfs[stride] = step_dfs[stride][["PID", "task_id", "score", *filtered_cols]]

    # ── Collect per‑fold importance vectors ───────────────────────────────
    importances_fold = {}
    for feat in feature_set:
        importances_fold[feat] = []

    for fold_idx, outer_res in enumerate(tqdm(outer_results, desc="Outer folds")):
        best_params = outer_res["best_params"]
        stride = best_params["stride"]
        df_stride = step_dfs[stride]

        # Split according to the original PIDs of this fold
        pid_train = set(outer_res["PID_train"])
        pid_test  = set(outer_res["PID_test"])
        train_mask = df_stride["PID"].isin(pid_train)
        test_mask  = df_stride["PID"].isin(pid_test)

        featnames = step_dfs[stride].columns.drop(["PID", "task_id", "score"]).tolist()

        X_train = df_stride.loc[train_mask, featnames].to_numpy(dtype=float)
        y_train = df_stride.loc[train_mask, "score"].to_numpy()
        X_test  = df_stride.loc[test_mask,  featnames].to_numpy(dtype=float)
        y_test  = df_stride.loc[test_mask,  "score"].to_numpy()

        # Fit & compute importance
        pipe = build_pipeline(best_params, random_state=args.random_state)
        pipe.fit(X_train, y_train)
        fold_imp = pc_to_feature_importance(pipe, X_test, y_test, featnames, random_state=args.random_state)

        for i, feat in enumerate(featnames):
            # print(f"{feat} {fold_imp[i]}")
            importances_fold[feat].append(fold_imp[i])

    mean_imp = []
    std_imp = []

    for feat in feature_set:
        importances_fold[feat] = np.array(importances_fold[feat])
        if len(importances_fold[feat]) > 0:
            mean_imp.append(np.mean(importances_fold[feat]))
            std_imp.append(np.std(importances_fold[feat]))
        else:
            mean_imp.append(0)
            std_imp.append(0)

    importance_df = (
        pd.DataFrame({"feature": feature_set,
                      "importance_mean": mean_imp,
                      "importance_std": std_imp})
          .sort_values("importance_mean", ascending=False)
          .reset_index(drop=True)
    )

    csv_path = Path(args.output_dir) / f"{args.experiment_label}_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\nSaved full importance table → {csv_path}\n")

    # ── Bar plot of Top‑k ────────────────────────────────────────────────
    topk = min(args.topk, len(importance_df))
    top_df = importance_df.head(topk)[::-1]  # reverse for nicer horizontal plot

    plt.figure(figsize=(6, 0.35 * topk + 1))
    plt.barh(top_df["feature"], top_df["importance_mean"])
    plt.xlabel("Relative importance (mean across folds)")
    plt.title(f"Top‑{topk} feature contributions – {args.experiment_label}")
    plt.tight_layout()
    fig_path = Path(args.output_dir) / f"{args.experiment_label}_top{topk}_features.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved bar plot → {fig_path}\n")


if __name__ == "__main__":
    main()
