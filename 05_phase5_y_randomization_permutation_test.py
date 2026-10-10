#!/usr/bin/env python
r"""
05_phase5_y_randomization_permutation_test.py

"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase5_PermutationTest")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "Data"
WT_DIR = DATA_DIR / "Fold_Feature_Matrices_WholeTumor"
HAB_DIR = DATA_DIR / "Fold_Feature_Matrices_Habitat"
RESULTS_DIR = BASE_DIR / "Results"
TABLES_DIR = RESULTS_DIR / "Tables"
FIG_DIR = RESULTS_DIR / "Figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Dynamically import Phase 3 module functions
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
p3 = importlib.import_module("03_phase3_nested_cv_prediction")


def single_permutation_iteration(
    perm_idx: int,
    perm_seed: int,
    df_wt: List[pd.DataFrame],
    df_hab: List[pd.DataFrame],
    df_merged: List[pd.DataFrame],
    wt_cols: List[str],
    hab_cols: List[str],
    early_cols: List[str],
    candidate_configs: List[Dict[str, Any]],
    canonical_ids: List[str],
    orig_labels: np.ndarray,
    fold_assignments: np.ndarray,
) -> Dict[str, Any]:
    r"""
    Executes a single end-to-end nested CV iteration on permuted outcome labels.
    Completely isolated feature selection, inner tuning, refitting, and out-of-fold scoring.
    """
    t_start = time.time()
    rng = np.random.default_rng(perm_seed)

    # Stratified label permutation: permute labels strictly within each fold to preserve exact balance
    perm_y_arr = np.zeros(len(canonical_ids), dtype=int)
    for f in range(5):
        f_mask = (fold_assignments == f)
        fold_y = orig_labels[f_mask].copy()
        perm_y_arr[f_mask] = rng.permutation(fold_y)

    y_series_perm = pd.Series(perm_y_arr, index=canonical_ids)
    y_true_perm = y_series_perm.values

    # 1. Whole-Tumor Alone (in-fold selection from 293 candidate features)
    p_wt, _, _, _, wt_tr_all, wt_te_all, _ = p3.run_nested_cv(
        df_wt, wt_cols, candidate_configs, y_series_perm, canonical_ids
    )
    auc_wt = float(roc_auc_score(y_true_perm, p_wt))

    # 2. Habitat Alone (in-fold selection from 885 candidate features)
    p_hab, _, _, _, hab_tr_all, hab_te_all, _ = p3.run_nested_cv(
        df_hab, hab_cols, candidate_configs, y_series_perm, canonical_ids
    )
    auc_hab = float(roc_auc_score(y_true_perm, p_hab))

    # 3. Early Fusion (in-fold selection from 1,178 candidate features)
    p_early, _, _, _, _, _, _ = p3.run_nested_cv(
        df_merged, early_cols, candidate_configs, y_series_perm, canonical_ids
    )
    auc_early = float(roc_auc_score(y_true_perm, p_early))

    # 4. Multi-Scale Late Fusion (50% Whole-Tumor + 50% Habitat out-of-fold blend)
    p_blend = pd.Series(index=canonical_ids, dtype=float)
    for f in range(5):
        test_mask = (fold_assignments == f)
        test_ids = [canonical_ids[i] for i in np.where(test_mask)[0]]
        b_te = 0.50 * wt_te_all[f].loc[test_ids].values + 0.50 * hab_te_all[f].loc[test_ids].values
        p_blend.loc[test_ids] = b_te
    auc_late = float(roc_auc_score(y_true_perm, p_blend.values))

    elapsed = time.time() - t_start

    return {
        "permutation_idx": int(perm_idx),
        "seed": int(perm_seed),
        "WholeTumor_Nested": float(auc_wt),
        "Habitat_Alone_Nested": float(auc_hab),
        "Early_Fusion_Merged": float(auc_early),
        "MultiScale_Late_Fusion": float(auc_late),
        "elapsed_sec": float(elapsed),
    }


def compute_dynamic_observed_aucs(
    df_wt: List[pd.DataFrame],
    df_hab: List[pd.DataFrame],
    df_merged: List[pd.DataFrame],
    wt_cols: List[str],
    hab_cols: List[str],
    early_cols: List[str],
    candidate_configs: List[Dict[str, Any]],
    canonical_ids: List[str],
    orig_labels: np.ndarray,
    fold_assignments: np.ndarray,
) -> Dict[str, float]:
    r"""
    Dynamically computes observed baseline AUCs using the identical current CV engine.
    Ensures zero stale or hardcoded results are compared against.
    """
    logger.info("Computing observed baseline AUCs dynamically using current CV engine...")
    y_series_orig = pd.Series(orig_labels, index=canonical_ids)

    # 1. Whole-Tumor Alone
    p_wt_obs, _, _, _, wt_tr_all, wt_te_all, _ = p3.run_nested_cv(
        df_wt, wt_cols, candidate_configs, y_series_orig, canonical_ids
    )
    obs_auc_wt = float(roc_auc_score(orig_labels, p_wt_obs))

    # 2. Habitat Alone
    p_hab_obs, _, _, _, hab_tr_all, hab_te_all, _ = p3.run_nested_cv(
        df_hab, hab_cols, candidate_configs, y_series_orig, canonical_ids
    )
    obs_auc_hab = float(roc_auc_score(orig_labels, p_hab_obs))

    # 3. Early Fusion
    p_early_obs, _, _, _, _, _, _ = p3.run_nested_cv(
        df_merged, early_cols, candidate_configs, y_series_orig, canonical_ids
    )
    obs_auc_early = float(roc_auc_score(orig_labels, p_early_obs))

    # 4. Multi-Scale Late Fusion
    p_blend_obs = pd.Series(index=canonical_ids, dtype=float)
    for f in range(5):
        test_mask = (fold_assignments == f)
        test_ids = [canonical_ids[i] for i in np.where(test_mask)[0]]
        b_te = 0.50 * wt_te_all[f].loc[test_ids].values + 0.50 * hab_te_all[f].loc[test_ids].values
        p_blend_obs.loc[test_ids] = b_te
    obs_auc_late = float(roc_auc_score(orig_labels, p_blend_obs.values))

    observed = {
        "WholeTumor_Nested": obs_auc_wt,
        "Habitat_Alone_Nested": obs_auc_hab,
        "Early_Fusion_Merged": obs_auc_early,
        "MultiScale_Late_Fusion": obs_auc_late,
    }

    logger.info("Dynamically Computed Observed AUCs on Current Pipeline (max_iter=10000):")
    for k, v in observed.items():
        logger.info(f"  - {k:25s}: {v:.4f}")

    return observed


def run_y_randomization_experiment(
    n_permutations: int = 200,
    n_jobs: int = 6,
    random_seed: int = 2026,
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Orchestrates the parallel Y-randomization experiment with incremental disk checkpointing.
    Guarantees zero progress loss upon pause, interruption, or crash.
    """
    logger.info("=" * 115)
    logger.info(f"STARTING Y-RANDOMIZATION PERMUTATION TEST (N = {n_permutations} ITERATIONS, N_JOBS = {n_jobs})")
    logger.info("Null Hypothesis: Mean AUC ~ 0.500 when outcome labels are randomly permuted across the cohort.")
    logger.info("=" * 115)

    df_wt = [pd.read_csv(WT_DIR / f"fold_{f}_features.csv").set_index("lesion_id") for f in range(5)]
    df_hab = [pd.read_csv(HAB_DIR / f"fold_{f}_features.csv").set_index("lesion_id") for f in range(5)]
    canonical_ids = list(df_wt[0].index)

    # Reindex all DataFrames strictly to canonical_ids
    for f in range(5):
        df_wt[f] = df_wt[f].reindex(canonical_ids)
        df_hab[f] = df_hab[f].reindex(canonical_ids)

    df0 = df_wt[0]
    orig_labels = df0["Status"].str.strip().str.lower().map({"enlargement": 1, "stable": 0, "shrinkage": 0}).values
    fold_assignments = df0["fold"].values

    wt_cols = [c for c in df0.columns if c.startswith("WholeTumor_") and not any(p in c for p in p3.NON_RADIOMIC)]
    hab_cols = [c for c in df_hab[0].columns if (c.startswith(("Hab1_", "Hab2_", "Hab3_")) or "Volume_Fraction" in c or "ILR" in c) and not any(p in c for p in p3.NON_RADIOMIC)]
    early_cols = wt_cols + hab_cols

    df_merged = []
    for f in range(5):
        m_f = pd.concat([df_wt[f][wt_cols], df_hab[f][hab_cols]], axis=1)
        m_f["fold"] = df_wt[f]["fold"]
        m_f["Status"] = df_wt[f]["Status"]
        df_merged.append(m_f)

    candidate_configs = []
    for k in [4, 6, 8]:
        for fam in ["ridge", "linear_svm"]:
            for C in [0.5, 1.0, 2.0]:
                candidate_configs.append({"k": k, "family": fam, "C": C})

    # Dynamically compute observed benchmark values from real clinical labels
    observed_aucs = compute_dynamic_observed_aucs(
        df_wt, df_hab, df_merged, wt_cols, hab_cols, early_cols,
        candidate_configs, canonical_ids, orig_labels, fold_assignments
    )

    raw_csv_path = TABLES_DIR / "production_y_randomization_permutation_raw_iterations.csv"

    # Step 1: Inspect disk checkpoint
    existing_records: List[Dict[str, Any]] = []
    completed_indices = set()

    expected_cols = ["permutation_idx", "seed", "WholeTumor_Nested", "Habitat_Alone_Nested", "Early_Fusion_Merged", "MultiScale_Late_Fusion"]

    if resume and raw_csv_path.exists():
        try:
            prev_df = pd.read_csv(raw_csv_path)
            if all(col in prev_df.columns for col in expected_cols) and len(prev_df) > 0:
                prev_df = prev_df.drop_duplicates(subset=["permutation_idx"], keep="last")
                prev_df = prev_df[prev_df["permutation_idx"].between(1, n_permutations)]
                existing_records = prev_df.to_dict("records")
                completed_indices = set(prev_df["permutation_idx"].astype(int))
                logger.info(f"Resume active: Found {len(completed_indices)} completed permutations in checkpoint CSV.")
        except Exception as e:
            logger.warning(f"Could not read previous raw checkpoint CSV ({e}); starting fresh.")
            existing_records = []
            completed_indices = set()

    if not resume or not raw_csv_path.exists() or len(existing_records) == 0:
        # Create empty template CSV
        init_df = pd.DataFrame(columns=expected_cols + ["elapsed_sec"])
        init_df.to_csv(raw_csv_path, index=False)
        existing_records = []
        completed_indices = set()

    # Determine pending iteration indices
    pending_indices = [idx for idx in range(1, n_permutations + 1) if idx not in completed_indices]
    logger.info(f"Experiment scope: {n_permutations} total permutations. Already finished: {len(completed_indices)}. Pending: {len(pending_indices)}.")

    if len(pending_indices) > 0:
        n_workers = joblib.cpu_count() if n_jobs == -1 else n_jobs
        logger.info(f"Launching remaining {len(pending_indices)} iterations across {n_workers} workers (batch_size=1, generator mode)...")
        start_time = time.time()

        # Joblib Parallel generator: yields results immediately as each worker completes a task
        parallel_engine = joblib.Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator")
        gen = parallel_engine(
            joblib.delayed(single_permutation_iteration)(
                perm_idx=idx,
                perm_seed=random_seed + (idx - 1),
                df_wt=df_wt,
                df_hab=df_hab,
                df_merged=df_merged,
                wt_cols=wt_cols,
                hab_cols=hab_cols,
                early_cols=early_cols,
                candidate_configs=candidate_configs,
                canonical_ids=canonical_ids,
                orig_labels=orig_labels,
                fold_assignments=fold_assignments,
            )
            for idx in pending_indices
        )

        recent_times: List[float] = []
        for i, res in enumerate(gen, 1):
            existing_records.append(res)

            # Append immediately to disk CSV and flush
            row_df = pd.DataFrame([res])
            row_df.to_csv(raw_csv_path, mode="a", header=False, index=False)

            recent_times.append(res["elapsed_sec"])
            if len(recent_times) > 10:
                recent_times.pop(0)
            avg_t = float(np.mean(recent_times))
            rem_tasks = len(pending_indices) - i
            eta_min = (rem_tasks * avg_t / max(1, n_jobs)) / 60.0

            total_done = len(existing_records)
            pct = 100.0 * total_done / n_permutations
            logger.info(
                f"[Iter {res['permutation_idx']:3d}/{n_permutations}] took {res['elapsed_sec']:4.1f}s | "
                f"Null AUCs: WT={res['WholeTumor_Nested']:.4f}, Hab={res['Habitat_Alone_Nested']:.4f}, "
                f"Early={res['Early_Fusion_Merged']:.4f}, Late={res['MultiScale_Late_Fusion']:.4f} | "
                f"Overall: {total_done}/{n_permutations} ({pct:4.1f}%) | ETA: ~{eta_min:4.1f} min"
            )

        elapsed = time.time() - start_time
        logger.info(f"Completed all pending permutations in {elapsed / 60.0:.1f} minutes.")

    # Deduplicate and sort all iterations
    perm_df = pd.DataFrame(existing_records).drop_duplicates(subset=["permutation_idx"]).sort_values("permutation_idx").reset_index(drop=True)
    perm_df.to_csv(raw_csv_path, index=False)
    logger.info(f"Verified {len(perm_df)} total completed permutations in: {raw_csv_path.name}")

    # Compute empirical p-values and summary statistics
    summary_rows = []
    for model_key, obs_auc in observed_aucs.items():
        null_vals = perm_df[model_key].values
        n_exceed = int(np.sum(null_vals >= obs_auc))
        p_emp = float((1 + n_exceed) / (1 + len(perm_df)))
        null_mean = float(np.mean(null_vals))
        null_std = float(np.std(null_vals))
        z_score = float((obs_auc - null_mean) / (null_std if null_std > 1e-6 else 1.0))
        q95 = float(np.percentile(null_vals, 95))
        q99 = float(np.percentile(null_vals, 99))
        max_null = float(np.max(null_vals))

        summary_rows.append({
            "Model_Key": model_key,
            "Observed_AUC": obs_auc,
            "Null_Mean_AUC": null_mean,
            "Null_SD_AUC": null_std,
            "Null_95th_Percentile": q95,
            "Null_99th_Percentile": q99,
            "Max_Null_AUC": max_null,
            "Exceeded_Count": n_exceed,
            "Total_Permutations": len(perm_df),
            "Permutation_p_value": p_emp,
            "Z_Score": z_score,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "production_y_randomization_permutation_summary.csv", index=False)
    summary_df.to_excel(TABLES_DIR / "production_y_randomization_permutation_summary.xlsx", index=False)

    return perm_df, summary_df


def generate_permutation_distribution_figure(
    perm_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):
    r"""
    Renders publication-grade 3-panel figure showing empirical null distributions.
    """
    logger.info("Generating publication figure for Y-randomization null distributions...")

    models_to_plot = [
        ("WholeTumor_Nested", "Whole-Tumor Radiomics (Nested CV)", "#2E7D32", "(A)"),
        ("Habitat_Alone_Nested", "Habitat Radiomics (Nested CV)", "#EF6C00", "(B)"),
        ("MultiScale_Late_Fusion", "Multi-Scale Late Fusion (Ensemble)", "#C62828", "(C)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for idx, (m_key, title_name, col_color, panel_tag) in enumerate(models_to_plot):
        ax = axes[idx]
        null_vals = perm_df[m_key].values
        row = summary_df[summary_df["Model_Key"] == m_key].iloc[0]
        obs_auc = row["Observed_AUC"]
        null_mean = row["Null_Mean_AUC"]
        p_val = row["Permutation_p_value"]
        q95 = row["Null_95th_Percentile"]

        # Plot Null Distribution Histogram & KDE
        sns.histplot(
            null_vals, bins=16, kde=True, ax=ax,
            color="#9E9E9E", edgecolor="black", alpha=0.55, stat="density",
            line_kws={"linewidth": 2.0, "color": "#424242"}
        )

        # Theoretical chance line at 0.50
        ax.axvline(0.50, color="black", linestyle=":", linewidth=1.5, alpha=0.8, label="Theoretical Chance (0.500)")

        # Empirical 95th percentile alpha threshold
        ax.axvline(q95, color="#FF9800", linestyle="--", linewidth=1.8, label=f"Null 95% Cutoff ({q95:.3f})")

        # Observed test AUC vertical marker
        ax.axvline(obs_auc, color=col_color, linestyle="-", linewidth=2.8, label=f"Observed Test AUC ({obs_auc:.3f})")

        # Annotations
        stats_text = (
            f"Observed AUC: {obs_auc:.4f}\n"
            f"Null Mean: {null_mean:.3f} \u00b1 {row['Null_SD_AUC']:.3f}\n"
            f"Exceeded: {int(row['Exceeded_Count'])} / {int(row['Total_Permutations'])}\n"
            f"Permutation p = {p_val:.4f}"
        )
        ax.text(
            0.05, 0.93, stats_text, transform=ax.transAxes,
            fontsize=9.5, fontweight="semibold", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#BDBDBD", alpha=0.92)
        )

        ax.set_title(f"{panel_tag} {title_name}", fontsize=11.5, fontweight="bold", pad=12)
        ax.set_xlabel("Cross-Validated ROC-AUC", fontsize=10, fontweight="semibold")
        ax.set_ylabel("Probability Density", fontsize=10, fontweight="semibold")
        ax.set_xlim([0.30, 0.85])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8.2, frameon=True)

    fig.suptitle(
        "Target Y-Randomization Permutation Test: Evaluation of Predictive Association Against Empirical Null Distribution",
        fontsize=13.5, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_png = FIG_DIR / "Figure_Permutation_Test_Null_Distribution.png"
    out_pdf = FIG_DIR / "Figure_Permutation_Test_Null_Distribution.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_png.name} and {out_pdf.name}")


def main():
    parser = argparse.ArgumentParser(description="Y-Randomization Permutation Test (Bullet-Proof Engine)")
    parser.add_argument("--n_permutations", type=int, default=200, help="Number of label permutations (default: 200)")
    parser.add_argument("--n_jobs", type=int, default=6, help="Parallel CPU workers (default: 6)")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed (default: 2026)")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume; start clean from iteration 1")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2-iteration smoke test to verify pipeline")
    args = parser.parse_args()

    n_perms = 2 if args.smoke_test else args.n_permutations
    resume = False if (args.no_resume or args.smoke_test) else True

    perm_df, summary_df = run_y_randomization_experiment(
        n_permutations=n_perms,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
        resume=resume,
    )

    generate_permutation_distribution_figure(perm_df, summary_df)

    print("\n" + "=" * 115)
    print("Y-RANDOMIZATION PERMUTATION TEST RESULTS SUMMARY")
    print("=" * 115)
    cols = ["Model_Key", "Observed_AUC", "Null_Mean_AUC", "Null_SD_AUC", "Null_95th_Percentile", "Exceeded_Count", "Total_Permutations", "Permutation_p_value", "Z_Score"]
    print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPermutation test execution complete.")


if __name__ == "__main__":
    main()
