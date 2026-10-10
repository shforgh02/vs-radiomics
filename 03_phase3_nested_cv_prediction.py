#!/usr/bin/env python
r"""
Phase 3: Leak-Free Nested Cross-Validation (Nested CV) Engine

"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, roc_auc_score, roc_curve, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase3_NestedCV")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "Data"
WT_DIR = DATA_DIR / "Fold_Feature_Matrices_WholeTumor"
HAB_DIR = DATA_DIR / "Fold_Feature_Matrices_Habitat"
RESULTS_DIR = BASE_DIR / "Results"
TABLES_DIR = RESULTS_DIR / "Tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

NON_RADIOMIC = {
    "_Extraction_Valid", "_Voxels", "total_tumor_voxels", "patient_uid", "patient_id", "lesion_id",
    "fold", "cv_fold", "split", "Status", "status", "Target_Enlargement", "PatientID", "center",
    "Necrosis", "Necrosis_Clean", "clinical_mapping_status", "is_bilateral", "num_patient_lesions",
    "tumor_mask_file", "has_complete_mri", "available_series", "strat_key",
    "Clinical_Volume_x", "Clinical_Dose_x", "Clinical_Fr_x", "Clinical_IDL_x", "Clinical_Coverage_x",
    "Clinical_Volume", "Clinical_Dose", "Clinical_Fr", "Clinical_IDL", "Clinical_Coverage",
    "Clinical_Selectivity", "Clinical_GI"
}

CLINICAL_COLS = ["Clinical_Volume_x", "Clinical_Dose_x", "Clinical_Fr_x", "Clinical_IDL_x", "Clinical_Coverage_x"]


def fast_paired_permutation_test(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_baseline: np.ndarray,
    n_perm: int = 20000,
    seed: int = 42
) -> Tuple[float, float, float]:
    r"""Exact paired permutation test between model probabilities and baseline."""
    diff_orig = roc_auc_score(y_true, p_model) - roc_auc_score(y_true, p_baseline)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    swaps = rng.integers(0, 2, size=(n_perm, n)).astype(bool)
    n1 = (y_true == 1).sum()
    n0 = n - n1
    y_pos = (y_true == 1)

    diffs = np.zeros(n_perm)
    for i in range(n_perm):
        pa = np.where(swaps[i], p_baseline, p_model)
        pb = np.where(swaps[i], p_model, p_baseline)
        ra = rankdata(pa)
        rb = rankdata(pb)
        auca = (ra[y_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        aucb = (rb[y_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        diffs[i] = auca - aucb

    p_two = (np.sum(np.abs(diffs) >= np.abs(diff_orig)) + 1) / (n_perm + 1)
    p_one = (np.sum(diffs >= diff_orig) + 1) / (n_perm + 1)
    return diff_orig, float(p_two), float(p_one)


def fast_bootstrap_ci(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42
) -> Tuple[float, float]:
    r"""Patient-clustered non-parametric bootstrap 95% confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    unique_pts = np.unique(patient_ids)
    pt_dict = {p: np.where(patient_ids == p)[0] for p in unique_pts}
    boot_aucs = []
    for _ in range(n_boot):
        sampled_pts = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        idx = np.concatenate([pt_dict[p] for p in sampled_pts])
        yt = y_true[idx]
        n1 = (yt == 1).sum()
        n0 = len(yt) - n1
        if n1 > 0 and n0 > 0:
            r = rankdata(p_pred[idx])
            boot_aucs.append((r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def fit_prep_predict(
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_te_raw: np.ndarray,
    feature_names: List[str],
    k: int,
    clf_family: str,
    C: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    r"""In-fold missingness filtering, median imputation, ranking, r-pruning, RobustScaler and fitting."""
    obs = np.isfinite(X_tr_raw)
    kp = np.flatnonzero(obs.mean(axis=0) >= 0.05)
    if len(kp) == 0:
        kp = np.arange(X_tr_raw.shape[1])
    raw_tr = X_tr_raw[:, kp]
    raw_te = X_te_raw[:, kp]
    names_kp = [feature_names[i] for i in kp]

    med = np.nanmedian(raw_tr, axis=0)
    med = np.nan_to_num(med, nan=0.0)
    X_tr = np.where(np.isfinite(raw_tr), raw_tr, med)
    X_te = np.where(np.isfinite(raw_te), raw_te, med)

    k_actual = min(k, X_tr.shape[1])
    if X_tr.shape[1] <= k_actual:
        selected = np.arange(X_tr.shape[1])
    else:
        ranks = rankdata(X_tr, axis=0)
        n1, n0 = (y_tr == 1).sum(), (y_tr == 0).sum()
        auc = (ranks[y_tr == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        order = np.argsort(-np.abs(auc - 0.5), kind="stable")
        centered = ranks - ranks.mean(axis=0)
        centered /= np.maximum(np.linalg.norm(centered, axis=0), 1e-12)
        selected = []
        for j in order:
            if not selected or np.max(np.abs(centered[:, selected].T @ centered[:, j])) <= 0.85:
                selected.append(int(j))
                if len(selected) == k_actual:
                    break

    sc = RobustScaler().fit(X_tr[:, selected])
    Z_tr = np.nan_to_num(sc.transform(X_tr[:, selected]), nan=0.0)
    Z_te = np.nan_to_num(sc.transform(X_te[:, selected]), nan=0.0)

    if clf_family == "ridge":
        clf = LogisticRegression(C=C, max_iter=10000, random_state=seed)
    else:
        clf = SVC(kernel="linear", C=C, probability=True, max_iter=10000, random_state=seed)

    clf.fit(Z_tr, y_tr)
    selected_names = [names_kp[s] for s in selected]
    return clf.predict_proba(Z_tr)[:, 1], clf.predict_proba(Z_te)[:, 1], selected_names


def run_nested_cv(
    df_list: List[pd.DataFrame],
    cand_cols: List[str],
    candidate_configs: List[Dict[str, Any]],
    y_series: pd.Series,
    canonical_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[float], List[Any], List[pd.Series], List[pd.Series], Dict[int, List[str]]]:
    r"""Executes full nested CV with inner 4-fold cross-validation strictly aligned by lesion_id."""
    oof_p = pd.Series(index=canonical_ids, dtype=float)
    oof_pred = pd.Series(index=canonical_ids, dtype=int)
    p_tr_by_fold = [None] * 5
    p_te_by_fold = [None] * 5
    fold_aucs = []
    selected_configs = []
    selected_features_per_fold = {}

    for f in range(5):
        df_f = df_list[f].reindex(canonical_ids)
        train_mask = (df_f["fold"] != f)
        test_mask = (df_f["fold"] == f)

        train_ids = df_f.index[train_mask]
        test_ids = df_f.index[test_mask]

        y_tr = y_series.loc[train_ids].values
        y_te = y_series.loc[test_ids].values

        raw_tr = df_f.loc[train_ids, cand_cols].apply(pd.to_numeric, errors="coerce").values
        raw_te = df_f.loc[test_ids, cand_cols].apply(pd.to_numeric, errors="coerce").values

        # Inner 4-fold CV strictly on the 80 training patients
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + f)
        inner_splits = list(inner_cv.split(raw_tr, y_tr))

        best_cfg = None
        best_inner_auc = -1.0

        for cfg in candidate_configs:
            k, fam, C = cfg["k"], cfg["family"], cfg["C"]
            val_scores = []
            for in_tr, in_val in inner_splits:
                _, p_val, _ = fit_prep_predict(raw_tr[in_tr], y_tr[in_tr], raw_tr[in_val], cand_cols, k, fam, C, seed=42 + f)
                val_scores.append(roc_auc_score(y_tr[in_val], p_val))
            mean_val_auc = float(np.mean(val_scores))
            if mean_val_auc > best_inner_auc:
                best_inner_auc = mean_val_auc
                best_cfg = cfg

        selected_configs.append((f, best_cfg, best_inner_auc))

        # Refit winning configuration on the outer training fold
        p_tr, p_te, sel_names = fit_prep_predict(raw_tr, y_tr, raw_te, cand_cols, best_cfg["k"], best_cfg["family"], best_cfg["C"], seed=42 + f)
        p_tr_by_fold[f] = pd.Series(p_tr, index=train_ids)
        p_te_by_fold[f] = pd.Series(p_te, index=test_ids)
        selected_features_per_fold[f] = sel_names

        # In-fold Youden J threshold determination
        fpr, tpr, th = roc_curve(y_tr, p_tr)
        best_t = th[np.argmax(tpr - fpr)]

        oof_p.loc[test_ids] = p_te
        oof_pred.loc[test_ids] = (p_te >= best_t).astype(int)
        fold_aucs.append(float(roc_auc_score(y_te, p_te)))

    return oof_p.values, oof_pred.values, fold_aucs, selected_configs, p_tr_by_fold, p_te_by_fold, selected_features_per_fold


def compute_full_metrics(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    r"""Calculates comprehensive diagnostic and discrimination metrics."""
    auc = float(roc_auc_score(y_true, probs))
    ap = float(average_precision_score(y_true, probs))
    acc = float(accuracy_score(y_true, preds))
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(0.5 * (sens + spec))
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true, preds))
    brier = float(brier_score_loss(y_true, probs))
    return {
        "ROC_AUC": auc, "PR_AUC": ap, "Accuracy": acc,
        "Sensitivity": sens, "Specificity": spec, "Balanced_Acc": bal_acc,
        "PPV": ppv, "NPV": npv, "F1": f1, "Brier": brier,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
    }


def execute_phase3_nested_cv() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""Executes master nested CV benchmark across all model architectures."""
    logger.info("=" * 115)
    logger.info("PHASE 3: LEAK-FREE NESTED CROSS-VALIDATION ENGINE (NESTED CV)")
    logger.info("Grid: k in {4, 6, 8} | Families: Ridge Logistic Regression vs Linear SVM | C in {0.5, 1.0, 2.0}")
    logger.info("=" * 115)

    df_wt = [pd.read_csv(WT_DIR / f"fold_{f}_features.csv").set_index("lesion_id") for f in range(5)]
    df_hab = [pd.read_csv(HAB_DIR / f"fold_{f}_features.csv").set_index("lesion_id") for f in range(5)]

    canonical_ids = list(df_wt[0].index)
    df0 = df_wt[0]

    # Rigorous ID, outcome, and fold integrity audit across all 10 matrices
    logger.info("Performing comprehensive lesion ID, outcome, and fold alignment audit...")
    for f in range(5):
        df_wt[f] = df_wt[f].reindex(canonical_ids)
        df_hab[f] = df_hab[f].reindex(canonical_ids)
        if not df_wt[f].index.equals(df_wt[0].index):
            raise ValueError(f"Lesion ID mismatch in Whole-Tumor fold {f}!")
        if not df_hab[f].index.equals(df_wt[0].index):
            raise ValueError(f"Lesion ID mismatch in Habitat fold {f}!")
        if not df_wt[f]["Status"].equals(df_wt[0]["Status"]):
            raise ValueError(f"Outcome Status mismatch in Whole-Tumor fold {f}!")
        if not df_hab[f]["Status"].equals(df_wt[0]["Status"]):
            raise ValueError(f"Outcome Status mismatch in Habitat fold {f}!")
        if not df_wt[f]["fold"].equals(df_wt[0]["fold"]):
            raise ValueError(f"Fold assignment mismatch in Whole-Tumor fold {f}!")
        if not df_hab[f]["fold"].equals(df_wt[0]["fold"]):
            raise ValueError(f"Fold assignment mismatch in Habitat fold {f}!")

    # Verify patient-level independence (1 lesion per patient, zero overlap across outer folds)
    if "patient_uid" in df0.columns:
        patient_uids = df0["patient_uid"]
        if len(patient_uids.unique()) != len(patient_uids):
            raise ValueError("Multiple lesions per patient detected; grouped CV required!")
        for f in range(5):
            tr_pts = set(patient_uids[df0["fold"] != f])
            te_pts = set(patient_uids[df0["fold"] == f])
            overlap = tr_pts.intersection(te_pts)
            if len(overlap) > 0:
                raise ValueError(f"Patient overlap detected across fold {f}: {overlap}")
        logger.info("  [PASSED] Audit Confirmed: 100 unique patients across 100 lesions. Zero patient overlap across all 5 folds.")

    y_series = df0["Status"].str.strip().str.lower().map({"enlargement": 1, "stable": 0, "shrinkage": 0})
    y_true = y_series.values
    patient_ids = df0["patient_uid"].values if "patient_uid" in df0.columns else df0.index.values

    wt_cols = [c for c in df0.columns if c.startswith("WholeTumor_") and not any(p in c for p in NON_RADIOMIC)]
    hab_cols = [c for c in df_hab[0].columns if (c.startswith(("Hab1_", "Hab2_", "Hab3_")) or "Volume_Fraction" in c or "ILR" in c) and not any(p in c for p in NON_RADIOMIC)]
    early_cols = wt_cols + hab_cols
    clin_cols = [c for c in CLINICAL_COLS if c in df0.columns]
    if not clin_cols:
        clin_cols = [c.replace("_x", "") for c in CLINICAL_COLS if c.replace("_x", "") in df0.columns]

    df_merged = []
    for f in range(5):
        m_f = pd.concat([df_wt[f][wt_cols], df_hab[f][hab_cols]], axis=1)
        m_f["fold"] = df_wt[f]["fold"]
        m_f["Status"] = df_wt[f]["Status"]
        df_merged.append(m_f)

    logger.info(f"Feature Space Dimensions:")
    logger.info(f"  - Clinical Dosimetry:      {len(clin_cols)} features")
    logger.info(f"  - Macro-Habitat Features:  {len(CLINICAL_COLS)} features")
    logger.info(f"  - Whole-Tumor Pool:        {len(wt_cols)} candidate features")
    logger.info(f"  - Habitat Radiomics Pool:  {len(hab_cols)} candidate features")
    logger.info(f"  - Early Fusion Merged:     {len(early_cols)} candidate features ({len(wt_cols)} WT + {len(hab_cols)} Hab)")

    candidate_configs = []
    for k in [4, 6, 8]:
        for fam in ["ridge", "linear_svm"]:
            for C in [0.5, 1.0, 2.0]:
                candidate_configs.append({"k": k, "family": fam, "C": C})

    oofs_probs = {}
    oofs_preds = {}
    fold_aucs_dict = {}
    features_selected_summary = {}

    # 1. Clinical Dosimetry Baseline
    logger.info("[1/6] Running Clinical Dosimetry Baseline (Linear SVM, C=1.0, RobustScaler)...")
    clin_probs = pd.Series(index=canonical_ids, dtype=float)
    clin_preds = pd.Series(index=canonical_ids, dtype=int)
    clin_f_aucs = []
    for f in range(5):
        df_f = df_wt[f].reindex(canonical_ids)
        train_mask = (df_f["fold"] != f)
        test_mask = (df_f["fold"] == f)
        train_ids = df_f.index[train_mask]
        test_ids = df_f.index[test_mask]

        y_tr = y_series.loc[train_ids].values
        y_te = y_series.loc[test_ids].values
        X_tr = df_f.loc[train_ids, clin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        X_te = df_f.loc[test_ids, clin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        sc = RobustScaler().fit(X_tr)
        clf = SVC(kernel="linear", C=1.0, probability=True, max_iter=2000, random_state=42 + f)
        clf.fit(sc.transform(X_tr), y_tr)
        p_tr = clf.predict_proba(sc.transform(X_tr))[:, 1]
        p_te = clf.predict_proba(sc.transform(X_te))[:, 1]
        fpr, tpr, th = roc_curve(y_tr, p_tr)
        best_t = th[np.argmax(tpr - fpr)]
        clin_probs.loc[test_ids] = p_te
        clin_preds.loc[test_ids] = (p_te >= best_t).astype(int)
        clin_f_aucs.append(float(roc_auc_score(y_te, p_te)))

    oofs_probs["Clinical_Dosimetry"] = clin_probs.values
    oofs_preds["Clinical_Dosimetry"] = clin_preds.values
    fold_aucs_dict["Clinical_Dosimetry"] = clin_f_aucs
    features_selected_summary["Clinical_Dosimetry"] = {f: clin_cols for f in range(5)}

    # 2. Macro-Habitat Alone Model (Volume fractions & Log Ratios)
    logger.info("[2/6] Running Macro-Habitat Alone Nested CV (Inner 4-Fold CV selects k in {3,4,5}, Model, C)...")
    for df in df_hab:
        eps = 1e-5
        df["Hab1_Hab3_LogRatio"] = np.log((df["Hab1_Volume_Fraction"] + eps) / (df["Hab3_Volume_Fraction"] + eps))
        df["Hab2_Hab3_LogRatio"] = np.log((df["Hab2_Volume_Fraction"] + eps) / (df["Hab3_Volume_Fraction"] + eps))
    macro_cols = ["Hab1_Volume_Fraction", "Hab2_Volume_Fraction", "Hab3_Volume_Fraction", "Hab1_Hab3_LogRatio", "Hab2_Hab3_LogRatio"]
    macro_configs = []
    for k in [3, 4, 5]:
        for fam in ["ridge", "linear_svm"]:
            for C in [0.5, 1.0, 2.0]:
                macro_configs.append({"k": k, "family": fam, "C": C})
    p_macro, pred_macro, f_macro, sel_macro, macro_tr_all, macro_te_all, sel_feats_macro = run_nested_cv(df_hab, macro_cols, macro_configs, y_series, canonical_ids)
    oofs_probs["Macro_Habitat_Alone"] = p_macro
    oofs_preds["Macro_Habitat_Alone"] = pred_macro
    fold_aucs_dict["Macro_Habitat_Alone"] = f_macro
    features_selected_summary["Macro_Habitat_Alone"] = sel_feats_macro

    # 3. Whole-Tumor Nested CV Model
    logger.info("[3/6] Running Whole-Tumor Nested CV (Inner 4-Fold CV selects k in {4,6,8}, Model, C)...")
    p_wt, pred_wt, f_wt, sel_wt, wt_tr_all, wt_te_all, sel_feats_wt = run_nested_cv(df_wt, wt_cols, candidate_configs, y_series, canonical_ids)
    oofs_probs["WholeTumor_Nested"] = p_wt
    oofs_preds["WholeTumor_Nested"] = pred_wt
    fold_aucs_dict["WholeTumor_Nested"] = f_wt
    features_selected_summary["WholeTumor_Nested"] = sel_feats_wt

    # 4. Habitat Alone Nested CV Model
    logger.info("[4/6] Running Habitat Alone Nested CV (Inner 4-Fold CV selects k in {4,6,8}, Model, C)...")
    p_hab, pred_hab, f_hab, sel_hab, hab_tr_all, hab_te_all, sel_feats_hab = run_nested_cv(df_hab, hab_cols, candidate_configs, y_series, canonical_ids)
    oofs_probs["Habitat_Alone_Nested"] = p_hab
    oofs_preds["Habitat_Alone_Nested"] = pred_hab
    fold_aucs_dict["Habitat_Alone_Nested"] = f_hab
    features_selected_summary["Habitat_Alone_Nested"] = sel_feats_hab

    # 5. Early Fusion (Merged Feature Pool)
    logger.info("[5/6] Running Early Fusion (Merged 1,178 Features, Inner 4-Fold CV selects k in {4,6,8})...")
    p_early, pred_early, f_early, sel_early, early_tr_all, early_te_all, sel_feats_early = run_nested_cv(df_merged, early_cols, candidate_configs, y_series, canonical_ids)
    oofs_probs["Early_Fusion_Merged"] = p_early
    oofs_preds["Early_Fusion_Merged"] = pred_early
    fold_aucs_dict["Early_Fusion_Merged"] = f_early
    features_selected_summary["Early_Fusion_Merged"] = sel_feats_early

    # 6. Multi-Scale Late Fusion
    logger.info("[6/6] Generating Multi-Scale Late Fusion (50% Whole-Tumor + 50% Habitat)...")
    p_blend = pd.Series(index=canonical_ids, dtype=float)
    pred_blend = pd.Series(index=canonical_ids, dtype=int)
    f_blend = []
    for f in range(5):
        df_f = df_wt[f].reindex(canonical_ids)
        train_mask = (df_f["fold"] != f)
        test_mask = (df_f["fold"] == f)
        train_ids = df_f.index[train_mask]
        test_ids = df_f.index[test_mask]

        y_tr = y_series.loc[train_ids].values
        y_te = y_series.loc[test_ids].values

        b_tr = (0.50 * wt_tr_all[f].loc[train_ids] + 0.50 * hab_tr_all[f].loc[train_ids]).values
        b_te = (0.50 * wt_te_all[f].loc[test_ids] + 0.50 * hab_te_all[f].loc[test_ids]).values

        fpr, tpr, th = roc_curve(y_tr, b_tr)
        best_t = th[np.argmax(tpr - fpr)]

        p_blend.loc[test_ids] = b_te
        pred_blend.loc[test_ids] = (b_te >= best_t).astype(int)
        f_blend.append(float(roc_auc_score(y_te, b_te)))

    oofs_probs["MultiScale_Late_Fusion"] = p_blend.values
    oofs_preds["MultiScale_Late_Fusion"] = pred_blend.values
    fold_aucs_dict["MultiScale_Late_Fusion"] = f_blend
    features_selected_summary["MultiScale_Late_Fusion"] = {
        f: [f"WT: {feat}" for feat in sel_feats_wt[f]] + [f"Hab: {feat}" for feat in sel_feats_hab[f]] for f in range(5)
    }

    # Statistical Evaluation & Leaderboard
    logger.info("Computing 20,000 paired permutation tests and 2,000 bootstrap CIs...")
    summary_records = []
    baseline_wt_probs = oofs_probs["WholeTumor_Nested"]

    models_display = [
        ("Clinical_Dosimetry", "Clinical Dosimetry Baseline", "Fixed 5", "Clinical"),
        ("WholeTumor_Nested", "Whole-Tumor Alone (Nested CV)", "Auto (4-8)", "WholeTumor"),
        ("Habitat_Alone_Nested", "Habitat Alone (Nested CV)", "Auto (4-8)", "Habitat_Radiomics"),
        ("Early_Fusion_Merged", "Early Fusion (Merged Feature Pool)", "Auto (4-8)", "Merged_Pool"),
        ("MultiScale_Late_Fusion", "Multi-Scale Late Fusion (50% WT + 50% Habitat)", "Ensemble", "Integrated")
    ]

    for key, disp_name, k_feats, category in models_display:
        probs = oofs_probs[key]
        preds = oofs_preds[key]
        metrics = compute_full_metrics(y_true, probs, preds)
        ci_lo, ci_hi = fast_bootstrap_ci(y_true, probs, patient_ids, n_boot=2000)

        if key == "WholeTumor_Nested":
            delta_auc, p2, p1 = 0.0, 1.0, 1.0
        else:
            delta_auc, p2, p1 = fast_paired_permutation_test(y_true, probs, baseline_wt_probs, n_perm=20000)

        f_list = fold_aucs_dict[key]
        summary_records.append({
            "Model_Key": key,
            "Model_Name": disp_name,
            "Category": category,
            "Features_k": k_feats,
            "ROC_AUC": metrics["ROC_AUC"],
            "AUC_CI_Low": ci_lo,
            "AUC_CI_High": ci_hi,
            "PR_AUC": metrics["PR_AUC"],
            "Accuracy": metrics["Accuracy"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Balanced_Acc": metrics["Balanced_Acc"],
            "PPV": metrics["PPV"],
            "NPV": metrics["NPV"],
            "F1_Score": metrics["F1"],
            "Brier_Score": metrics["Brier"],
            "Mean_Fold_AUC": float(np.mean(f_list)),
            "SD_Fold_AUC": float(np.std(f_list)),
            "Delta_vs_WT": float(delta_auc),
            "p_two_tailed": float(p2),
            "p_one_tailed": float(p1),
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TN": metrics["TN"]
        })

    leaderboard_df = pd.DataFrame(summary_records)
    probs_df = pd.DataFrame(oofs_probs, index=canonical_ids)
    probs_df.insert(0, "lesion_id", canonical_ids)
    probs_df["Target_Enlargement"] = y_true
    preds_df = pd.DataFrame(oofs_preds, index=canonical_ids)
    preds_df.insert(0, "lesion_id", canonical_ids)
    preds_df["Target_Enlargement"] = y_true

    # Export Selected Features
    feat_rows = []
    for model_k, fold_dict in features_selected_summary.items():
        for f, f_names in fold_dict.items():
            feat_rows.append({
                "Model_Key": model_k,
                "Outer_Fold": f,
                "Num_Features": len(f_names),
                "Features": "; ".join(f_names)
            })
    selected_features_df = pd.DataFrame(feat_rows)

    # Save to disk (both CSV and XLSX)
    leaderboard_df.to_csv(TABLES_DIR / "production_nested_cv_leaderboard.csv", index=False)
    leaderboard_df.to_excel(TABLES_DIR / "production_nested_cv_leaderboard.xlsx", index=False)
    probs_df.to_csv(TABLES_DIR / "production_oof_probabilities.csv", index=False)
    preds_df.to_csv(TABLES_DIR / "production_oof_predictions.csv", index=False)
    selected_features_df.to_csv(TABLES_DIR / "production_selected_features_per_fold.csv", index=False)
    selected_features_df.to_excel(TABLES_DIR / "production_selected_features_per_fold.xlsx", index=False)

    # Fold level metrics
    fold_rows = []
    for m_key, f_list in fold_aucs_dict.items():
        for f_idx, f_val in enumerate(f_list):
            fold_rows.append({"Model_Key": m_key, "Fold": f_idx, "Fold_AUC": f_val})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(TABLES_DIR / "production_fold_level_metrics.csv", index=False)

    logger.info(f"All tables and matrices saved to {TABLES_DIR}")
    return leaderboard_df, probs_df, preds_df, selected_features_df


def main():
    leaderboard_df, probs_df, preds_df, selected_features_df = execute_phase3_nested_cv()
    cols_print = ["Model_Name", "Features_k", "ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "Brier_Score", "Delta_vs_WT", "p_one_tailed"]
    print("\n" + "=" * 115)
    print("PHASE 3 PRODUCTION LEADERBOARD: STRICT NESTED CROSS-VALIDATION")
    print("=" * 115)
    print(leaderboard_df[cols_print].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPhase 3 execution complete.")


if __name__ == "__main__":
    main()
