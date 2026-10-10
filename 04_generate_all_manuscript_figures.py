#!/usr/bin/env python
r"""
Phase 4: Publication Figure Generator.

"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import convolve, distance_transform_edt, generate_binary_structure, label, median_filter
import seaborn as sns
import SimpleITK as sitk
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.multitest import multipletests

# Typography and Styling for High-Impact Journals
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["figure.autolayout"] = False
mpl.rcParams["axes.edgecolor"] = "#222222"
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase4_Figures")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
ALIGNED_DIR = DATA_DIR / "Aligned_NIfTI"
METADATA_DIR = DATA_DIR / "Metadata"
HAB_DIR = DATA_DIR / "Fold_Feature_Matrices_Habitat"
WT_DIR = DATA_DIR / "Fold_Feature_Matrices_WholeTumor"
RESULTS_DIR = BASE_DIR / "Results"
TABLES_DIR = RESULTS_DIR / "Tables"
FIG_DIR = RESULTS_DIR / "Figures"
GALLERY_DIR = FIG_DIR / "Figure1_Candidate_Gallery"

for d in [FIG_DIR, GALLERY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Import Phase 2 module dynamically
sys.path.insert(0, str(BASE_DIR / "Scripts"))
try:
    phase2_mod = importlib.import_module("02_phase2_habitat_radiomics")
except ModuleNotFoundError:
    try:
        phase2_mod = importlib.import_module("02_phase2_habitat_loader")
    except ModuleNotFoundError:
        phase2_mod = importlib.import_module("02_phase2_radiomic_feature_extraction")


def normalize_intensity(img_2d: np.ndarray) -> np.ndarray:
    r"""Robust min-max normalization for visualization."""
    p1, p99 = np.percentile(img_2d, (1, 99))
    if p99 > p1:
        norm = np.clip((img_2d - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm = np.zeros_like(img_2d, dtype=np.float32)
    return norm


def fcm_2class(
    data_1d: np.ndarray,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-5,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    r"""Literature-standard 2-class Fuzzy C-Means (FCM) clustering (Huang et al., 2022/2023)."""
    X = data_1d.reshape(-1, 1).astype(np.float64)
    c0 = np.percentile(data_1d, 35)
    c1 = np.percentile(data_1d, 65)
    centers = np.array([min(c0, c1), max(c0, c1)], dtype=np.float64)
    power = 2.0 / (m - 1.0)
    for _ in range(max_iter):
        dist = np.abs(X - centers.reshape(1, -1)) + 1e-8
        inv_dist = 1.0 / (dist ** power)
        u = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        u_m = u ** m
        new_centers = np.sum(u_m * X, axis=0) / (np.sum(u_m, axis=0) + 1e-8)
        if np.max(np.abs(new_centers - centers)) < tol:
            break
        centers = new_centers
    order = np.argsort(centers)
    centers = centers[order]
    dist = np.abs(X - centers.reshape(1, -1)) + 1e-8
    inv_dist = 1.0 / (dist ** power)
    u = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
    labels = np.argmax(u, axis=1).astype(np.int16)
    return labels, centers


def decompose_2hab_literature(lesion_dir: Path):
    r"""Decomposes a single lesion into Solid (0) and Cystic (1) sub-compartments using Huang et al."""
    mask_sitk = sitk.ReadImage(str(lesion_dir / "mask.nii.gz"))
    mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
    tumor_idx = np.where(mask_arr > 0)
    total_vox = len(tumor_idx[0])

    t1gd = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))
    t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T1.nii.gz"), sitk.sitkFloat32))
    t2 = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T2.nii.gz"), sitk.sitkFloat32))

    p1_t1gd, p99_t1gd = np.percentile(t1gd, (1, 99))
    p1_t2, p99_t2 = np.percentile(t2, (1, 99))
    t1gd_n = np.clip((t1gd - p1_t1gd) / (p99_t1gd - p1_t1gd + 1e-6), 0.05, 1.0)
    t2_n = np.clip((t2 - p1_t2) / (p99_t2 - p1_t2 + 1e-6), 0.0, 1.0)

    ratio = np.zeros_like(t1gd_n)
    ratio[mask_arr > 0] = t2_n[mask_arr > 0] / t1gd_n[mask_arr > 0]

    coords = np.argwhere(mask_arr > 0)
    z_min, z_max = coords[:, 0].min(), coords[:, 0].max() + 1
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max() + 1
    x_min, x_max = coords[:, 2].min(), coords[:, 2].max() + 1

    sub_ratio = ratio[z_min:z_max, y_min:y_max, x_min:x_max]
    sub_mask = mask_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    filt_ratio = np.zeros_like(sub_ratio)
    for s in range(sub_ratio.shape[0]):
        filt_ratio[s] = median_filter(sub_ratio[s], size=3)

    r_vox = filt_ratio[sub_mask > 0]
    labels, centers = fcm_2class(r_vox)

    sub_labels = np.full_like(sub_mask, -1, dtype=np.int16)
    sub_labels[sub_mask > 0] = labels

    footprint = np.ones((3, 3, 3), dtype=np.int32)
    counts = np.zeros((2, *sub_labels.shape), dtype=np.int32)
    for h_idx in range(2):
        counts[h_idx] = convolve(((sub_labels == h_idx) & (sub_mask > 0)).astype(np.int32), footprint, mode='constant', cval=0)
    smoothed = np.where(sub_mask > 0, np.argmax(counts, axis=0), -1).astype(np.int16)

    s_struct = generate_binary_structure(3, 3)
    cleaned = smoothed.copy()
    for h_idx in range(2):
        bin_hab = (cleaned == h_idx) & (sub_mask > 0)
        labeled_cc, num_cc = label(bin_hab, structure=s_struct)
        if num_cc > 0:
            cc_sizes = np.bincount(labeled_cc.ravel())
            small_mask = np.isin(labeled_cc, np.where((cc_sizes < 27) & (cc_sizes > 0))[0])
            if np.any(small_mask):
                cleaned[small_mask] = 1 - h_idx

    final_3d = np.full_like(mask_arr, -1, dtype=np.int16)
    final_3d[z_min:z_max, y_min:y_max, x_min:x_max] = np.where(sub_mask > 0, cleaned, -1)

    s_vox = int(np.sum((final_3d == 0) & (mask_arr > 0)))
    c_vox = int(np.sum((final_3d == 1) & (mask_arr > 0)))
    macro_2h = {
        "Total_Tumor_Voxels": total_vox,
        "Solid_Voxels": s_vox,
        "Cystic_Voxels": c_vox,
        "Solid_Volume_Fraction": s_vox / total_vox,
        "Cystic_Volume_Fraction": c_vox / total_vox,
    }
    return final_3d, mask_arr, ratio, filt_ratio, z_min, z_max, y_min, y_max, x_min, x_max, macro_2h


def generate_single_case_card_5panel(
    pid: str,
    label_text: str,
    z_slice: int,
    fold_idx: int,
    pad: int = 25,
    save_path: Path = None
):
    r"""Generates a high-definition 5-panel card displaying mpMRI and 3-Habitat model."""
    p_path = ALIGNED_DIR / pid
    hab_fold_path = p_path / f"habitat_3class_fold_{fold_idx}.nii.gz"
    if not hab_fold_path.exists():
        hab_fold_path = p_path / "habitat_3class.nii.gz"

    if hab_fold_path.exists():
        hab_sitk = sitk.ReadImage(str(hab_fold_path))
        final_3d_3h = sitk.GetArrayFromImage(hab_sitk).astype(np.int16)
        mask_sitk = sitk.ReadImage(str(p_path / "mask.nii.gz"))
        mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
    else:
        model_3h = phase2_mod.load_spatial_gmm_model(fold_idx)
        final_3d_3h, mask_arr, macro_3h = phase2_mod.decompose_lesion_habitats(p_path, model_3h)

    t1_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1.nii.gz"), sitk.sitkFloat32))
    t2_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T2.nii.gz"), sitk.sitkFloat32))
    t1gd_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))

    t1 = t1_vol[z_slice, :, :]
    t2 = t2_vol[z_slice, :, :]
    t1gd = t1gd_vol[z_slice, :, :]
    mask = mask_arr[z_slice, :, :]

    hab_3h_sl = np.where(mask > 0, final_3d_3h[z_slice, :, :] + 1, 0)
    t1_n, t2_n, t1gd_n = normalize_intensity(t1), normalize_intensity(t2), normalize_intensity(t1gd)

    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        coords = np.argwhere(mask_arr.sum(axis=0) > 0)
    ymin, xmin = max(0, coords.min(axis=0)[0] - pad), max(0, coords.min(axis=0)[1] - pad)
    ymax, xmax = min(mask.shape[0], coords.max(axis=0)[0] + pad), min(mask.shape[1], coords.max(axis=0)[1] + pad)
    box = (slice(ymin, ymax), slice(xmin, xmax))

    cmap_hab3 = ListedColormap(["none", "#1f77b4", "#2ca02c", "#d62728"])

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=300)
    titles = [
        "Native T1-Weighted", "T2-Weighted", "Contrast T1+Gd",
        "Whole-Tumor ROI Mask", "3-Habitat Model"
    ]

    imgs = [t1_n, t2_n, t1gd_n]
    for col_idx in range(3):
        axes[col_idx].imshow(imgs[col_idx][box], cmap="gray", origin="lower")
        axes[col_idx].set_xticks([]); axes[col_idx].set_yticks([])
        axes[col_idx].set_title(titles[col_idx], fontsize=11, fontweight="bold")
    axes[0].set_ylabel(label_text, fontsize=11, fontweight="bold", labelpad=8)

    # ROI Contour
    axes[3].imshow(t1gd_n[box], cmap="gray", origin="lower")
    axes[3].imshow(np.ma.masked_where(mask[box] == 0, mask[box]), cmap="autumn", alpha=0.55, origin="lower")
    axes[3].contour(mask[box], levels=[0.5], colors=["yellow"], linewidths=1.5)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    axes[3].set_title(titles[3], fontsize=11, fontweight="bold")

    # 3-Habitat Overlay
    axes[4].imshow(t1gd_n[box], cmap="gray", origin="lower")
    axes[4].imshow(np.ma.masked_where(hab_3h_sl[box] == 0, hab_3h_sl[box]), cmap=cmap_hab3, alpha=0.75, origin="lower", vmin=0, vmax=3)
    axes[4].contour(mask[box], levels=[0.5], colors=["white"], linewidths=1.0, linestyles="--")
    axes[4].set_xticks([]); axes[4].set_yticks([])
    axes[4].set_title(titles[4], fontsize=11, fontweight="bold")

    h1_pct = macro_3h['Hab1_Volume_Fraction'] * 100
    h2_pct = macro_3h['Hab2_Volume_Fraction'] * 100
    h3_pct = macro_3h['Hab3_Volume_Fraction'] * 100
    axes[4].text(0.04, 0.05, f"Habitat 1: {h1_pct:.1f}% | Habitat 2: {h2_pct:.1f}% | Habitat 3: {h3_pct:.1f}%",
                 transform=axes[4].transAxes, fontsize=8.0, fontweight="bold",
                 color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75, lw=0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_figure_1_and_gallery():
    r"""Generates Master Figure 1 (5 Columns) and Candidate Gallery (8 cases)."""
    logger.info("[1/9] Generating Figure 1 (5 Columns: 3-Habitat Model) and Candidate Gallery...")

    # Master Figure 1 Cases: Sample 26 and Sample 78
    master_pts = [
        {
            "id": "26_tumor_mask",
            "name": "Case_26_Large_Stable",
            "label": "Case 1: Sample 26\n(8.1 cc, Stable Non-Enlargement)",
            "z": 105,
            "pad": 35,
            "fold": 3
        },
        {
            "id": "78_tumor_mask",
            "name": "Case_78_Small_Shrinkage",
            "label": "Case 2: Sample 78\n(3.1 cc, Shrinkage Non-Enlargement)",
            "z": 91,
            "pad": 25,
            "fold": 0
        }
    ]

    gallery_cases = [
        {"id": "26_tumor_mask", "name": "Gallery_26_Large_Stable", "label": "Sample 26 (8.1 cc, Stable)", "z": 105, "pad": 35, "fold": 3},
        {"id": "78_tumor_mask", "name": "Gallery_78_Small_Shrinkage", "label": "Sample 78 (3.1 cc, Shrinkage)", "z": 91, "pad": 25, "fold": 0},
        {"id": "8_tumor_mask", "name": "Gallery_08_Large_Shrinkage", "label": "Sample 8 (23.4 cc, Shrinkage)", "z": 95, "pad": 35, "fold": 2},
        {"id": "60_tumor_mask", "name": "Gallery_60_Medium_Stable", "label": "Sample 60 (3.7 cc, Stable)", "z": 97, "pad": 30, "fold": 2},
        {"id": "88_tumor_mask", "name": "Gallery_88_Medium_Enlargement", "label": "Sample 88 (6.5 cc, Enlargement)", "z": 98, "pad": 30, "fold": 0},
        {"id": "42_tumor_mask", "name": "Gallery_42_Medium_Enlargement", "label": "Sample 42 (3.9 cc, Enlargement)", "z": 103, "pad": 25, "fold": 0},
        {"id": "14_tumor_mask", "name": "Gallery_14_Medium_Stable", "label": "Sample 14 (2.1 cc, Stable)", "z": 103, "pad": 25, "fold": 2},
        {"id": "50_tumor_mask", "name": "Gallery_50_Small_Shrinkage", "label": "Sample 50 (0.4 cc, Shrinkage)", "z": 98, "pad": 20, "fold": 4},
    ]

    # Generate Candidate Gallery
    for pt in gallery_cases:
        card_path = GALLERY_DIR / f"{pt['name']}.png"
        generate_single_case_card_5panel(pt["id"], pt["label"], pt["z"], pt["fold"], pt["pad"], card_path)

    # Master Figure 1: 2 rows x 5 columns
    fig = plt.figure(figsize=(19, 8.8), dpi=300)
    gs = fig.add_gridspec(2, 5, hspace=0.22, wspace=0.08, left=0.06, right=0.98, top=0.91, bottom=0.12)
    cmap_hab3 = ListedColormap(["none", "#1f77b4", "#2ca02c", "#d62728"])

    col_titles = [
        "(A) Native T1-Weighted",
        "(B) T2-Weighted",
        "(C) Contrast T1+Gd",
        "(D) Whole-Tumor ROI Mask",
        "(E) 3-Habitat Model"
    ]

    for row_idx, pt in enumerate(master_pts):
        pid, z, fold_idx = pt["id"], pt["z"], pt["fold"]
        p_path = ALIGNED_DIR / pid
        hab_fold_path = p_path / f"habitat_3class_fold_{fold_idx}.nii.gz"
        if not hab_fold_path.exists():
            hab_fold_path = p_path / "habitat_3class.nii.gz"

        if hab_fold_path.exists():
            hab_sitk = sitk.ReadImage(str(hab_fold_path))
            final_3d_3h = sitk.GetArrayFromImage(hab_sitk).astype(np.int16)
            mask_sitk = sitk.ReadImage(str(p_path / "mask.nii.gz"))
            mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
        else:
            model_3h = phase2_mod.load_spatial_gmm_model(fold_idx)
            final_3d_3h, mask_arr, macro_3h = phase2_mod.decompose_lesion_habitats(p_path, model_3h)

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1.nii.gz"), sitk.sitkFloat32))[z, :, :]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T2.nii.gz"), sitk.sitkFloat32))[z, :, :]
        t1gd = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))[z, :, :]
        mask = mask_arr[z, :, :]

        hab_3h_sl = np.where(mask > 0, final_3d_3h[z, :, :] + 1, 0)
        t1_n, t2_n, t1gd_n = normalize_intensity(t1), normalize_intensity(t2), normalize_intensity(t1gd)

        coords = np.argwhere(mask > 0)
        pad = pt["pad"]
        ymin, xmin = max(0, coords.min(axis=0)[0] - pad), max(0, coords.min(axis=0)[1] - pad)
        ymax, xmax = min(mask.shape[0], coords.max(axis=0)[0] + pad), min(mask.shape[1], coords.max(axis=0)[1] + pad)
        box = (slice(ymin, ymax), slice(xmin, xmax))

        imgs = [t1_n, t2_n, t1gd_n]
        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(imgs[col_idx][box], cmap="gray", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(pt["label"], fontsize=11, fontweight="bold", labelpad=10)

        # Whole-Tumor ROI
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.imshow(t1gd_n[box], cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(mask[box] == 0, mask[box]), cmap="autumn", alpha=0.55, origin="lower")
        ax.contour(mask[box], levels=[0.5], colors=["yellow"], linewidths=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[3], fontsize=12, fontweight="bold", pad=8)

        # 3-Habitat Model
        ax = fig.add_subplot(gs[row_idx, 4])
        ax.imshow(t1gd_n[box], cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(hab_3h_sl[box] == 0, hab_3h_sl[box]), cmap=cmap_hab3, alpha=0.75, origin="lower", vmin=0, vmax=3)
        ax.contour(mask[box], levels=[0.5], colors=["white"], linewidths=1.0, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[4], fontsize=12, fontweight="bold", pad=8)

        h1_pct = macro_3h['Hab1_Volume_Fraction'] * 100
        h2_pct = macro_3h['Hab2_Volume_Fraction'] * 100
        h3_pct = macro_3h['Hab3_Volume_Fraction'] * 100
        ax.text(0.04, 0.05, f"Habitat 1: {h1_pct:.1f}% | Habitat 2: {h2_pct:.1f}% | Habitat 3: {h3_pct:.1f}%",
                transform=ax.transAxes, fontsize=8.2, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75, lw=0))

    # Publication legends (Strictly Habitat 1, 2, 3 without descriptions)
    legend_elements = [
        patches.Patch(facecolor="#1f77b4", edgecolor="k", label="Habitat 1 (Blue)"),
        patches.Patch(facecolor="#2ca02c", edgecolor="k", label="Habitat 2 (Green)"),
        patches.Patch(facecolor="#d62728", edgecolor="k", label="Habitat 3 (Red)"),
        patches.Patch(facecolor="yellow", edgecolor="k", label="Whole-Tumor ROI Contour")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=11, frameon=True,
               facecolor="#f8f9fa", edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Multi-Parametric MRI Spatial Habitat Decomposition (3-Habitat Model)",
                 fontsize=14, fontweight="bold", y=0.97)

    out_png = FIG_DIR / "Figure 1 - Habitat_Discovery_Spatial_Mapping.png"
    out_pdf = FIG_DIR / "Figure 1 - Habitat_Discovery_Spatial_Mapping.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure 1.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved Figure 1 (Master 5-Column) and Candidate Gallery ({len(gallery_cases)} cases).")


def generate_figure_2():
    r"""
    Generates Figure 2: Macroscopic Habitat Volume Distributions & Log-Ratios.
    Contains strictly the 5 panels for the 3-Habitat Model (Panels A-E).
    """
    logger.info("[2/9] Generating Figure 2 (Macroscopic Distributions: 3-Habitat Model)...")
    macro_3h_csv = METADATA_DIR / "phase2_habitat_macro_metrics.csv"
    df_3h = pd.read_csv(macro_3h_csv)
    df_3h["Hab1_Hab3_LogRatio"] = np.log((df_3h["Hab1_Voxels"] + 1.0) / (df_3h["Hab3_Voxels"] + 1.0))
    df_3h["Hab2_Hab3_LogRatio"] = np.log((df_3h["Hab2_Voxels"] + 1.0) / (df_3h["Hab3_Voxels"] + 1.0))

    df_3h["Group"] = df_3h["Status"].apply(lambda s: "Enlargement" if s == "Enlargement" else "Non-Enlargement")
    enl_3h = df_3h[df_3h["Group"] == "Enlargement"]
    non_3h = df_3h[df_3h["Group"] == "Non-Enlargement"]

    metrics_3h = [
        ("Hab1_Volume_Fraction", "Habitat 1\nVolume Fraction", "(A)"),
        ("Hab2_Volume_Fraction", "Habitat 2\nVolume Fraction", "(B)"),
        ("Hab3_Volume_Fraction", "Habitat 3\nVolume Fraction", "(C)"),
        ("Hab1_Hab3_LogRatio", "log(Habitat 1 / Habitat 3)", "(D)"),
        ("Hab2_Hab3_LogRatio", "log(Habitat 2 / Habitat 3)", "(E)")
    ]

    p_vals_3h = []
    stats_3h = {}
    for col, _, _ in metrics_3h:
        u_stat, p_val = stats.mannwhitneyu(enl_3h[col], non_3h[col], alternative="two-sided")
        r_bis = 1.0 - (2.0 * u_stat) / (len(enl_3h) * len(non_3h))
        stats_3h[col] = {"p": p_val, "r": r_bis}
        p_vals_3h.append(p_val)

    # Validated Benjamini-Hochberg FDR (statsmodels)
    _, q_vals, _, _ = multipletests(p_vals_3h, method="fdr_bh")
    for idx, (col, _, _) in enumerate(metrics_3h):
        stats_3h[col]["q"] = q_vals[idx]

    # 1 row x 5 columns layout
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2), dpi=300)
    palette = {"Non-Enlargement": "#1976D2", "Enlargement": "#D32F2F"}

    for idx, (col, label, tag) in enumerate(metrics_3h):
        ax = axes[idx]
        sns.boxplot(
            data=df_3h, x="Group", y=col, hue="Group", order=["Non-Enlargement", "Enlargement"],
            palette=palette, width=0.45, ax=ax, legend=False,
            fliersize=0, boxprops=dict(alpha=0.65, edgecolor="black", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=2.0)
        )
        sns.stripplot(
            data=df_3h, x="Group", y=col, hue="Group", order=["Non-Enlargement", "Enlargement"],
            palette=palette, alpha=0.65, jitter=0.2, size=5, ax=ax, edgecolor="gray", linewidth=0.5, legend=False
        )
        ax.set_title(f"{tag} {label}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(label.replace("\n", " "), fontsize=9.5, fontweight="semibold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Enlargement\n(n=55)", "Enlargement\n(n=45)"], fontsize=9.5, fontweight="bold")

        y_max, y_min = df_3h[col].max(), df_3h[col].min()
        y_range = y_max - y_min
        bracket_y = y_max + 0.12 * y_range
        text_y = y_max + 0.16 * y_range
        ax.plot([0, 0, 1, 1], [bracket_y - 0.03 * y_range, bracket_y, bracket_y, bracket_y - 0.03 * y_range], lw=1.2, c="black")

        st = stats_3h[col]
        sig_str = f"p = {st['p']:.4f}\nFDR q = {st['q']:.4f}\nr = {abs(st['r']):.3f}"
        ax.text(0.5, text_y, sig_str, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#111111")
        ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.44 * y_range)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle("Intra-Tumoral Spatial Habitat Volume Distributions and Log-Ratios: Enlargement vs. Non-Enlargement",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_png = FIG_DIR / "Figure 2 - Enlargement_Macroscopic_Habitat_Distributions.png"
    out_pdf = FIG_DIR / "Figure 2 - Enlargement_Macroscopic_Habitat_Distributions.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure 2.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved Figure 2 (5 Panels 3-Habitat): {out_png.name}")


def generate_figure_3():
    r"""Generates Figure 3: Strict Nested CV Predictive Performance (5 Core Models across 4 Panels)."""
    logger.info("[3/9] Generating Figure 3 (Strict Nested CV 5-Model 4-Panel Performance)...")
    probs_df = pd.read_csv(TABLES_DIR / "production_oof_probabilities.csv")
    preds_df = pd.read_csv(TABLES_DIR / "production_oof_predictions.csv")
    lb_df = pd.read_csv(TABLES_DIR / "production_nested_cv_leaderboard.csv")

    y_true = probs_df["Target_Enlargement"].values

    models_meta = [
        ("Clinical_Dosimetry", "Clinical Dosimetry Baseline", "#78909C", ":", 1.8),
        ("WholeTumor_Nested", "Whole-Tumor Alone (Nested CV)", "#2E7D32", "-", 2.2),
        ("Habitat_Alone_Nested", "Habitat Alone (Nested CV)", "#EF6C00", "-", 2.2),
        ("Early_Fusion_Merged", "Early Fusion (Merged Pool)", "#8E24AA", "-.", 2.2),
        ("MultiScale_Late_Fusion", "Multi-Scale Late Fusion (50/50)", "#C62828", "-", 2.6)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    ax_roc, ax_pr, ax_cal, ax_dca = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: ROC Curves
    for key, name, color, ls, lw in models_meta:
        if key not in probs_df.columns:
            continue
        p = probs_df[key].values
        fpr, tpr, _ = roc_curve(y_true, p)
        row = lb_df[lb_df["Model_Key"] == key].iloc[0]
        label = f"{name} (AUC = {row['ROC_AUC']:.3f} [{row['AUC_CI_Low']:.3f}, {row['AUC_CI_High']:.3f}])"
        ax_roc.plot(fpr, tpr, label=label, color=color, linestyle=ls, linewidth=lw)

    ax_roc.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Chance (AUC = 0.500)")
    ax_roc.set_title("(A) Out-of-Fold Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=10, fontweight="semibold")
    ax_roc.set_ylabel("Sensitivity (True Positive Rate)", fontsize=10, fontweight="semibold")
    ax_roc.legend(loc="lower right", fontsize=8.0, frameon=True)
    ax_roc.grid(True, linestyle="--", alpha=0.3)
    ax_roc.set_xlim([-0.02, 1.02]); ax_roc.set_ylim([-0.02, 1.02])

    # Panel B: Precision-Recall Curves
    prevalence = y_true.mean()
    for key, name, color, ls, lw in models_meta:
        if key not in probs_df.columns:
            continue
        p = probs_df[key].values
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        ax_pr.plot(rec, prec, label=f"{name} (AP = {ap:.3f})", color=color, linestyle=ls, linewidth=lw)

    ax_pr.axhline(prevalence, color="k", linestyle=":", alpha=0.5, label=f"No Skill Prevalence ({prevalence:.2f})")
    ax_pr.set_title("(B) Precision-Recall Curves (Enlargement Event)", fontsize=11, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="semibold")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, fontweight="semibold")
    ax_pr.legend(loc="lower left", fontsize=8.0, frameon=True)
    ax_pr.grid(True, linestyle="--", alpha=0.3)
    ax_pr.set_xlim([-0.02, 1.02]); ax_pr.set_ylim([0.25, 1.02])

    # Panel C: Reliability Calibration Curves
    for key, name, color, ls, lw in models_meta:
        if key not in probs_df.columns:
            continue
        p = probs_df[key].values
        brier = brier_score_loss(y_true, p)
        bins = np.linspace(0, 1, 6)
        bin_idx = np.digitize(p, bins) - 1
        obs_props = []
        pred_means = []
        for b in range(len(bins) - 1):
            mask_b = (bin_idx == b)
            if mask_b.sum() > 0:
                obs_props.append(y_true[mask_b].mean())
                pred_means.append(p[mask_b].mean())
        ax_cal.plot(pred_means, obs_props, "o-", label=f"{name} (Brier = {brier:.4f})", color=color, linestyle=ls, linewidth=lw, markersize=5)

    ax_cal.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Ideal Calibration")
    ax_cal.set_title("(C) Reliability Calibration Curves", fontsize=11, fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="semibold")
    ax_cal.set_ylabel("Observed Fraction of Enlargement", fontsize=10, fontweight="semibold")
    ax_cal.legend(loc="upper left", fontsize=8.0, frameon=True)
    ax_cal.grid(True, linestyle="--", alpha=0.3)
    ax_cal.set_xlim([-0.02, 1.02]); ax_cal.set_ylim([-0.02, 1.02])

    # Panel D: Decision Curve Analysis (DCA Net Benefit)
    thresholds = np.linspace(0.10, 0.85, 50)
    N = len(y_true)
    treat_all = [(y_true.sum() / N) - ((1.0 - (y_true.sum() / N)) * (pt / (1.0 - pt))) for pt in thresholds]
    ax_dca.plot(thresholds, treat_all, "k:", alpha=0.5, label="Treat All")
    ax_dca.axhline(0, color="gray", linestyle="-", alpha=0.5, label="Treat None")

    for key, name, color, ls, lw in models_meta:
        if key not in probs_df.columns:
            continue
        p = probs_df[key].values
        net_benefits = []
        for pt in thresholds:
            y_pred_thresh = (p >= pt).astype(int)
            tp = np.logical_and(y_pred_thresh == 1, y_true == 1).sum()
            fp = np.logical_and(y_pred_thresh == 1, y_true == 0).sum()
            nb = (tp / N) - (fp / N) * (pt / (1.0 - pt))
            net_benefits.append(nb)
        ax_dca.plot(thresholds, net_benefits, label=name, color=color, linestyle=ls, linewidth=lw)

    ax_dca.set_title("(D) Decision Curve Analysis (Clinical Net Benefit)", fontsize=11, fontweight="bold")
    ax_dca.set_xlabel("Threshold Probability", fontsize=10, fontweight="semibold")
    ax_dca.set_ylabel("Net Benefit", fontsize=10, fontweight="semibold")
    ax_dca.legend(loc="upper right", fontsize=8.0, frameon=True)
    ax_dca.grid(True, linestyle="--", alpha=0.3)
    ax_dca.set_xlim([0.10, 0.85]); ax_dca.set_ylim([-0.05, 0.45])

    plt.suptitle("Out-of-Fold Predictive Performance under Strict 5-Fold Nested Cross-Validation",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_png = FIG_DIR / "Figure 3 - Enlargement_Predictive_Performance.png"
    out_pdf = FIG_DIR / "Figure 3 - Enlargement_Predictive_Performance.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure 3.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out_png.name}")


def generate_figure_4():
    r"""Generates Figure 4: Intratumoral & Whole-Tumor Feature Discovery and FDR Volcano."""
    logger.info("[4/9] Generating Figure 4 (Feature Discovery & FDR Volcano)...")
    fdr_df = pd.read_csv(TABLES_DIR / "production_univariable_fdr_features.csv")
    sel_feats_df = pd.read_csv(TABLES_DIR / "production_selected_features_per_fold.csv")

    volcano_df = fdr_df.copy()
    if "neg_log10_q" not in volcano_df.columns:
        q_col = "q_val" if "q_val" in volcano_df.columns else "FDR_q"
        volcano_df["neg_log10_q"] = -np.log10(np.clip(volcano_df[q_col], 1e-10, 1.0))
    if "Compartment" not in volcano_df.columns:
        volcano_df["Compartment"] = "Whole-Tumor"
        volcano_df.loc[volcano_df["Feature"].str.startswith("Hab1_"), "Compartment"] = "Habitat 1"
        volcano_df.loc[volcano_df["Feature"].str.startswith("Hab2_"), "Compartment"] = "Habitat 2"
        volcano_df.loc[volcano_df["Feature"].str.startswith("Hab3_"), "Compartment"] = "Habitat 3"

    wt_selected = []
    for _, row in sel_feats_df[sel_feats_df["Model_Key"] == "WholeTumor_Nested"].iterrows():
        wt_selected.extend([f.strip() for f in row["Features"].split(";")])
    wt_freq = pd.Series(wt_selected).value_counts().head(6)

    hab_selected = []
    for _, row in sel_feats_df[sel_feats_df["Model_Key"] == "Habitat_Alone_Nested"].iterrows():
        hab_selected.extend([f.strip() for f in row["Features"].split(";")])
    hab_freq = pd.Series(hab_selected).value_counts().head(6)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), dpi=300)
    ax_vol, ax_wt, ax_hab = axes[0], axes[1], axes[2]

    # Panel A: Volcano Landscape
    comp_colors = {
        "Whole-Tumor": "#43A047",
        "Habitat 1": "#1E88E5",
        "Habitat 2": "#FB8C00",
        "Habitat 3": "#E53935"
    }
    for comp, color in comp_colors.items():
        sub = volcano_df[volcano_df["Compartment"] == comp]
        ax_vol.scatter(sub["AUC"], sub["neg_log10_q"], c=color, label=comp, alpha=0.60, s=28, edgecolors="none")

    ax_vol.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.7, label="FDR q = 0.05 Threshold")
    ax_vol.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax_vol.set_title("(A) False Discovery Rate (FDR) Volcano Landscape\n(1,178 Candidate Radiomic Features)", fontsize=11, fontweight="bold")
    ax_vol.set_xlabel("Univariable ROC-AUC", fontsize=10, fontweight="semibold")
    ax_vol.set_ylabel(r"$-\log_{10}(\mathrm{FDR}\ q\mathrm{-value})$", fontsize=10, fontweight="semibold")
    ax_vol.legend(loc="upper left", fontsize=8.2, frameon=True)
    ax_vol.grid(True, linestyle="--", alpha=0.3)

    # Panel B: Top Whole-Tumor Features
    y_wt = np.arange(len(wt_freq))
    clean_wt_labels = [l.replace("WholeTumor_", "").replace("original_", "").replace("firstorder_", "") for l in wt_freq.index]
    ax_wt.barh(y_wt, wt_freq.values, align="center", color="#43A047", alpha=0.85, edgecolor="black", height=0.55)
    ax_wt.set_yticks(y_wt)
    ax_wt.set_yticklabels(clean_wt_labels, fontsize=8.5, fontweight="semibold")
    ax_wt.invert_yaxis()
    ax_wt.set_xlabel("Selection Frequency across 5 Folds", fontsize=10, fontweight="semibold")
    ax_wt.set_title("(B) Top Whole-Tumor Features\n(Discovered by Autonomous Inner-CV)", fontsize=11, fontweight="bold")
    ax_wt.set_xlim([0, 5.5]); ax_wt.set_xticks([1, 2, 3, 4, 5])
    ax_wt.grid(axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(wt_freq.values):
        ax_wt.text(v + 0.1, i, f"{v}/5 folds", va="center", fontsize=8.0, fontweight="bold")

    # Panel C: Top Habitat Features (Strictly Habitat 1, 2, 3)
    y_hab = np.arange(len(hab_freq))
    clean_hab_labels = [l.replace("original_", "").replace("firstorder_", "") for l in hab_freq.index]
    bar_colors_hab = ["#E53935" if "Hab3" in l else ("#FB8C00" if "Hab2" in l else "#1E88E5") for l in hab_freq.index]
    ax_hab.barh(y_hab, hab_freq.values, align="center", color=bar_colors_hab, alpha=0.85, edgecolor="black", height=0.55)
    ax_hab.set_yticks(y_hab)
    ax_hab.set_yticklabels(clean_hab_labels, fontsize=8.5, fontweight="semibold")
    ax_hab.invert_yaxis()
    ax_hab.set_xlabel("Selection Frequency across 5 Folds", fontsize=10, fontweight="semibold")
    ax_hab.set_title("(C) Top Habitat Radiomic Features\n(Discovered by Autonomous Inner-CV)", fontsize=11, fontweight="bold")
    ax_hab.set_xlim([0, 5.5]); ax_hab.set_xticks([1, 2, 3, 4, 5])
    ax_hab.grid(axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(hab_freq.values):
        ax_hab.text(v + 0.1, i, f"{v}/5 folds", va="center", fontsize=8.0, fontweight="bold")

    plt.suptitle("Multi-Scale Biological Feature Discovery, False Discovery Rate (FDR), and Fold Stability",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_png = FIG_DIR / "Figure 4 - Intratumoral_Habitat_FDR_Discovery.png"
    out_pdf = FIG_DIR / "Figure 4 - Intratumoral_Habitat_FDR_Discovery.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure 4.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out_png.name}")


def generate_figure_5():
    r"""Generates Figure 5: Subgroup Volumetric Scalability Analysis."""
    logger.info("[5/9] Generating Figure 5 (Robust Low vs High Volume Scalability)...")
    probs_df = pd.read_csv(TABLES_DIR / "production_oof_probabilities.csv")
    df_wt0 = pd.read_csv(WT_DIR / "fold_0_features.csv")
    if "lesion_id" in probs_df.columns and "lesion_id" in df_wt0.columns:
        probs_df = probs_df.set_index("lesion_id")
        df_wt0 = df_wt0.set_index("lesion_id")
        probs_df = probs_df.reindex(df_wt0.index)

    vols = df_wt0["WholeTumor_original_shape_MeshVolume"].values / 1000.0
    med_vol = float(np.median(vols))
    y_true = probs_df["Target_Enlargement"].values

    subgroups = [
        (f"Low Volume (< {med_vol:.2f} cc, n=50)", vols < med_vol),
        (f"High Volume (>= {med_vol:.2f} cc, n=50)", vols >= med_vol)
    ]

    models_to_eval = [
        ("Clinical_Dosimetry", "Clinical Dosimetry", "#78909C"),
        ("WholeTumor_Nested", "Whole-Tumor Alone", "#2E7D32"),
        ("Habitat_Alone_Nested", "Habitat Alone", "#EF6C00"),
        ("Early_Fusion_Merged", "Early Fusion", "#8E24AA"),
        ("MultiScale_Late_Fusion", "Multi-Scale Late Fusion", "#C62828")
    ]

    subgroup_records = []
    for g_name, mask in subgroups:
        yt_sub = y_true[mask]
        n_sub = len(yt_sub)
        n_enl = int(yt_sub.sum())
        for key, disp_name, col in models_to_eval:
            if key not in probs_df.columns:
                continue
            p_sub = probs_df[key].values[mask]
            auc_sub = roc_auc_score(yt_sub, p_sub)
            subgroup_records.append({
                "Subgroup": g_name,
                "Model": disp_name,
                "AUC": auc_sub,
                "N": n_sub,
                "Enlarged": n_enl,
                "Color": col
            })

    sub_df = pd.DataFrame(subgroup_records)
    sub_df.to_csv(TABLES_DIR / "production_volume_subgroup_scalability.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    sns.barplot(data=sub_df, x="Subgroup", y="AUC", hue="Model",
                palette=[c[2] for c in models_to_eval if c[0] in probs_df.columns], ax=ax, edgecolor="black", alpha=0.85)

    ax.set_title("Cross-Fold Predictive Discrimination Stratified by Tumor Volume (N = 50 Low vs. N = 50 High)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Out-of-Fold ROC-AUC", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Baseline Volumetric Stratum", fontsize=10, fontweight="semibold")
    ax.set_ylim([0.45, 0.88])
    ax.axhline(0.50, color="gray", linestyle=":", alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Model Architecture", fontsize=8.8, title_fontsize=9.5, loc="lower right", frameon=True)

    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    out_png = FIG_DIR / "Figure 5 - Tumor_Volume_Scalability.png"
    out_pdf = FIG_DIR / "Figure 5 - Tumor_Volume_Scalability.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure 5.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out_png.name}")


def generate_figure_s1():
    r"""
    Generates Figure S1: Dual-Stage Spatial Topological Regularization Pipeline (3-Habitat Model).
    Demonstrated on Sample 78 and Sample 26.
    """
    logger.info("[6/9] Generating Figure S1 (Spatial Topological Regularization for 3-Habitat Model)...")
    cases = [
        {"pid": "78_tumor_mask", "z": 91, "fold": 0, "pad": 22, "title_pfx": "Sample 78 (3.1 cc)"},
        {"pid": "26_tumor_mask", "z": 105, "fold": 3, "pad": 30, "title_pfx": "Sample 26 (8.1 cc)"}
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=300)
    cmap_hab3 = ListedColormap(["none", "#1f77b4", "#2ca02c", "#d62728"])

    for row_idx, cs in enumerate(cases):
        pid, z, fold_idx, pad = cs["pid"], cs["z"], cs["fold"], cs["pad"]
        p_path = ALIGNED_DIR / pid
        model_3h = phase2_mod.load_spatial_gmm_model(fold_idx)

        mask_sitk = sitk.ReadImage(str(p_path / "mask.nii.gz"))
        mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
        tumor_idx = np.where(mask_arr > 0)

        t1gd_img = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))
        t1_img = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1.nii.gz"), sitk.sitkFloat32))
        t2_img = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T2.nii.gz"), sitk.sitkFloat32))

        z_coords, y_coords, x_coords = tumor_idx
        margin = 5
        d, h, w = mask_arr.shape
        z_min, z_max = max(0, z_coords.min() - margin), min(d, z_coords.max() + margin)
        y_min, y_max = max(0, y_coords.min() - margin), min(h, y_coords.max() + margin)
        x_min, x_max = max(0, x_coords.min() - margin), min(w, x_coords.max() + margin)

        sub_mask = mask_arr[z_min:z_max, y_min:y_max, x_min:x_max]
        sub_dtb = distance_transform_edt(sub_mask)
        max_dtb = np.max(sub_dtb)
        sub_dtb_norm = sub_dtb / (max_dtb if max_dtb > 1e-6 else 1.0)
        dtb_norm_vox = sub_dtb_norm[z_coords - z_min, y_coords - y_min, x_coords - x_min].astype(np.float32)

        X_vox = np.column_stack([t1gd_img[tumor_idx], t1_img[tumor_idx], t2_img[tumor_idx], dtb_norm_vox])
        raw_labels = model_3h.predict(X_vox)
        sub_labels = np.full_like(sub_mask, -1, dtype=np.int16)
        sub_labels[z_coords - z_min, y_coords - y_min, x_coords - x_min] = raw_labels

        sub_smoothed = phase2_mod.tumor_constrained_mode_filter(sub_labels, sub_mask)
        sub_final = phase2_mod.erase_small_habitat_islands(sub_smoothed, sub_mask, min_size=27)

        rel_z = z - z_min
        raw_sl = np.where(sub_mask[rel_z] > 0, sub_labels[rel_z] + 1, 0)
        final_sl = np.where(sub_mask[rel_z] > 0, sub_final[rel_z] + 1, 0)

        t1gd_slice = t1gd_img[z, :, :]
        mask_slice = mask_arr[z, :, :]
        t1gd_sub = t1gd_slice[y_min:y_max, x_min:x_max]
        mask_sub = mask_slice[y_min:y_max, x_min:x_max]

        coords_sl = np.argwhere(mask_sub > 0)
        ymin, xmin = max(0, coords_sl.min(axis=0)[0] - pad), max(0, coords_sl.min(axis=0)[1] - pad)
        ymax, xmax = min(mask_sub.shape[0], coords_sl.max(axis=0)[0] + pad), min(mask_sub.shape[1], coords_sl.max(axis=0)[1] + pad)
        box = (slice(ymin, ymax), slice(xmin, xmax))

        t1gd_norm = normalize_intensity(t1gd_sub)

        # Col 0: T1+Gd with Contour
        tag_col0 = "(A)" if row_idx == 0 else "(D)"
        tag_col1 = "(B)" if row_idx == 0 else "(E)"
        tag_col2 = "(C)" if row_idx == 0 else "(F)"

        axes[row_idx, 0].imshow(t1gd_norm[box], cmap="gray", origin="lower")
        axes[row_idx, 0].contour(mask_sub[box], levels=[0.5], colors=["yellow"], linewidths=2.0)
        axes[row_idx, 0].set_title(f"{tag_col0} Co-Registered Contrast T1+Gd MRI\nwith Manual ROI Contour", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].text(-0.06, 0.5, cs["title_pfx"], transform=axes[row_idx, 0].transAxes,
                              fontsize=11.5, fontweight="bold", va="center", ha="right", rotation=90)

        # Col 1: Raw 3-Hab
        axes[row_idx, 1].imshow(t1gd_norm[box], cmap="gray", origin="lower")
        axes[row_idx, 1].imshow(np.ma.masked_where(raw_sl[box] == 0, raw_sl[box]), cmap=cmap_hab3, alpha=0.75, origin="lower", vmin=0, vmax=3)
        axes[row_idx, 1].set_title(f"{tag_col1} Raw 3-Habitat Spatial-GMM Voxel Clustering\n(Presence of High-Frequency Noise)", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Regularized 3-Hab
        axes[row_idx, 2].imshow(t1gd_norm[box], cmap="gray", origin="lower")
        axes[row_idx, 2].imshow(np.ma.masked_where(final_sl[box] == 0, final_sl[box]), cmap=cmap_hab3, alpha=0.75, origin="lower", vmin=0, vmax=3)
        axes[row_idx, 2].set_title(f"{tag_col2} Regularized 3-Habitat Spatial Mapping\n(3D Mode Filter + Island Erasure)", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

    leg_handles = [
        patches.Patch(facecolor="#1f77b4", edgecolor="k", label="Habitat 1 (Blue)"),
        patches.Patch(facecolor="#2ca02c", edgecolor="k", label="Habitat 2 (Green)"),
        patches.Patch(facecolor="#d62728", edgecolor="k", label="Habitat 3 (Red)"),
        patches.Patch(facecolor="yellow", edgecolor="k", label="Whole-Tumor ROI Contour")
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=4, fontsize=10.5, frameon=True,
               facecolor="#f8f9fa", edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.015))

    plt.suptitle("Dual-Stage Spatial Topological Regularization Pipeline for 3-Habitat Model",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    out_png = FIG_DIR / "Figure S1 - Spatial_Topological_Regularization.png"
    out_pdf = FIG_DIR / "Figure S1 - Spatial_Topological_Regularization.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure S1.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved Figure S1 (3-Habitat Regularization): {out_png.name}")


def generate_figure_s2():
    r"""Generates Figure S2: Cross-Fold Generalization Stability Boxplots (5 Core Models)."""
    logger.info("[7/9] Generating Figure S2 (Cross-Fold Stability Boxplots for 5 Core Models)...")
    fold_df = pd.read_csv(TABLES_DIR / "production_fold_level_metrics.csv")

    model_name_map = {
        "Clinical_Dosimetry": "Clinical Dosimetry",
        "WholeTumor_Nested": "Whole-Tumor Alone",
        "Habitat_Alone_Nested": "Habitat Alone",
        "Early_Fusion_Merged": "Early Fusion",
        "MultiScale_Late_Fusion": "Multi-Scale Late Fusion"
    }

    sub_fold_df = fold_df[fold_df["Model_Key"].isin(model_name_map.keys())].copy()
    sub_fold_df["Model_Display"] = sub_fold_df["Model_Key"].map(model_name_map)

    palette = ["#78909C", "#2E7D32", "#EF6C00", "#8E24AA", "#C62828"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)

    sns.boxplot(data=sub_fold_df, x="Model_Display", y="Fold_AUC", palette=palette, ax=ax,
                width=0.45, boxprops=dict(alpha=0.7, edgecolor="black"), medianprops=dict(color="black", lw=2))
    sns.stripplot(data=sub_fold_df, x="Model_Display", y="Fold_AUC", palette=palette, ax=ax,
                  size=8, jitter=0.15, edgecolor="black", linewidth=1.0)

    ax.set_title("Cross-Fold Generalization Stability across 5 Outer Cross-Validation Folds", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Outer Fold ROC-AUC", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Predictive Model Architecture", fontsize=10, fontweight="semibold")
    ax.axhline(0.50, color="gray", linestyle=":", alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_png = FIG_DIR / "Figure S2 - Cross_Fold_Stability.png"
    out_pdf = FIG_DIR / "Figure S2 - Cross_Fold_Stability.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure S2.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out_png.name}")


def generate_figure_s3():
    r"""Generates Figure S3: Inter-Feature Orthogonality Correlation Heatmap."""
    logger.info("[8/9] Generating Figure S3 (Feature Orthogonality Correlation Heatmap)...")
    sel_df = pd.read_csv(TABLES_DIR / "production_selected_features_per_fold.csv")
    df_hab0 = pd.read_csv(HAB_DIR / "fold_0_features.csv").set_index("lesion_id")

    hab_feats = []
    for _, row in sel_df[sel_df["Model_Key"] == "Habitat_Alone_Nested"].iterrows():
        hab_feats.extend([f.strip() for f in row["Features"].split(";")])
    unique_top = pd.Series(hab_feats).value_counts().head(8).index.tolist()

    sub_data = df_hab0[unique_top].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr_mat = sub_data.corr(method="spearman")

    clean_labels = [l.replace("original_", "").replace("firstorder_", "").replace("glszm_", "").replace("glcm_", "") for l in unique_top]

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=clean_labels, yticklabels=clean_labels, ax=ax,
                cbar_kws={"label": "Spearman Rank Correlation (r)"},
                linewidths=0.5, linecolor="gray")

    ax.set_title("Inter-Feature Orthogonality Matrix of Top Discovered Habitat Features (|r| <= 0.85 Pruned)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8.5, fontweight="semibold")
    plt.yticks(fontsize=8.5, fontweight="semibold")

    plt.tight_layout()
    out_png = FIG_DIR / "Figure S3 - Inter_Feature_Correlation_Heatmap.png"
    out_pdf = FIG_DIR / "Figure S3 - Inter_Feature_Correlation_Heatmap.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure S3.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out_png.name}")


def generate_figure_s4_literature_ablation():
    r"""
    Generates Figure S4: Literature 2-Habitat Benchmark Walkthrough (Huang et al.).
    Walkthrough of the T2W/T1WC Contrast-Ratio Fuzzy C-Means Decomposition Pipeline.
    """
    logger.info("[9/9] Generating Figure S4 (Literature 2-Habitat Benchmark Walkthrough)...")
    cases = [
        {"pid": "17_tumor_mask", "label": "Case 1: Sample 17\n(18.7 cm³, Enlargement)", "z": 86, "pad": 35},
        {"pid": "14_tumor_mask", "label": "Case 2: Sample 14\n(8.7 cm³, Stable)", "z": 101, "pad": 30}
    ]

    fig, axes = plt.subplots(2, 6, figsize=(22, 8.5), dpi=300)
    titles = [
        "Native T1-Weighted", "T2-Weighted", "Contrast T1+Gd",
        "Synthetic Ratio Map (R)", "FCM 2-Habitat Clustering", "Isolated Cystic Overlay"
    ]

    cmap_2h = ListedColormap(["#1565c0", "#e65100"])
    cmap_cystic = ListedColormap(["#e65100"])

    for row_idx, cs in enumerate(cases):
        pid, z, pad = cs["pid"], cs["z"], cs["pad"]
        p_path = ALIGNED_DIR / pid
        final_3d, mask_arr, ratio_map, filt_ratio, z_min, z_max, y_min, y_max, x_min, x_max, macro_2h = decompose_2hab_literature(p_path)

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1.nii.gz"), sitk.sitkFloat32))[z, :, :]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T2.nii.gz"), sitk.sitkFloat32))[z, :, :]
        t1gd = sitk.GetArrayFromImage(sitk.ReadImage(str(p_path / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))[z, :, :]
        mask = mask_arr[z, :, :]

        t1_n, t2_n, t1gd_n = normalize_intensity(t1), normalize_intensity(t2), normalize_intensity(t1gd)

        coords = np.argwhere(mask > 0)
        ymin, xmin = max(0, coords.min(axis=0)[0] - pad), max(0, coords.min(axis=0)[1] - pad)
        ymax, xmax = min(mask.shape[0], coords.max(axis=0)[0] + pad), min(mask.shape[1], coords.max(axis=0)[1] + pad)
        box = (slice(ymin, ymax), slice(xmin, xmax))

        rel_z = z - z_min
        r_slice = np.where(mask > 0, ratio_map[z, :, :], 0.0)
        p98_r = np.percentile(r_slice[mask > 0], 98) if mask.sum() > 0 else 1.0
        r_norm = np.clip(r_slice / (p98_r + 1e-6), 0.0, 1.0)

        hab_2h_sl = np.where(mask > 0, final_3d[z, :, :] + 1, 0)
        cystic_sl = np.where((mask > 0) & (final_3d[z, :, :] == 1), 1, 0)

        # Col 0: T1
        axes[row_idx, 0].imshow(t1_n[box], cmap="gray", origin="lower")
        axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])
        if row_idx == 0: axes[row_idx, 0].set_title(titles[0], fontsize=11, fontweight="bold")
        axes[row_idx, 0].set_ylabel(cs["label"], fontsize=11, fontweight="bold", labelpad=8)

        # Col 1: T2
        axes[row_idx, 1].imshow(t2_n[box], cmap="gray", origin="lower")
        axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])
        if row_idx == 0: axes[row_idx, 1].set_title(titles[1], fontsize=11, fontweight="bold")

        # Col 2: T1+Gd
        axes[row_idx, 2].imshow(t1gd_n[box], cmap="gray", origin="lower")
        axes[row_idx, 2].contour(mask[box], levels=[0.5], colors=["yellow"], linewidths=1.2)
        axes[row_idx, 2].set_xticks([]); axes[row_idx, 2].set_yticks([])
        if row_idx == 0: axes[row_idx, 2].set_title(titles[2], fontsize=11, fontweight="bold")

        # Col 3: Ratio map
        axes[row_idx, 3].imshow(t1gd_n[box], cmap="gray", origin="lower")
        axes[row_idx, 3].imshow(np.ma.masked_where(mask[box] == 0, r_norm[box]), cmap="magma", alpha=0.75, origin="lower", vmin=0, vmax=1)
        axes[row_idx, 3].set_xticks([]); axes[row_idx, 3].set_yticks([])
        if row_idx == 0: axes[row_idx, 3].set_title(titles[3], fontsize=11, fontweight="bold")

        # Col 4: FCM 2-Habitat Clustering
        axes[row_idx, 4].imshow(t1gd_n[box], cmap="gray", origin="lower")
        axes[row_idx, 4].imshow(np.ma.masked_where(hab_2h_sl[box] == 0, hab_2h_sl[box] - 1), cmap=cmap_2h, alpha=0.75, origin="lower", vmin=0, vmax=1)
        axes[row_idx, 4].set_xticks([]); axes[row_idx, 4].set_yticks([])
        if row_idx == 0: axes[row_idx, 4].set_title(titles[4], fontsize=11, fontweight="bold")

        s_pct = np.sum((final_3d[z] == 0) & (mask > 0)) / np.sum(mask > 0) * 100
        c_pct = np.sum((final_3d[z] == 1) & (mask > 0)) / np.sum(mask > 0) * 100
        axes[row_idx, 4].text(0.04, 0.05, f"Solid: {s_pct:.1f}% | Cystic: {c_pct:.1f}%",
                              transform=axes[row_idx, 4].transAxes, fontsize=8.5, fontweight="bold",
                              color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75, lw=0))

        # Col 5: Isolated Cystic Overlay
        axes[row_idx, 5].imshow(t1gd_n[box], cmap="gray", origin="lower")
        axes[row_idx, 5].imshow(np.ma.masked_where(cystic_sl[box] == 0, cystic_sl[box]), cmap=cmap_cystic, alpha=0.85, origin="lower", vmin=0, vmax=1)
        axes[row_idx, 5].contour(mask[box], levels=[0.5], colors=["white"], linewidths=1.0, linestyles="--")
        axes[row_idx, 5].set_xticks([]); axes[row_idx, 5].set_yticks([])
        if row_idx == 0: axes[row_idx, 5].set_title(titles[5], fontsize=11, fontweight="bold")

    leg_elements = [
        patches.Patch(facecolor="#1565c0", edgecolor="k", label="Solid Compartment (Lower Ratio)"),
        patches.Patch(facecolor="#e65100", edgecolor="k", label="Cystic / Fluid Compartment (Higher Ratio)"),
        patches.Patch(facecolor="yellow", edgecolor="k", label="Manual Whole-Tumor ROI Contour")
    ]
    fig.legend(handles=leg_elements, loc="lower center", ncol=3, fontsize=11, frameon=True,
               facecolor="#f8f9fa", edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.015))

    fig.suptitle("Literature-Inspired Two-Habitat Model Walkthrough: Contrast-Ratio Fuzzy C-Means (Huang et al.)",
                 fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    out_png = FIG_DIR / "Figure S4 - Literature_Two_Habitat_Ablation.png"
    out_pdf = FIG_DIR / "Figure S4 - Literature_Two_Habitat_Ablation.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "Figure S4.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved Figure S4 (Literature Ablation): {out_png.name}")


def main():
    logger.info("=" * 80)
    logger.info("GENERATING ALL PUBLICATION FIGURES & CANDIDATE GALLERY (300 DPI)")
    logger.info("=" * 80)
    generate_figure_1_and_gallery()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_s1()
    generate_figure_s2()
    generate_figure_s3()
    generate_figure_s4_literature_ablation()
    logger.info("All publication figures successfully created in: " + str(FIG_DIR))


if __name__ == "__main__":
    main()
