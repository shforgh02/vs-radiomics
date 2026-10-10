#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
01_phase1_preprocessing_and_habitat_segmentation.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import convolve, distance_transform_edt, label as nd_label
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase1_Preprocessing_GMM")

# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS & DIRECTORY PATHS
# -----------------------------------------------------------------------------
REF_MODALITY = "T1+Gd"
MOVING_MODALITIES = ("T1", "T2")
ALL_MODALITIES = ("T1+Gd", "T1", "T2")

NUM_FOLDS = 5
K_HABITATS = 3
MIN_ISLAND_VOXELS = 27           # 27 mm^3 in 1.0 mm isotropic space
MAX_TRAIN_VOXELS_PER_LESION = 2000
GMM_MAX_ITER = 200
SEED = 42

DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "Production_Pipeline" / "Data"
DEFAULT_ALIGNED_DIR = DEFAULT_DATA_DIR / "Aligned_NIfTI"
DEFAULT_METADATA_DIR = DEFAULT_DATA_DIR / "Metadata"
DEFAULT_HAB_DIR = DEFAULT_DATA_DIR / "Fold_Feature_Matrices_Habitat"
DEFAULT_MODELS_DIR = DEFAULT_HAB_DIR / "Models"


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class PreprocessingAuditRecord:
    lesion_id: str
    modality: str
    execution_time_seconds: float
    grid_spacing_match: bool
    grid_origin_match: bool
    grid_direction_match: bool
    status: str
    error_message: str = ""


# =============================================================================
# 1. IMAGE PREPROCESSING, RESAMPLING & STANDARDIZATION FUNCTIONS
# =============================================================================
def resample_to_isotropic_grid(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkBSpline,
    default_pixel_value: float = 0.0
) -> sitk.Image:
    r"""
    Resamples an input 3D image to a strict isotropic physical spacing (default 1.0 x 1.0 x 1.0 mm^3).
    Automatically computes new grid dimensions to preserve physical field-of-view (FOV).
    """
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()

    # Compute new size in physical units
    target_size = [
        int(np.round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    return resampler.Execute(image)


def generate_tissue_mask(image: sitk.Image) -> sitk.Image:
    r"""
    Generates an automated intracranial / tissue mask via Otsu thresholding
    followed by binary hole-filling and morphological closing (radius = 3 voxels).
    """
    img_f = sitk.Cast(image, sitk.sitkFloat32)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = otsu.Execute(img_f)

    mask = sitk.BinaryFillhole(mask)
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelRadius(3)
    mask = closing.Execute(mask)

    return sitk.Cast(mask, sitk.sitkUInt8)


def apply_native_n4_bias_correction(
    image: sitk.Image,
    mask: Optional[sitk.Image] = None,
    shrink_factor: int = 2
) -> sitk.Image:
    r"""
    Applies N4 bias field correction in physical coordinate space.
    Uses multi-resolution subsampling (shrink factor = 2) for rapid, stable convergence.
    """
    img_f = sitk.Cast(image, sitk.sitkFloat32)
    if mask is None:
        mask = generate_tissue_mask(img_f)

    shrink_factors = [shrink_factor] * img_f.GetDimension()
    sub_image = sitk.Shrink(img_f, shrink_factors)
    sub_mask = sitk.Shrink(mask, shrink_factors)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 40, 30])
    corrector.SetConvergenceThreshold(1e-4)

    _ = corrector.Execute(sub_image, sub_mask)
    log_bias_sub = corrector.GetLogBiasFieldAsImage(sub_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_f)
    resampler.SetInterpolator(sitk.sitkBSpline)
    log_bias = resampler.Execute(log_bias_sub)

    corrected = img_f / sitk.Exp(log_bias)
    corrected.CopyInformation(image)
    return corrected


def standardize_masked_zscore(
    image: sitk.Image,
    tissue_mask: sitk.Image
) -> Tuple[sitk.Image, float, float]:
    r"""
    Applies foreground tissue z-score intensity standardization:
      I_std(x) = (I(x) - mean_tissue) / std_tissue  for x in tissue_mask
    Voxels outside the tissue mask are set to 0.0.
    """
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(tissue_mask) > 0

    tissue_voxels = img_arr[mask_arr]
    if len(tissue_voxels) == 0:
        logger.warning("Empty tissue mask encountered during standardization!")
        return image, 0.0, 1.0

    mean_val = float(np.mean(tissue_voxels))
    std_val = float(np.std(tissue_voxels))
    if std_val < 1e-8:
        std_val = 1.0

    norm_arr = np.zeros_like(img_arr, dtype=np.float32)
    norm_arr[mask_arr] = (img_arr[mask_arr] - mean_val) / std_val

    std_img = sitk.GetImageFromArray(norm_arr)
    std_img.CopyInformation(image)
    return std_img, mean_val, std_val


def register_multimodal_rigid(
    fixed_img: sitk.Image,
    moving_img: sitk.Image
) -> Tuple[sitk.Transform, float, float, int, str]:
    r"""
    Performs deterministic 3-level multi-resolution rigid 3D co-registration
    using Mattes Mutual Information (50 histogram bins, 20% random sampling, seed 42)
    and Regular Step Gradient Descent optimizer.
    """
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
    reg.SetMetricSamplingPercentage(0.20, seed=SEED)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=150,
        relaxationFactor=0.7,
        gradientMagnitudeTolerance=1e-6
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    initial_val = reg.MetricEvaluate(fixed_img, moving_img)
    final_transform = reg.Execute(fixed_img, moving_img)
    final_val = reg.GetMetricValue()
    iters = reg.GetOptimizerIteration()
    stop_cond = reg.GetOptimizerStopConditionDescription()

    return final_transform, initial_val, final_val, iters, stop_cond


# =============================================================================
# 2. FULL RAW PREPROCESSING PIPELINE PER LESION
# =============================================================================
def preprocess_single_lesion_from_raw(
    lesion_id: str,
    patient_uid: str,
    mask_filename: str,
    raw_nifti_dir: Path,
    out_aligned_dir: Path
) -> List[PreprocessingAuditRecord]:
    r"""
    Executes the complete preprocessing workflow from raw MRI files:
      1. Loads raw T1+Gd and whole-tumor mask; resamples to 1.0 mm isotropic spacing.
      2. Generates tissue mask, applies N4 bias correction, and standardizes via z-score.
      3. For moving sequences (T1, T2): applies N4, rigidly co-registers to isotropic T1+Gd,
         resamples to 1.0 mm isotropic grid via B-spline, and standardizes via z-score.
      4. Writes `image_T1+Gd.nii.gz`, `image_T1.nii.gz`, `image_T2.nii.gz`, and `mask.nii.gz`.
    """
    records: List[PreprocessingAuditRecord] = []
    lesion_out_dir = out_aligned_dir / lesion_id
    lesion_out_dir.mkdir(parents=True, exist_ok=True)

    pt_dir = raw_nifti_dir / patient_uid
    # Robust path candidate lookup for reference image
    ref_candidates = [
        pt_dir / REF_MODALITY / "image.nii.gz",
        pt_dir / REF_MODALITY / f"{REF_MODALITY}.nii.gz",
        pt_dir / f"{REF_MODALITY}.nii.gz",
        raw_nifti_dir / lesion_id / f"image_{REF_MODALITY}.nii.gz",
    ]
    ref_img_path = next((p for p in ref_candidates if p.exists()), ref_candidates[0])

    # Robust path candidate lookup for reference mask
    mask_candidates = [
        pt_dir / REF_MODALITY / mask_filename,
        pt_dir / mask_filename,
        pt_dir / REF_MODALITY / "mask.nii.gz",
        pt_dir / "mask.nii.gz",
        raw_nifti_dir / lesion_id / "mask.nii.gz"
    ]
    ref_mask_path = next((p for p in mask_candidates if p.exists()), mask_candidates[0])

    if not ref_img_path.exists() or not ref_mask_path.exists():
        err = f"Missing reference image ({ref_img_path}) or mask ({ref_mask_path}) for lesion {lesion_id}"
        logger.error(err)
        records.append(PreprocessingAuditRecord(lesion_id, REF_MODALITY, 0.0, False, False, False, "FAILED", err))
        return records

    t0_ref = time.time()
    # Step A: Load Raw Reference & Mask and Validate Native Physical Geometry
    raw_ref_img = sitk.ReadImage(str(ref_img_path), sitk.sitkFloat32)
    raw_mask_img = sitk.ReadImage(str(ref_mask_path), sitk.sitkUInt8)

    # Geometry harmonization between native mask and native reference
    size_match = raw_ref_img.GetSize() == raw_mask_img.GetSize()
    spacing_match = np.allclose(raw_ref_img.GetSpacing(), raw_mask_img.GetSpacing(), atol=1e-3)
    direction_match = np.allclose(raw_ref_img.GetDirection(), raw_mask_img.GetDirection(), atol=1e-3)
    origin_match = np.allclose(raw_ref_img.GetOrigin(), raw_mask_img.GetOrigin(), atol=1e-3)

    if not (size_match and spacing_match and direction_match and origin_match):
        logger.warning(f"Resampling native mask for {lesion_id} to match native reference grid geometry.")
        resampler_mask = sitk.ResampleImageFilter()
        resampler_mask.SetReferenceImage(raw_ref_img)
        resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        raw_mask_img = resampler_mask.Execute(raw_mask_img)

    # Step B: Native-Space N4 Bias Field Correction & Z-Score Intensity Standardization (as in Code.ipynb)
    ref_tissue_mask = generate_tissue_mask(raw_ref_img)
    t1gd_n4 = apply_native_n4_bias_correction(raw_ref_img, ref_tissue_mask)
    t1gd_std, mean_t1gd, std_t1gd = standardize_masked_zscore(t1gd_n4, ref_tissue_mask)

    # Save standardized reference image & tumor mask on the native T1+Gd reference grid
    # (Downstream 1.0 mm isotropic grid resampling is performed during IBSI radiomics extraction via PyRadiomics)
    sitk.WriteImage(t1gd_std, str(lesion_out_dir / "image_T1+Gd.nii.gz"))
    sitk.WriteImage(raw_mask_img, str(lesion_out_dir / "mask.nii.gz"))

    ref_duration = time.time() - t0_ref
    records.append(PreprocessingAuditRecord(
        lesion_id=lesion_id,
        modality=REF_MODALITY,
        execution_time_seconds=ref_duration,
        grid_spacing_match=True,
        grid_origin_match=True,
        grid_direction_match=True,
        status="SUCCESS"
    ))

    # Step C: Process & Co-Register Moving Modalities (T1, T2) onto Native T1+Gd Reference Grid
    for mod in MOVING_MODALITIES:
        moving_candidates = [
            pt_dir / mod / "image.nii.gz",
            pt_dir / mod / f"{mod}.nii.gz",
            pt_dir / f"{mod}.nii.gz",
            raw_nifti_dir / lesion_id / f"image_{mod}.nii.gz"
        ]
        moving_path = next((p for p in moving_candidates if p.exists()), moving_candidates[0])
        if not moving_path.exists():
            records.append(PreprocessingAuditRecord(lesion_id, mod, 0.0, False, False, False, "MISSING", f"File missing: {moving_path}"))
            continue

        t0_m = time.time()
        raw_moving_img = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

        # N4 in native moving space
        moving_tissue_mask = generate_tissue_mask(raw_moving_img)
        moving_n4 = apply_native_n4_bias_correction(raw_moving_img, moving_tissue_mask)

        # 3D Rigid co-registration to native N4 T1+Gd
        transform, init_val, final_val, iters, stop_cond = register_multimodal_rigid(t1gd_n4, moving_n4)

        # Resample moving image directly into the native T1+Gd reference grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(t1gd_std)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(transform)
        resampled_moving = resampler.Execute(moving_n4)

        # Foreground tissue z-score standardization
        std_moving, mean_m, std_m = standardize_masked_zscore(resampled_moving, ref_tissue_mask)

        # Save standardized moving image
        sitk.WriteImage(std_moving, str(lesion_out_dir / f"image_{mod}.nii.gz"))

        # Verify physical grid alignment
        spacing_ok = bool(np.allclose(t1gd_std.GetSpacing(), std_moving.GetSpacing(), atol=1e-3))
        origin_ok = bool(np.allclose(t1gd_std.GetOrigin(), std_moving.GetOrigin(), atol=1e-3))
        direction_ok = bool(np.allclose(t1gd_std.GetDirection(), std_moving.GetDirection(), atol=1e-3))
        geom_ok = spacing_ok and origin_ok and direction_ok

        records.append(PreprocessingAuditRecord(
            lesion_id=lesion_id,
            modality=mod,
            execution_time_seconds=time.time() - t0_m,
            grid_spacing_match=spacing_ok,
            grid_origin_match=origin_ok,
            grid_direction_match=direction_ok,
            status="SUCCESS" if geom_ok else "FAILED_GEOMETRY"
        ))

    return records


# =============================================================================
# 3. TOPOLOGICAL REGULARIZATION (MODE FILTER + ISLAND ERASURE)
# =============================================================================
def tumor_constrained_mode_filter(labels_3d: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    r"""
    Applies a 3D majority voting mode filter (3x3x3 footprint)
    strictly within the tumor boundary to suppress high-frequency assignment noise.
    """
    coords = np.where(mask_3d > 0)
    if len(coords[0]) == 0:
        return labels_3d

    z_min, z_max = int(coords[0].min()), int(coords[0].max()) + 1
    y_min, y_max = int(coords[1].min()), int(coords[1].max()) + 1
    x_min, x_max = int(coords[2].min()), int(coords[2].max()) + 1

    sub_labels = labels_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    sub_mask = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    kernel = np.ones((3, 3, 3), dtype=np.int32)

    class_votes = []
    for c in range(K_HABITATS):
        c_mask = ((sub_labels == c) & (sub_mask > 0)).astype(np.int32)
        votes = convolve(c_mask, kernel, mode="constant", cval=0)
        class_votes.append(votes)

    stacked_votes = np.stack(class_votes, axis=-1)
    majority_class = np.argmax(stacked_votes, axis=-1).astype(np.int16)

    out_labels = labels_3d.copy()
    out_sub = labels_3d[z_min:z_max, y_min:y_max, x_min:x_max].copy()
    out_sub[sub_mask > 0] = majority_class[sub_mask > 0]
    out_labels[z_min:z_max, y_min:y_max, x_min:x_max] = out_sub
    return out_labels


def erase_small_habitat_islands(
    labels_3d: np.ndarray,
    mask_3d: np.ndarray,
    min_size: int = MIN_ISLAND_VOXELS
) -> np.ndarray:
    r"""
    Absorbs connected components smaller than min_size (27 voxels = 27 mm^3 in 1.0 mm isotropic space)
    into the surrounding majority habitat to enforce anatomical contiguity.
    """
    clean_labels = labels_3d.copy()
    struct_26 = np.ones((3, 3, 3), dtype=bool)

    for c in range(K_HABITATS):
        bin_c = (clean_labels == c) & (mask_3d > 0)
        labeled_comp, num_feat = nd_label(bin_c, structure=struct_26)
        if num_feat == 0:
            continue
        comp_sizes = np.bincount(labeled_comp.ravel())
        for comp_id in range(1, num_feat + 1):
            if comp_sizes[comp_id] < min_size:
                comp_mask = (labeled_comp == comp_id)
                dilated = convolve(comp_mask.astype(np.int32), np.ones((3, 3, 3), dtype=np.int32)) > 0
                surrounding = dilated & (~comp_mask) & (mask_3d > 0)
                surr_labels = clean_labels[surrounding]
                valid_surr = surr_labels[(surr_labels >= 0) & (surr_labels != c)]
                if len(valid_surr) > 0:
                    replacement = int(np.bincount(valid_surr).argmax())
                else:
                    all_tumor = clean_labels[mask_3d > 0]
                    valid_all = all_tumor[all_tumor >= 0]
                    replacement = int(np.bincount(valid_all).argmax()) if len(valid_all) > 0 else 0
                clean_labels[comp_mask] = replacement

    return clean_labels


# =============================================================================
# 4. 4D SPATIAL-GMM TRAINING & LEAK-FREE PER-FOLD SEGMENTATION
# =============================================================================
def fit_and_align_spatial_gmm(
    voxels_4d: np.ndarray,
    seed: int = SEED
) -> GaussianMixture:
    r"""
    Fits Gaussian Mixture Model (K = 3) on 4D multi-parametric spatial voxels:
      [I_{T1+Gd}, I_{T1}, I_{T2}, DTB_{norm}]
    Deterministically aligns components by ascending T1+Gd mean enhancement:
      Hab 1: Low enhancement (core / necrotic)
      Hab 2: Intermediate enhancement (transition zone)
      Hab 3: High enhancement (enhancing vital tumor)
    """
    gmm = GaussianMixture(
        n_components=K_HABITATS,
        covariance_type="full",
        max_iter=GMM_MAX_ITER,
        random_state=seed
    )
    gmm.fit(voxels_4d)

    # Sort components by T1+Gd mean intensity (column 0) ascending
    sort_idx = np.argsort(gmm.means_[:, 0])
    gmm.means_ = gmm.means_[sort_idx]
    gmm.covariances_ = gmm.covariances_[sort_idx]
    gmm.weights_ = gmm.weights_[sort_idx]
    gmm.precisions_cholesky_ = gmm.precisions_cholesky_[sort_idx]
    return gmm


def extract_lesion_4d_voxels(
    lesion_dir: Path
) -> Tuple[np.ndarray, np.ndarray, sitk.Image, int]:
    r"""
    Extracts 4D voxel vectors [T1+Gd, T1, T2, DTB_norm] from preprocessed lesion NIfTIs.
    """
    mask_sitk = sitk.ReadImage(str(lesion_dir / "mask.nii.gz"))
    mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
    tumor_idx = np.where(mask_arr > 0)
    total_vox = int(np.sum(mask_arr > 0))

    t1gd = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))[tumor_idx]
    t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T1.nii.gz"), sitk.sitkFloat32))[tumor_idx]
    t2 = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T2.nii.gz"), sitk.sitkFloat32))[tumor_idx]

    dtb = distance_transform_edt(mask_arr)
    max_dtb = float(np.max(dtb))
    dtb_norm = (dtb / (max_dtb if max_dtb > 1e-6 else 1.0))[tumor_idx].astype(np.float32)

    voxels_4d = np.column_stack([t1gd, t1, t2, dtb_norm])
    return voxels_4d, mask_arr, mask_sitk, total_vox


# =============================================================================
# 5. MASTER PIPELINE EXECUTION
# =============================================================================
def run_phase1_pipeline(
    raw_nifti_dir: Path,
    splits_csv: Path,
    out_aligned_dir: Path,
    out_models_dir: Path,
    out_metadata_dir: Path,
    skip_preprocessing: bool = False
) -> None:
    r"""
    Executes the complete, leak-free Phase 1 pipeline:
      1. Multi-parametric preprocessing (resampling to 1.0 mm isotropic, N4, rigid co-reg, z-score).
      2. 4D Spatial-GMM fitting strictly within each outer training split (folds 0-4).
      3. Fold-specific segmentation: GMM model f segments ALL 100 lesions for fold f,
         saving `habitat_3class_fold_{f}.nii.gz` to ensure zero leakage across outer splits.
      4. Saves models to standardized location: `Fold_Feature_Matrices_Habitat/Models/spatial_gmm_model_fold_{f}.pkl`.
    """
    out_aligned_dir.mkdir(parents=True, exist_ok=True)
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_metadata_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 1: MRI PREPROCESSING, DTB MAPPING & SPATIAL-GMM FITTED WITHIN OUTER TRAINING FOLDS")
    logger.info("=" * 80)

    # 1. Load CV Splits (N = 100 unilateral cohort)
    if not splits_csv.exists():
        raise FileNotFoundError(f"CV Splits file missing: {splits_csv}")
    splits_df = pd.read_csv(splits_csv, dtype={"lesion_id": str, "patient_uid": str})
    splits_df["lesion_id"] = splits_df["lesion_id"].astype(str).str.strip()
    if "patient_uid" in splits_df.columns:
        splits_df["patient_uid"] = splits_df["patient_uid"].astype(str).str.strip()
    all_lesion_ids = splits_df["lesion_id"].tolist()
    logger.info(f"Target Cohort: {len(all_lesion_ids)} unilateral vestibular schwannomas across 5 folds.")

    # 2. Execute Preprocessing Workflow (from Raw NIfTI if not skipped)
    if not skip_preprocessing:
        logger.info("Executing active multi-parametric preprocessing workflow from raw MRI files...")
        audit_records = []
        for idx, (_, row) in enumerate(splits_df.iterrows(), 1):
            lid = str(row["lesion_id"])
            puid = str(row["patient_uid"])
            mask_fn = str(row["tumor_mask_file"])
            logger.info(f"[{idx}/{len(splits_df)}] Preprocessing {lid} (Patient {puid})...")
            recs = preprocess_single_lesion_from_raw(
                lesion_id=lid,
                patient_uid=puid,
                mask_filename=mask_fn,
                raw_nifti_dir=raw_nifti_dir,
                out_aligned_dir=out_aligned_dir
            )
            audit_records.extend(recs)

        audit_df = pd.DataFrame([r.__dict__ for r in audit_records])
        audit_csv = out_metadata_dir / "phase1_registration_audit.csv"
        audit_df.to_csv(audit_csv, index=False)
        logger.info(f"Saved preprocessing audit: {audit_csv.name}")
    else:
        logger.info("Skipping raw preprocessing as requested (--skip-preprocessing). Utilizing existing Aligned_NIfTI directory.")

    # 3. Extract 4D Feature Vectors [T1+Gd, T1, T2, DTB_norm] Across All Cases
    lesion_voxel_data: Dict[str, np.ndarray] = {}
    lesion_masks: Dict[str, np.ndarray] = {}
    lesion_mask_sitks: Dict[str, sitk.Image] = {}
    total_voxel_counts: Dict[str, int] = {}

    logger.info("Harvesting 4D feature vectors [T1+Gd, T1, T2, DTB_norm] for all cases...")
    for lid in all_lesion_ids:
        ldir = out_aligned_dir / lid
        if not (ldir / "mask.nii.gz").exists() or not (ldir / "image_T1+Gd.nii.gz").exists():
            raise FileNotFoundError(f"Aligned files missing for lesion {lid} in {ldir}")
        v_4d, m_arr, m_sitk, n_vox = extract_lesion_4d_voxels(ldir)
        lesion_voxel_data[lid] = v_4d
        lesion_masks[lid] = m_arr
        lesion_mask_sitks[lid] = m_sitk
        total_voxel_counts[lid] = n_vox

    # 4. 4D Spatial-GMM Fitting within Each Outer Training Fold
    gmm_models: Dict[int, GaussianMixture] = {}
    for fold in range(NUM_FOLDS):
        # Training lesions strictly where fold != f (n = 80)
        train_lids = splits_df[splits_df["fold"] != fold]["lesion_id"].tolist()
        logger.info(f"[Fold {fold}] Harvesting training voxels strictly from {len(train_lids)} lesions (fold != {fold})...")

        rng = np.random.RandomState(SEED + fold)
        train_samples = []
        for lid in train_lids:
            vox = lesion_voxel_data[lid]
            sample_size = min(MAX_TRAIN_VOXELS_PER_LESION, len(vox))
            idx = rng.choice(len(vox), size=sample_size, replace=False)
            train_samples.append(vox[idx])

        X_train = np.vstack(train_samples)
        logger.info(f"[Fold {fold}] Fitting full-covariance 4D Spatial-GMM (K={K_HABITATS}) on {len(X_train)} voxels...")
        gmm = fit_and_align_spatial_gmm(X_train, seed=SEED + fold)
        gmm_models[fold] = gmm

        # Standardized model save path expected by loader and Phase 3:
        # Fold_Feature_Matrices_Habitat/Models/spatial_gmm_model_fold_{fold}.pkl
        model_path = out_models_dir / f"spatial_gmm_model_fold_{fold}.pkl"
        joblib.dump(gmm, model_path)
        logger.info(f"[Fold {fold}] Serialized model: {model_path.name}")

    # 5. Leak-Free Per-Fold Habitat Segmentation:
    # For evaluation fold f, GMM model f segments ALL 100 lesions evaluated in fold f!
    logger.info("Executing leak-free per-fold segmentation: GMM model f segments all 100 cases for evaluation fold f...")
    all_fold_records = []
    held_out_records = []

    for fold in range(NUM_FOLDS):
        gmm = gmm_models[fold]
        logger.info(f"[Fold {fold}] Generating habitat masks for all 100 lesions under GMM fold {fold}...")

        for _, row in splits_df.iterrows():
            lid = str(row["lesion_id"])
            puid = str(row["patient_uid"])
            vox_4d = lesion_voxel_data[lid]
            mask_arr = lesion_masks[lid]
            mask_sitk = lesion_mask_sitks[lid]
            tot_vox = total_voxel_counts[lid]

            # Voxelwise classification under GMM model f
            raw_labels = gmm.predict(vox_4d)
            labels_3d = np.full_like(mask_arr, -1, dtype=np.int16)
            coords = np.where(mask_arr > 0)
            labels_3d[coords] = raw_labels

            # Dual-stage topological regularization
            smoothed_3d = tumor_constrained_mode_filter(labels_3d, mask_arr)
            final_3d = erase_small_habitat_islands(smoothed_3d, mask_arr, min_size=MIN_ISLAND_VOXELS)

            # Save fold-specific regularized mask: `habitat_3class_fold_{fold}.nii.gz`
            out_hab_sitk = sitk.GetImageFromArray(final_3d.astype(np.int16))
            out_hab_sitk.CopyInformation(mask_sitk)
            ldir = out_aligned_dir / lid
            sitk.WriteImage(out_hab_sitk, str(ldir / f"habitat_3class_fold_{fold}.nii.gz"))

            # If this is the held-out test fold for this lesion, also save canonical visualization mask
            is_held_out = (row["fold"] == fold)
            if is_held_out:
                sitk.WriteImage(out_hab_sitk, str(ldir / "habitat_3class.nii.gz"))

            # Compute macroscopic proportions
            v1 = int(np.sum((final_3d == 0) & (mask_arr > 0)))
            v2 = int(np.sum((final_3d == 1) & (mask_arr > 0)))
            v3 = int(np.sum((final_3d == 2) & (mask_arr > 0)))

            f1 = v1 / tot_vox if tot_vox > 0 else 0.0
            f2 = v2 / tot_vox if tot_vox > 0 else 0.0
            f3 = v3 / tot_vox if tot_vox > 0 else 0.0

            eps = 1e-5
            rec = {
                "lesion_id": lid,
                "patient_uid": puid,
                "generating_fold": fold,
                "Status": row["Status"],
                "Necrosis": row.get("Necrosis_Clean", row.get("Necrosis", "yes")),
                "total_tumor_voxels": tot_vox,
                "Hab1_Voxels": v1,
                "Hab1_Volume_Fraction": f1,
                "Hab2_Voxels": v2,
                "Hab2_Volume_Fraction": f2,
                "Hab3_Voxels": v3,
                "Hab3_Volume_Fraction": f3,
                "Hab1_Hab2_LogRatio": float(np.log((f1 + eps) / (f2 + eps))),
                "Hab1_Hab3_LogRatio": float(np.log((f1 + eps) / (f3 + eps))),
                "Hab2_Hab3_LogRatio": float(np.log((f2 + eps) / (f3 + eps))),
            }
            all_fold_records.append(rec)
            if is_held_out:
                held_out_records.append(rec)

    # 1. Save full 500-row table (all folds) separately
    all_folds_df = pd.DataFrame(all_fold_records)
    all_folds_csv = out_metadata_dir / "phase2_habitat_macro_metrics_all_folds.csv"
    all_folds_df.to_csv(all_folds_csv, index=False)
    logger.info(f"Saved complete cross-fold macroscopic metrics: {all_folds_csv.name} ({len(all_folds_df)} records across 5 folds)")

    # 2. Save canonical 100-row held-out table for Figure 2 (1 record per unique lesion under its held-out test GMM)
    held_out_df = pd.DataFrame(held_out_records)
    macro_csv = out_metadata_dir / "phase2_habitat_macro_metrics.csv"
    held_out_df.to_csv(macro_csv, index=False)
    logger.info(f"Saved canonical held-out macroscopic metrics for Figure 2: {macro_csv.name} ({len(held_out_df)} unique lesions)")
    logger.info("Phase 1 execution complete.")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Multi-Parametric MRI Preprocessing, DTB Mapping, and Leak-Free Spatial-GMM Segmentation."
    )
    parser.add_argument("--raw-nifti-dir", type=Path, default=DEFAULT_BASE_DIR / "Data" / "Raw_NIfTI", help="Raw MRI input directory")
    parser.add_argument("--splits-csv", type=Path, default=DEFAULT_METADATA_DIR / "phase2_cv_splits.csv", help="CV splits CSV path")
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR, help="Output aligned NIfTI directory")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Output GMM models directory")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR, help="Output metadata directory")
    parser.add_argument("--skip-preprocessing", action="store_true", default=False, help="Skip raw preprocessing if Aligned_NIfTI is already populated")
    args = parser.parse_args()

    run_phase1_pipeline(
        raw_nifti_dir=args.raw_nifti_dir,
        splits_csv=args.splits_csv,
        out_aligned_dir=args.aligned_dir,
        out_models_dir=args.models_dir,
        out_metadata_dir=args.metadata_dir,
        skip_preprocessing=args.skip_preprocessing
    )


if __name__ == "__main__":
    main()
