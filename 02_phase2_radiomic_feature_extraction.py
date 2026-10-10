#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
02_phase2_radiomic_feature_extraction.py

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

try:
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    radiomics.setVerbosity(logging.ERROR)
except ImportError:
    sitk = None
    featureextractor = None

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase2_Feature_Extraction")

# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS & DIRECTORY PATHS
# -----------------------------------------------------------------------------
REF_MODALITY = "T1+Gd"
MODALITIES = ("T1+Gd", "T1", "T2")
K_HABITATS = 3
MIN_VOXELS_FOR_RADIOMICS = 50     # Zero-imputation threshold
NUM_FOLDS = 5
DEFAULT_NUM_WORKERS = 6

DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "Production_Pipeline" / "Data"
DEFAULT_ALIGNED_DIR = DEFAULT_DATA_DIR / "Aligned_NIfTI"
DEFAULT_METADATA_DIR = DEFAULT_DATA_DIR / "Metadata"
DEFAULT_RESULTS_DIR = DEFAULT_BASE_DIR / "Production_Pipeline" / "Results"
DEFAULT_TABLES_DIR = DEFAULT_RESULTS_DIR / "Tables"

DEFAULT_WT_MATRICES_DIR = DEFAULT_DATA_DIR / "Fold_Feature_Matrices_WholeTumor"
DEFAULT_HAB_MATRICES_DIR = DEFAULT_DATA_DIR / "Fold_Feature_Matrices_Habitat"


# =============================================================================
# 1. CANONICAL RADIOMIC FEATURE SCHEMA (EXACTLY 293 STEMS PER COMPARTMENT)
# =============================================================================
SHAPE_FEATURE_NAMES = [
    "Elongation", "Flatness", "LeastAxisLength", "MajorAxisLength",
    "Maximum2DDiameterColumn", "Maximum2DDiameterRow", "Maximum2DDiameterSlice",
    "Maximum3DDiameter", "MeshVolume", "MinorAxisLength", "Sphericity",
    "SurfaceArea", "SurfaceVolumeRatio", "VoxelVolume"
]  # 14 shape features

TEXTURE_CLASSES_AND_FEATURES = {
    "firstorder": [
        "10Percentile", "90Percentile", "Energy", "Entropy", "InterquartileRange",
        "Kurtosis", "Maximum", "MeanAbsoluteDeviation", "Mean", "Median",
        "Minimum", "Range", "RobustMeanAbsoluteDeviation", "RootMeanSquared",
        "Skewness", "TotalEnergy", "Uniformity", "Variance"
    ],  # 18 first-order features
    "glcm": [
        "Autocorrelation", "ClusterProminence", "ClusterShade", "ClusterTendency",
        "Contrast", "Correlation", "DifferenceAverage", "DifferenceEntropy",
        "DifferenceVariance", "Id", "Idm", "Idmn", "Idn", "Imc1", "Imc2",
        "InverseVariance", "JointAverage", "JointEnergy", "JointEntropy",
        "MCC", "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares"
    ],  # 24 GLCM features
    "glrlm": [
        "GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
        "HighGrayLevelRunEmphasis", "LongRunEmphasis", "LongRunHighGrayLevelEmphasis",
        "LongRunLowGrayLevelEmphasis", "LowGrayLevelRunEmphasis", "RunEntropy",
        "RunLengthNonUniformity", "RunLengthNonUniformityNormalized", "RunPercentage",
        "RunVariance", "ShortRunEmphasis", "ShortRunHighGrayLevelEmphasis",
        "ShortRunLowGrayLevelEmphasis"
    ],  # 16 GLRLM features
    "glszm": [
        "GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
        "HighGrayLevelZoneEmphasis", "LargeAreaEmphasis", "LargeAreaHighGrayLevelEmphasis",
        "LargeAreaLowGrayLevelEmphasis", "LowGrayLevelZoneEmphasis", "SizeZoneNonUniformity",
        "SizeZoneNonUniformityNormalized", "SmallAreaEmphasis", "SmallAreaHighGrayLevelEmphasis",
        "SmallAreaLowGrayLevelEmphasis", "ZoneEntropy", "ZonePercentage", "ZoneVariance"
    ],  # 16 GLSZM features
    "gldm": [
        "DependenceEntropy", "DependenceNonUniformity", "DependenceNonUniformityNormalized",
        "DependenceVariance", "GrayLevelNonUniformity", "GrayLevelVariance",
        "HighGrayLevelEmphasis", "LargeDependenceEmphasis", "LargeDependenceHighGrayLevelEmphasis",
        "LargeDependenceLowGrayLevelEmphasis", "LowGrayLevelEmphasis", "SmallDependenceEmphasis",
        "SmallDependenceHighGrayLevelEmphasis", "SmallDependenceLowGrayLevelEmphasis"
    ],  # 14 GLDM features
    "ngtdm": [
        "Busyness", "Coarseness", "Complexity", "Contrast", "Strength"
    ]  # 5 NGTDM features
}  # Total texture features per sequence = 18 + 24 + 16 + 16 + 14 + 5 = 93


def build_canonical_compartment_stems() -> List[str]:
    r"""Builds the canonical list of exactly 293 radiomic feature stems per compartment."""
    stems = []
    # 14 shape features
    for s in SHAPE_FEATURE_NAMES:
        stems.append(f"original_shape_{s}")
    # 93 texture features per sequence (3 sequences = 279 texture features)
    for mod in MODALITIES:
        for f_class, f_list in TEXTURE_CLASSES_AND_FEATURES.items():
            for fn in f_list:
                stems.append(f"{mod}_original_{f_class}_{fn}")
    assert len(stems) == 293, f"Expected 293 feature stems, got {len(stems)}"
    return stems


CANONICAL_293_STEMS = build_canonical_compartment_stems()


def build_all_1178_eligible_features() -> List[str]:
    r"""Builds the full eligible list of exactly 1,178 candidate radiomic features."""
    eligible = []
    # 1. Whole-Tumor (293 features)
    for stem in CANONICAL_293_STEMS:
        eligible.append(f"WholeTumor_{stem}")

    # 2. 3-Habitat Sub-Compartments (3 x 293 = 879 features)
    for h in (1, 2, 3):
        for stem in CANONICAL_293_STEMS:
            eligible.append(f"Hab{h}_{stem}")

    # 3. Macroscopic Proportions & Log-Ratios (6 features)
    macro_features = [
        "Hab1_Volume_Fraction", "Hab2_Volume_Fraction", "Hab3_Volume_Fraction",
        "Hab1_Hab2_LogRatio", "Hab1_Hab3_LogRatio", "Hab2_Hab3_LogRatio"
    ]
    eligible.extend(macro_features)

    assert len(eligible) == 1178, f"Expected exactly 1,178 eligible features, got {len(eligible)}"
    return eligible


ELIGIBLE_1178_RADIOMIC_FEATURES = build_all_1178_eligible_features()


# =============================================================================
# 2. THREAD-SAFE PYRADIOMICS EXTRACTOR FACTORY
# =============================================================================
_thread_local = threading.local()


def get_thread_extractor() -> Any:
    r"""Provides an isolated, IBSI-standardized PyRadiomics extractor per thread."""
    if not hasattr(_thread_local, "extractor"):
        if featureextractor is None:
            raise ImportError("PyRadiomics is not installed. Run 'pip install pyradiomics'.")
        settings = {
            "binWidth": 0.25,
            "voxelArrayShift": 0.0,
            "interpolator": sitk.sitkBSpline,
            "resampledPixelSpacing": [1.0, 1.0, 1.0],
            "padDistance": 5,
            "normalize": False,  # Pre-standardized via foreground tissue z-score
        }
        ext = featureextractor.RadiomicsFeatureExtractor(**settings)
        ext.disableAllImageTypes()
        ext.enableImageTypeByName("Original")
        ext.disableAllFeatures()
        ext.enableFeatureClassByName("shape")
        ext.enableFeatureClassByName("firstorder")
        ext.enableFeatureClassByName("glcm")
        ext.enableFeatureClassByName("glrlm")
        ext.enableFeatureClassByName("glszm")
        ext.enableFeatureClassByName("gldm")
        ext.enableFeatureClassByName("ngtdm")
        _thread_local.extractor = ext
    return _thread_local.extractor


# =============================================================================
# 3. EXTRACTION WORKERS: WHOLE-TUMOR & FOLD-AWARE HABITATS
# =============================================================================
def validate_feature_value(
    val: Any,
    feature_name: str = "",
    compartment: str = "",
    lesion_id: str = ""
) -> float:
    r"""
    Validates that an extracted feature value is present, convertible, and finite numeric float.
    Raises ValueError if None, non-numeric, or non-finite (NaN / Inf).
    """
    if val is None:
        raise ValueError(f"[{compartment}] Feature '{feature_name}' is None for lesion '{lesion_id}'.")
    try:
        f = float(val)
    except (TypeError, ValueError) as err:
        raise ValueError(f"[{compartment}] Feature '{feature_name}' cannot be converted to float for lesion '{lesion_id}': {err}")
    if np.isnan(f) or np.isinf(f):
        raise ValueError(f"[{compartment}] Feature '{feature_name}' returned non-finite value ({f}) for lesion '{lesion_id}'.")
    return f


def safe_float(
    val: Any,
    feature_name: str = "",
    compartment: str = "",
    lesion_id: str = ""
) -> float:
    r"""
    Validates and converts a value to finite float.
    Strictly raises ValueError on NaN, Inf, None, or conversion error to prevent silent zero-corruption.
    """
    return validate_feature_value(val, feature_name=feature_name, compartment=compartment, lesion_id=lesion_id)


def get_zero_imputed_wholetumor_dict() -> Dict[str, float]:
    r"""Returns a complete dictionary of all 293 Whole-Tumor radiomic features populated with 0.0."""
    return {f"WholeTumor_{stem}": 0.0 for stem in CANONICAL_293_STEMS}


def get_zero_imputed_habitat_dict() -> Dict[str, float]:
    r"""Returns a complete dictionary of all 885 Habitat candidate features populated with 0.0."""
    d: Dict[str, float] = {}
    for h in (1, 2, 3):
        d[f"Hab{h}_Voxels"] = 0.0
        d[f"Hab{h}_Volume_Fraction"] = 0.0
        d[f"Hab{h}_Extraction_Valid"] = 0.0
        for stem in CANONICAL_293_STEMS:
            d[f"Hab{h}_{stem}"] = 0.0
    d["Hab1_Volume_Fraction"] = 0.0
    d["Hab2_Volume_Fraction"] = 0.0
    d["Hab3_Volume_Fraction"] = 0.0
    d["Hab1_Hab2_LogRatio"] = 0.0
    d["Hab1_Hab3_LogRatio"] = 0.0
    d["Hab2_Hab3_LogRatio"] = 0.0
    return d


def extract_wholetumor_radiomics_single(
    lesion_id: str,
    aligned_dir: Path
) -> Dict[str, float]:
    r"""
    Extracts exactly 293 Whole-Tumor radiomic features adhering to canonical schema.
    Requires every expected feature key and a finite numeric value; raises explicit exceptions
    (KeyError, ValueError, RuntimeError) on any missing keys, non-finite values, or extraction failures.
    """
    ldir = aligned_dir / lesion_id
    mask_path = ldir / "mask.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing whole-tumor mask for lesion {lesion_id}: {mask_path}")

    img_paths = {m: ldir / f"image_{m}.nii.gz" for m in MODALITIES}
    for m, p in img_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {m} image for lesion {lesion_id}: {p}")

    mask_sitk = sitk.ReadImage(str(mask_path))
    img_dict = {m: sitk.ReadImage(str(p), sitk.sitkFloat32) for m, p in img_paths.items()}

    extractor = get_thread_extractor()
    features: Dict[str, float] = {}

    # Extract 14 shape features from mask
    try:
        shape_res = extractor.execute(img_dict[REF_MODALITY], mask_sitk, label=1)
    except Exception as e:
        raise RuntimeError(f"Whole-tumor shape extraction technical failure for {lesion_id}: {e}")

    for s in SHAPE_FEATURE_NAMES:
        key = f"original_shape_{s}"
        if key not in shape_res:
            raise KeyError(f"[WholeTumor] Missing required shape feature '{key}' for lesion '{lesion_id}'.")
        features[f"WholeTumor_{key}"] = validate_feature_value(shape_res[key], key, "WholeTumor", lesion_id)

    # Extract 93 texture features per modality (3 x 93 = 279 texture features)
    for mod in MODALITIES:
        try:
            res = extractor.execute(img_dict[mod], mask_sitk, label=1)
        except Exception as e:
            raise RuntimeError(f"Whole-tumor texture extraction technical failure for {lesion_id} ({mod}): {e}")

        for f_class, f_list in TEXTURE_CLASSES_AND_FEATURES.items():
            for fn in f_list:
                pyrad_key = f"original_{f_class}_{fn}"
                if pyrad_key not in res:
                    raise KeyError(f"[WholeTumor] Missing required texture feature '{pyrad_key}' ({mod}) for lesion '{lesion_id}'.")
                features[f"WholeTumor_{mod}_{pyrad_key}"] = validate_feature_value(
                    res[pyrad_key], pyrad_key, f"WholeTumor_{mod}", lesion_id
                )

    if len(features) != 293:
        raise ValueError(f"[WholeTumor] Feature count mismatch for {lesion_id}: expected 293, got {len(features)}")

    return features


def extract_habitat_radiomics_for_fold(
    lesion_id: str,
    fold: int,
    aligned_dir: Path
) -> Dict[str, float]:
    r"""
    Extracts exactly 885 Habitat radiomic features for a given lesion under evaluation fold f
    using `habitat_3class_fold_{fold}.nii.gz`.
    
    Adheres strictly to the evaluation split:
      - Requires the exact fold-specific mask `habitat_3class_fold_{fold}.nii.gz`.
      - Does NOT fall back to generic `habitat_3class.nii.gz` to avoid representation leakage.
      - Genuinely small sub-compartments (< 50 voxels) are legitimately zero-imputed with
        Hab*_Extraction_Valid = 0.0.
      - Any technical failure (missing files, read errors, extraction crash on valid habitat)
        raises an explicit exception so the pipeline can stop before producing corrupted matrices.
    """
    ldir = aligned_dir / lesion_id
    hab_mask_path = ldir / f"habitat_3class_fold_{fold}.nii.gz"
    if not hab_mask_path.exists():
        raise FileNotFoundError(
            f"[Fold {fold}] Required fold-specific habitat mask missing for lesion {lesion_id}: {hab_mask_path.name}. "
            f"Silently falling back to generic mask is prohibited to prevent representation leakage."
        )

    mask_path = ldir / "mask.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"[Fold {fold}] Whole-tumor mask missing for lesion {lesion_id}: {mask_path}")

    img_paths = {m: ldir / f"image_{m}.nii.gz" for m in MODALITIES}
    for m, p in img_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"[Fold {fold}] Missing {m} image for lesion {lesion_id}: {p}")

    hab_sitk = sitk.ReadImage(str(hab_mask_path))
    hab_arr = sitk.GetArrayFromImage(hab_sitk)

    mask_sitk = sitk.ReadImage(str(mask_path))
    mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
    tot_vox = int(np.sum(mask_arr > 0))

    img_dict = {m: sitk.ReadImage(str(p), sitk.sitkFloat32) for m, p in img_paths.items()}

    extractor = get_thread_extractor()
    features: Dict[str, float] = {}
    hab_fractions = {}

    for h_idx in range(K_HABITATS):
        hab_num = h_idx + 1
        sub_mask_arr = ((hab_arr == h_idx) & (mask_arr > 0)).astype(np.uint8)
        hab_vox = int(np.sum(sub_mask_arr))
        frac = hab_vox / tot_vox if tot_vox > 0 else 0.0
        hab_fractions[hab_num] = frac

        features[f"Hab{hab_num}_Voxels"] = float(hab_vox)
        features[f"Hab{hab_num}_Volume_Fraction"] = float(frac)

        # COMPLETE ZERO-IMPUTATION FOR GENUINELY SMALL / ABSENT HABITATS (< 50 voxels)
        if hab_vox < MIN_VOXELS_FOR_RADIOMICS:
            features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
            for stem in CANONICAL_293_STEMS:
                features[f"Hab{hab_num}_{stem}"] = 0.0
            continue

        # VALID HABITAT (>= 50 voxels): MUST EXTRACT CLEANLY
        features[f"Hab{hab_num}_Extraction_Valid"] = 1.0
        sub_sitk = sitk.GetImageFromArray(sub_mask_arr)
        sub_sitk.CopyInformation(mask_sitk)

        # 14 shape features
        try:
            shape_res = extractor.execute(img_dict[REF_MODALITY], sub_sitk, label=1)
        except Exception as e:
            features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
            raise RuntimeError(f"[Fold {fold}] Hab{hab_num} shape extraction technical failure for {lesion_id}: {e}")

        for s in SHAPE_FEATURE_NAMES:
            key = f"original_shape_{s}"
            if key not in shape_res:
                features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
                raise KeyError(f"[Fold {fold}] Missing required shape feature '{key}' in Hab{hab_num} for lesion '{lesion_id}'.")
            try:
                features[f"Hab{hab_num}_{key}"] = validate_feature_value(
                    shape_res[key], key, f"Hab{hab_num}", lesion_id
                )
            except Exception:
                features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
                raise

        # 93 texture features per modality
        for mod in MODALITIES:
            try:
                res = extractor.execute(img_dict[mod], sub_sitk, label=1)
            except Exception as e:
                features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
                raise RuntimeError(f"[Fold {fold}] Hab{hab_num} texture extraction technical failure for {lesion_id} ({mod}): {e}")

            for f_class, f_list in TEXTURE_CLASSES_AND_FEATURES.items():
                for fn in f_list:
                    pyrad_key = f"original_{f_class}_{fn}"
                    if pyrad_key not in res:
                        features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
                        raise KeyError(f"[Fold {fold}] Missing required texture feature '{pyrad_key}' ({mod}) in Hab{hab_num} for lesion '{lesion_id}'.")
                    try:
                        features[f"Hab{hab_num}_{mod}_{pyrad_key}"] = validate_feature_value(
                            res[pyrad_key], pyrad_key, f"Hab{hab_num}_{mod}", lesion_id
                        )
                    except Exception:
                        features[f"Hab{hab_num}_Extraction_Valid"] = 0.0
                        raise

    # 6 Physiological Macroscopic Features
    eps = 1e-5
    f1 = hab_fractions.get(1, 0.0)
    f2 = hab_fractions.get(2, 0.0)
    f3 = hab_fractions.get(3, 0.0)
    features["Hab1_Volume_Fraction"] = float(f1)
    features["Hab2_Volume_Fraction"] = float(f2)
    features["Hab3_Volume_Fraction"] = float(f3)
    features["Hab1_Hab2_LogRatio"] = validate_feature_value(np.log((f1 + eps) / (f2 + eps)), "Hab1_Hab2_LogRatio", "Macro", lesion_id)
    features["Hab1_Hab3_LogRatio"] = validate_feature_value(np.log((f1 + eps) / (f3 + eps)), "Hab1_Hab3_LogRatio", "Macro", lesion_id)
    features["Hab2_Hab3_LogRatio"] = validate_feature_value(np.log((f2 + eps) / (f3 + eps)), "Hab2_Hab3_LogRatio", "Macro", lesion_id)

    return features


# =============================================================================
# HELPER FUNCTIONS: SPATIAL GMM MODEL LOADING & LESION DECOMPOSITION
# (Supplied for seamless downstream figure generation and modular compatibility)
# =============================================================================
def load_spatial_gmm_model(
    fold: int,
    models_dir: Optional[Path] = None
) -> Any:
    r"""Loads the pre-trained Spatial-GMM model for a given outer fold."""
    import joblib
    target_dir = models_dir or (DEFAULT_HAB_MATRICES_DIR / "Models")
    model_path = target_dir / f"spatial_gmm_model_fold_{fold}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def tumor_constrained_mode_filter(labels_3d: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    r"""Applies a 3x3x3 majority mode filter constrained strictly within the tumor envelope."""
    from scipy.ndimage import convolve
    footprint = np.ones((3, 3, 3), dtype=np.int32)
    counts = np.zeros((K_HABITATS, *labels_3d.shape), dtype=np.int32)
    for h in range(K_HABITATS):
        counts[h] = convolve(((labels_3d == h) & (mask_arr > 0)).astype(np.int32), footprint, mode='constant', cval=0)

    majority = np.argmax(counts, axis=0)
    smoothed = np.where(mask_arr > 0, majority, -1)
    return smoothed.astype(np.int16)


def erase_small_habitat_islands(labels_3d: np.ndarray, mask_arr: np.ndarray, min_size: int = 27) -> np.ndarray:
    r"""Removes spurious micro-clusters smaller than min_size voxels, reassigning to dominant neighbor."""
    from scipy.ndimage import convolve, generate_binary_structure, label as nd_label
    cleaned = labels_3d.copy()
    s_struct = generate_binary_structure(3, 3)
    footprint = np.ones((3, 3, 3), dtype=np.int32)

    for h_idx in range(K_HABITATS):
        bin_hab = (cleaned == h_idx) & (mask_arr > 0)
        labeled_cc, num_cc = nd_label(bin_hab, structure=s_struct)
        if num_cc == 0:
            continue
        cc_sizes = np.bincount(labeled_cc.ravel())
        small_mask = np.isin(labeled_cc, np.where((cc_sizes < min_size) & (cc_sizes > 0))[0])
        if np.any(small_mask):
            counts = np.zeros((K_HABITATS, *labels_3d.shape), dtype=np.int32)
            for other_h in range(K_HABITATS):
                if other_h != h_idx:
                    counts[other_h] = convolve(((cleaned == other_h) & (mask_arr > 0)).astype(np.int32), footprint, mode='constant', cval=0)
            best_other = np.argmax(counts, axis=0)
            cleaned[small_mask] = best_other[small_mask]

    return cleaned


def decompose_lesion_habitats(
    lesion_dir: Path,
    gmm_model: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    r"""
    Decomposes a single lesion into 3 spatial-multiparametric habitat sub-compartments.
    """
    from scipy.ndimage import distance_transform_edt
    if sitk is None:
        raise ImportError("SimpleITK is required for lesion habitat decomposition.")

    mask_sitk = sitk.ReadImage(str(lesion_dir / "mask.nii.gz"))
    mask_arr = (sitk.GetArrayFromImage(mask_sitk) > 0).astype(np.uint8)
    tumor_idx = np.where(mask_arr > 0)
    total_voxels = int(np.sum(mask_arr > 0))
    if total_voxels == 0:
        raise ValueError(f"No positive tumor voxels found in {lesion_dir}")

    t1gd_img = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T1+Gd.nii.gz"), sitk.sitkFloat32))
    t1_img = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T1.nii.gz"), sitk.sitkFloat32))
    t2_img = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_dir / "image_T2.nii.gz"), sitk.sitkFloat32))

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

    X_vox = np.column_stack([
        t1gd_img[tumor_idx],
        t1_img[tumor_idx],
        t2_img[tumor_idx],
        dtb_norm_vox
    ])

    raw_labels = gmm_model.predict(X_vox)

    sub_labels = np.full_like(sub_mask, -1, dtype=np.int16)
    sub_labels[z_coords - z_min, y_coords - y_min, x_coords - x_min] = raw_labels

    sub_smoothed = tumor_constrained_mode_filter(sub_labels, sub_mask)
    sub_final = erase_small_habitat_islands(sub_smoothed, sub_mask, min_size=27)

    final_3d = np.full_like(mask_arr, -1, dtype=np.int16)
    final_3d[z_min:z_max, y_min:y_max, x_min:x_max] = np.where(sub_mask > 0, sub_final, -1)

    macro_metrics = {"total_tumor_voxels": total_voxels}
    for h_idx in range(K_HABITATS):
        hab_vox = int(np.sum((final_3d == h_idx) & (mask_arr > 0)))
        frac = hab_vox / total_voxels
        macro_metrics[f"Hab{h_idx + 1}_Volume_Fraction"] = frac
        macro_metrics[f"Hab{h_idx + 1}_Voxels"] = hab_vox

    eps = 1e-5
    v1 = macro_metrics["Hab1_Volume_Fraction"]
    v2 = macro_metrics["Hab2_Volume_Fraction"]
    v3 = macro_metrics["Hab3_Volume_Fraction"]
    macro_metrics["Hab1_Hab2_LogRatio"] = float(np.log((v1 + eps) / (v2 + eps)))
    macro_metrics["Hab1_Hab3_LogRatio"] = float(np.log((v1 + eps) / (v3 + eps)))
    macro_metrics["Hab2_Hab3_LogRatio"] = float(np.log((v2 + eps) / (v3 + eps)))

    return final_3d, mask_arr, macro_metrics


# =============================================================================
# 4. RESTRICTED UNIVARIABLE FDR DISCOVERY AUDIT
# =============================================================================
def compute_univariable_fdr_audit(
    feature_df: pd.DataFrame,
    eligible_feature_list: List[str],
    outcome_col: str = "Status"
) -> pd.DataFrame:
    r"""
    Executes univariable False Discovery Rate screening strictly restricted to the
    explicit list of eligible radiomic candidate features (exactly 1,178 features):
      - Excludes identifiers, clinical variables, and voxel counts.
      - Two-sided Mann-Whitney U test statistic and unadjusted p-value.
      - Directional AUC: max(AUC, 1 - AUC).
      - Benjamini-Hochberg FDR correction (q < 0.05).
    """
    logger.info(f"Running restricted univariable FDR audit on exactly {len(eligible_feature_list)} radiomic features...")
    assert len(eligible_feature_list) == 1178, f"Expected 1,178 features, got {len(eligible_feature_list)}"

    y_raw = feature_df[outcome_col].astype(str).str.strip().str.lower()
    y = np.where(y_raw.isin(["enlargement", "1", "true"]), 1, 0)
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    assert n_pos > 0 and n_neg > 0, "Both enlargement and non-enlargement cases required"

    audit_rows = []
    for f in eligible_feature_list:
        if f not in feature_df.columns:
            logger.warning(f"Feature {f} missing from feature_df; populating with zeros.")
            vals = np.zeros(len(feature_df), dtype=float)
        else:
            vals = pd.to_numeric(feature_df[f], errors="coerce").fillna(0.0).values

        v_pos = vals[pos_mask]
        v_neg = vals[neg_mask]

        try:
            stat, p_val = mannwhitneyu(v_pos, v_neg, alternative="two-sided")
            auc = stat / (n_pos * n_neg)
            dir_auc = max(auc, 1.0 - auc)
        except Exception:
            stat, p_val, dir_auc = 0.0, 1.0, 0.5

        # Infer compartment, sequence, and feature class
        if f.startswith("WholeTumor_"): comp = "Whole-Tumor"
        elif f.startswith("Hab1_"): comp = "Habitat 1"
        elif f.startswith("Hab2_"): comp = "Habitat 2"
        elif f.startswith("Hab3_"): comp = "Habitat 3"
        else: comp = "Habitat"

        if "Volume_Fraction" in f or "LogRatio" in f:
            seq = "Morphologic"
            f_class = "Macro Proportions"
        elif "shape" in f:
            seq = "Morphologic"
            f_class = "Shape-3D"
        else:
            seq = "T1+Gd" if "T1+Gd" in f else ("T1 (Native)" if "_T1_" in f else "T2")
            if "firstorder" in f: f_class = "First-Order"
            elif "glcm" in f: f_class = "GLCM"
            elif "glrlm" in f: f_class = "GLRLM"
            elif "glszm" in f: f_class = "GLSZM"
            elif "gldm" in f: f_class = "GLDM"
            elif "ngtdm" in f: f_class = "NGTDM"
            else: f_class = "Texture"

        audit_rows.append({
            "Feature": f,
            "Compartment": comp,
            "Sequence": seq,
            "Feature_Class": f_class,
            "AUC": dir_auc,
            "Delta_AUC": abs(dir_auc - 0.5),
            "Mann_Whitney_U": stat,
            "Unadjusted_p": p_val,
            "p_val": p_val
        })

    audit_df = pd.DataFrame(audit_rows)
    # Benjamini-Hochberg FDR correction
    audit_df = audit_df.sort_values("Unadjusted_p").reset_index(drop=True)
    m = len(audit_df)
    audit_df["FDR_q"] = audit_df["Unadjusted_p"] * m / (audit_df.index + 1)
    audit_df["FDR_q"] = np.minimum.accumulate(audit_df["FDR_q"].values[::-1])[::-1]
    audit_df["FDR_q"] = np.clip(audit_df["FDR_q"], 0.0, 1.0)
    audit_df["q_val"] = audit_df["FDR_q"]
    audit_df["neg_log10_q"] = -np.log10(np.maximum(audit_df["FDR_q"], 1e-15))
    audit_df["Significant_q05"] = audit_df["FDR_q"] < 0.05

    return audit_df


# =============================================================================
# 5. MASTER CONTROLLER: MULTITHREADED EXTRACTION & FOLD ASSEMBLY
# =============================================================================
def run_phase2_pipeline(
    aligned_dir: Path,
    splits_csv: Path,
    out_wt_dir: Path,
    out_hab_dir: Path,
    out_tables_dir: Path,
    num_workers: int = DEFAULT_NUM_WORKERS,
    skip_extraction: bool = False
) -> None:
    r"""
    Executes complete Phase 2 radiomic extraction:
      1. Whole-Tumor radiomics (extracted once for all 100 cases).
      2. Fold-Aware Habitat radiomics (for fold f, extracted from `habitat_3class_fold_{f}.nii.gz`).
      3. Assembles `Fold_Feature_Matrices_WholeTumor/fold_{f}_features.csv` and
         `Fold_Feature_Matrices_Habitat/fold_{f}_features.csv` retaining all clinical columns.
      4. Executes restricted FDR discovery audit on exactly 1,178 eligible radiomic candidates.
    """
    out_wt_dir.mkdir(parents=True, exist_ok=True)
    out_hab_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 2: IBSI RADIOMIC FEATURE EXTRACTION, ZERO-IMPUTATION & FDR AUDIT")
    logger.info("=" * 80)

    # 1. Load CV Splits
    if not splits_csv.exists():
        raise FileNotFoundError(f"Missing CV splits file: {splits_csv}")
    splits_df = pd.read_csv(splits_csv, dtype={"lesion_id": str, "patient_uid": str})
    splits_df["lesion_id"] = splits_df["lesion_id"].astype(str).str.strip()
    if "patient_uid" in splits_df.columns:
        splits_df["patient_uid"] = splits_df["patient_uid"].astype(str).str.strip()
    all_lids = splits_df["lesion_id"].tolist()
    logger.info(f"Target Cohort: {len(all_lids)} lesions. Worker threads: {num_workers}.")

    clinical_base_cols = [
        "Clinical_Volume", "Clinical_Dose", "Clinical_Fr", "Clinical_IDL",
        "Clinical_Coverage", "Clinical_Selectivity", "Clinical_GI"
    ]

    if not skip_extraction:
        # 2. Extract Whole-Tumor Radiomics Across All 100 Lesions (Independent of GMM)
        logger.info("Extracting Whole-Tumor IBSI Radiomics (293 features per lesion)...")
        wt_data: Dict[str, Dict[str, float]] = {}
        wt_failures: List[Tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(extract_wholetumor_radiomics_single, lid, aligned_dir): lid for lid in all_lids}
            for future in futures:
                lid = futures[future]
                try:
                    wt_data[lid] = future.result()
                except Exception as e:
                    logger.error(f"Whole-tumor extraction technical failure for {lid}: {e}")
                    wt_failures.append((lid, str(e)))

        if wt_failures:
            raise RuntimeError(
                f"Aborting Phase 2: {len(wt_failures)} technical failures occurred during Whole-Tumor extraction. "
                f"Analysis matrices will NOT be generated from corrupted/incomplete runs."
            )

        wt_df = pd.DataFrame.from_dict(wt_data, orient="index").reset_index().rename(columns={"index": "lesion_id"})
        wt_df["lesion_id"] = wt_df["lesion_id"].astype(str).str.strip()
        if set(wt_df["lesion_id"]) != set(splits_df["lesion_id"]):
            missing = set(splits_df["lesion_id"]) - set(wt_df["lesion_id"])
            raise KeyError(f"Whole-Tumor extraction missing lesions: {missing}")
        if len(wt_df) != len(splits_df):
            raise ValueError(f"Whole-Tumor extraction count mismatch: {len(wt_df)} vs {len(splits_df)}")
        wt_merged = splits_df.merge(wt_df, on="lesion_id", how="inner")
        
        # Ensure Phase 3 clinical column compatibility (_x suffixed aliases alongside unsuffixed)
        for col in clinical_base_cols:
            if col in splits_df.columns:
                wt_merged[f"{col}_x"] = splits_df[col].values

        wt_feature_cols = [c for c in wt_merged.columns if c not in splits_df.columns or c.startswith("WholeTumor_")]
        if wt_merged[wt_feature_cols].isna().any().any():
            nan_cols = [c for c in wt_feature_cols if wt_merged[c].isna().any()]
            raise ValueError(f"Non-finite / missing values detected in Whole-Tumor features: {nan_cols}")

        # Save Whole-Tumor Matrices for Folds 0 through 4
        for f in range(NUM_FOLDS):
            wt_fold_csv = out_wt_dir / f"fold_{f}_features.csv"
            wt_merged.to_csv(wt_fold_csv, index=False)
        logger.info(f"Saved all 5 Whole-Tumor fold matrices ({wt_merged.shape[1]} total columns).")

        # 3. Leak-Free Per-Fold Habitat Radiomics Extraction
        # For evaluation fold f, extracts habitat features under GMM model f (fitted on outer training fold) for all 100 cases
        logger.info("Extracting Fold-Aware Habitat Radiomics (GMM fitted within outer training folds)...")
        hab_dfs_by_fold: Dict[int, pd.DataFrame] = {}

        for f in range(NUM_FOLDS):
            logger.info(f"[Fold {f}] Extracting habitat features from habitat_3class_fold_{f}.nii.gz for all 100 lesions...")
            hab_fold_data: Dict[str, Dict[str, float]] = {}
            hab_failures: List[Tuple[str, str]] = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(extract_habitat_radiomics_for_fold, lid, f, aligned_dir): lid for lid in all_lids}
                for future in futures:
                    lid = futures[future]
                    try:
                        hab_fold_data[lid] = future.result()
                    except Exception as e:
                        logger.error(f"[Fold {f}] Habitat extraction technical failure for {lid}: {e}")
                        hab_failures.append((lid, str(e)))

            if hab_failures:
                raise RuntimeError(
                    f"Aborting Phase 2: {len(hab_failures)} technical failures occurred in Habitat extraction for fold {f}. "
                    f"Analysis matrices will NOT be generated from corrupted/incomplete runs."
                )

            hab_fold_df = pd.DataFrame.from_dict(hab_fold_data, orient="index").reset_index().rename(columns={"index": "lesion_id"})
            hab_fold_df["lesion_id"] = hab_fold_df["lesion_id"].astype(str).str.strip()
            if set(hab_fold_df["lesion_id"]) != set(splits_df["lesion_id"]):
                missing = set(splits_df["lesion_id"]) - set(hab_fold_df["lesion_id"])
                raise KeyError(f"[Fold {f}] Habitat extraction missing lesions: {missing}")
            if len(hab_fold_df) != len(splits_df):
                raise ValueError(f"[Fold {f}] Habitat extraction count mismatch: {len(hab_fold_df)} vs {len(splits_df)}")
            hab_merged_f = splits_df.merge(hab_fold_df, on="lesion_id", how="inner")

            # Ensure Phase 3 clinical column compatibility (_x suffixed aliases alongside unsuffixed)
            for col in clinical_base_cols:
                if col in splits_df.columns:
                    hab_merged_f[f"{col}_x"] = splits_df[col].values

            hab_feature_cols = [c for c in hab_merged_f.columns if c not in splits_df.columns or c.startswith("Hab")]
            if hab_merged_f[hab_feature_cols].isna().any().any():
                nan_cols = [c for c in hab_feature_cols if hab_merged_f[c].isna().any()]
                raise ValueError(f"[Fold {f}] Non-finite / missing values detected in Habitat features: {nan_cols}")

            # Save Habitat Matrix for Fold f
            hab_fold_csv = out_hab_dir / f"fold_{f}_features.csv"
            hab_merged_f.to_csv(hab_fold_csv, index=False)
            hab_dfs_by_fold[f] = hab_merged_f
            logger.info(f"[Fold {f}] Saved Habitat fold matrix: {hab_fold_csv.name} ({hab_merged_f.shape[1]} columns).")

    else:
        logger.info("Skipping active radiomic extraction (--skip-extraction). Loading existing fold matrices from disk...")
        wt_merged = pd.read_csv(out_wt_dir / "fold_0_features.csv", dtype={"lesion_id": str, "patient_uid": str})
        wt_merged["lesion_id"] = wt_merged["lesion_id"].astype(str).str.strip()
        if "patient_uid" in wt_merged.columns:
            wt_merged["patient_uid"] = wt_merged["patient_uid"].astype(str).str.strip()

        for col in clinical_base_cols:
            if col in wt_merged.columns and f"{col}_x" not in wt_merged.columns:
                wt_merged[f"{col}_x"] = wt_merged[col].values

        hab_dfs_by_fold = {}
        for f in range(NUM_FOLDS):
            h_df = pd.read_csv(out_hab_dir / f"fold_{f}_features.csv", dtype={"lesion_id": str, "patient_uid": str})
            h_df["lesion_id"] = h_df["lesion_id"].astype(str).str.strip()
            if "patient_uid" in h_df.columns:
                h_df["patient_uid"] = h_df["patient_uid"].astype(str).str.strip()
            for col in clinical_base_cols:
                if col in h_df.columns and f"{col}_x" not in h_df.columns:
                    h_df[f"{col}_x"] = h_df[col].values
            hab_dfs_by_fold[f] = h_df

    # 4. Assemble Full Early-Fusion Dataset and Run Restricted FDR Audit
    # For leak-free cross-cohort FDR evaluation, each patient's habitat features come strictly
    # from their held-out test fold GMM model!
    
    # Strictly align Whole-Tumor data by lesion_id with splits_df
    wt_rows = []
    for _, row in splits_df.iterrows():
        lid = str(row["lesion_id"]).strip()
        row_match = wt_merged[wt_merged["lesion_id"] == lid]
        if len(row_match) == 0:
            raise KeyError(f"[FDR Audit] Lesion '{lid}' not found in Whole-Tumor feature matrix.")
        if len(row_match) > 1:
            raise ValueError(f"[FDR Audit] Duplicate matches ({len(row_match)}) found for lesion '{lid}' in Whole-Tumor matrix.")
        wt_rows.append(row_match.iloc[0])
    wt_aligned_df = pd.DataFrame(wt_rows).reset_index(drop=True)

    # Strictly align Habitat data by lesion_id with splits_df from held-out test fold matrices
    oof_hab_rows = []
    for _, row in splits_df.iterrows():
        lid = str(row["lesion_id"]).strip()
        test_f = int(row["fold"])
        if test_f not in hab_dfs_by_fold:
            raise KeyError(f"[FDR Audit] Fold {test_f} missing from habitat feature matrices.")
        hab_df_f = hab_dfs_by_fold[test_f]
        
        row_match = hab_df_f[hab_df_f["lesion_id"] == lid]
        if len(row_match) == 0:
            raise KeyError(f"[FDR Audit] Lesion '{lid}' not found in habitat feature matrix for held-out fold {test_f}.")
        if len(row_match) > 1:
            raise ValueError(f"[FDR Audit] Duplicate matches ({len(row_match)}) found for lesion '{lid}' in fold {test_f}.")
        oof_hab_rows.append(row_match.iloc[0])

    oof_hab_df = pd.DataFrame(oof_hab_rows).reset_index(drop=True)

    eval_parts = [splits_df.reset_index(drop=True)]
    wt_feature_cols = [c for c in wt_aligned_df.columns if c.startswith("WholeTumor_") and c not in splits_df.columns]
    if wt_feature_cols:
        eval_parts.append(wt_aligned_df[wt_feature_cols].reset_index(drop=True))
    hab_feature_cols = [c for c in oof_hab_df.columns if c.startswith("Hab") and c not in splits_df.columns]
    if hab_feature_cols:
        eval_parts.append(oof_hab_df[hab_feature_cols].reset_index(drop=True))
    df_fdr_eval = pd.concat(eval_parts, axis=1)

    logger.info("Executing Restricted Univariable FDR Discovery Audit on exactly 1,178 candidate features...")
    fdr_audit_df = compute_univariable_fdr_audit(
        feature_df=df_fdr_eval,
        eligible_feature_list=ELIGIBLE_1178_RADIOMIC_FEATURES,
        outcome_col="Status"
    )

    fdr_out_csv = out_tables_dir / "production_univariable_fdr_features.csv"
    fdr_audit_df.to_csv(fdr_out_csv, index=False)
    sig_count = int(fdr_audit_df["Significant_q05"].sum())
    logger.info(f"Saved univariable FDR audit: {fdr_out_csv.name} ({len(fdr_audit_df)} features tested, {sig_count} with q < 0.05).")
    logger.info("Phase 2 execution complete.")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: High-Throughput IBSI Feature Extraction, Zero-Imputation, and Fold Matrix Assembly."
    )
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR, help="Aligned NIfTI directory")
    parser.add_argument("--splits-csv", type=Path, default=DEFAULT_METADATA_DIR / "phase2_cv_splits.csv", help="CV splits CSV path")
    parser.add_argument("--out-wt-dir", type=Path, default=DEFAULT_WT_MATRICES_DIR, help="Output Whole-Tumor fold matrices directory")
    parser.add_argument("--out-hab-dir", type=Path, default=DEFAULT_HAB_MATRICES_DIR, help="Output Habitat fold matrices directory")
    parser.add_argument("--out-tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Output Results/Tables directory")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker threads")
    parser.add_argument("--skip-extraction", action="store_true", default=False, help="Skip extraction and audit existing fold matrices")
    args = parser.parse_args()

    run_phase2_pipeline(
        aligned_dir=args.aligned_dir,
        splits_csv=args.splits_csv,
        out_wt_dir=args.out_wt_dir,
        out_hab_dir=args.out_hab_dir,
        out_tables_dir=args.out_tables_dir,
        num_workers=args.num_workers,
        skip_extraction=args.skip_extraction
    )


if __name__ == "__main__":
    main()
