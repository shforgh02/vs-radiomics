# Whole-Tumor and Habitat MRI Radiomics

Research code for comparing whole-tumor and habitat radiomics in predicting 12-month volumetric enlargement of vestibular schwannomas after radiosurgery. Inputs are pretreatment T1, contrast-enhanced T1, and T2 MRI with tumor masks.

## Workflow

1. **Preprocessing and habitats:** native-space N4 correction, rigid registration, intensity standardization, and three-component spatial Gaussian mixture modeling using MRI intensities and normalized distance to the tumor boundary. Habitat models are fitted within outer training folds.
2. **Feature extraction:** whole-tumor and habitat radiomics, with 1-mm resampling during PyRadiomics extraction and fold-specific habitat feature matrices.
3. **Prediction:** five-fold outer validation with four-fold inner feature selection and tuning; comparison with a clinical/dosimetric baseline and secondary early/late fusion.
4. **Figures:** performance, habitat distributions, and exploratory analyses.
5. **Permutation testing:** outcome labels shuffled within outer folds, repeating feature selection, tuning, and classifier fitting.

## Use

Keep the five scripts together under `Production_Pipeline/Scripts/`, using their original filenames without download suffixes. Install the dependencies imported by the scripts, including SimpleITK, PyRadiomics, NumPy, pandas, SciPy, scikit-learn, joblib, matplotlib, seaborn, and openpyxl. Configure input/output paths and supply compatible MRI files, masks, clinical metadata, and predefined cross-validation splits. Run Phases 1–3 before Phases 4 and 5. Results are written under `Production_Pipeline/Results/`.

Patient data, feature matrices, and trained models are not included. Running the analysis requires compatible inputs supplied separately. This code supports research and is not validated for clinical decision-making.
