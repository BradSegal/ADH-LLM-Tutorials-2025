# Data Pipeline Specification

## 1. Overview

This document provides a definitive specification for the data pipeline responsible for processing the PhysioNet/CinC 2019 Sepsis Challenge dataset. The pipeline's sole responsibility is to convert the raw, multi-file, pipe-delimited dataset into a single, clean, preprocessed, and cached Parquet file. This final file serves as the reliable input for the PyTorch `Dataset` used in all sequence modeling tasks.

The entire pipeline is designed to be **idempotent and cache-aware**. It will only perform expensive processing or downloading steps if the required artifacts are not already present.

## 2. Data Source

*   **Source:** [PhysioNet/Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data](https://physionet.org/content/challenge-2019/1.0.0/)
*   **Access:** Requires a free PhysioNet account for credentialed access. The download process will be handled programmatically, but users will be prompted for their credentials.
*   **Citation:** Please cite the original publication when using this data:
    > Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D., & Sharma, A. (2020). Early prediction of sepsis from clinical data: The physionet/computing in cardiology challenge 2019 (version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-gu25.

## 3. Raw Data Schema

The raw data is distributed as a set of pipe-delimited (`.psv`) files, one per patient. Each file contains 41 columns. The key columns are:

| Column Name   | Data Type | Description                                                                 |
| :------------ | :-------- | :-------------------------------------------------------------------------- |
| `HR`          | `float64` | Heart Rate (beats per minute)                                               |
| `O2Sat`       | `float64` | Pulse Oximetry (%)                                                          |
| `Temp`        | `float64` | Temperature (°C)                                                            |
| `SBP`         | `float64` | Systolic Blood Pressure (mm Hg)                                             |
| `MAP`         | `float64` | Mean Arterial Pressure (mm Hg)                                              |
| `DBP`         | `float64` | Diastolic Blood Pressure (mm Hg)                                            |
| `Resp`        | `float64` | Respiration Rate (breaths per minute)                                       |
| `EtCO2`       | `float64` | End-tidal CO2 (mm Hg)                                                       |
| `BaseExcess`  | `float64` | Measure of metabolic acidosis/alkalosis (mmol/L)                            |
| `HCO3`        | `float64` | Bicarbonate (mmol/L)                                                        |
| `FiO2`        | `float64` | Fraction of Inspired Oxygen                                                 |
| `pH`          | `float64` | Blood pH                                                                    |
| `PaCO2`       | `float64` | Partial Pressure of CO2 from Arterial Blood (mm Hg)                         |
| `SaO2`        | `float64` | Oxygen Saturation from Arterial Blood (%)                                   |
| `AST`         | `float64` | Aspartate Aminotransferase (IU/L)                                           |
| `BUN`         | `float64` | Blood Urea Nitrogen (mg/dL)                                                 |
| `Alkalinephos`| `float64` | Alkaline Phosphatase (IU/L)                                                 |
| `Calcium`     | `float64` | Calcium (mg/dL)                                                             |
| `Chloride`    | `float64` | Chloride (mmol/L)                                                           |
| `Creatinine`  | `float64` | Creatinine (mg/dL)                                                          |
| `Bilirubin_direct`|`float64`| Direct Bilirubin (mg/dL)                                                   |
| `Glucose`     | `float64` | Serum Glucose (mg/dL)                                                       |
| `Lactate`     | `float64` | Lactate (mmol/L)                                                            |
| `Magnesium`   | `float64` | Magnesium (mg/dL)                                                           |
| `Phosphate`   | `float64` | Phosphate (mg/dL)                                                           |
| `Potassium`   | `float64` | Potassium (mmol/L)                                                          |
| `Bilirubin_total`|`float64`| Total Bilirubin (mg/dL)                                                    |
| `TroponinI`   | `float64` | Troponin I (ng/mL)                                                          |
| `Hct`         | `float64` | Hematocrit (%)                                                              |
| `Hgb`         | `float64` | Hemoglobin (g/dL)                                                           |
| `PTT`         | `float64` | Partial Thromboplastin Time (seconds)                                       |
| `WBC`         | `float64` | White Blood Cell Count (count*10^3/µL)                                      |
| `Fibrinogen`  | `float64` | Fibrinogen (mg/dL)                                                          |
| `Platelets`   | `float64` | Platelets (count*10^3/µL)                                                   |
| `Age`         | `int64`   | Age (years)                                                                 |
| `Gender`      | `int64`   | 0: Female, 1: Male                                                          |
| `Unit1`       | `float64` | Administrative identifier for ICU unit (MICU)                               |
| `Unit2`       | `float64` | Administrative identifier for ICU unit (SICU)                               |
| `HospAdmTime` | `float64` | Hours between hospital admit and ICU admit                                  |
| `ICULOS`      | `int64`   | ICU length-of-stay (hours)                                                  |
| `SepsisLabel` | `int64`   | 0: No sepsis, 1: Sepsis                                                     |

## 4. End-to-End Processing Logic

The pipeline is orchestrated by a single high-level function, `core.data.physionet_sepsis.get_sepsis_data()`. The logic proceeds in the following, strictly-ordered sequence:

1.  **Check for Cached Processed Data:** The function first checks for the existence of the final processed file (`data/processed/physionet_sepsis_preprocessed.parquet`). If it exists, it is loaded directly into a Pandas DataFrame and returned. The pipeline terminates here.

2.  **Check for Raw Data:** If the cached file does not exist, the function checks for the presence of the raw, extracted data in `data/raw/physionet_2019/`.

3.  **Download Raw Data:** If the raw data directory is missing or empty, the function `download_physionet_data()` is called. This function will prompt the user for their PhysioNet username and password, download the zip archive, and extract it to the `data/raw/` directory.

4.  **Parse and Concatenate Raw Files:** The `process_physionet_data()` function is called. It will iterate over all `.psv` files in the raw data directory. Each file is read into a DataFrame. A `patient_id` column is created from the filename. All individual patient DataFrames are concatenated into a single, large DataFrame.

5.  **Impute Missing Values (Patient-Wise):** The concatenated DataFrame is grouped by `patient_id`. Within each patient group, the following imputation strategy is applied in order:
    *   **Forward Fill:** `fillna(method='ffill')` is applied to propagate the last known observation forward. This is the primary imputation method.
    *   **Backward Fill:** `fillna(method='bfill')` is applied to fill any remaining `NaN` values at the *beginning* of a patient's record.
    *   **Global Mean/Median Fill (Contingency):** After the above steps, if any `NaN` values remain (e.g., a patient has no measurements for a given feature), they will be filled with the global median value for that column, calculated from the entire training dataset. A warning will be logged if this step is triggered.

6.  **Feature Scaling:**
    *   The data is split into training and validation/test sets (based on a predefined split, if available from the challenge, or a stratified split).
    *   A `sklearn.preprocessing.StandardScaler` is instantiated.
    *   The scaler is **fit ONLY on the training data**.
    *   The fitted scaler is used to **transform the training, validation, and test sets**.
    *   The fitted scaler object is serialized and saved to disk (e.g., `data/processed/scaler.joblib`). This is critical for ensuring that the same scaling is applied at inference time.

7.  **Cache Processed Data:** The final, preprocessed (imputed and scaled) DataFrame is saved to `data/processed/physionet_sepsis_preprocessed.parquet`. The Parquet format is chosen for its efficiency and for preserving data types.

## 5. Processed Data Contract

The output of the entire pipeline, and the input to the `SepsisDataset` class, is a single Pandas DataFrame with the following guaranteed properties:

*   **No Missing Values:** The DataFrame will contain zero `NaN` values.
*   **Scaled Features:** All 34 physiological features (from `HR` to `Platelets`) will be standardized (zero mean, unit variance).
*   **Consistent Data Types:** All columns will have the data types as specified in the schema above.
*   **Columns:** The DataFrame will contain all 41 original columns plus the `patient_id` column.

This DataFrame is the "source of truth" for all subsequent modeling steps.