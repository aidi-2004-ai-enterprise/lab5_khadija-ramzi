# Company Bankruptcy Prediction

## Project Overview

This project contains the end-to-end training pipeline for predicting company bankruptcy using financial ratios.
The pipeline includes:

1. Exploratory Data Analysis (EDA)
2. Feature preprocessing and selection
3. Model training (Logistic Regression, XGBoost, CatBoost)
4. Hyperparameter tuning
5. Model evaluation (ROC-AUC, PR-AUC, Recall, F1, Calibration, Brier score)
6. Interpretability with SHAP
7. Population Stability Index (PSI) & drift analysis

** Note: All results (plots, metrics, tables) are saved into the results/ and plots/ folders.

## Reproducibility

- Random seeds are fixed (random_state=42) for train/test splits and all model training.
- Results may vary slightly due to parallelization in XGBoost/CatBoost.
- Ensure consistent environment setup using the package list below.

## Dependencies

- Install required packages before running the pipeline:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap joblib

- For PSI/drift plots):
pip install scipy

## Setup Folders

- Before running the pipeline, create the following folders to store outputs:

```sh
mkdir results
mkdir plots
```

## Running the Pipeline

1. Place the dataset in:
project/Company_Bankruptcy_Prediction.csv

2. Run the training pipeline:
python project/training_pipeline.py

3. Outputs will be saved in:

results/ → metrics tables & summary reports

plots/ → ROC, calibration, SHAP, PSI plots

## Report

See insights_report.md for:

- EDA insights
- Preprocessing choices
- Hyperparameter tuning results
- Model comparison
- Interpretability & drift analysis

## Demo Video

**lab5_khadija-ramzi\project\Company Bankruptcy Prediction Demo.mp4