# Company Bankruptcy Prediction - Insights Report

## EDA visualizations and insights
- Confirmed major class imbalance: around 3% bankrupt vs. around 97% non-bankrupt with stratified split applied.
- Most financial ratios are highly skewed; extreme values preserved as they may signal genuine financial distress.
- Correlation heatmap revealed multicollinearity among profitability and leverage ratios.
- Class distribution plot and top-variance histograms generated to illustrate imbalance and feature spread.
- Outliers and skewness are consistent with bankruptcy scenarios (e.g., firms on the brink of collapse often show extreme debt ratios).

## Selected features and preprocessing steps
 - All features numeric therefore, no categorical encoding needed.
- Logistic Regression uses StandardScaler and XGBoost and CatBoost use raw features (no scaling needed).
- Imbalance addressed via class_weight="balanced" (LR) and scale_pos_weight (XGB/CB).
- Outliers retained; SMOTE avoided to preserve realistic ratio patterns.
- Random seed fixed at 42 for reproducibility across preprocessing, model training, and evaluation.

## Hyperparameter tuning results
- Hyperparameters were set manually, guided by Lab 4 insights and prior experimentation.
- Logistic Regression: tuned C, solver, and class_weight="balanced" to handle class imbalance.
- XGBoost: parameters (max_depth, learning_rate, n_estimators, and regularization terms) selected based on best-performing ranges from earlier trials.
- CatBoost: configured with fixed iterations, depth, learning_rate, and l2_leaf_reg, following Lab 4 tuning rationale.
- Choices were documented and locked in the code to ensure reproducibility and to avoid overfitting from excessive hyperparameter searches.

## Model comparison table (metrics across train/test)
- Compared LR, XGB, and CB on PR-AUC, ROC-AUC, Recall, F1-score, and Brier Score.
- XGBoost achieved the highest PR-AUC (0.52), making it the strongest candidate for imbalanced classification (since bankruptcies are rare).
- Logistic Regression had the highest recall (0.82), meaning it caught more bankrupt firms but at the cost of lower precision (low F1).
- CatBoost gave a balanced trade-off between recall (0.52) and F1 (0.52), with good calibration (Brier 0.026).
- Brier scores show both XGBoost and CatBoost are better calibrated than Logistic Regression (lower = better).
- Overall, XGBoost is best for predictive power, while CatBoost is the most balanced if recall is prioritized.

## Calibration curves, ROC-AUC curves, Brier Scores
- Calibration curves (see figure) show Logistic Regression under-predicts probabilities, while XGBoost tracks closer to diagonal reliability.
- CatBoost oscillates but improves recall at moderate thresholds.
- Brier score reinforces calibration: XGBoost 0.024 < CatBoost 0.026 < LR 0.092.
- ROC/PR curves confirm overall ordering: XGB > CB > LR

## SHAP summary plots
- SHAP values (XGBoost best model):
- Quick Ratio and Total Debt / Net Worth are the strongest predictors.
- Measures of leverage (borrowing dependency) and profitability (retained earnings to total assets) also play a strong role.
- Companies with weaker liquidity (low Quick Ratio) and higher leverage (more debt relative to equity) are more likely to go bankrupt.
- The SHAP plots provide clear, visual explanations of how financial health indicators affect the model’s decisions

## PSI results and drift analysis
- All PSI values were < 0.1, indicating no significant drift.
- Top features with the highest PSI were:
 1. Operating profit per person (0.021)
 2. Working Capital to Total Assets (0.0196)
 3. Cash Flow Rate (0.0193)
 4. Even the highest observed values fall within the “no drift” zone (<0.1).

- Proves that training and testing populations are stable, and the models are unlikely to suffer performance degradation due to distribution shifts.
- Monitoring should continue in production. If PSI values exceed 0.25, retraining will be required.
- The stability aligns with the overall robust generalization observed in ROC-AUC and PR-AUC scores across models.

