"""
training_pipeline.py

Author: Fathima Khadija Ramzi

Description: Single end-to-end pipeline for Company Bankruptcy Prediction
AIDI 2004 - Lab 05
"""

# -------------------
# Import Libraries
# -------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import recall_score, brier_score_loss, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import shap

# -------------------
# Dataset path
# -------------------
DATA_PATH = r"C:\Users\framz\OneDrive\Documents\GitHub\lab5_khadija-ramzi\project\Company_Bankruptcy_Prediction.csv"

# -------------------
# 1) Load Data
# -------------------
df = pd.read_csv(DATA_PATH)
target = "Bankrupt?"

# -------------------
# 2) EDA Insights
# -------------------
print("\n=== Data Overview ===")
print(df.head())
print("\nClass balance:")
print(df[target].value_counts(normalize=True))

plt.figure(figsize=(6,4))
sns.countplot(x=target, data=df)
plt.title("Target Class Distribution")
plt.savefig("plots/class_distribution.png", dpi=300)
plt.close()

# -------------------
# 3) Train-Test Split
# -------------------
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Class imbalance weight
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / max(pos, 1)

# -------------------
# 4) Preprocessing for Logistic Regression
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 5) Evaluation function
# -------------------
def eval_model(name, clf, Xtr, Xte):
    clf.fit(Xtr, y_train)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xte)[:, 1]
    else:
        scores = clf.decision_function(Xte)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "Brier": brier_score_loss(y_test, proba)
    }
    return metrics, proba

# -------------------
# 6) Train Models
# -------------------
results = {}
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, loss_function="Logloss",
        scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False, 
        train_dir="results/catboost_logs"
    )
}

for name, model in models.items():
    Xtr, Xte = (X_train_scaled, X_test_scaled) if name == "Logistic Regression" else (X_train, X_test)
    metrics, proba = eval_model(name, model, Xtr, Xte)
    results[name] = {"metrics": metrics, "proba": proba}

# -------------------
# 7) Model Comparison Table
# -------------------
df_results = pd.DataFrame([v["metrics"] for v in results.values()])
df_results.to_csv("results/model_comparison.csv", index=False)
print("\n=== Model Comparison ===")
print(df_results)

df_results.to_csv("results/model_comparison.csv", index=False)
with open("results/model_comparison.md", "w", encoding="utf-8") as f:
    f.write(df_results.to_markdown(index=False))

# -------------------
# 8) Calibration Curves & ROC Curves
# -------------------
plt.figure(figsize=(8,6))
for name, res in results.items():
    prob_true, prob_pred = calibration_curve(y_test, res["proba"], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves")
plt.legend()
plt.savefig("plots/calibration_curves.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
for name, res in results.items():
    RocCurveDisplay.from_predictions(y_test, res["proba"], name=name)
plt.title("ROC Curves")
plt.savefig("plots/roc_curves.png", dpi=300)
plt.close()

# -------------------
# 9) SHAP Explainability for Best Model
# -------------------
best_model_name = df_results.sort_values(by="PR-AUC", ascending=False).iloc[0]["Model"]
print(f"\nBest Model: {best_model_name}")
best_model = models[best_model_name]

# Train on the same matrices we evaluated with
Xtr, Xte = (X_train, X_test) if best_model_name != "Logistic Regression" else (X_train_scaled, X_test_scaled)
best_model.fit(Xtr, y_train)

if best_model_name in ("XGBoost", "CatBoost"):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)  # use *original* feature space for plots
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig("plots/shap_summary.png", dpi=300)
    plt.close()
else:
    # Logistic Regression: show top coefficients (absolute value)
    coefs = best_model.coef_.ravel()
    coef_df = pd.DataFrame({"feature": X.columns, "coef": coefs})
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
    top = coef_df.head(20).iloc[::-1]  # plot top 20, smallest at bottom
    plt.figure(figsize=(7,6))
    plt.barh(top["feature"], top["coef"])
    plt.title("Logistic Regression – Top 20 |Coefficients|")
    plt.tight_layout()
    plt.savefig("plots/lr_top_coeffs.png", dpi=300)
    plt.close()
# -------------------
# 10) PSI Drift Analysis
# -------------------

def psi_single_feature(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """PSI for one feature using quantile cuts from expected (train)."""
    # Build cuts on expected (train) quantiles
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = expected.quantile(quantiles).values
    # Ensure unique, otherwise fallback to linear cuts
    cuts = np.unique(cuts)
    if cuts.size < 3:
        cuts = np.linspace(expected.min(), expected.max(), buckets + 1)

    # Bin both series using *same* cuts
    exp_bins = np.clip(np.digitize(expected, cuts, right=True) - 1, 0, cuts.size - 2)
    act_bins = np.clip(np.digitize(actual,   cuts, right=True) - 1, 0, cuts.size - 2)

    # Counts → percents
    exp_counts = np.bincount(exp_bins, minlength=cuts.size - 1).astype(float)
    act_counts = np.bincount(act_bins, minlength=cuts.size - 1).astype(float)
    exp_pct = exp_counts / max(exp_counts.sum(), 1.0)
    act_pct = act_counts / max(act_counts.sum(), 1.0)

    # PSI
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = (act_pct - exp_pct) * np.log((act_pct + 1e-12) / (exp_pct + 1e-12))
    contrib[np.isnan(contrib)] = 0.0
    return float(np.sum(contrib))

# Compute PSI for all features
def psi_table(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:    
    rows = []
    for col in X_train_df.columns:
        psi_val = psi_single_feature(X_train_df[col], X_test_df[col], buckets=10)
        rows.append({"Feature": col, "PSI": psi_val})
    out = pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
    out.to_csv("results/psi_train_vs_test.csv", index=False)
    print("\nTop features by PSI (train vs test):")
    print(out.head(top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAverage PSI across all features: {out['PSI'].mean():.4f}")
    return out

# Call it with your *unscaled* dataframes to reflect the real feature space:
psi_df = psi_table(X_train, X_test, top_n=15)