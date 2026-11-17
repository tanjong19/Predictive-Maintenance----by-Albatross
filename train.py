import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import shap


# =========================
# 1. CONFIG
# =========================

DATA_PATH = "data/predictive_maintenance_full.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Machines: {df['machine_id'].nunique()}")
print("\nFailure distribution:")
print(df["failure"].value_counts())

# =========================
# 3. SORT TEMPORALLY
# =========================

# Convert timestamp to elapsed seconds if possible
try:
    df["timestamp_seconds"] = pd.to_timedelta(
        "00:" + df["timestamp"].astype(str)
    ).dt.total_seconds()
except Exception:
    df["timestamp_seconds"] = range(len(df))

df = df.sort_values(
    ["machine_id", "episode_id", "timestamp_seconds"]
).reset_index(drop=True)

# =========================
# 4. FEATURE ENGINEERING
# =========================

sensor_cols = [
    "temperature",
    "vibration",
    "motor_current",
    "rpm",
    "workload"
]

# Previous-value changes
for col in sensor_cols:
    df[f"{col}_change"] = (
        df.groupby("machine_id")[col].diff()
    )

# Rolling features using past observations
for col in sensor_cols:
    df[f"{col}_mean_5"] = (
        df.groupby("machine_id")[col]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    df[f"{col}_std_5"] = (
        df.groupby("machine_id")[col]
        .transform(lambda x: x.rolling(5, min_periods=1).std())
    )

# Remove rows where temporal features cannot be calculated
df = df.fillna(0)

# =========================
# 5. SELECT FEATURES
# =========================

features = [
    "temperature",
    "vibration",
    "motor_current",
    "rpm",
    "operating_hours",
    "workload",

    "temperature_change",
    "vibration_change",
    "motor_current_change",
    "rpm_change",
    "workload_change",

    "temperature_mean_5",
    "temperature_std_5",
    "vibration_mean_5",
    "vibration_std_5",
    "motor_current_mean_5",
    "motor_current_std_5",
    "rpm_mean_5",
    "rpm_std_5",
    "workload_mean_5",
    "workload_std_5",
]

X = df[features]
y = df["failure"]

# =========================
# 6. TIME-BASED TRAIN/TEST
# =========================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# =========================
# 7. TRAIN XGBOOST
# =========================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 8. PREDICTION
# =========================

y_probability = model.predict_proba(X_test)[:, 1]

y_pred = (y_probability >= 0.5).astype(int)

# =========================
# 9. EVALUATION
# =========================

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")

if y_test.nunique() == 2:
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_probability):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# 10. FEATURE IMPORTANCE
# =========================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nTop Features:")
print(importance.head(10))

importance.head(15).sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150)
plt.close()

# =========================
# 11. SHAP
# =========================

print("\nGenerating SHAP explanation...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150)
plt.close()

# =========================
# 12. SAVE MODEL
# =========================

model_path = f"{MODEL_DIR}/xgboost_failure_model.pkl"

joblib.dump(model, model_path)

with open(
    f"{MODEL_DIR}/feature_columns.json",
    "w"
) as f:
    json.dump(features, f, indent=4)

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

print(f"Model saved to: {model_path}")
print("Features saved to: models/feature_columns.json")
print("SHAP saved to: models/shap_summary.png")
print("Feature importance saved to: models/feature_importance.png")