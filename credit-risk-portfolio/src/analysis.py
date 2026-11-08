"""Reproducible credit default analysis and baseline model."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
MODEL_PATH = REPORTS / "credit_default_model.joblib"
METRICS_PATH = REPORTS / "metrics.json"
PREDICTIONS_PATH = REPORTS / "test_predictions.csv"

TARGET_CANDIDATES = ["Y", "default payment next month", "default.payment.next.month"]

def load_data():
    """Fetch the UCI dataset; normalise target and feature names."""
    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        target_col = next((c for c in TARGET_CANDIDATES if c in y.columns), y.columns[0])
        y = y[target_col]
    data = X.copy()
    data["default_next_month"] = pd.to_numeric(y, errors="coerce").values
    data = data.dropna(subset=["default_next_month"]).reset_index(drop=True)
    data["default_next_month"] = data["default_next_month"].astype(int)
    # The UCI feature table commonly labels the identifier as ID; it is not a predictive feature.
    id_cols = [c for c in data.columns if str(c).strip().lower() in {"id", "index"}]
    data = data.drop(columns=id_cols, errors="ignore")
    return data

def build_pipeline(X):
    categorical = [c for c in X.columns if any(k in str(c).upper() for k in ["PAY_", "SEX", "EDUCATION", "MARRIAGE"])]
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42))
    ])

def run():
    data = load_data()
    X = data.drop(columns=["default_next_month"])
    y = data["default_next_month"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "dataset_rows": int(len(data)),
        "feature_count": int(X.shape[1]),
        "default_rate": float(y.mean()),
        "test_size": int(len(y_test)),
        "threshold": 0.50,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0)
    }
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    out = X_test.copy()
    out["actual_default"] = y_test.values
    out["predicted_probability"] = probabilities
    out["predicted_default_050"] = predictions
    out.to_csv(PREDICTIONS_PATH, index=False)
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved test predictions: {PREDICTIONS_PATH}")
    return data, model, metrics

if __name__ == "__main__":
    run()
