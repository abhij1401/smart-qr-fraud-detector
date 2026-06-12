"""
train.py
--------
Loads the feature CSV, trains a Random Forest classifier, prints
evaluation metrics, and saves the model to disk.

Random Forest was chosen because:
  • It handles mixed numeric features well with no scaling needed
  • It's resistant to overfitting (ensemble of decision trees)
  • It gives a feature_importances_ array useful for debugging
"""

import pandas as pd
import joblib
import os

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      classification_report)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
FEATURES_CSV  = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "model.pkl")


def load_data(path: str):
    """
    Read features.csv and split it into:
      X  – feature matrix  (all columns except 'label')
      y  – target vector   (the 'label' column)
    """
    df = pd.read_csv(path)

    # Every column except 'label' is a feature
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    print(f"Loaded {len(df):,} rows | {len(feature_cols)} features")
    return X, y, feature_cols


def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Fit a Random Forest on the training data and return it.

    Key hyperparameters:
      n_estimators  – number of trees; more = more stable but slower
      max_depth     – limits tree depth to reduce overfitting
      random_state  – ensures reproducible results
      n_jobs        – use all CPU cores for speed
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    print("Training Random Forest …")
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, feature_cols) -> None:
    """
    Print a full set of classification metrics on the held-out test set.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # probability of class 1 (phishing)

    print("\n── Evaluation on test set ─────────────────────────────────")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    print("\n── Full classification report ─────────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Phishing"]))

    # Show which features mattered most
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("── Top feature importances ────────────────────────────────")
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:<22} {imp:.4f}  {bar}")


def save_model(model, path: str) -> None:
    """Persist the trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n✅ Model saved → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load data
    X, y, feature_cols = load_data(FEATURES_CSV)

    # 2. Split: 80 % train, 20 % test (stratify keeps class balance equal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    evaluate(model, X_test, y_test, feature_cols)

    # 5. Save
    save_model(model, MODEL_PATH)