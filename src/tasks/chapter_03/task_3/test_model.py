"""
Test script for Task 3 — Titanic Logistic Regression

Loads the saved pipeline, evaluates it on a held-out Titanic test split,
and prints a full metrics report.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .titanic_classifier import find_latest_model, load_and_transform


def main():
    import joblib

    print("Loading Titanic dataset...")
    X, y, features = load_and_transform()

    # Reproduce same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test samples: {len(X_test)}")

    model_path = find_latest_model()
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    print(f"Model loaded from: {model_path}")
    print(f"Trained accuracy  : {model_data['accuracy']:.2%}")

    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"\nAccuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(f"              Predicted No  Predicted Yes")
    print(f"Actual No       {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"Actual Yes      {cm[1][0]:>6}        {cm[1][1]:>6}")

    return acc


if __name__ == "__main__":
    main()
