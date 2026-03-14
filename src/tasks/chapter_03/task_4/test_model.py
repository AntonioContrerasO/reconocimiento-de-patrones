"""
Test script for Task 4 — Spam Classifier

Loads the saved best pipeline, evaluates it on a held-out test split
of the SpamAssassin corpus, and prints a full metrics report.

NOTE: The EmailPreprocessor import below is required so joblib can
      deserialize the saved pipeline that contains the custom transformer.
"""

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Must be imported before joblib.load so pickle can find the class
from .spam_classifier import EmailPreprocessor  # noqa: F401
from .spam_classifier import find_latest_model, load_emails, DATA_DIR


def main():
    print("Loading SpamAssassin corpus from disk...")
    spam_raw = load_emails("spam")
    ham_raw  = load_emails("ham")

    if not spam_raw or not ham_raw:
        print("ERROR: Dataset not found. Run spam_classifier.py first to download it.")
        return

    X = spam_raw + ham_raw
    y = np.array([1] * len(spam_raw) + [0] * len(ham_raw))

    # Reproduce same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test samples : {len(X_test)}  (spam={y_test.sum()}, ham={(y_test == 0).sum()})")

    model_path = find_latest_model()
    model_data = joblib.load(model_path)
    pipeline   = model_data["pipeline"]
    print(f"Model loaded : {model_path}")
    print(f"Classifier   : {model_data['name']}")
    print(f"Train accuracy (saved): {model_data['accuracy']:.2%}")

    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"\nAccuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f}  (of flagged as spam, how many really are)")
    print(f"Recall    : {rec:.4f}  (of actual spam, how many were caught)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(f"              Predicted Ham  Predicted Spam")
    print(f"Actual Ham      {cm[0][0]:>7}        {cm[0][1]:>7}")
    print(f"Actual Spam     {cm[1][0]:>7}        {cm[1][1]:>7}")

    return acc


if __name__ == "__main__":
    main()
