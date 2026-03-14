"""
Test script for Task 2 — MNIST KNN with Data Augmentation

Loads the saved augmented model, evaluates it on the MNIST test set,
and prints a full metrics report.
"""

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .mnist_augmented import load_model


def main():
    print("Loading MNIST test set...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    X_test, y_test = X[60000:], y[60000:]
    print(f"Test samples: {len(X_test)}")

    model_data = load_model()
    pca   = model_data["pca"]
    model = model_data["model"]

    print(f"Augmentation used: {model_data.get('augmentation', 'N/A')}")

    print("\nTransforming test set with PCA...")
    X_test_pca = pca.transform(X_test)

    print("Running predictions...")
    y_pred = model.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Params   : {model_data['best_params']}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(cm)

    return acc


if __name__ == "__main__":
    main()
