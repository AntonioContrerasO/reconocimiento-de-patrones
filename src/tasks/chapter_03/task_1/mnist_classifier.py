"""
MNIST Classifier with KNeighborsClassifier

Achieves >97% accuracy on the test set using grid search for hyperparameter tuning.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Path to models directory (relative to project root)
MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "models"


def get_model_path(accuracy: float) -> Path:
    return MODELS_DIR / f"model_task1_{accuracy * 100:.0f}pct.joblib"


def find_latest_model() -> Path:
    matches = sorted(MODELS_DIR.glob("model_task1_*pct.joblib"))
    if not matches:
        raise FileNotFoundError("No task-1 model found. Run main() first.")
    return matches[-1]


def main():
    # Fetch MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split into train and test sets (MNIST standard split: 60k train, 10k test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Apply PCA to reduce dimensions (784 -> 100)
    # Using 100 components retains more variance for better accuracy
    print("\nApplying PCA...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Reduced dimensions: {X_train.shape[1]} -> {X_train_pca.shape[1]}")
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")

    # Use a subset for faster grid search
    sample_size = 15000
    indices = np.random.choice(len(X_train_pca), sample_size, replace=False)
    X_sample, y_sample = X_train_pca[indices], y_train[indices]

    # Define the parameter grid
    param_grid = {
        "n_neighbors": [3, 4, 5],
        "weights": ["uniform", "distance"],
    }

    # n_jobs=-1 uses all CPU cores
    knn = KNeighborsClassifier(n_jobs=-1)

    # GridSearchCV with 3-fold cross-validation
    grid_search = GridSearchCV(
        knn, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1
    )

    print(f"\nRunning grid search on {sample_size} samples...")
    grid_search.fit(X_sample, y_sample)

    # Best parameters
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (on {sample_size} samples): {grid_search.best_score_:.4f}")

    # Train final model on full PCA-transformed training data
    best_params = grid_search.best_params_
    print(f"\nTraining final model on full dataset with: {best_params}")

    final_knn = KNeighborsClassifier(**best_params, n_jobs=-1)
    final_knn.fit(X_train_pca, y_train)

    # Predict on test set
    print("Evaluating on test set...")
    y_pred = final_knn.predict(X_test_pca)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    if test_accuracy > 0.97:
        print("Success! Achieved >97% accuracy on the test set.")
    else:
        print("Accuracy is below 97%. Consider tuning hyperparameters further.")

    # Save the model and PCA transformer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": final_knn,
        "pca": pca,
        "accuracy": test_accuracy,
        "best_params": best_params,
    }
    model_path = get_model_path(test_accuracy)
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to: {model_path}")

    return test_accuracy


def load_model():
    """Load the saved model and PCA transformer."""
    model_path = find_latest_model()
    model_data = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Accuracy: {model_data['accuracy']:.2%}")
    print(f"Parameters: {model_data['best_params']}")
    return model_data


def predict(images):
    """
    Predict digits from images using the saved model.

    Args:
        images: numpy array of shape (n_samples, 784) with pixel values 0-255

    Returns:
        Predicted digit labels
    """
    model_data = load_model()
    pca = model_data["pca"]
    model = model_data["model"]

    # Transform with PCA and predict
    images_pca = pca.transform(images)
    return model.predict(images_pca)


if __name__ == "__main__":
    main()
