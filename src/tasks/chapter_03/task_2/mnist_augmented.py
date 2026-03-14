"""
MNIST Classifier with Data Augmentation

Shifts images in 4 directions (left, right, up, down) to expand the training set.
This technique improves model accuracy by providing more training examples.
"""

from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Paths
MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "models"


def get_model_path(accuracy: float) -> Path:
    return MODELS_DIR / f"model_task2_{accuracy * 100:.0f}pct.joblib"


def find_latest_model() -> Path:
    matches = sorted(MODELS_DIR.glob("model_task2_*pct.joblib"))
    if not matches:
        raise FileNotFoundError("No task-2 model found. Run main() first.")
    return matches[-1]
IMAGES_DIR = Path(__file__).parent / "original_images"
TRANSFORMED_DIR = Path(__file__).parent / "transformed_images"


def shift_image(image, direction):
    """
    Shift an MNIST image by one pixel in the specified direction.

    Args:
        image: 1D array of 784 pixels (28x28 flattened)
        direction: one of 'left', 'right', 'up', 'down'

    Returns:
        Shifted image as 1D array
    """
    image_2d = image.reshape(28, 28)

    if direction == "left":
        shifted = shift(image_2d, [0, -1], cval=0)
    elif direction == "right":
        shifted = shift(image_2d, [0, 1], cval=0)
    elif direction == "up":
        shifted = shift(image_2d, [-1, 0], cval=0)
    elif direction == "down":
        shifted = shift(image_2d, [1, 0], cval=0)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return shifted.reshape(784)


def augment_dataset(X, y):
    """
    Create augmented dataset by adding 4 shifted copies of each image.

    Args:
        X: Training images (n_samples, 784)
        y: Training labels

    Returns:
        Augmented X and y arrays (5x the original size)
    """
    directions = ["left", "right", "up", "down"]
    augmented_images = [X]
    augmented_labels = [y]

    print("Augmenting dataset...")
    for i, direction in enumerate(directions):
        print(f"  Shifting {direction}... ({i+1}/4)")
        shifted = np.array([shift_image(img, direction) for img in X])
        augmented_images.append(shifted)
        augmented_labels.append(y)

    X_augmented = np.vstack(augmented_images)
    y_augmented = np.hstack(augmented_labels)

    return X_augmented, y_augmented


def save_sample_images(X, y):
    """
    Save 8 random images and their transformations for visualization.
    """
    import matplotlib.pyplot as plt

    # Create directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing images
    for f in IMAGES_DIR.glob("*.png"):
        f.unlink()
    for f in TRANSFORMED_DIR.glob("*.png"):
        f.unlink()

    # Select 8 random images
    np.random.seed(42)
    indices = np.random.choice(len(X), 8, replace=False)

    directions = ["left", "right", "up", "down"]

    for i, idx in enumerate(indices, start=1):
        image = X[idx]
        label = y[idx]

        # Save original
        plt.figure(figsize=(2, 2))
        plt.imshow(image.reshape(28, 28), cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.savefig(IMAGES_DIR / f"image_{i}.png", bbox_inches="tight", dpi=100)
        plt.close()

        # Save transformed versions
        fig, axes = plt.subplots(1, 4, figsize=(8, 2))
        for j, direction in enumerate(directions):
            shifted = shift_image(image, direction)
            axes[j].imshow(shifted.reshape(28, 28), cmap="gray")
            axes[j].set_title(direction)
            axes[j].axis("off")
        plt.suptitle(f"Image {i} (Label: {label})")
        plt.savefig(TRANSFORMED_DIR / f"image_{i}_shifted.png", bbox_inches="tight", dpi=100)
        plt.close()

    print(f"Saved 8 original images to: {IMAGES_DIR}")
    print(f"Saved transformed images to: {TRANSFORMED_DIR}")


def main():
    # Fetch MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    print(f"Dataset shape: {X.shape}")

    # Split into train and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Save sample images for visualization
    save_sample_images(X_train, y_train)

    # Augment the training set
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train)
    print(f"\nAugmented training set: {X_train_aug.shape[0]} samples (5x original)")

    # Apply PCA
    print("\nApplying PCA...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_aug)
    X_test_pca = pca.transform(X_test)

    print(f"Reduced dimensions: {X_train_aug.shape[1]} -> {X_train_pca.shape[1]}")
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")

    # Use best parameters from task_1
    best_params = {"n_neighbors": 4, "weights": "distance"}
    print(f"\nTraining KNN with params: {best_params}")

    knn = KNeighborsClassifier(**best_params, n_jobs=-1)
    knn.fit(X_train_pca, y_train_aug)

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = knn.predict(X_test_pca)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    if test_accuracy > 0.97:
        print("Success! Data augmentation improved the model!")

    # Save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": knn,
        "pca": pca,
        "accuracy": test_accuracy,
        "best_params": best_params,
        "augmentation": "shifted (left, right, up, down)",
    }
    model_path = get_model_path(test_accuracy)
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to: {model_path}")

    return test_accuracy


def load_model():
    """Load the saved augmented model."""
    model_path = find_latest_model()
    model_data = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Accuracy: {model_data['accuracy']:.2%}")
    print(f"Parameters: {model_data['best_params']}")
    print(f"Augmentation: {model_data['augmentation']}")
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

    images_pca = pca.transform(images)
    return model.predict(images_pca)


if __name__ == "__main__":
    main()
