"""
Titanic Survival Classifier with Logistic Regression

Uses feature engineering and variable transformations to predict survival.
Dataset fetched from OpenML (same source as Kaggle's Titanic competition).
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "models"


def get_model_path(accuracy: float) -> Path:
    return MODELS_DIR / f"model_task3_{accuracy * 100:.0f}pct.joblib"


def find_latest_model() -> Path:
    matches = sorted(MODELS_DIR.glob("model_task3_*pct.joblib"))
    if not matches:
        raise FileNotFoundError("No task-3 model found. Run main() first.")
    return matches[-1]


def load_and_transform():
    """Load Titanic data and apply feature transformations."""
    print("Loading Titanic dataset from OpenML...")
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    df = titanic.frame

    # --- Feature engineering ---

    # 1. Sex -> binary
    df["is_female"] = (df["sex"] == "female").astype(int)

    # 2. Pclass as numeric (already is, but make explicit)
    df["pclass"] = df["pclass"].astype(float)

    # 3. Age: fill missing with median, then bin into groups
    df["age"] = df["age"].astype(float)
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    df["age_bin"] = np.digitize(df["age"], bins=[12, 18, 35, 60])
    # 0=child(<12), 1=teen, 2=young adult, 3=adult, 4=senior

    # 4. Fare: fill missing with median, log-transform to reduce skew
    df["fare"] = df["fare"].astype(float)
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["log_fare"] = np.log1p(df["fare"])

    # 5. Family size = siblings/spouses + parents/children + self
    df["sibsp"] = df["sibsp"].astype(float)
    df["parch"] = df["parch"].astype(float)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # 6. Embarked: fill missing, one-hot encode
    df["embarked"] = df["embarked"].fillna("S")
    df["embarked_C"] = (df["embarked"] == "C").astype(int)
    df["embarked_Q"] = (df["embarked"] == "Q").astype(int)

    features = [
        "pclass",
        "is_female",
        "age_bin",
        "log_fare",
        "family_size",
        "is_alone",
        "embarked_C",
        "embarked_Q",
    ]

    X = df[features].values
    y = df["survived"].astype(int).values

    print(f"Dataset shape: {X.shape}")
    print(f"Survival rate: {y.mean():.2%}")
    print(f"Features used: {features}")
    return X, y, features


def main():
    X, y, features = load_and_transform()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))

    # Show model coefficients
    coefs = pipeline.named_steps["clf"].coef_[0]
    print("Feature coefficients (logistic regression):")
    for feat, coef in sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:15s}: {coef:+.4f}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(accuracy)
    joblib.dump({"pipeline": pipeline, "features": features, "accuracy": accuracy}, model_path)
    print(f"\nModel saved to: {model_path}")

    return accuracy


if __name__ == "__main__":
    main()
