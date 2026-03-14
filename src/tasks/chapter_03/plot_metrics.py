"""
Generate metric plots for all Chapter 03 tasks.

Saves PNG files to:  plots/chapter_03/

Tasks 1 & 2 use a 2 000-sample MNIST subset to keep prediction time reasonable.
Tasks 3 & 4 use their full test splits.

Usage:
    make plots
    # or directly:
    uv run python -m src.tasks.chapter_03.plot_metrics
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from ...utils.plots import save_accuracy_summary, save_task_report

# Must be imported at module level (not inside a function) so that when joblib
# unpickles the saved pipeline, pickle can resolve __main__.EmailPreprocessor
# against the currently running __main__ module (this file).
from .task_4.spam_classifier import EmailPreprocessor as EmailPreprocessor  # noqa: F401

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "plots" / "chapter_03"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MNIST_PLOT_SAMPLES = 2_000   # cap so KNN prediction stays fast


# ── helpers ──────────────────────────────────────────────────────────────────
def _find(pattern: str) -> Path:
    matches = sorted(MODELS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No model matching '{pattern}' in {MODELS_DIR}. "
            "Run the corresponding training script first."
        )
    return matches[-1]


# ── per-task collectors ──────────────────────────────────────────────────────
def collect_task1():
    print("Task 1 — loading MNIST + KNN model...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_test, y_test = mnist.data[60000:], mnist.target[60000:]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), MNIST_PLOT_SAMPLES, replace=False)
    X_sub, y_sub = X_test[idx], y_test[idx]

    data = joblib.load(_find("model_task1_*pct.joblib"))
    X_pca = data["pca"].transform(X_sub)
    y_pred = data["model"].predict(X_pca)

    class_names = [str(i) for i in range(10)]
    return y_sub, y_pred, class_names, data["accuracy"]


def collect_task2():
    print("Task 2 — loading MNIST + augmented KNN model...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_test, y_test = mnist.data[60000:], mnist.target[60000:]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), MNIST_PLOT_SAMPLES, replace=False)
    X_sub, y_sub = X_test[idx], y_test[idx]

    data = joblib.load(_find("model_task2_*pct.joblib"))
    X_pca = data["pca"].transform(X_sub)
    y_pred = data["model"].predict(X_pca)

    class_names = [str(i) for i in range(10)]
    return y_sub, y_pred, class_names, data["accuracy"]


def collect_task3():
    print("Task 3 — loading Titanic model...")
    from .task_3.titanic_classifier import load_and_transform

    X, y, features = load_and_transform()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    data = joblib.load(_find("model_task3_*pct.joblib"))
    y_pred = data["pipeline"].predict(X_test)

    class_names = ["Did not survive", "Survived"]
    return y_test, y_pred, class_names, data["accuracy"]


def collect_task4():
    print("Task 4 — loading Spam model...")
    from .task_4.spam_classifier import load_emails

    spam_raw = load_emails("spam")
    ham_raw  = load_emails("ham")
    X = spam_raw + ham_raw
    y = np.array([1] * len(spam_raw) + [0] * len(ham_raw))
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    data = joblib.load(_find("model_task4_*pct.joblib"))
    y_pred = data["pipeline"].predict(X_test)

    class_names = ["Ham", "Spam"]
    return y_test, y_pred, class_names, data["accuracy"]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    tasks = [
        ("Task 1 — MNIST KNN",            collect_task1),
        ("Task 2 — MNIST Augmented KNN",  collect_task2),
        ("Task 3 — Titanic LogReg",        collect_task3),
        ("Task 4 — Spam Classifier",       collect_task4),
    ]

    accuracy_summary = {}
    saved_files = []

    for task_name, collector in tasks:
        try:
            y_true, y_pred, class_names, acc = collector()
            out = save_task_report(y_true, y_pred, class_names, task_name, acc, OUTPUT_DIR)
            saved_files.append(out)
            accuracy_summary[task_name] = acc
            print(f"  Saved: {out.name}")
        except FileNotFoundError as e:
            print(f"  SKIPPED ({e})")

    if accuracy_summary:
        out = save_accuracy_summary(accuracy_summary, OUTPUT_DIR)
        saved_files.append(out)
        print(f"  Saved: {out.name}")

    print(f"\nAll plots written to: {OUTPUT_DIR}")
    return saved_files


if __name__ == "__main__":
    main()
