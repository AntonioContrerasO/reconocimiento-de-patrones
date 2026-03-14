"""
Reusable plotting utilities for model evaluation metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(ax, y_true, y_pred, class_names, title):
    """Heatmap of a confusion matrix on the given axes."""
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )


def plot_metrics_bar(ax, y_true, y_pred, class_names, title):
    """Grouped bar chart of precision, recall and F1 per class."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def save_task_report(y_true, y_pred, class_names, task_name, accuracy, output_dir: Path):
    """
    Save a two-panel figure (confusion matrix + metrics bar) for one task.

    Returns the output file path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{task_name}  —  Accuracy: {accuracy:.2%}", fontsize=13, fontweight="bold")

    plot_confusion_matrix(axes[0], y_true, y_pred, class_names, "Confusion Matrix")
    plot_metrics_bar(axes[1], y_true, y_pred, class_names, "Precision / Recall / F1")

    plt.tight_layout()
    out = output_dir / f"{task_name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_accuracy_summary(task_accuracies: dict, output_dir: Path):
    """
    Bar chart comparing accuracy across all tasks.

    task_accuracies: {"Task 1 — ...": 0.97, ...}
    """
    labels = list(task_accuracies.keys())
    values = list(task_accuracies.values())
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors[: len(labels)], alpha=0.85, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.2%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Model Accuracy — Chapter 03 Tasks", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = output_dir / "accuracy_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
