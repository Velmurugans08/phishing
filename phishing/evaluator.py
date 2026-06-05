"""
evaluator.py
------------
This module evaluates trained models using standard classification metrics:
    - Accuracy Score
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix (with visualization)
    - Model Comparison Bar Chart

It produces clear visual reports saved as PNG images in the 'reports/' folder.

Author: Phishing Detection Project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------------
# Directory for saving evaluation reports / charts
# ---------------------------------------------------------------
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Computes all classification metrics for a given model's predictions.

    Args:
        y_true (array-like): True labels (ground truth).
        y_pred (array-like): Predicted labels from the model.
        model_name (str): Name of the model for display purposes.

    Returns:
        dict: Dictionary containing all computed metric values.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print(f"{'=' * 50}")

    # Full sklearn classification report
    print("\n  Detailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Safe Email (0)", "Phishing Email (1)"]
    ))

    return {
        "model_name": model_name,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1_score":   f1
    }


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Creates and saves a styled confusion matrix heatmap.

    A confusion matrix shows:
        - True Positives  (TP): Phishing correctly identified
        - True Negatives  (TN): Safe emails correctly identified
        - False Positives (FP): Safe emails wrongly flagged as phishing
        - False Negatives (FN): Phishing emails missed

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Model name for the chart title.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw the heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=1.5,
        linecolor="#333355",
        ax=ax,
        annot_kws={"size": 18, "weight": "bold", "color": "white"},
        cbar_kws={"shrink": 0.8}
    )

    # Labels
    ax.set_xlabel("Predicted Label", fontsize=13, color="white", labelpad=12)
    ax.set_ylabel("True Label", fontsize=13, color="white", labelpad=12)
    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=15, color="white", pad=15, fontweight="bold"
    )
    ax.set_xticklabels(["Safe (0)", "Phishing (1)"], color="white", fontsize=11)
    ax.set_yticklabels(["Safe (0)", "Phishing (1)"], color="white", fontsize=11, rotation=0)
    ax.tick_params(colors="white")

    # Colorbar text color
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()

    safe_name  = model_name.lower().replace(" ", "_")
    save_path  = os.path.join(REPORTS_DIR, f"confusion_matrix_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()

    print(f"[CHART] Confusion matrix saved: {save_path}")
    return save_path


def plot_metrics_bar(metrics_dict, model_name="Model"):
    """
    Creates a bar chart showing Accuracy, Precision, Recall, and F1-Score
    for a single model.

    Args:
        metrics_dict (dict): Output from compute_metrics().
        model_name (str): Model name for the chart title.
    """
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values  = [
        metrics_dict["accuracy"],
        metrics_dict["precision"],
        metrics_dict["recall"],
        metrics_dict["f1_score"]
    ]
    colors  = ["#4cc9f0", "#7209b7", "#f72585", "#4361ee"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.bar(labels, [v * 100 for v in values], color=colors,
                  width=0.5, edgecolor="none", zorder=3)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val * 100:.1f}%",
            ha="center", va="bottom",
            color="white", fontsize=12, fontweight="bold"
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", color="white", fontsize=12)
    ax.set_title(
        f"Evaluation Metrics — {model_name}",
        color="white", fontsize=14, fontweight="bold", pad=12
    )
    ax.tick_params(colors="white", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444466")
    ax.spines["bottom"].set_color("#444466")
    ax.yaxis.grid(True, color="#2a2a4a", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(REPORTS_DIR, f"metrics_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()

    print(f"[CHART] Metrics bar chart saved: {save_path}")
    return save_path


def plot_model_comparison(lr_metrics, nb_metrics):
    """
    Creates a grouped bar chart comparing Logistic Regression
    vs Naive Bayes across all four metrics.

    Args:
        lr_metrics (dict): Metrics dict for Logistic Regression.
        nb_metrics (dict): Metrics dict for Naive Bayes.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_values = [
        lr_metrics["accuracy"], lr_metrics["precision"],
        lr_metrics["recall"],   lr_metrics["f1_score"]
    ]
    nb_values = [
        nb_metrics["accuracy"], nb_metrics["precision"],
        nb_metrics["recall"],   nb_metrics["f1_score"]
    ]

    x       = np.arange(len(metric_names))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars_lr = ax.bar(x - width / 2, [v * 100 for v in lr_values],
                     width, label="Logistic Regression",
                     color="#4cc9f0", edgecolor="none", zorder=3)
    bars_nb = ax.bar(x + width / 2, [v * 100 for v in nb_values],
                     width, label="Naive Bayes",
                     color="#f72585", edgecolor="none", zorder=3)

    # Value labels
    for bar in bars_lr:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center", color="white", fontsize=10, fontweight="bold"
        )
    for bar in bars_nb:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center", color="white", fontsize=10, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color="white", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Score (%)", color="white", fontsize=12)
    ax.set_title(
        "Model Comparison: Logistic Regression vs Naive Bayes",
        color="white", fontsize=14, fontweight="bold", pad=14
    )
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444466")
    ax.spines["bottom"].set_color("#444466")
    ax.yaxis.grid(True, color="#2a2a4a", zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(
        facecolor="#0f3460", edgecolor="#444466",
        labelcolor="white", fontsize=11
    )

    plt.tight_layout()

    save_path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()

    print(f"[CHART] Model comparison chart saved: {save_path}")
    return save_path


def run_full_evaluation(training_results):
    """
    Runs the complete evaluation pipeline for both models.
    Generates all charts and prints all metrics.

    Args:
        training_results (dict): The dict returned by train_and_save_all()
                                  from model_trainer.py.

    Returns:
        dict: Metrics for both models.
    """
    print("\n" + "=" * 60)
    print("  RUNNING FULL MODEL EVALUATION")
    print("=" * 60)

    X_test    = training_results["X_test"]
    y_test    = training_results["y_test"]
    lr_model  = training_results["lr_model"]
    nb_model  = training_results["nb_model"]

    # Get predictions
    lr_pred = lr_model.predict(X_test)
    nb_pred = nb_model.predict(X_test)

    # Compute metrics
    lr_metrics = compute_metrics(y_test, lr_pred, "Logistic Regression")
    nb_metrics = compute_metrics(y_test, nb_pred, "Naive Bayes")

    # Generate all charts
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    plot_confusion_matrix(y_test, nb_pred, "Naive Bayes")
    plot_metrics_bar(lr_metrics, "Logistic Regression")
    plot_metrics_bar(nb_metrics, "Naive Bayes")
    plot_model_comparison(lr_metrics, nb_metrics)

    print(f"\n[DONE] All evaluation charts saved in the '{REPORTS_DIR}/' folder.")

    return {"lr_metrics": lr_metrics, "nb_metrics": nb_metrics}


# ---------------------------------------------------------------
# Run evaluation standalone (trains first, then evaluates)
# ---------------------------------------------------------------
if __name__ == "__main__":
    from model_trainer import train_and_save_all
    results = train_and_save_all()
    run_full_evaluation(results)
