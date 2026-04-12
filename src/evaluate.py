"""
Evaluation module for model performance.
"""

from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """
    Evaluate model using multiple metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix plot.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return fig


def plot_model_comparison(all_results: Dict[str, Dict[str, float]]):
    """
    Create comparison plot for multiple models.

    Args:
        all_results (Dict[str, Dict[str, float]]):
            {
                "model1": {"accuracy": ..., "precision": ..., ...},
                "model2": {...}
            }

    Returns:
        matplotlib.figure.Figure: Comparison plot
    """

    df = pd.DataFrame(all_results).T

    df.reset_index(inplace=True)
    df.rename(columns={"index": "model"}, inplace=True)

    df_melted = df.melt(id_vars="model", var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df_melted,
        x="model",
        y="value",
        hue="metric",
        ax=ax
    )

    ax.set_title("Model Comparison (Accuracy, Precision, Recall, F1)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")

    plt.xticks(rotation=30)

    return fig
