import matplotlib
import matplotlib.figure

matplotlib.use("Agg")

from src.evaluate import evaluate_model, plot_confusion_matrix, plot_model_comparison


def test_evaluate_model_metrics_values():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]

    metrics = evaluate_model(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f1_score"], 6) == round(2 / 3, 6)


def test_plot_confusion_matrix_returns_figure():
    fig = plot_confusion_matrix([1, 0, 1, 0], [1, 0, 0, 0])

    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes[0].get_title() == "Confusion Matrix"


def test_plot_model_comparison_returns_figure():
    all_results = {
        "model_a": {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
        },
        "model_b": {
            "accuracy": 0.85,
            "precision": 0.78,
            "recall": 0.72,
            "f1_score": 0.74,
        },
    }

    fig = plot_model_comparison(all_results)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert "Model Comparison" in fig.axes[0].get_title()
