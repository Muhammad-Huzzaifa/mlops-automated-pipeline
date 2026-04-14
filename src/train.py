"""
Training pipeline with parameterized experiments.
"""

import os
import tempfile
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data_loader import DataLoader
from src.models import get_param_grid, build_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_model_comparison
from src.registry import promote_best_model


def train():
    """
    Train models with multiple hyperparameter configurations.

    Returns:
        dict: Model performance results
    """

    loader = DataLoader()

    df = loader.load_raw_data("dataset.csv")
    df = loader.preprocess(df)
    loader.save_processed_data(df, "processed.csv")

    X, y = loader.split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )
    X_train, X_test = loader.scale_features(X_train, X_test)
    loader.save_split_data(X_train, X_test, y_train, y_test)

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    param_grid = get_param_grid()
    results = {}

    with mlflow.start_run():

        for model_name, param_list in param_grid.items():
            for i, params in enumerate(param_list):

                run_name = f"{model_name}_run_{i+1}"

                with mlflow.start_run(run_name=run_name, nested=True):

                    model = build_model(model_name, params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # log parameters
                    mlflow.log_param("model_name", model_name)
                    for key, value in params.items():
                        mlflow.log_param(key, value)

                    # log metrics
                    metrics = evaluate_model(y_test, y_pred)
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)

                    # log confusion matrix
                    fig = plot_confusion_matrix(y_test, y_pred)
                    fig.savefig(os.path.join(Config.RESULTS_DIR, "confusion_matrix.png"))
                    mlflow.log_figure(fig, os.path.join(Config.RESULTS_DIR, "confusion_matrix.png"))

                    # log model artifacts explicitly so registry/deploy can load from runs:/.../model
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        local_model_dir = os.path.join(tmp_dir, "model")
                        mlflow.sklearn.save_model(model, local_model_dir)
                        mlflow.log_artifacts(local_model_dir, artifact_path="model")

                    results[run_name] = metrics
        
        # log comparison plot
        fig = plot_model_comparison(results)
        fig.savefig(os.path.join(Config.RESULTS_DIR, "model_comparison.png"))
        mlflow.log_figure(fig, os.path.join(Config.RESULTS_DIR, "model_comparison.png"))

    promote_best_model()


if __name__ == "__main__":
    train()
