"""
MLflow model registry.
"""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import Config


def promote_best_model():
    """
    Promote best model to production.
    """

    client = MlflowClient()

    exp = client.get_experiment_by_name(Config.EXPERIMENT_NAME)
    runs = client.search_runs(exp.experiment_id)

    best_run = max(runs, key=lambda x: x.data.metrics["accuracy"])

    model_uri = f"runs:/{best_run.info.run_id}/model"

    model_name = "fraud_model"

    mlflow.register_model(model_uri, model_name)

    latest = client.get_latest_versions(model_name)[-1]

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production"
    )

    return model_uri
