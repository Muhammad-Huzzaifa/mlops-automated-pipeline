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
    valid_runs = [
        run for run in runs
        if "accuracy" in run.data.metrics
        and "mlflow.parentRunId" in run.data.tags
    ]
    if not valid_runs:
        raise Exception("No valid child runs with accuracy found.")
    
    best_run = max(valid_runs, key=lambda x: x.data.metrics["accuracy"])
    best_accuracy = best_run.data.metrics["accuracy"]
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best accuracy: {best_accuracy}")

    model_uri = f"runs:/{best_run.info.run_id}/model"

    model_name = "fraud_model"

    mlflow.register_model(model_uri, model_name)

    latest = client.get_latest_versions(model_name)[-1]

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production"
    )
    print(f"Model version {latest.version} promoted to Production") 

    return model_uri
