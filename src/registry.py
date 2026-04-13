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
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )

    valid_runs = []
    for run in runs:
        if "accuracy" not in run.data.metrics:
            continue
        if "mlflow.parentRunId" not in run.data.tags:
            continue

        artifacts = client.list_artifacts(run.info.run_id)
        artifact_paths = [a.path for a in artifacts]
        if "model" not in artifact_paths:
            continue
        
        valid_runs.append(run)

    if not valid_runs:
        raise Exception("No valid child runs with accuracy found.")
    
    best_run = max(valid_runs, key=lambda x: x.data.metrics["accuracy"])
    best_accuracy = best_run.data.metrics["accuracy"]
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best accuracy: {best_accuracy}")

    model_uri = f"runs:/{best_run.info.run_id}/model"

    model_name = "fraud_model"

    result = mlflow.register_model(model_uri, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )
    print(f"Model version {result.version} promoted to Production") 

    return model_uri
