"""
MLflow model registry.
"""

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import Config


def promote_best_model():
    """
    Promote best model to production.
    """

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

    client = MlflowClient()

    exp = client.get_experiment_by_name(Config.EXPERIMENT_NAME)
    if exp is None:
        raise Exception(f"Experiment '{Config.EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.accuracy DESC"]
    )

    valid_runs = []
    for run in runs:
        if "accuracy" not in run.data.metrics:
            continue
        if "mlflow.parentRunId" not in run.data.tags:
            continue
        
        valid_runs.append(run)

    if not valid_runs:
        valid_runs = [run for run in runs if "accuracy" in run.data.metrics]

    if not valid_runs:
        raise Exception("No valid runs with accuracy found.")
    
    model_name = "fraud_model"
    candidate_runs = sorted(
        valid_runs,
        key=lambda x: x.data.metrics["accuracy"],
        reverse=True,
    )
    errors = []

    for run in candidate_runs:
        run_id = run.info.run_id
        run_accuracy = run.data.metrics["accuracy"]
        model_uri = f"runs:/{run_id}/model"

        try:
            print(f"Trying run ID: {run_id} (accuracy={run_accuracy})")
            try:
                result = mlflow.register_model(model_uri, model_name)
            except MlflowException as exc:
                if "Unable to find a logged_model" not in str(exc):
                    raise

                try:
                    client.get_registered_model(model_name)
                except Exception:
                    client.create_registered_model(model_name)

                result = client.create_model_version(
                    name=model_name,
                    source=f"{run.info.artifact_uri}/model",
                    run_id=run_id,
                )

            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage="Production"
            )
            print(f"Model version {result.version} promoted to Production")
            return model_uri

        except Exception as exc:
            errors.append(f"run_id={run_id}, accuracy={run_accuracy}, error={exc}")

    raise Exception("Unable to promote any candidate run. " + " | ".join(errors))

if __name__ == "__main__":
    promote_best_model()
