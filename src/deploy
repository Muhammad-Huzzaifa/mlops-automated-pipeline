"""
Deployment module: Deploy latest Production model to Hugging Face.
"""

import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi

from src.config import Config


def get_production_model_uri(model_name: str) -> str:
    """
    Get latest Production model URI from MLflow.

    Args:
        model_name (str): Registered model name

    Returns:
        str: Model URI
    """

    client = MlflowClient()

    versions = client.get_latest_versions(
        name=model_name,
        stages=["Production"]
    )

    if not versions:
        raise Exception("No model found in Production stage.")

    version = versions[0]

    return f"models:/{model_name}/{version.version}"


def load_model_from_mlflow(model_uri: str):
    """
    Load model from MLflow.

    Args:
        model_uri (str): MLflow model URI

    Returns:
        model: Loaded model
    """

    return mlflow.sklearn.load_model(model_uri)


def save_model_locally(model, path: str):
    """
    Save model locally.

    Args:
        model: Trained model
        path (str): Save path
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def deploy_to_huggingface(local_model_path: str):
    """
    Upload model to Hugging Face.

    Args:
        local_model_path (str): Path to saved model
    """

    api = HfApi()

    # create repo if not exists
    api.create_repo(
        repo_id=Config.HF_REPO_ID,
        token=Config.HF_TOKEN,
        exist_ok=True
    )

    # upload model
    api.upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo="model.pkl",
        repo_id=Config.HF_REPO_ID,
        token=Config.HF_TOKEN,
        commit_message="Updating production model"
    )


def deploy():
    """
    Full deployment pipeline:
    - Fetch Production model from MLflow
    - Save locally
    - Upload to Hugging Face
    """

    MODEL_NAME = "fraud_model"

    model_uri = get_production_model_uri(MODEL_NAME)
    print(f"Using model from: {model_uri}")

    model = load_model_from_mlflow(model_uri)

    local_path = os.path.join(Config.MODEL_DIR, "model.pkl")
    save_model_locally(model, local_path)

    deploy_to_huggingface(local_path)

    print("Model deployed successfully!")
