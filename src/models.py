"""
Model factory and parameter configurations.
"""

from typing import Dict, List, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_param_grid() -> Dict[str, List[Dict[str, Any]]]:
    """
    Define parameter grid for experiments.

    Returns:
        Dict[str, List[Dict]]: Model-wise parameter sets
    """

    return {
        "LogisticRegression": [
            {"C": 1.0, "max_iter": 200},
            {"C": 0.5, "max_iter": 300}
        ],
        "RandomForest": [
            {"n_estimators": 50},
            {"n_estimators": 150}
        ]
    }


def build_model(model_name: str, params: Dict[str, Any]):
    """
    Build model with given parameters.

    Args:
        model_name (str): Model name
        params (dict): Hyperparameters

    Returns:
        model: Initialized model
    """

    if model_name == "LogisticRegression":
        return LogisticRegression(**params)

    if model_name == "RandomForest":
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model: {model_name}")
