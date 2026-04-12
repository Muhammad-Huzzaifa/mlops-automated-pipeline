"""
Configuration module for project.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration class."""

    RAW_DATA_DIR = "data/raw/"
    PROCESSED_DATA_DIR = "data/processed/"
    MODEL_DIR = "models/"
    RESULTS_DIR = "results/"

    TARGET_COLUMN = "is_fraud"

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_USERNAME = os.getenv("DAGSHUB_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("DAGSHUB_PASSWORD")
    EXPERIMENT_NAME = "credit_card_fraud_detection"

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("HF_REPO_ID")

    DRIFT_THRESHOLD = 1
    DRIFT_DETECTING_COLUMNS = ['amt', 'city_pop', 'age', 'distance', 'hour', 'month']
    CRITICAL_COLUMNS = ['amt', 'distance']
