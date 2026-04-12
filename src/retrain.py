"""
Retraining pipeline with drift detection.
"""

import pandas as pd

from src.data_loader import DataLoader
from src.drift import detect_drift
from src.train import train
from src.registry import promote_best_model


def retrain():
    """
    Run retraining if drift detected.
    """

    loader = DataLoader()

    old_data = loader.load_processed_data("processed.csv")

    new_raw = loader.load_raw_data("incoming.csv")
    new_data = loader.preprocess(new_raw)

    drift = detect_drift(old_data, new_data)

    if not drift:
        print("No drift detected. Skipping retraining.")
        return

    print("Drift detected. Updating dataset...")

    updated = pd.concat([old_data, new_data])
    loader.save_processed_data(updated, "processed.csv")

    train()
    promote_best_model()
