"""
Retraining pipeline with drift detection.
"""

import pandas as pd

from src.data_loader import DataLoader
from src.drift import detect_drift
from src.train import train


def retrain():
    """
    Run retraining if drift detected.
    """

    loader = DataLoader()

    try:
        old_data = loader.load_processed_data("processed.csv")
    except FileNotFoundError:
        old_raw = loader.load_raw_data("dataset.csv")
    
    old_data = loader.preprocess(old_raw)

    try:
        new_raw = loader.load_raw_data("incoming.csv")
    except FileNotFoundError:
        print("No new data found. Skipping retraining.")
        return
    
    new_data = loader.preprocess(new_raw)

    drift = detect_drift(old_data, new_data)

    if not drift:
        print("No drift detected. Skipping retraining.")
        return

    print("Drift detected. Updating dataset...")

    updated = pd.concat([old_data, new_data])
    loader.save_processed_data(updated, "processed.csv")

    train()


if __name__ == "__main__":
    retrain()
