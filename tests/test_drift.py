import numpy as np
import pandas as pd

from src import drift


def test_detect_drift_per_column_detects_clear_shift():
    rng = np.random.default_rng(42)
    old_data = pd.DataFrame({"amt": rng.normal(0, 1, 500)})
    new_data = pd.DataFrame({"amt": rng.normal(3, 1, 500)})

    assert bool(drift.detect_drift_per_column(old_data, new_data, "amt"))


def test_detect_drift_returns_true_when_critical_column_drifts(monkeypatch):
    old_data = pd.DataFrame({"amt": [1, 2], "distance": [1, 1], "hour": [5, 6]})
    new_data = pd.DataFrame({"amt": [3, 4], "distance": [1, 1], "hour": [5, 6]})

    monkeypatch.setattr(drift.Config, "DRIFT_DETECTING_COLUMNS", ["amt", "distance", "hour"])
    monkeypatch.setattr(drift.Config, "CRITICAL_COLUMNS", ["amt"])
    monkeypatch.setattr(drift.Config, "DRIFT_THRESHOLD", 2)

    def fake_detect_per_column(_old, _new, col):
        return col == "amt"

    monkeypatch.setattr(drift, "detect_drift_per_column", fake_detect_per_column)

    assert drift.detect_drift(old_data, new_data) is True


def test_detect_drift_returns_false_when_below_threshold_and_no_critical(monkeypatch):
    old_data = pd.DataFrame({"amt": [1, 2], "distance": [1, 1], "hour": [5, 6]})
    new_data = pd.DataFrame({"amt": [1, 2], "distance": [1, 1], "hour": [5, 6]})

    monkeypatch.setattr(drift.Config, "DRIFT_DETECTING_COLUMNS", ["amt", "distance", "hour"])
    monkeypatch.setattr(drift.Config, "CRITICAL_COLUMNS", ["amt", "distance"])
    monkeypatch.setattr(drift.Config, "DRIFT_THRESHOLD", 3)

    def fake_detect_per_column(_old, _new, col):
        return col == "hour"

    monkeypatch.setattr(drift, "detect_drift_per_column", fake_detect_per_column)

    assert drift.detect_drift(old_data, new_data) is False
