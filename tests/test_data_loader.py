import pandas as pd

from src.data_loader import DataLoader


def _raw_df_sample():
    return pd.DataFrame(
        {
            "merchant": ["m1", "m2"],
            "trans_num": ["t1", "t2"],
            "trans_date_trans_time": ["2020-01-01 01:30:00", "2020-01-02 23:15:00"],
            "dob": ["2000-01-01", "1990-06-15"],
            "lat": [40.0, 41.0],
            "long": [-75.0, -74.0],
            "merch_lat": [40.1, 41.0],
            "merch_long": [-75.1, -74.2],
            "category": ["food", "tech"],
            "job": ["engineer", "teacher"],
            "city": ["A", "B"],
            "state": ["X", "Y"],
            "amt": [10.0, 20.0],
            "is_fraud": [0, 1],
        }
    )


def test_preprocess_creates_engineered_features_and_drops_raw_columns():
    loader = DataLoader()
    processed = loader.preprocess(_raw_df_sample())

    assert "merchant" not in processed.columns
    assert "trans_num" not in processed.columns
    assert "trans_date_trans_time" not in processed.columns
    assert "dob" not in processed.columns

    assert "hour" in processed.columns
    assert "day" in processed.columns
    assert "month" in processed.columns
    assert "age" in processed.columns
    assert "distance" in processed.columns


def test_split_features_target_uses_config_target():
    loader = DataLoader()
    processed = loader.preprocess(_raw_df_sample())

    X, y = loader.split_features_target(processed)

    assert "is_fraud" not in X.columns
    assert y.name == "is_fraud"
    assert len(X) == len(y) == 2


def test_scale_features_returns_arrays_with_expected_shapes():
    loader = DataLoader()
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    X_test = pd.DataFrame({"a": [4.0], "b": [40.0]})

    X_train_scaled, X_test_scaled = loader.scale_features(X_train, X_test)

    assert X_train_scaled.shape == (3, 2)
    assert X_test_scaled.shape == (1, 2)
