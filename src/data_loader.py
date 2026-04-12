"""
Data loading and preprocessing module.
"""

import os
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Config


class DataLoader:
    """
    Class for loading and preprocessing data.
    """

    def __init__(self) -> None:
        """Initialize DataLoader with config paths."""
        self.raw_dir = Config.RAW_DATA_DIR
        self.processed_dir = Config.PROCESSED_DATA_DIR
        self.target = Config.TARGET_COLUMN


    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Load raw dataset from raw directory.

        Args:
            filename (str): File name inside raw directory.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        path = os.path.join(self.raw_dir, filename)
        return pd.read_csv(path)


    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed dataset.

        Args:
            filename (str): File name inside processed directory.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        path = os.path.join(self.processed_dir, filename)
        return pd.read_csv(path)


    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw dataframe.

        Args:
            df (pd.DataFrame): Raw dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        
        df = df.copy()

        drop_cols = ["merchant", "trans_num"]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"]
        )
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day"] = df["trans_date_trans_time"].dt.day
        df["month"] = df["trans_date_trans_time"].dt.month
        df = df.drop(columns=["trans_date_trans_time"])

        df["dob"] = pd.to_datetime(df["dob"])
        df["age"] = 2026 - df["dob"].dt.year
        df = df.drop(columns=["dob"])

        df["distance"] = (
            (df["lat"] - df["merch_lat"]) ** 2
            + (df["long"] - df["merch_long"]) ** 2
        ) ** 0.5

        categorical_cols = ["category", "job", "city", "state"]
        df = pd.get_dummies(
            df,
            columns=[col for col in categorical_cols if col in df.columns],
            drop_first=True
        )

        return df


    def split_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target.

        Args:
            df (pd.DataFrame): Processed dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: X and y
        """
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return X, y


    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple:
        """
        Scale features using StandardScaler.

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features

        Returns:
            Tuple: Scaled X_train and X_test
        """
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled


    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str
    ) -> None:
        """
        Save processed dataframe.

        Args:
            df (pd.DataFrame): Dataframe to save
            filename (str): File name
        """
        path = os.path.join(self.processed_dir, filename)
        df.to_csv(path, index=False)


    def save_split_data(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ) -> None:
        """
        Save train-test splits.

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """

        pd.DataFrame(X_train).to_csv(
            os.path.join(self.processed_dir, "X_train.csv"),
            index=False
        )
        pd.DataFrame(X_test).to_csv(
            os.path.join(self.processed_dir, "X_test.csv"),
            index=False
        )
        pd.DataFrame(y_train).to_csv(
            os.path.join(self.processed_dir, "y_train.csv"),
            index=False
        )
        pd.DataFrame(y_test).to_csv(
            os.path.join(self.processed_dir, "y_test.csv"),
            index=False
        )
