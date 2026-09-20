from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV dataset with a binary target column named ``label``."""
    data = pd.read_csv(csv_path)
    if "label" not in data.columns:
        raise ValueError("Expected a target column named 'label'.")

    X = data.drop(columns=["label"]).to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=float)
    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    test_size: float = 0.2,
):
    """Create a reproducible train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def scale_train_test(X_train_raw: np.ndarray, X_test_raw: np.ndarray):
    """Fit the scaler on training data only, then transform train and test."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, scaler
