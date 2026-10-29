"""Leakage-safe train/holdout splitting for KNN regression."""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_holdout_split(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return training features, holdout features, training target, holdout target."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target, str) or target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))
    if len(data) < 5:
        raise ValueError("KNN data splitting requires at least 5 samples")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    y = data[target]
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("target contains missing or non-finite values")

    X = data.drop(columns=[target])
    train, holdout = train_test_split(
        data.index.to_numpy(), test_size=test_size, random_state=random_state, shuffle=True
    )
    train_index = pd.Index(train)
    holdout_index = pd.Index(holdout)
    if train_index.isin(holdout_index).any() or holdout_index.isin(train_index).any():
        raise RuntimeError("training and holdout indices overlap")

    return (
        X.loc[train_index],
        X.loc[holdout_index],
        y.loc[train_index],
        y.loc[holdout_index],
    )


split_data = train_holdout_split
