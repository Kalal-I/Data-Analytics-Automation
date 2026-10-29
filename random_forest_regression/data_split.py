"""Train/holdout splitting for Random Forest regression."""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_holdout_split(data: pd.DataFrame, target: str, test_size: float = 0.20,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return training features, holdout features, training target, holdout target."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target, str) or target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))
    if len(data) < 10:
        raise ValueError("random forest workflow requires at least 10 samples")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    y = data[target]
    try:
        y_values = y.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("target must be numeric") from exc
    if y.isna().any() or not np.isfinite(y_values).all():
        raise ValueError("target contains missing or non-finite values")

    X = data.drop(columns=[target])
    train_indices, holdout_indices = train_test_split(
        data.index.to_numpy(), test_size=test_size, random_state=random_state, shuffle=True
    )
    train_index = pd.Index(train_indices)
    holdout_index = pd.Index(holdout_indices)
    if len(train_index) < 5 or len(holdout_index) < 1:
        raise ValueError("the split must leave at least 5 training samples and 1 holdout sample")
    if train_index.isin(holdout_index).any() or holdout_index.isin(train_index).any():
        raise RuntimeError("training and holdout indices overlap")
    return X.loc[train_index], X.loc[holdout_index], y.loc[train_index], y.loc[holdout_index]


split_data = train_holdout_split
