"""Stratified train/holdout splitting for Random Forest classification."""
from numbers import Integral
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_holdout_split(data: pd.DataFrame, target: str, test_size: float = 0.20,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return stratified training and holdout features and targets."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target, str) or target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer")
    if len(data) < 10:
        raise ValueError("random forest classification requires at least 10 samples")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    y = data[target]
    if y.isna().any():
        raise ValueError("target contains missing values")
    counts = y.value_counts()
    holdout_count = max(1, int(round(len(data) * test_size)))
    if counts.min() < 2 or holdout_count < len(counts) or len(data) - holdout_count < len(counts):
        raise ValueError(
            "class sample counts are insufficient for a stratified train/holdout split; "
            "class counts: {}".format(counts.to_dict())
        )

    train_indices, holdout_indices = train_test_split(
        np.asarray(data.index.to_numpy()),
        test_size=test_size,
        random_state=int(random_state),
        shuffle=True,
        stratify=y.to_numpy(),
    )
    train_index = pd.Index(train_indices)
    holdout_index = pd.Index(holdout_indices)
    if len(train_index) < len(counts) or len(holdout_index) < len(counts):
        raise ValueError("the split must leave every class represented in both partitions")
    if train_index.isin(holdout_index).any() or holdout_index.isin(train_index).any():
        raise RuntimeError("training and holdout indices overlap")

    X = data.drop(columns=[target])
    return (
        X.loc[train_index],
        X.loc[holdout_index],
        y.loc[train_index],
        y.loc[holdout_index],
    )


split_data = train_holdout_split
