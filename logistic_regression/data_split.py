"""Leakage-safe stratified splitting for logistic regression."""
from numbers import Integral
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
    """Return stratified training/holdout features and targets."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target, str) or target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer")
    if len(data) < 5:
        raise ValueError("classification data splitting requires at least 5 samples")
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    y = data[target]
    if y.isna().any():
        raise ValueError("target contains missing values")
    counts = y.value_counts()
    if counts.empty:
        raise ValueError("target must contain at least one class")
    holdout_count = max(1, int(round(len(data) * test_size)))
    can_stratify = counts.min() >= 2 and holdout_count >= len(counts) and len(data) - holdout_count >= len(counts)

    indices = np.arange(len(data))
    try:
        train_indices, holdout_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=int(random_state),
            shuffle=True,
            stratify=y.to_numpy() if can_stratify else None,
        )
    except ValueError as exc:
        raise ValueError("unable to create a class-preserving train/holdout split: {}".format(exc)) from exc

    train_index = pd.Index(train_indices)
    holdout_index = pd.Index(holdout_indices)
    if train_index.isin(holdout_index).any() or holdout_index.isin(train_index).any():
        raise RuntimeError("training and holdout indices overlap")

    X = data.drop(columns=[target]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    return (
        X.iloc[train_indices].copy(),
        X.iloc[holdout_indices].copy(),
        y.iloc[train_indices].copy(),
        y.iloc[holdout_indices].copy(),
    )


split_data = train_holdout_split
