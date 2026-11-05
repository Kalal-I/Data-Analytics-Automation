"""Stratified cross-validation assignments for Random Forest classification."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldAssignment:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def create_cv_folds(X, y, n_splits: int = 5, shuffle: bool = True,
                    random_state: int = 42):
    """Create deterministic, reusable stratified training-only folds."""
    if y is None:
        raise TypeError("y is required for stratified classification folds")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    y_array = np.asarray(y)
    if pd.isna(y_array).any():
        raise ValueError("y contains missing values")
    counts = pd.Series(y_array).value_counts()
    insufficient = counts[counts < n_splits]
    if not insufficient.empty:
        raise ValueError(
            "each class must have at least {} samples; insufficient class counts: {}".format(
                n_splits, insufficient.to_dict()
            )
        )

    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=shuffle,
        random_state=int(random_state) if shuffle else None,
    )
    return [
        FoldAssignment(
            np.asarray(train_indices, dtype=int),
            np.asarray(validation_indices, dtype=int),
        )
        for train_indices, validation_indices in splitter.split(np.asarray(X), y_array)
    ]


create_folds = create_cv_folds
