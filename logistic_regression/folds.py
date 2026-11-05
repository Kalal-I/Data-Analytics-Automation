"""Fixed Stratified K-Fold assignments for logistic regression."""
from dataclasses import dataclass
from numbers import Integral
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldAssignment:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def create_cv_folds(
    X,
    y=None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[FoldAssignment]:
    """Create deterministic, reusable stratified positional assignments."""
    if y is None:
        raise TypeError("y is required to create stratified classification folds")
    n_samples = len(X)
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if n_samples < int(n_splits):
        raise ValueError("training partition has {} samples, but {} folds require at least {} samples".format(n_samples, n_splits, n_splits))
    if len(y) != n_samples:
        raise ValueError("X and y must contain the same number of samples")
    y_array = np.asarray(y)
    if pd.isna(y_array).any():
        raise ValueError("y contains missing values")
    counts = pd.Series(y_array).value_counts()
    insufficient = counts[counts < int(n_splits)]
    if not insufficient.empty:
        details = ", ".join("{}={}".format(repr(index), int(value)) for index, value in insufficient.items())
        raise ValueError("each class must have at least {} samples; insufficient: {}".format(n_splits, details))

    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=shuffle,
        random_state=int(random_state) if shuffle else None,
    )
    return [
        FoldAssignment(np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int))
        for train_indices, validation_indices in splitter.split(np.asarray(X), y_array)
    ]


create_folds = create_cv_folds
