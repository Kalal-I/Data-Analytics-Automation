"""Cross-validation assignments created exclusively on training data."""
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class FoldAssignment:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def create_cv_folds(X, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> List[FoldAssignment]:
    """Create deterministic, reusable K-fold assignments for X."""
    n_samples = len(X)
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if n_samples < n_splits:
        raise ValueError("training partition has {} samples, but {} folds require at least {} samples".format(n_samples, n_splits, n_splits))
    assignments = []
    for train_indices, validation_indices in KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(np.asarray(X)):
        assignments.append(FoldAssignment(np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int)))
    return assignments


create_folds = create_cv_folds
