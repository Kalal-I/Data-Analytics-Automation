"""Training-only cross-validation assignments for Random Forest regression."""
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class FoldAssignment:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def create_cv_folds(X, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
    """Create deterministic reusable folds within a training partition."""
    n_samples = len(X)
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if n_samples < n_splits:
        raise ValueError("training partition has {} samples, but {} folds require at least {} samples".format(n_samples, n_splits, n_splits))
    return [
        FoldAssignment(np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int))
        for train_indices, validation_indices in KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        ).split(np.asarray(X))
    ]


create_folds = create_cv_folds
