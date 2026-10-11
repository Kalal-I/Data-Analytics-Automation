"""Split a dataset and train a linear regression model on its training split."""

from collections.abc import Sequence
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from linear_regression.linear_model import (
    _as_finite_numeric,
    _selected_features,
    train_linear_regression,
)


def _data_fingerprint(data: pd.DataFrame) -> str:
    """Hash values, index, columns, and dtypes to detect stale saved folds."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in data.columns]).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in data.dtypes]).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Saved folds require data that can be hashed; use scalar DataFrame cells."
        ) from error
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _cross_validation_metrics(
    training_data: pd.DataFrame,
    target_column: str,
    features: Sequence[str] | str | None,
    random_state: int,
    cv_folds: list[dict[str, list[int]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Calculate metrics for fixed CV folds and their population standard deviations."""
    selected_features = _selected_features(
        training_data, target_column, features
    )
    predictors = _as_finite_numeric(training_data, selected_features)
    target = _as_finite_numeric(training_data, [target_column])[target_column]

    if len(training_data) < 10:
        raise ValueError(
            "At least 10 training rows are required for five-fold R^2 "
            "cross-validation (at least two validation rows per fold)."
        )

    if cv_folds is None:
        fold_splitter = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
        cv_splits = list(fold_splitter.split(predictors))
    else:
        cv_splits = [
            (
                np.asarray(fold["train_indices"], dtype=int),
                np.asarray(fold["validation_indices"], dtype=int),
            )
            for fold in cv_folds
        ]

    fold_scores = {"r2": [], "mae": [], "rmse": []}
    for train_indices, validation_indices in cv_splits:
        model = LinearRegression()
        model.fit(predictors.iloc[train_indices], target.iloc[train_indices])
        predictions = model.predict(predictors.iloc[validation_indices])
        actual = target.iloc[validation_indices]
        fold_scores["r2"].append(float(r2_score(actual, predictions)))
        fold_scores["mae"].append(float(mean_absolute_error(actual, predictions)))
        fold_scores["rmse"].append(
            float(np.sqrt(mean_squared_error(actual, predictions)))
        )

    metrics: dict[str, dict[str, Any]] = {}
    for name, scores in fold_scores.items():
        if not np.isfinite(scores).all():
            raise ValueError(
                "Five-fold cross-validation produced an undefined {} score; "
                "check for constant target values in a validation fold.".format(
                    name.upper()
                )
            )
        metrics[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=0)),
            "fold_scores": scores,
        }
    return metrics


def _validate_saved_folds(
    data: pd.DataFrame, folds: dict[str, Any]
) -> tuple[list[int], list[int], list[dict[str, list[int]]]]:
    """Validate persisted positional splits before using them."""
    if not isinstance(folds, dict):
        raise TypeError("folds must be a dictionary returned by the fold utility.")
    if folds.get("row_count") != len(data):
        raise ValueError("Saved folds do not match the current number of data rows.")
    if folds.get("data_fingerprint") != _data_fingerprint(data):
        raise ValueError(
            "Saved folds were created for different data or row ordering."
        )

    train_indices = folds.get("train_indices")
    test_indices = folds.get("test_indices")
    cv_folds = folds.get("cv_folds")
    if not isinstance(train_indices, list) or not isinstance(test_indices, list):
        raise ValueError("Saved folds must contain training and test row positions.")
    if not all(isinstance(index, int) and not isinstance(index, bool)
               for index in train_indices + test_indices):
        raise ValueError("Saved row positions must be integers.")
    expected_positions = set(range(len(data)))
    if (
        len(set(train_indices)) != len(train_indices)
        or len(set(test_indices)) != len(test_indices)
        or set(train_indices).intersection(test_indices)
        or set(train_indices).union(test_indices) != expected_positions
    ):
        raise ValueError("Saved training and test positions must partition all rows.")
    if len(train_indices) < 10:
        raise ValueError(
            "At least 10 training rows are required for five-fold R^2 "
            "cross-validation (at least two validation rows per fold)."
        )
    if not isinstance(cv_folds, list) or len(cv_folds) != 5:
        raise ValueError("Saved folds must contain exactly five CV folds.")

    expected_training_positions = set(range(len(train_indices)))
    validation_seen: list[int] = []
    normalized_folds: list[dict[str, list[int]]] = []
    for fold in cv_folds:
        if not isinstance(fold, dict):
            raise ValueError("Each saved CV fold must be a dictionary.")
        fold_train = fold.get("train_indices")
        fold_validation = fold.get("validation_indices")
        if not isinstance(fold_train, list) or not isinstance(fold_validation, list):
            raise ValueError("Each CV fold must have train and validation positions.")
        if not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in fold_train + fold_validation
        ):
            raise ValueError("CV row positions must be integers.")
        if (
            set(fold_train).intersection(fold_validation)
            or set(fold_train).union(fold_validation)
            != expected_training_positions
            or len(set(fold_train)) != len(fold_train)
            or len(set(fold_validation)) != len(fold_validation)
            or len(fold_validation) < 2
        ):
            raise ValueError("Saved CV fold positions are invalid.")
        validation_seen.extend(fold_validation)
        normalized_folds.append(
            {
                "train_indices": fold_train,
                "validation_indices": fold_validation,
            }
        )
    if sorted(validation_seen) != list(range(len(train_indices))):
        raise ValueError(
            "Each training row must appear in validation exactly once across folds."
        )
    return train_indices, test_indices, normalized_folds


def split_and_train(
    data: pd.DataFrame,
    target_column: str,
    features: Sequence[str] | str | None = None,
    random_state: int = 42,
    folds: dict[str, Any] | None = None,
) -> tuple[LinearRegression, pd.DataFrame, pd.Series, dict[str, dict[str, Any]]]:
    """Train on an 80% split, optionally using previously saved row folds.

    Five-fold cross-validation and its metrics are calculated using only the
    80% training partition. Returned standard deviations are population
    standard deviations across the five fold scores.

    ``folds`` may be supplied by ``create_folds_and_train`` to guarantee the
    same outer split and CV fold membership across feature-selection runs.
    Returns ``(model, X_test, y_test, cross_validation_metrics)``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string.")
    if target_column not in data.columns:
        raise ValueError("Target column {!r} was not found.".format(target_column))
    if data.columns.has_duplicates:
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(
            "DataFrame column names must be unique; duplicates found: {!r}.".format(
                duplicates
            )
        )
    if len(data) < 2:
        raise ValueError("At least two rows are required for an 80/20 split.")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer.")

    selected_features = _selected_features(data, target_column, features)

    if folds is None:
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=random_state
        )
        cv_folds = None
    else:
        train_indices, test_indices, cv_folds = _validate_saved_folds(data, folds)
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
    metrics = _cross_validation_metrics(
        train_data,
        target_column=target_column,
        features=features,
        random_state=random_state,
        cv_folds=cv_folds,
    )
    model = train_linear_regression(
        train_data,
        target_column=target_column,
        features=features,
    )
    return (
        model,
        test_data[selected_features].copy(),
        test_data[target_column].copy(),
        metrics,
    )
