"""Create, persist, and reuse a fixed holdout split and CV fold assignment."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from linear_regression.split_data import (
    _data_fingerprint,
    _selected_features,
    _validate_saved_folds,
    split_and_train,
)

DEFAULT_FOLDS_FILE = Path(__file__).with_name("stored_folds.json")


def create_or_load_folds(
    data: pd.DataFrame,
    folds_file: str | Path = DEFAULT_FOLDS_FILE,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load an existing compatible fold file or create and persist one.

    Stored row positions are tied to the exact DataFrame values, columns,
    dtypes, and index. If the data changes, this function refuses to silently
    generate a different evaluation split; choose a new file or explicitly
    remove the old one to start a new experiment.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if len(data) < 2:
        raise ValueError("At least two rows are required to create a split.")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer.")
    path = Path(folds_file)
    fingerprint = _data_fingerprint(data)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                stored = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Could not read saved fold data from {!s}.".format(path)
            ) from error
        if not isinstance(stored, dict):
            raise ValueError("Saved fold file must contain a JSON object.")
        if stored.get("data_fingerprint") != fingerprint:
            raise ValueError(
                "Saved folds belong to different data or row ordering. "
                "Use another fold file to create a new assignment."
            )
        _validate_saved_folds(data, stored)
        return stored

    train_indices, test_indices = train_test_split(
        np.arange(len(data), dtype=int),
        test_size=0.2,
        random_state=random_state,
    )
    if len(train_indices) < 10:
        raise ValueError(
            "The 80% training split must contain at least 10 rows for "
            "five-fold R^2 cross-validation."
        )

    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_folds = [
        {
            "train_indices": fold_train.tolist(),
            "validation_indices": fold_validation.tolist(),
        }
        for fold_train, fold_validation in splitter.split(train_indices)
    ]
    folds: dict[str, Any] = {
        "schema_version": 1,
        "row_count": len(data),
        "data_fingerprint": fingerprint,
        "random_state": random_state,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "cv_folds": cv_folds,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(folds, file, indent=2, allow_nan=False)
        temporary_path.replace(path)
    except OSError as error:
        raise OSError("Could not save fold data to {!s}.".format(path)) from error
    return folds


def create_folds_and_train(
    data: pd.DataFrame,
    target_column: str,
    features: list[str] | tuple[str, ...] | str | None = None,
    random_state: int = 42,
    folds_file: str | Path = DEFAULT_FOLDS_FILE,
):
    """Reuse a persisted split/fold assignment to train a selected feature set."""
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
    _selected_features(data, target_column, features)
    folds = create_or_load_folds(
        data,
        folds_file=folds_file,
        random_state=random_state,
    )
    return split_and_train(
        data,
        target_column=target_column,
        features=features,
        random_state=random_state,
        folds=folds,
    )
