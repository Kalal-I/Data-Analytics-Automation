"""Fit a multiple linear regression model to training data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype
from sklearn.linear_model import LinearRegression


def _selected_features(
    data: pd.DataFrame,
    target_column: str,
    features: Sequence[str] | str | None,
) -> list[str]:
    """Validate and return the requested predictor column names."""
    if features is None:
        selected = [column for column in data.columns if column != target_column]
    elif isinstance(features, str):
        selected = [features]
    elif isinstance(features, (list, tuple)):
        selected = list(features)
        if not selected:
            raise ValueError("features cannot be empty.")
    else:
        raise TypeError("features must be a string, a list/tuple of strings, or None.")

    if not all(isinstance(column, str) for column in selected):
        raise TypeError("Every feature name must be a string.")
    if len(set(selected)) != len(selected):
        raise ValueError("features must not contain duplicate column names.")
    if target_column in selected:
        raise ValueError("The target column cannot also be used as a feature.")

    missing = [column for column in selected if column not in data.columns]
    if missing:
        raise ValueError("Feature columns not found in data: {!r}.".format(missing))
    if not selected:
        raise ValueError("At least one feature column is required.")
    return selected


def _as_finite_numeric(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to finite float values or raise a clear error."""
    converted = {}
    for column in columns:
        try:
            values = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Column {!r} must contain only numeric values. "
                "Encode categorical features and handle invalid values first.".format(
                    column
                )
            ) from error

        if is_complex_dtype(values.dtype):
            raise ValueError(
                "Column {!r} contains complex values; real-valued regression "
                "requires real numeric data.".format(column)
            )
        try:
            numeric_values = values.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Column {!r} cannot be represented as real float values.".format(
                    column
                )
            ) from error
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                "Column {!r} contains missing or non-finite values; "
                "handle them before training.".format(column)
            )
        converted[column] = numeric_values

    return pd.DataFrame(converted, index=data.index)


def train_linear_regression(
    training_data: pd.DataFrame,
    target_column: str,
    features: Sequence[str] | str | None = None,
) -> LinearRegression:
    """Fit and return a multiple linear regression model.

    If ``features`` is omitted, all columns other than ``target_column`` are
    used. Predictors and target must be finite, real numeric data; this
    function does not encode categories or impute missing values.
    """
    if not isinstance(training_data, pd.DataFrame):
        raise TypeError("training_data must be a pandas DataFrame.")
    if training_data.empty:
        raise ValueError("training_data must contain at least one row.")
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string.")
    if training_data.columns.has_duplicates:
        duplicates = training_data.columns[
            training_data.columns.duplicated()
        ].tolist()
        raise ValueError(
            "DataFrame column names must be unique; duplicates found: {!r}.".format(
                duplicates
            )
        )
    if target_column not in training_data.columns:
        raise ValueError(
            "Target column {!r} was not found.".format(target_column)
        )

    selected = _selected_features(training_data, target_column, features)
    predictors = _as_finite_numeric(training_data, selected)
    target_frame = _as_finite_numeric(training_data, [target_column])
    target = target_frame[target_column]

    model = LinearRegression()
    model.fit(predictors, target)
    return model
