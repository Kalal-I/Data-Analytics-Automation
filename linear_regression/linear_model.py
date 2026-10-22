"""Fit a multiple linear regression model to training data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype
from sklearn.linear_model import ElasticNet, Lasso, Ridge

RegularizedRegressor = Lasso | Ridge | ElasticNet


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


def _create_regressor(
    model_type: str,
    alpha: float,
    l1_ratio: float | None = None,
) -> RegularizedRegressor:
    """Validate a regularization configuration and create its estimator."""
    if not isinstance(model_type, str):
        raise TypeError("model_type must be a string.")
    if model_type not in {"lasso", "ridge", "elastic_net"}:
        raise ValueError(
            "model_type must be 'lasso', 'ridge', or 'elastic_net'."
        )
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise TypeError("alpha must be a positive finite number.")
    try:
        alpha_value = float(alpha)
    except OverflowError as error:
        raise ValueError("alpha must be a positive finite number.") from error
    if not np.isfinite(alpha_value) or alpha_value <= 0:
        raise ValueError("alpha must be a positive finite number.")

    if model_type == "lasso":
        return Lasso(alpha=alpha_value, max_iter=10000)
    if model_type == "ridge":
        return Ridge(alpha=alpha_value)

    if l1_ratio is None:
        raise ValueError("l1_ratio is required for Elastic Net.")
    if isinstance(l1_ratio, bool) or not isinstance(
        l1_ratio, (int, float, np.integer, np.floating)
    ):
        raise TypeError("l1_ratio must be a number satisfying 0 < l1_ratio <= 1.")
    try:
        l1_ratio_value = float(l1_ratio)
    except OverflowError as error:
        raise ValueError("l1_ratio must satisfy 0 < l1_ratio <= 1.") from error
    if not np.isfinite(l1_ratio_value) or not 0 < l1_ratio_value <= 1:
        raise ValueError("l1_ratio must satisfy 0 < l1_ratio <= 1.")
    return ElasticNet(
        alpha=alpha_value,
        l1_ratio=l1_ratio_value,
        max_iter=10000,
    )


def train_linear_regression(
    training_data: pd.DataFrame,
    target_column: str,
    model_type: str,
    alpha: float,
    l1_ratio: float | None = None,
) -> RegularizedRegressor:
    """Fit and return a regularized regression model.

    Every column other than ``target_column`` is used as a predictor.
    Predictors and target must be finite, real numeric data; this function
    does not encode categories or impute missing values.
    """
    model = _create_regressor(model_type, alpha, l1_ratio)
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

    selected = _selected_features(training_data, target_column, None)
    predictors = _as_finite_numeric(training_data, selected)
    target_frame = _as_finite_numeric(training_data, [target_column])
    target = target_frame[target_column]

    model.fit(predictors, target)
    model.training_row_count_ = len(training_data)
    model.model_type_ = model_type
    model.regularization_alpha_ = float(alpha)
    model.regularization_l1_ratio_ = (
        float(l1_ratio) if model_type == "elastic_net" else None
    )
    return model
