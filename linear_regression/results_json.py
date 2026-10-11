"""Serialize linear-regression training and evaluation results as strict JSON."""

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _json_float(value: Any, field: str) -> float:
    """Convert a numeric scalar and reject values not representable in JSON."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite number.".format(field)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number.".format(field))
    return result


def split_result_to_json(
    split_result: tuple[Any, pd.DataFrame, pd.Series, dict[str, Any]],
    target_column: str,
    folds: dict[str, Any] | None = None,
) -> str:
    """Convert ``split_and_train`` output into a detailed strict JSON report.

    The report includes selected features, fitted coefficients, CV fold
    metrics, and predictions plus metrics on the held-out test partition. If
    ``folds`` is supplied, the persisted row positions and split metadata are
    included too.
    """
    if not isinstance(split_result, tuple) or len(split_result) != 4:
        raise TypeError(
            "split_result must be the four-item tuple returned by split_and_train."
        )
    model, x_test, y_test, cv_metrics = split_result
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string.")
    if not isinstance(x_test, pd.DataFrame) or not isinstance(y_test, pd.Series):
        raise TypeError("split_result must contain a DataFrame and target Series.")
    if len(x_test) != len(y_test):
        raise ValueError("Test predictors and targets must have matching row counts.")

    feature_names = [str(name) for name in model.feature_names_in_]
    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(feature_names) != len(coefficients):
        raise ValueError("Model feature names do not match its coefficients.")

    predictions = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    actual_values = np.asarray(y_test, dtype=float).reshape(-1)
    if not np.isfinite(predictions).all() or not np.isfinite(actual_values).all():
        raise ValueError("Test predictions and actual values must be finite.")

    if len(actual_values) >= 2:
        test_metrics: dict[str, float | None] = {
            "r2": _json_float(r2_score(actual_values, predictions), "test R^2"),
            "mae": _json_float(
                mean_absolute_error(actual_values, predictions), "test MAE"
            ),
            "rmse": _json_float(
                np.sqrt(mean_squared_error(actual_values, predictions)), "test RMSE"
            ),
        }
    else:
        test_metrics = {"r2": None, "mae": None, "rmse": None}

    split_details: dict[str, Any] = {
        "holdout_row_count": len(y_test),
        "cross_validation_fold_count": 5,
    }
    if folds is not None:
        if not isinstance(folds, dict):
            raise TypeError("folds must be a dictionary.")
        train_indices = folds.get("train_indices")
        test_indices = folds.get("test_indices")
        cv_folds = folds.get("cv_folds")
        if not (
            isinstance(train_indices, list)
            and isinstance(test_indices, list)
            and isinstance(cv_folds, list)
            and len(cv_folds) == 5
        ):
            raise ValueError("folds must contain the saved split and five CV folds.")
        split_details.update(
            {
                "training_row_count": len(train_indices),
                "holdout_row_positions": test_indices,
                "training_row_positions": train_indices,
                "random_state": folds.get("random_state"),
                "data_fingerprint": folds.get("data_fingerprint"),
                "cross_validation_folds": cv_folds,
            }
        )

    report = {
        "schema_version": 1,
        "target_column": target_column,
        "features": feature_names,
        "data_split": split_details,
        "training": {
            "model": "LinearRegression",
            "intercept": _json_float(model.intercept_, "model intercept"),
            "coefficients": {
                name: _json_float(coefficient, "coefficient for {!r}".format(name))
                for name, coefficient in zip(feature_names, coefficients)
            },
            "cross_validation": cv_metrics,
        },
        "holdout": {
            "row_count": len(y_test),
            "metrics": test_metrics,
            "rows": [
                {
                    "index": str(index),
                    "actual": _json_float(actual, "actual target"),
                    "predicted": _json_float(prediction, "prediction"),
                }
                for index, actual, prediction in zip(
                    y_test.index, actual_values, predictions
                )
            ],
        },
    }
    try:
        return json.dumps(report, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Training results could not be serialized as strict JSON.") from error
