"""Serialize linear-regression training and evaluation results as strict JSON."""

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from linear_regression.split_data import _adjusted_r2


def _json_float(value: Any, field: str) -> float:
    """Convert a numeric scalar and reject values not representable in JSON."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite number.".format(field)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number.".format(field))
    return result


def _json_index_value(value: Any) -> dict[str, Any]:
    """Encode supported index labels without conflating their Python types.

    JSON row indices use tagged values for null, boolean, integer, finite
    floating-point, string, and tuple labels. Tuple values support MultiIndex
    labels recursively. Other label types are rejected rather than stringified.
    """
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "value": [_json_index_value(item) for item in value],
        }
    if isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("Missing datetime index labels cannot be serialized.")
        return {"type": "datetime", "value": timestamp.isoformat(), "fold": 0}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Index labels must be finite when floating-point.")
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, datetime):
        if pd.isna(value):
            raise ValueError("Missing datetime index labels cannot be serialized.")
        return {"type": "datetime", "value": value.isoformat(), "fold": value.fold}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    raise TypeError(
        "Unsupported index label type {!r}; supported labels are null, "
        "booleans, integers, finite floats, strings, dates, datetimes, "
        "and tuples of supported labels.".format(type(value).__name__)
    )


def split_result_to_json(
    split_result: tuple[Any, pd.DataFrame, pd.Series, dict[str, Any]],
    target_column: str,
    folds: dict[str, Any] | None = None,
    include_holdout_stats: bool = False,
) -> str:
    """Convert ``split_and_train`` output into a detailed strict JSON report.

    The report includes selected features, fitted coefficients, and CV fold
    metrics. Holdout predictions and metrics are included only when
    ``include_holdout_stats`` is true. If ``folds`` is supplied, the persisted
    row positions and split metadata are included too. Index labels in
    holdout rows are encoded as tagged JSON objects to preserve types;
    supported labels are null, booleans, integers, finite floats, strings,
    dates, datetimes, and recursively supported tuples. Other index label
    types are rejected. Holdout R² is null when it is undefined because there
    are fewer than two rows or the observed target is constant. Adjusted R² is
    null when R² is undefined or the holdout has no more than one additional
    observation beyond the predictor count; MAE and RMSE are still reported
    whenever at least one holdout row is available.
    """
    if not isinstance(split_result, tuple) or len(split_result) != 4:
        raise TypeError(
            "split_result must be the four-item tuple returned by split_and_train."
        )
    model, x_test, y_test, cv_metrics = split_result
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string.")
    if not isinstance(include_holdout_stats, bool):
        raise TypeError("include_holdout_stats must be a boolean.")
    if not isinstance(x_test, pd.DataFrame) or not isinstance(y_test, pd.Series):
        raise TypeError("split_result must contain a DataFrame and target Series.")
    if len(x_test) != len(y_test):
        raise ValueError("Test predictors and targets must have matching row counts.")
    if not x_test.index.equals(y_test.index):
        raise ValueError("Test predictors and targets must have matching indexes.")

    feature_names = [str(name) for name in model.feature_names_in_]
    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(feature_names) != len(coefficients):
        raise ValueError("Model feature names do not match its coefficients.")
    if list(x_test.columns) != feature_names:
        raise ValueError("Test predictor columns do not match the fitted model.")

    split_details: dict[str, Any] = {
        "training_row_count": model.training_row_count_,
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
                "holdout_row_positions": test_indices,
                "training_row_positions": train_indices,
                "random_state": folds.get("random_state"),
                "data_fingerprint": folds.get("data_fingerprint"),
                "cross_validation_folds": cv_folds,
            }
        )

    model_type = getattr(model, "model_type_", None)
    model_names = {
        "lasso": "Lasso",
        "ridge": "Ridge",
        "elastic_net": "Elastic Net",
    }
    if model_type not in model_names:
        raise ValueError("The fitted model is missing supported model metadata.")
    hyperparameters = {
        "alpha": _json_float(
            model.regularization_alpha_,
            "model regularization alpha",
        )
    }
    if model_type == "elastic_net":
        hyperparameters["l1_ratio"] = _json_float(
            model.regularization_l1_ratio_,
            "model l1_ratio",
        )

    report = {
        "schema_version": 1,
        "target_column": target_column,
        "features": feature_names,
        "data_split": split_details,
        "training": {
            "model": model_names[model_type],
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "intercept": _json_float(model.intercept_, "model intercept"),
            "coefficients": {
                name: _json_float(coefficient, "coefficient for {!r}".format(name))
                for name, coefficient in zip(feature_names, coefficients)
            },
            "cross_validation": cv_metrics,
        },
    }
    if include_holdout_stats:
        predictions = (
            np.asarray(model.predict(x_test), dtype=float).reshape(-1)
            if len(x_test)
            else np.empty(0, dtype=float)
        )
        actual_values = np.asarray(y_test, dtype=float).reshape(-1)
        if not np.isfinite(predictions).all() or not np.isfinite(actual_values).all():
            raise ValueError("Test predictions and actual values must be finite.")
        test_r2 = (
            _json_float(r2_score(actual_values, predictions), "test R^2")
            if len(actual_values) >= 2 and np.unique(actual_values).size >= 2
            else None
        )
        test_adjusted_r2 = _adjusted_r2(
            test_r2, len(actual_values), len(feature_names)
        )
        test_metrics: dict[str, float | None] = {
            "r2": test_r2,
            "adjusted_r2": (
                _json_float(test_adjusted_r2, "test adjusted R^2")
                if test_adjusted_r2 is not None
                else None
            ),
            "mae": (
                _json_float(
                    mean_absolute_error(actual_values, predictions), "test MAE"
                )
                if len(actual_values)
                else None
            ),
            "rmse": (
                _json_float(
                    np.sqrt(mean_squared_error(actual_values, predictions)),
                    "test RMSE",
                )
                if len(actual_values)
                else None
            ),
        }
        report["holdout"] = {
            "row_count": len(y_test),
            "metrics": test_metrics,
            "rows": [
                {
                    "index": _json_index_value(index),
                    "actual": _json_float(actual, "actual target"),
                    "predicted": _json_float(prediction, "prediction"),
                }
                for index, actual, prediction in zip(
                    y_test.index, actual_values, predictions
                )
            ],
        }
    report["feature_removal_tests"] = getattr(
        model, "feature_removal_tests_", []
    )
    try:
        return json.dumps(report, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Training results could not be serialized as strict JSON.") from error
