"""Automatic validation and filtering of classification predictors."""
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def analyze_features(data: pd.DataFrame, target: str):
    """Return selected features and reasons for automatically excluded columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target, str) or target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))
    if data.columns.has_duplicates:
        raise ValueError("DataFrame column names must be unique")

    selected = []
    excluded = []
    for column in data.columns:
        if column == target:
            excluded.append({"column": column, "reason": "target column"})
            continue
        values = data[column]
        missing = values.isna()
        non_null = values[~missing]
        if non_null.empty:
            excluded.append({"column": column, "reason": "column contains only missing values"})
            continue
        if non_null.nunique(dropna=True) <= 1:
            excluded.append({
                "column": column,
                "reason": "constant column (all non-null values are identical)",
            })
            continue

        if is_numeric_dtype(values.dtype) and not is_bool_dtype(values.dtype):
            numeric_values = values.to_numpy(dtype=float, na_value=np.nan)
            if np.isneginf(numeric_values).any() or np.isposinf(numeric_values).any():
                raise ValueError(
                    "feature {!r} contains positive or negative infinity; "
                    "preprocess it before modeling".format(column)
                )

        if len(non_null) == 0:
            excluded.append({"column": column, "reason": "column has no usable values"})
            continue
        selected.append(column)

    if not selected:
        raise ValueError("no usable predictors remain")
    return {"selected_features": selected, "excluded_features": excluded}


def select_features(data: pd.DataFrame, target: str, return_exclusions=False):
    """Retain usable predictors and optionally return the exclusion report."""
    report = analyze_features(data, target)
    if return_exclusions:
        return report["selected_features"], report["excluded_features"]
    return report["selected_features"]


validate_features = select_features
