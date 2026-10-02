import math

import pandas as pd
from pandas.api.types import is_numeric_dtype


def detect_outliers(data, method="zscore", columns=None, threshold=3.0, multiplier=1.5):
    """Return a copy with rows outside Z-score or IQR bounds removed."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(method, str):
        raise TypeError("method must be a string: 'zscore' or 'iqr'")

    method = method.lower()
    if method not in {"zscore", "iqr"}:
        raise ValueError("method must be 'zscore' or 'iqr'")

    if columns is None:
        selected_columns = [
            column for column in data.columns if is_numeric_dtype(data[column].dtype)
        ]
    elif isinstance(columns, str):
        selected_columns = [columns]
    elif isinstance(columns, (list, tuple)):
        if not columns:
            raise ValueError("columns cannot be empty")
        if not all(isinstance(column, str) for column in columns):
            raise TypeError("columns must contain only strings")
        selected_columns = list(columns)
    else:
        raise TypeError("columns must be a string, a list of strings, or None")

    missing_columns = [column for column in selected_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"columns not found in data: {missing_columns}")
    non_numeric_columns = [
        column for column in selected_columns if not is_numeric_dtype(data[column].dtype)
    ]
    if non_numeric_columns:
        raise TypeError(f"outlier detection requires numeric columns: {non_numeric_columns}")

    if method == "zscore":
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a positive finite number")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a positive finite number")
    else:
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise TypeError("multiplier must be a positive finite number")
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("multiplier must be a positive finite number")

    if not selected_columns:
        return data.copy()

    outlier_mask = pd.Series(False, index=data.index)
    for column in selected_columns:
        values = data[column]
        if method == "zscore":
            standard_deviation = values.std(ddof=0)
            if pd.isna(standard_deviation) or standard_deviation == 0:
                continue
            column_mask = ((values - values.mean()).abs() / standard_deviation) > threshold
        else:
            first_quartile = values.quantile(0.25)
            third_quartile = values.quantile(0.75)
            interquartile_range = third_quartile - first_quartile
            lower_bound = first_quartile - multiplier * interquartile_range
            upper_bound = third_quartile + multiplier * interquartile_range
            column_mask = (values < lower_bound) | (values > upper_bound)
        outlier_mask |= column_mask.fillna(False)

    return data.loc[~outlier_mask].copy()