import pandas as pd
from pandas.api.types import is_scalar


def _normalize_markers(missing_values):
    if missing_values is None:
        return []
    if isinstance(missing_values, (list, tuple, set)):
        markers = list(missing_values)
        if not markers:
            raise ValueError("missing_values cannot be empty")
    elif is_scalar(missing_values):
        markers = [missing_values]
    else:
        raise TypeError("missing_values must be a scalar value or a list of scalar values")

    if not all(is_scalar(marker) for marker in markers):
        raise TypeError("missing_values must contain only scalar values")
    return markers


def remove_missing_values(data, columns=None, missing_values=None):
    """Drop rows with NA or optional marker values in the selected columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if columns is None:
        selected_columns = list(data.columns)
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

    markers = _normalize_markers(missing_values)
    result = data.copy()
    if markers:
        for column in selected_columns:
            result[column] = result[column].mask(result[column].isin(markers))

    return result.dropna(subset=selected_columns).copy()