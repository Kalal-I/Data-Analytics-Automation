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


def fill_missing_values(data, strategy="mean", columns=None, missing_values=None):
    """Fill NA and optional marker values with mean, median, or mode."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(strategy, str):
        raise TypeError("strategy must be a string: 'mean', 'median', or 'mode'")

    strategy = strategy.lower()
    if strategy not in {"mean", "median", "mode"}:
        raise ValueError("strategy must be 'mean', 'median', or 'mode'")

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
    for column in selected_columns:
        values = result[column]
        if markers:
            values = values.mask(values.isin(markers))

        if not values.isna().any():
            result[column] = values
            continue

        if strategy == "mode":
            modes = values.mode(dropna=True)
            if modes.empty:
                raise ValueError(f"cannot compute mode for all-null column '{column}'")
            fill_value = modes.iloc[0]
        else:
            try:
                values = pd.to_numeric(values, errors="raise")
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"{strategy} imputation requires a numeric column: '{column}'"
                ) from error
            fill_value = getattr(values, strategy)()
            if pd.isna(fill_value):
                raise ValueError(f"cannot compute {strategy} for all-null column '{column}'")

        result[column] = values.fillna(fill_value)

    return result


def fill_missing_values_by_group(data, target_column, group_column, missing_values=None):
    """Fill a numeric column's missing values with means from each group."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target_column, str) or not isinstance(group_column, str):
        raise TypeError("target_column and group_column must be strings")

    missing_columns = [
        column for column in (target_column, group_column) if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"columns not found in data: {missing_columns}")
    if target_column == group_column:
        raise ValueError("target_column and group_column must be different")

    markers = _normalize_markers(missing_values)
    values = data[target_column].copy()
    if markers:
        values = values.mask(values.isin(markers))

    try:
        values = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"grouped mean imputation requires a numeric column: '{target_column}'"
        ) from error

    group_means = values.groupby(data[group_column], dropna=True).transform("mean")
    filled_values = values.fillna(group_means)
    if filled_values.isna().any():
        raise ValueError(
            f"some missing values in '{target_column}' have no available group mean"
        )

    result = data.copy()
    result[target_column] = filled_values
    return result