import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler


def min_max_scale(data, columns=None):
    """Scale selected numeric columns to [0, 1], or all numeric columns by default."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

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
        raise TypeError(f"scaling requires numeric columns: {non_numeric_columns}")
    if not selected_columns:
        return data.copy()

    result = data.copy()
    scaled_values = MinMaxScaler().fit_transform(data[selected_columns])
    for index, column in enumerate(selected_columns):
        result[column] = scaled_values[:, index]
    return result