import pandas as pd


def one_hot_encode(data, columns=None):
    """One-hot encode selected columns, or columns with fewer than 10 values."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if columns is None:
        selected_columns = [
            column for column in data.columns if data[column].nunique(dropna=True) < 10
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

    if not selected_columns:
        return data.copy()

    return pd.get_dummies(data, columns=selected_columns, dtype=int)