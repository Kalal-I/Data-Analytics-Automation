import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype
from sklearn.preprocessing import LabelEncoder


def label_encode(data, columns=None):
    """Label encode selected columns, or all string and categorical columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if columns is None:
        selected_columns = [
            column
            for column in data.columns
            if is_object_dtype(data[column].dtype)
            or is_string_dtype(data[column].dtype)
            or is_categorical_dtype(data[column].dtype)
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

    result = data.copy()
    for column in selected_columns:
        values = data[column]
        present_values = values.dropna()
        if present_values.empty:
            result[column] = pd.Series(pd.NA, index=data.index, dtype="Int64")
            continue

        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(present_values)
        encoded_column = pd.Series(pd.NA, index=data.index, dtype="Int64")
        encoded_column.loc[present_values.index] = encoded_values
        result[column] = encoded_column

    return result