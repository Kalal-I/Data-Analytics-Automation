import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype
from sklearn.preprocessing import OrdinalEncoder


def ordinal_encode(data, columns=None, categories=None):
    """Ordinal encode selected columns, optionally with a specified category order."""
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
    if categories is not None:
        if not isinstance(categories, dict):
            raise TypeError("categories must be a dictionary mapping columns to ordered values")
        unexpected_columns = [column for column in categories if column not in selected_columns]
        if unexpected_columns:
            raise ValueError(f"category orders provided for unselected columns: {unexpected_columns}")
        for column, order in categories.items():
            if not isinstance(order, (list, tuple)) or not order:
                raise TypeError(f"category order for '{column}' must be a non-empty list or tuple")

    if not selected_columns:
        return data.copy()

    result = data.copy()
    for column in selected_columns:
        values = data[column]
        present_positions = values.notna().to_numpy().nonzero()[0]
        encoded_values = pd.Series(pd.NA, index=data.index, dtype="Int64")
        if len(present_positions) == 0:
            result[column] = encoded_values
            continue

        if categories is not None and column in categories:
            order = categories[column]
            if not all(pd.api.types.is_scalar(value) for value in order):
                raise TypeError(f"category order for '{column}' must contain only scalar values")
            if len(pd.Index(order).unique()) != len(order):
                raise ValueError(f"category order for '{column}' cannot contain duplicates")
            categorical_values = pd.Categorical(
                values.iloc[present_positions], categories=order, ordered=True
            )
            encoded_column = categorical_values.codes
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            encoded_column = encoder.fit_transform(
                values.iloc[present_positions].to_frame()
            ).ravel().astype(int)

        encoded_values.iloc[present_positions] = encoded_column
        result[column] = encoded_values

    return result