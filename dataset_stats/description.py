import json
import os

import pandas as pd


def to_json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def get_missing_percentage(values):
    if len(values) == 0:
        return 0.0
    return round(float(values.isna().sum()) / len(values) * 100, 2)


def get_column_report(values, classification_values):
    non_null_values = values.dropna()
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_numeric_count = int(numeric_values.notna().sum())

    if pd.api.types.is_numeric_dtype(values.dtype) or valid_numeric_count > 0:
        valid_values = numeric_values.dropna()
        bad_characters = 0
        if not pd.api.types.is_numeric_dtype(values.dtype):
            bad_characters = len(non_null_values) - valid_numeric_count

        return "Numerical", {
            "Max value": to_json_value(valid_values.max()) if not valid_values.empty else None,
            "Min value": to_json_value(valid_values.min()) if not valid_values.empty else None,
            "Mean": to_json_value(valid_values.mean()) if not valid_values.empty else None,
            "Median": to_json_value(valid_values.median()) if not valid_values.empty else None,
            "Missing percentage": get_missing_percentage(values),
            "Bad characters": bad_characters
        }

    is_non_categorical = (
        classification_values.nunique() == len(classification_values)
    )
    if is_non_categorical:
        return "Non-categorical string", {
            "Missing percentage": get_missing_percentage(values),
            "Duplicates": int(non_null_values.duplicated().sum())
        }

    categories = non_null_values.unique()
    return "Categorical", {
        "Number of unique values": int(non_null_values.nunique()),
        "Categories": [to_json_value(value) for value in categories[:10]],
        "Missing percentage": get_missing_percentage(values)
    }


def generate_report(data, file_names, target_column):
    """Return JSON reports for DataFrame input and its target column."""
    if isinstance(data, pd.DataFrame):
        dataframes = [data]
        file_names = [file_names]
    elif isinstance(data, (list, tuple)):
        dataframes = data
        if isinstance(file_names, str):
            raise ValueError("Provide one filename for each DataFrame.")
        file_names = list(file_names)
    else:
        raise TypeError("Input must be a DataFrame or a list of DataFrames.")

    if not isinstance(target_column, str):
        raise TypeError("Target column name must be a string.")
    if len(dataframes) != len(file_names):
        raise ValueError("Provide one filename for each DataFrame.")
    if not all(isinstance(name, str) for name in file_names):
        raise TypeError("Each filename must be a string.")

    reports = []
    for dataframe, file_name in zip(dataframes, file_names):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Every item in the input must be a DataFrame.")
        if target_column not in dataframe.columns:
            raise ValueError(
                "Target column '{}' was not found in {}.".format(
                    target_column, file_name
                )
            )

        unique_rows = dataframe.drop_duplicates()
        duplicate_count = int(dataframe.duplicated().sum())
        duplicate_percentage = (
            round(duplicate_count / len(dataframe) * 100, 2)
            if len(dataframe) else 0.0
        )
        report = {
            "File name": os.path.basename(file_name),
            "Overall report": {
                "Number of columns": len(dataframe.columns),
                "Number of rows": len(dataframe.index),
                "Duplicate rows percentage": duplicate_percentage
            },
            "Numerical columns": {},
            "Categorical columns": {},
            "Non-categorical string columns": {},
            "Target column": {}
        }

        for column in dataframe.columns:
            values = dataframe[column]
            classification_values = unique_rows[column].dropna()
            column_type, details = get_column_report(
                values, classification_values
            )

            if column == target_column:
                report["Target column"] = {
                    "Name": column,
                    "Type": column_type,
                    "Statistics": details
                }
            elif column_type == "Numerical":
                report["Numerical columns"][column] = details
            elif column_type == "Categorical":
                report["Categorical columns"][column] = details
            else:
                report["Non-categorical string columns"][column] = details

        reports.append(report)

    return json.dumps({"Reports": reports}, indent=4, default=str)


if __name__ == "__main__":
    file_paths = [
        r"D:\Kavin\College work\Sem 5\ML\package\datasets\dataset_parsed.csv"
    ]
    dataframes = [pd.read_csv(path) for path in file_paths]
    target_column = input("Enter the target column name: ")
    print(generate_report(dataframes, file_paths, target_column))
