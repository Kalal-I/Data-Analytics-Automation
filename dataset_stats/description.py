import json
import os

import pandas as pd

from dataset_stats.column_analysis import (
    analyze_column,
    distinct_values,
    non_missing_values,
)

def to_json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def get_missing_percentage(values):
    if len(values) == 0:
        return 0.0
    return round(float(values.isna().sum()) / len(values) * 100, 2)


def get_column_report(values):
    """Return profile details using the shared dataset column classification."""
    analysis = analyze_column(values)
    non_null_values = non_missing_values(values)
    numeric_values = analysis.numeric_values
    valid_values = numeric_values.dropna()

    if analysis.numeric_candidate:
        return "Numerical", {
            "Max value": to_json_value(valid_values.max()) if not valid_values.empty else None,
            "Min value": to_json_value(valid_values.min()) if not valid_values.empty else None,
            "Mean": to_json_value(valid_values.mean()) if not valid_values.empty else None,
            "Median": to_json_value(valid_values.median()) if not valid_values.empty else None,
            "Missing percentage": get_missing_percentage(values),
            "Bad characters": analysis.invalid_count
        }

    distinct_non_null_values = distinct_values(non_null_values)
    if analysis.detected_type in {
        "all_missing",
        "constant",
        "non_categorical_string",
    }:
        return "Non-categorical string", {
            "Missing percentage": get_missing_percentage(values),
            "Duplicates": len(non_null_values) - len(distinct_non_null_values)
        }

    categories = distinct_non_null_values
    return "Categorical", {
        "Number of unique values": len(categories),
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
            column_type, details = get_column_report(values)

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