"""Tests for consistent classification across dataset reports."""

import json

import pandas as pd
import pytest

from dataset_stats.description import generate_report
from dataset_stats.model_statistics import generate_model_statistics


@pytest.mark.parametrize(
    ("values", "expected_general_type", "expected_model_type"),
    [
        (["1", "2", "bad", "3"], "Numerical", "numerical"),
        (["1", "bad"], "Non-categorical string", "non_categorical_string"),
        (["abc", "def", "3"], "Non-categorical string", "non_categorical_string"),
        (["red", "blue", "red"], "Categorical", "categorical"),
        (["alpha", "beta", "gamma"], "Non-categorical string", "non_categorical_string"),
        (["1", "inf", "bad"], "Numerical", "constant"),
    ],
)
def test_reports_use_the_same_numeric_string_detection_policy(
    values, expected_general_type, expected_model_type
):
    """Both reports classify strings using the shared strict-majority policy."""
    data = pd.DataFrame(
        {"feature": values, "target": list(range(1, len(values) + 1))}
    )

    general = json.loads(generate_report(data, "data.csv", "target"))["Reports"][0]
    if "feature" in general["Numerical columns"]:
        actual_general_type = "Numerical"
    elif "feature" in general["Categorical columns"]:
        actual_general_type = "Categorical"
    else:
        actual_general_type = "Non-categorical string"

    statistics = json.loads(generate_model_statistics(data, "target"))
    actual_model_type = statistics["features"][0]["detected_type"]

    assert actual_general_type == expected_general_type
    assert actual_model_type == expected_model_type


@pytest.mark.parametrize(
    ("feature_values", "expected_general_type"),
    [
        (pd.Series([None, None], dtype="float64"), "Numerical"),
        (pd.Series([None, None], dtype="object"), "Non-categorical string"),
    ],
)
def test_all_missing_feature_keeps_dtype_appropriate_general_classification(
    feature_values, expected_general_type
):
    """All-missing columns are not confused with categories or constants."""
    data = pd.DataFrame(
        {"feature": feature_values, "target": [1, 2]}
    )

    general = json.loads(generate_report(data, "data.csv", "target"))["Reports"][0]
    if "feature" in general["Numerical columns"]:
        actual_general_type = "Numerical"
    elif "feature" in general["Categorical columns"]:
        actual_general_type = "Categorical"
    else:
        actual_general_type = "Non-categorical string"

    statistics = json.loads(generate_model_statistics(data, "target"))

    assert actual_general_type == expected_general_type
    assert statistics["features"][0]["detected_type"] == "all_missing"
