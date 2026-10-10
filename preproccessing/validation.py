import inspect
import json
import math

from .drop_rows_with_missing_values import remove_missing_values
from .impute_missing_values import fill_missing_values, fill_missing_values_by_group
from .label_encoding import label_encode
from .min_max_scaler import min_max_scale
from .one_hot_encoding import one_hot_encode
from .ordinal_encoding import ordinal_encode
from .outlier_detection import detect_outliers
from .standard_scaler import standard_scale


_FUNCTIONS = {
    function.__name__: function
    for function in (
        label_encode,
        one_hot_encode,
        standard_scale,
        ordinal_encode,
        min_max_scale,
        detect_outliers,
        fill_missing_values,
        fill_missing_values_by_group,
        remove_missing_values,
    )
}
def _dataset_columns(description):
    if isinstance(description, str):
        description = json.loads(description)
    if not isinstance(description, dict) or not isinstance(description.get("Reports"), list):
        return None

    columns = {}
    targets = set()
    for report in description["Reports"]:
        if not isinstance(report, dict):
            return None

        for section, kind in (
            ("Numerical columns", "numeric"),
            ("Categorical columns", "categorical"),
            ("Non-categorical string columns", "string"),
        ):
            entries = report.get(section)
            if not isinstance(entries, dict):
                return None
            for name, details in entries.items():
                if not isinstance(name, str) or not isinstance(details, dict):
                    return None
                columns[name] = (kind, details.get("Missing percentage"))

        target = report.get("Target column")
        if not isinstance(target, dict):
            return None
        if target:
            name = target.get("Name")
            if not isinstance(name, str) or not name:
                return None
            details = target.get("Statistics")
            if not isinstance(details, dict):
                return None
            target_type = target.get("Type")
            if target_type == "Numerical":
                kind = "numeric"
            elif target_type == "Categorical":
                kind = "categorical"
            else:
                kind = "string"
            columns[name] = (kind, details.get("Missing percentage"))
            targets.add(name)
    return columns, targets


def _valid_markers(value):
    scalar = lambda item: item is None or isinstance(item, (str, int, float, bool))
    if scalar(value):
        return True
    return (
        isinstance(value, list)
        and bool(value)
        and all(scalar(item) for item in value)
    )


def _valid_operation(column, operation, columns):
    if not isinstance(operation, dict) or set(operation) != {"function", "inputs"}:
        return False

    name, inputs = operation["function"], operation["inputs"]
    function = _FUNCTIONS.get(name) if isinstance(name, str) else None
    if function is None or not isinstance(inputs, dict) or not inputs:
        return False

    parameters = inspect.signature(function).parameters
    allowed = set(parameters) - {"data"}
    required = {
        key for key, parameter in parameters.items()
        if key != "data" and parameter.default is inspect.Parameter.empty
    }
    if inputs.keys() - allowed or required - inputs.keys():
        return False

    if name == "fill_missing_values_by_group":
        selected = inputs.get("target_column")
        group = inputs.get("group_column")
        if (
            selected != column
            or not isinstance(group, str)
            or group not in columns
            or selected == group
        ):
            return False
        selected_columns = [selected]
    else:
        value = inputs.get("columns")
        selected_columns = [value] if isinstance(value, str) else value
        if selected_columns != [column]:
            return False

    if any(selected not in columns for selected in selected_columns):
        return False
    kind = columns[column][0]

    if name in {"standard_scale", "min_max_scale", "detect_outliers"} and kind != "numeric":
        return False
    if name in {"one_hot_encode", "label_encode", "ordinal_encode"} and kind != "categorical":
        return False

    if name == "fill_missing_values":
        strategy = inputs.get("strategy", "mean")
        if not isinstance(strategy, str) or strategy not in {"mean", "median", "mode"}:
            return False
        if strategy in {"mean", "median"} and kind != "numeric":
            return False

    if name == "detect_outliers":
        method = inputs.get("method", "zscore")
        if not isinstance(method, str) or method not in {"zscore", "iqr"}:
            return False
        if ("threshold" in inputs and method != "zscore") or (
            "multiplier" in inputs and method != "iqr"
        ):
            return False
        for parameter in ("threshold", "multiplier"):
            if parameter in inputs:
                value = inputs[parameter]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    return False

    if "missing_values" in inputs and not _valid_markers(inputs["missing_values"]):
        return False

    if name == "ordinal_encode" and "categories" in inputs:
        categories = inputs["categories"]
        if not isinstance(categories, dict) or set(categories) != {column}:
            return False
        for ordered_values in categories.values():
            if (
                not isinstance(ordered_values, list)
                or not ordered_values
                or not all(
                    value is None or isinstance(value, (str, int, float, bool))
                    for value in ordered_values
                )
                or len(ordered_values) != len(set(map(repr, ordered_values)))
            ):
                return False

    if name in {"fill_missing_values", "fill_missing_values_by_group", "remove_missing_values"}:
        missing = columns[column][1]
        if (
            not ("missing_values" in inputs and inputs["missing_values"] is not None)
            and isinstance(missing, (int, float))
            and missing <= 0
        ):
            return False

    return True


def validate_preprocessing_calls(preprocessing_calls, dataset_description):
    """Return only preprocessing operations valid for the profiler report."""
    try:
        parsed = _dataset_columns(dataset_description)
        if parsed is None or not isinstance(preprocessing_calls, dict):
            return {}
        columns, targets = parsed
        if not columns:
            return {}

        valid_calls = {}
        for column, operations in preprocessing_calls.items():
            if (
                not isinstance(column, str)
                or column not in columns
                or column in targets
                or not isinstance(operations, list)
            ):
                continue
            valid_operations = [
                operation
                for operation in operations
                if _valid_operation(column, operation, columns)
            ]
            if valid_operations:
                valid_calls[column] = valid_operations
        return valid_calls
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
        return {}
