"""Keep categorical encodings out of pre-model statistical analysis."""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

ENCODING_FUNCTIONS = frozenset(
    {"one_hot_encode", "label_encode", "ordinal_encode"}
)
PreprocessingOperations = dict[str, list[dict[str, Any]]]


def partition_deferred_encodings(
    operations: PreprocessingOperations,
) -> tuple[PreprocessingOperations, PreprocessingOperations]:
    """Separate categorical encodings from operations applied before statistics."""
    immediate_operations: PreprocessingOperations = {}
    deferred_operations: PreprocessingOperations = {}

    for column, column_operations in operations.items():
        for operation in column_operations:
            destination = (
                deferred_operations
                if operation["function"] in ENCODING_FUNCTIONS
                else immediate_operations
            )
            destination.setdefault(column, []).append(operation)

    return immediate_operations, deferred_operations


def apply_deferred_encodings(
    data: pd.DataFrame,
    operations: PreprocessingOperations,
    preprocessing_functions: Mapping[str, Callable[..., pd.DataFrame]],
) -> pd.DataFrame:
    """Apply staged categorical encodings after a model configuration is chosen."""
    for column_operations in operations.values():
        for operation in column_operations:
            function_name = operation["function"]
            if function_name not in ENCODING_FUNCTIONS:
                raise ValueError(
                    "Deferred operation {!r} is not a categorical encoding.".format(
                        function_name
                    )
                )
            function = preprocessing_functions[function_name]
            data = function(data, **operation["inputs"])
    return data
