"""Feature selection and validation for KNN regression."""
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_features(data: pd.DataFrame, target: str, features: Optional[Iterable[str]] = None) -> List[str]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))

    if features is None:
        selected = []
        for column in data.columns:
            if column == target or not is_numeric_dtype(data[column].dtype):
                continue
            values = data[column].to_numpy()
            if data[column].isna().any() or not np.isfinite(values.astype(float)).all():
                raise ValueError("feature {!r} contains missing or non-finite values; preprocess it before KNN".format(column))
            if data[column].nunique(dropna=False) > 1:
                selected.append(column)
        if not selected:
            raise ValueError("no usable numeric features remain after removing constant features")
        return selected

    if isinstance(features, str):
        raise TypeError("features must be a list of column names, not a string")
    selected = list(features)
    if not selected:
        raise ValueError("at least one feature must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("selected features must not contain duplicates")
    missing = [column for column in selected if column not in data.columns]
    if missing:
        raise KeyError("feature columns not found: {}".format(missing))
    if target in selected:
        raise ValueError("target column {!r} cannot be a feature".format(target))
    invalid_type = [column for column in selected if not is_numeric_dtype(data[column].dtype)]
    if invalid_type:
        raise TypeError("selected features must be numeric: {}".format(invalid_type))
    invalid_values = []
    for column in selected:
        if data[column].isna().any() or not np.isfinite(data[column].to_numpy(dtype=float)).all():
            invalid_values.append(column)
    if invalid_values:
        raise ValueError("selected features contain missing or non-finite values; preprocess them first: {}".format(invalid_values))
    return selected


select_features = validate_features
