"""Shared column typing and safe numeric conversion for dataset reports."""

import math
import numbers
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnAnalysis:
    """Reusable type and numeric-value facts for one DataFrame column."""

    numeric_values: pd.Series
    detected_type: str
    numeric_candidate: bool
    is_constant: bool
    unique_non_missing_count: int
    missing_count: int
    invalid_count: int
    non_finite_count: int
    precision_loss_count: int


def _is_missing(value: Any) -> bool:
    """Return whether a scalar value is missing, without truth-testing arrays."""
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _is_supported_scalar(value: Any) -> bool:
    """Recognize scalar inputs we can safely inspect as text or numeric values."""
    return isinstance(value, (str, bytes, numbers.Number, Decimal, np.generic))


def _exact_integer(value: Any) -> int | None:
    """Return the exact integer represented by an integral scalar."""
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, Decimal) and value.is_finite():
        if value == value.to_integral_value():
            return int(value)
    return None


def _integer_from_numeric_text(value: str) -> tuple[Decimal | None, int | None]:
    """Parse numeric text and retain its exact decimal/integer value if possible."""
    exact_decimal = Decimal(value)
    exact_integer = None
    if exact_decimal.is_finite() and exact_decimal == exact_decimal.to_integral_value():
        exact_integer = int(exact_decimal)
    return exact_decimal, exact_integer


def _conversion_loses_precision(
    original: Any,
    converted: float,
    exact_integer: int | None,
    exact_decimal: Decimal | None,
) -> bool:
    """Check whether conversion to float64 changed exact integer information."""
    if exact_integer is not None and int(converted) != exact_integer:
        return True
    if exact_decimal is not None and converted == 0 and exact_decimal != 0:
        return True
    return isinstance(original, np.floating) and np.longdouble(converted) != original


def _parse_real_number(value: Any) -> tuple[str, float | None]:
    """Classify a cell and return its finite float64 value when safe."""
    if _is_missing(value):
        return "missing", None
    if not _is_supported_scalar(value):
        return "invalid", None

    if isinstance(value, (complex, np.complexfloating)):
        try:
            real_part = float(value.real)
            imaginary_part = float(value.imag)
        except OverflowError:
            return "non_finite", None
        except (TypeError, ValueError):
            return "invalid", None
        if not math.isfinite(real_part) or not math.isfinite(imaginary_part):
            return "non_finite", None
        if imaginary_part != 0:
            return "invalid", None
        value = real_part

    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return "non_finite", None

    exact_integer = _exact_integer(value)
    exact_decimal = value if isinstance(value, Decimal) else None
    if isinstance(value, str):
        try:
            exact_decimal, text_integer = _integer_from_numeric_text(value)
            if exact_integer is None:
                exact_integer = text_integer
            value = float(value)
        except Exception:
            return "invalid", None

    try:
        numeric = float(value)
    except OverflowError:
        return "non_finite", None
    except Exception:
        return "invalid", None

    if not math.isfinite(numeric):
        return "non_finite", None
    if _conversion_loses_precision(value, numeric, exact_integer, exact_decimal):
        return "precision_loss", None
    return "finite", numeric


def _safe_unique_count(values: list[Any]) -> int:
    """Count values without hashing or invoking custom equality implementations."""
    keys: set[tuple[type[Any], str]] = set()
    for value in values:
        try:
            keys.add((type(value), repr(value)))
        except Exception:
            return len(values)
    return len(keys)


def distinct_values(values: list[Any]) -> list[Any]:
    """Return stable representatives of scalar values without hashing cells."""
    unique: list[Any] = []
    seen: set[tuple[type[Any], str]] = set()
    for value in values:
        try:
            key = (type(value), repr(value))
        except Exception:
            unique.append(value)
            continue
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def non_missing_values(values: pd.Series) -> list[Any]:
    """Return non-missing cells without array-valued truth tests."""
    return [value for value in values if not _is_missing(value)]


def analyze_column(values: pd.Series) -> ColumnAnalysis:
    """Classify a column and safely parse numeric candidates using one shared policy.

    Native numeric columns are numerical. An object/string column is numerical
    only when strictly more than half of non-missing cells are recognized as
    numeric, including non-finite or precision-loss numeric values. Unhashable
    or unsupported cells count as invalid and produce the ``unsupported`` type
    if the column does not meet the numeric threshold.
    """
    value_list = list(values)
    parsed: list[float] = []
    missing_count = 0
    invalid_count = 0
    non_finite_count = 0
    precision_loss_count = 0
    numeric_evidence_count = 0

    for value in value_list:
        status, numeric = _parse_real_number(value)
        if status == "finite":
            parsed.append(numeric if numeric is not None else np.nan)
            numeric_evidence_count += 1
        else:
            parsed.append(np.nan)
            if status == "missing":
                missing_count += 1
            elif status == "invalid":
                invalid_count += 1
                if isinstance(value, (complex, np.complexfloating)):
                    numeric_evidence_count += 1
            elif status == "non_finite":
                non_finite_count += 1
                numeric_evidence_count += 1
            elif status == "precision_loss":
                precision_loss_count += 1
                numeric_evidence_count += 1

    numeric_values = pd.Series(parsed, index=range(len(value_list)), dtype="float64")
    non_missing_cells = [
        value for value in value_list if not _is_missing(value)
    ]
    non_missing_count = len(non_missing_cells)
    numeric_candidate = pd.api.types.is_numeric_dtype(values.dtype) or (
        numeric_evidence_count > non_missing_count / 2
    )
    unique_count = _safe_unique_count(non_missing_cells)
    if numeric_candidate and numeric_values.notna().any():
        is_constant = numeric_values.nunique() == 1
    else:
        is_constant = non_missing_count > 0 and unique_count == 1

    if non_missing_count == 0:
        detected_type = "all_missing"
    elif numeric_candidate:
        detected_type = "constant" if is_constant else "numerical"
    elif any(not _is_supported_scalar(value) for value in non_missing_cells):
        detected_type = "unsupported"
    elif is_constant:
        detected_type = "constant"
    elif unique_count == non_missing_count:
        detected_type = "non_categorical_string"
    else:
        detected_type = "categorical"

    return ColumnAnalysis(
        numeric_values=numeric_values,
        detected_type=detected_type,
        numeric_candidate=numeric_candidate,
        is_constant=is_constant,
        unique_non_missing_count=unique_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        non_finite_count=non_finite_count,
        precision_loss_count=precision_loss_count,
    )
