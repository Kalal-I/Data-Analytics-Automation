"""Regression-specific statistical evidence for model and feature selection."""

import json
import math
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from dataset_stats.column_analysis import analyze_column

DEFAULT_CORRELATION_THRESHOLD = 0.8
_PEARSON_NEAR_CONSTANT_ULPS = 1 + 4 * np.finfo(float).eps


def _finite_float(value: Any) -> float | None:
    """Convert a scalar to a finite Python float, or ``None``."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _near_constant(values: pd.Series) -> bool:
    """Flag spreads no greater than one float64 ULP at the values' magnitude.

    This conservative representational-resolution check avoids rejecting
    ordinary small variations just because values have a large common offset.
    It cannot diagnose measurement precision or domain-specific meaningfulness.
    """
    array = values.to_numpy(dtype="float64")
    scale = float(np.max(np.abs(array)))
    if scale == 0:
        return True
    with np.errstate(over="ignore", invalid="ignore"):
        spread = float(np.max(array) - np.min(array))
    if math.isfinite(spread):
        return spread <= _PEARSON_NEAR_CONSTANT_ULPS * math.ulp(scale)
    relative_spread = float(np.max(array / scale) - np.min(array / scale))
    return (
        relative_spread
        <= _PEARSON_NEAR_CONSTANT_ULPS * (math.ulp(scale) / scale)
    )


def _correlation(
    left: pd.Series, right: pd.Series, method: str
) -> tuple[float | None, int, str | None]:
    """Return a stable correlation, paired count, and method-specific limitation."""
    paired = pd.concat(
        [left.reset_index(drop=True), right.reset_index(drop=True)], axis=1
    ).dropna()
    count = len(paired)
    if count < 2:
        return None, count, "fewer_than_two_valid_pairs"
    left_values = paired.iloc[:, 0]
    right_values = paired.iloc[:, 1]
    if left_values.nunique() < 2 or right_values.nunique() < 2:
        return None, count, "constant_feature_or_target_in_paired_data"

    if method == "pearson":
        if _near_constant(left_values) or _near_constant(right_values):
            return None, count, "near_constant_values_pearson_unreliable"
        left_values = pd.Series(_scaled_differences(left_values))
        right_values = pd.Series(_scaled_differences(right_values))
    elif method == "spearman":
        left_values = left_values.rank(method="average")
        right_values = right_values.rank(method="average")
    else:
        raise ValueError("Unsupported correlation method: {!r}.".format(method))

    value = _finite_float(left_values.corr(right_values, method="pearson"))
    if value is None:
        return None, count, "correlation_is_undefined"
    return value, count, None


def _scaled_differences(values: pd.Series) -> np.ndarray:
    """Center and scale values without overflowing or losing small offsets."""
    array = values.to_numpy(dtype="float64")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = array - array[0]
    if np.isfinite(differences).all():
        scale = float(np.max(np.abs(differences)))
        return differences / scale

    scale = float(np.max(np.abs(array)))
    normalized = array / scale
    differences = normalized - normalized[0]
    return differences / float(np.max(np.abs(differences)))


def _target_report(values: pd.Series) -> tuple[dict[str, Any], pd.Series]:
    """Report target numeric validity separately from full regression suitability."""
    analysis = analyze_column(values)
    numeric = analysis.numeric_values
    invalid_count = analysis.invalid_count
    non_finite_count = analysis.non_finite_count
    missing_count = analysis.missing_count
    precision_loss_count = analysis.precision_loss_count
    valid_count = int(numeric.notna().sum())
    unique_count = int(numeric.nunique())
    is_constant = valid_count > 0 and unique_count == 1
    numeric_data_valid = valid_count >= 2 and unique_count >= 2

    if valid_count == 0:
        reason = (
            "No target values can be represented safely as finite float64 numbers."
            if precision_loss_count
            else "No finite real numeric target values are available."
        )
    elif valid_count < 2:
        reason = "At least two finite real numeric target values are required."
    elif is_constant:
        reason = "A constant target cannot support regression."
    else:
        reason = None

    return (
        {
            "name": str(values.name),
            "valid_numeric_count": valid_count,
            "missing_count": missing_count,
            "invalid_non_numeric_count": invalid_count,
            "non_finite_count": non_finite_count,
            "precision_loss_count": precision_loss_count,
            "unique_value_count": unique_count,
            "is_constant": bool(is_constant),
            "numeric_data_valid": bool(numeric_data_valid),
            "suitable_for_regression": False if not numeric_data_valid else None,
            "regression_suitability_assessment": (
                "unsuitable"
                if not numeric_data_valid
                else "undetermined_from_target_alone"
            ),
            "unsuitable_reason": reason,
        },
        numeric,
    )


def _initial_feature_limitation(
    is_numeric_candidate: bool,
    numeric_values: pd.Series,
    precision_loss_count: int,
) -> str:
    """Describe why a feature cannot yet be used for correlation."""
    if not is_numeric_candidate:
        return "not_a_numeric_predictor"
    if numeric_values.notna().any():
        return "fewer_than_two_valid_pairs"
    if precision_loss_count:
        return "no_precision_safe_numeric_observations"
    return "no_finite_real_numeric_observations"


def _collect_features(
    data: pd.DataFrame, target_column: str
) -> tuple[
    list[dict[str, Any]],
    dict[Any, pd.Series],
    dict[Any, dict[str, Any]],
]:
    """Classify predictor columns and retain numeric data for pair analysis."""
    features: list[dict[str, Any]] = []
    numeric_predictors: dict[Any, pd.Series] = {}
    feature_reports: dict[Any, dict[str, Any]] = {}

    for column in data.columns:
        if column == target_column:
            continue

        # Reset labels only on this local copy so correlation pairs by row position.
        source_values = data[column].reset_index(drop=True)
        analysis = analyze_column(source_values)
        numeric_values = analysis.numeric_values
        detected_type = analysis.detected_type
        is_numeric_candidate = (
            analysis.numeric_candidate and detected_type != "all_missing"
        )

        feature_report: dict[str, Any] = {
            "name": str(column),
            "detected_type": detected_type,
            "valid_numeric_count": int(numeric_values.notna().sum()),
            "invalid_non_numeric_count": analysis.invalid_count,
            "non_finite_count": analysis.non_finite_count,
            "precision_loss_count": analysis.precision_loss_count,
            "usable_for_correlation": False,
            "correlation_limitation": _initial_feature_limitation(
                is_numeric_candidate,
                numeric_values,
                analysis.precision_loss_count,
            ),
        }
        features.append(feature_report)
        feature_reports[column] = feature_report
        if is_numeric_candidate:
            numeric_predictors[column] = numeric_values

    return features, numeric_predictors, feature_reports


def _feature_target_relationships(
    numeric_predictors: dict[Any, pd.Series],
    target_values: pd.Series,
    feature_reports: dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate Pearson and Spearman results and update feature usability."""
    relationships: list[dict[str, Any]] = []
    for column, predictor_values in numeric_predictors.items():
        pearson, paired_count, pearson_limitation = _correlation(
            predictor_values, target_values, "pearson"
        )
        spearman, _, spearman_limitation = _correlation(
            predictor_values, target_values, "spearman"
        )

        # If conversion lost every value, report that instead of a generic pair limit.
        if (
            not predictor_values.notna().any()
            and feature_reports[column]["precision_loss_count"]
        ):
            pearson_limitation = "no_precision_safe_numeric_observations"
            spearman_limitation = "no_precision_safe_numeric_observations"

        legacy_limitation = (
            pearson_limitation
            if pearson_limitation == spearman_limitation
            else None
        )
        relationships.append(
            {
                "feature": str(column),
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "valid_paired_observation_count": paired_count,
                "pearson_limitation": pearson_limitation,
                "spearman_limitation": spearman_limitation,
                "limitation": legacy_limitation,
            }
        )

        is_usable = pearson is not None or spearman is not None
        feature_reports[column]["usable_for_correlation"] = is_usable
        if is_usable:
            feature_reports[column]["correlation_limitation"] = None
        else:
            feature_reports[column]["correlation_limitation"] = (
                legacy_limitation or "see_method_specific_limitations"
            )

    return relationships


def _feature_pair_relationships(
    numeric_predictors: dict[Any, pd.Series],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Report every numeric feature pair and the subset meeting the threshold."""
    relationships: list[dict[str, Any]] = []
    highly_correlated_pairs: list[dict[str, Any]] = []

    for (left_name, left_values), (right_name, right_values) in combinations(
        numeric_predictors.items(), 2
    ):
        correlation, paired_count, limitation = _correlation(
            left_values, right_values, "pearson"
        )
        pair = {
            "feature_1": str(left_name),
            "feature_2": str(right_name),
            "pearson_correlation": correlation,
            "valid_paired_observation_count": paired_count,
            "pearson_limitation": limitation,
        }
        relationships.append(pair)

        # Undefined coefficients are not eligible for threshold comparisons.
        if correlation is not None and abs(correlation) >= threshold:
            highly_correlated_pairs.append(
                {
                    "feature_1": str(left_name),
                    "feature_2": str(right_name),
                    "pearson_correlation": correlation,
                    "valid_paired_observation_count": paired_count,
                    "threshold": threshold,
                }
            )

    return relationships, highly_correlated_pairs


def _build_warnings(
    target_report: dict[str, Any],
    relationships: list[dict[str, Any]],
) -> list[str]:
    """Create regression-specific caveats from target and correlation results."""
    warnings: list[str] = []
    if not target_report["numeric_data_valid"]:
        warnings.append(
            "The target does not contain enough finite real numeric variation: "
            + str(target_report["unsuitable_reason"])
        )
    else:
        warnings.append(
            "Target numeric validity is established, but regression suitability "
            "cannot be determined from target statistics alone."
        )

    if target_report["missing_count"]:
        warnings.append(
            "Missing target values are excluded from correlations; they are reported, not imputed."
        )
    if target_report["invalid_non_numeric_count"] or target_report["non_finite_count"]:
        warnings.append(
            "Invalid or non-finite target values are excluded from correlations and counted in the target report."
        )
    if target_report["precision_loss_count"]:
        warnings.append(
            "Target values that lose numeric precision in float64 are excluded from correlations."
        )

    has_feature_target_correlation = any(
        item["pearson_correlation"] is not None
        or item["spearman_correlation"] is not None
        for item in relationships
    )
    if not has_feature_target_correlation:
        warnings.append("No feature-target correlations could be computed.")
    return warnings


def generate_model_statistics(
    data: pd.DataFrame,
    target_column: str,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> str:
    """Return strict JSON with regression-specific correlations and limitations.

    Native numeric columns are numerical. For object/string columns, more than
    half of non-missing cells must parse as numeric (finite or non-finite) for
    the column to be treated as numerical; unsupported cells count as invalid.
    Complex scalar cells count as numeric-type evidence, but nonzero imaginary
    parts are invalid real observations. Finite real values are converted to
    float64 for calculations. Integer and extended-float precision losses are
    counted separately. Lists, mappings, arrays, and unsupported objects are
    invalid; columns containing them but not numerically detected are
    classified as ``unsupported``. Missing, invalid, non-finite, and
    precision-loss cells retain row positions and are excluded.

    Pearson near-constant detection is limited to spreads within one float64
    ULP at the input magnitude. This is a representational safeguard, not a
    judgement about measurement or domain-specific meaningfulness.

    Target numeric validity is reported separately from regression suitability:
    target-only statistics cannot establish that regression is appropriate.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string.")
    if data.columns.has_duplicates:
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(
            "DataFrame column names must be unique; duplicates found: {!r}.".format(
                duplicates
            )
        )
    if target_column not in data.columns:
        raise ValueError("Target column {!r} was not found.".format(target_column))
    if data.empty:
        raise ValueError("data must contain at least one row.")
    if isinstance(correlation_threshold, bool) or not isinstance(
        correlation_threshold, Real
    ):
        raise TypeError("correlation_threshold must be a real number between 0 and 1.")
    try:
        threshold = float(correlation_threshold)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "correlation_threshold must be finite and between 0 and 1."
        ) from error
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("correlation_threshold must be finite and between 0 and 1.")

    target_report, target_values = _target_report(data[target_column])
    features, numeric_predictors, feature_reports = _collect_features(
        data, target_column
    )
    relationships = _feature_target_relationships(
        numeric_predictors, target_values, feature_reports
    )
    feature_feature_relationships, highly_correlated_pairs = (
        _feature_pair_relationships(numeric_predictors, threshold)
    )
    warnings = _build_warnings(target_report, relationships)

    result = {
        "schema_version": 1,
        "target": target_report,
        "features": features,
        "feature_target_relationships": relationships,
        "feature_feature_relationships": feature_feature_relationships,
        "multicollinearity": {
            "method": "Pearson correlation with pairwise complete finite observations",
            "absolute_correlation_threshold": threshold,
            "threshold_is_heuristic": True,
            "highly_correlated_pairs": highly_correlated_pairs,
        },
        "warnings": warnings,
        "interpretation": (
            "Correlations describe association, not causation, and do not "
            "guarantee predictive usefulness. This evidence neither identifies "
            "every important feature nor determines the best regression model."
        ),
    }
    return json.dumps(result, indent=2, allow_nan=False)
