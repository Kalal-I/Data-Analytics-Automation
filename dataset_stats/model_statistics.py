"""Regression-specific statistical evidence for model and feature selection."""

import json
import math
from itertools import combinations
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression

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
    data: pd.DataFrame,
    target_column: str,
    one_hot_encoded_features: set[str],
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
        if column in one_hot_encoded_features:
            finite_values = numeric_values.dropna()
            binary_values_verified = bool(
                not finite_values.empty
                and finite_values.isin([0.0, 1.0]).all()
            )
            feature_report["encoding"] = {
                "type": "one_hot_binary",
                "binary_values_verified": binary_values_verified,
                "instruction": (
                    "Treat as a one-hot binary variable. Preserve valid "
                    "target associations, mutual information, and pairwise "
                    "correlations; qualify statistics whose assumptions do "
                    "not fit binary data."
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


def _normalized(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale finite values to avoid overflow in diagnostic calculations."""
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    if scale == 0:
        return np.zeros_like(values, dtype="float64"), 0.0
    return values / scale, scale


def _distribution_summary(values: pd.Series, name: str) -> dict[str, Any]:
    """Summarize valid numeric observations and IQR-rule outliers."""
    finite = values.dropna().to_numpy(dtype="float64")
    result: dict[str, Any] = {
        "name": name,
        "valid_observation_count": int(len(finite)),
        "skewness": None,
        "quantiles": {},
        "outliers": {
            "method": "Tukey 1.5-IQR fences",
            "lower_fence": None,
            "upper_fence": None,
            "count": None,
        },
        "variance": None,
    }
    if not len(finite):
        result["limitation"] = "no_finite_numeric_observations"
        return result

    normalized, scale = _normalized(finite)
    quantile_probabilities = {
        "q05": 0.05,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q95": 0.95,
    }
    normalized_quantiles = {
        label: float(np.quantile(normalized, probability))
        for label, probability in quantile_probabilities.items()
    }
    result["quantiles"] = {
        label: _finite_float(value * scale)
        for label, value in normalized_quantiles.items()
    }
    q25 = normalized_quantiles["q25"]
    q75 = normalized_quantiles["q75"]
    iqr = q75 - q25
    lower_normalized = q25 - 1.5 * iqr
    upper_normalized = q75 + 1.5 * iqr
    result["outliers"] = {
        "method": "Tukey 1.5-IQR fences",
        "lower_fence": _finite_float(lower_normalized * scale),
        "upper_fence": _finite_float(upper_normalized * scale),
        "count": int(
            np.count_nonzero(
                (normalized < lower_normalized) | (normalized > upper_normalized)
            )
        ),
    }
    variance_normalized = float(np.var(normalized, ddof=0))
    result["variance"] = _finite_float(variance_normalized * scale * scale)
    if len(normalized) >= 3 and variance_normalized > 0:
        mean = float(np.mean(normalized))
        standard_deviation = math.sqrt(variance_normalized)
        result["skewness"] = _finite_float(
            float(np.mean(((normalized - mean) / standard_deviation) ** 3))
        )
    else:
        result["skewness_limitation"] = (
            "fewer_than_three_observations"
            if len(normalized) < 3
            else "constant_values"
        )
    return result


def _feature_distribution_summaries(
    target_column: str,
    target_values: pd.Series,
    numeric_predictors: dict[Any, pd.Series],
) -> dict[str, Any]:
    return {
        "target": _distribution_summary(target_values, target_column),
        "features": [
            _distribution_summary(values, str(name))
            for name, values in numeric_predictors.items()
        ],
    }


def _mutual_information_summaries(
    numeric_predictors: dict[Any, pd.Series],
    target_values: pd.Series,
) -> dict[str, Any]:
    """Estimate numeric feature-target mutual information deterministically."""
    summaries: list[dict[str, Any]] = []
    for name, values in numeric_predictors.items():
        paired = pd.concat(
            [values.reset_index(drop=True), target_values.reset_index(drop=True)],
            axis=1,
        ).dropna()
        x = paired.iloc[:, 0].to_numpy(dtype="float64")
        y = paired.iloc[:, 1].to_numpy(dtype="float64")
        summary: dict[str, Any] = {
            "feature": str(name),
            "mutual_information": None,
            "valid_paired_observation_count": int(len(paired)),
            "method": "sklearn mutual_info_regression",
            "limitation": None,
        }
        if len(paired) < 3:
            summary["limitation"] = "fewer_than_three_valid_pairs"
        elif np.unique(x).size < 2 or np.unique(y).size < 2:
            summary["limitation"] = "constant_feature_or_target"
        else:
            x_scaled, _ = _normalized(x)
            y_scaled, _ = _normalized(y)
            estimate = mutual_info_regression(
                x_scaled.reshape(-1, 1),
                y_scaled,
                discrete_features=False,
                n_neighbors=min(3, len(paired) - 1),
                random_state=0,
            )[0]
            summary["mutual_information"] = _finite_float(estimate)
            if summary["mutual_information"] is None:
                summary["limitation"] = "mutual_information_is_non_finite"
        summaries.append(summary)
    return {
        "method": (
            "Non-negative k-nearest-neighbor estimate for numerical predictors; "
            "values are not directly comparable across sample sizes."
        ),
        "features": summaries,
    }


def _feature_cardinality_and_missingness(
    data: pd.DataFrame,
    target_column: str,
    target_values: pd.Series,
    numeric_predictors: dict[Any, pd.Series],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Report raw feature cardinality, effective sample size, and missingness."""
    cardinality: list[dict[str, Any]] = []
    for column in data.columns:
        if column == target_column:
            continue
        analysis = analyze_column(data[column].reset_index(drop=True))
        cardinality.append(
            {
                "feature": str(column),
                "detected_type": analysis.detected_type,
                "unique_non_missing_count": analysis.unique_non_missing_count,
                "non_missing_count": int(len(data) - analysis.missing_count),
                "cardinality_ratio": (
                    _finite_float(
                        analysis.unique_non_missing_count
                        / (len(data) - analysis.missing_count)
                    )
                    if len(data) > analysis.missing_count
                    else None
                ),
            }
        )

    numeric_feature_count = len(numeric_predictors)
    complete_case_count = 0
    if numeric_feature_count:
        complete = pd.DataFrame(
            {
                str(name): values.reset_index(drop=True)
                for name, values in numeric_predictors.items()
            }
        )
        complete[target_column] = target_values.reset_index(drop=True)
        complete_case_count = int(complete.dropna().shape[0])
    ratio = (
        _finite_float(complete_case_count / numeric_feature_count)
        if numeric_feature_count
        else None
    )
    cardinality_report = {
        "features": cardinality,
        "effective_sample_to_numeric_feature_ratio": {
            "complete_case_sample_count": complete_case_count,
            "numeric_feature_count": numeric_feature_count,
            "ratio": ratio,
            "limitation": (
                None
                if numeric_feature_count
                else "no_numeric_predictors"
            ),
        },
    }

    missingness = data.isna()
    pattern_counts: dict[tuple[str, ...], int] = {}
    for row in missingness.itertuples(index=False, name=None):
        pattern = tuple(
            str(column) for column, is_missing in zip(data.columns, row) if is_missing
        )
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    pattern_report = {
        "columns": [
            {
                "feature": str(column),
                "missing_count": int(missingness[column].sum()),
                "missing_percentage": _finite_float(
                    100.0 * float(missingness[column].mean())
                ),
            }
            for column in data.columns
        ],
        "row_patterns": [
            {
                "missing_features": list(pattern),
                "row_count": count,
            }
            for pattern, count in sorted(
                pattern_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }
    return cardinality_report, pattern_report


def _feature_scale_summaries(
    numeric_predictors: dict[Any, pd.Series],
    target_column: str,
    target_values: pd.Series,
) -> dict[str, Any]:
    """Summarize location and scale for numeric columns without imputing."""
    summaries: list[dict[str, Any]] = []
    for name, values in [(target_column, target_values), *numeric_predictors.items()]:
        finite = values.dropna().to_numpy(dtype="float64")
        summary: dict[str, Any] = {
            "name": str(name),
            "valid_observation_count": int(len(finite)),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "range": None,
        }
        if len(finite):
            normalized, scale = _normalized(finite)
            minimum = float(np.min(normalized))
            maximum = float(np.max(normalized))
            mean = float(np.mean(normalized))
            standard_deviation = float(np.std(normalized, ddof=0))
            summary.update(
                {
                    "minimum": _finite_float(minimum * scale),
                    "maximum": _finite_float(maximum * scale),
                    "mean": _finite_float(mean * scale),
                    "standard_deviation": _finite_float(standard_deviation * scale),
                    "range": _finite_float((maximum - minimum) * scale),
                }
            )
        else:
            summary["limitation"] = "no_finite_numeric_observations"
        summaries.append(summary)
    return {"columns": summaries}


def _nonlinear_relationship_screening(
    numeric_predictors: dict[Any, pd.Series],
    target_values: pd.Series,
    one_hot_encoded_features: set[str] | None = None,
) -> dict[str, Any]:
    """Screen quadratic and pairwise interaction terms using in-sample R²."""
    one_hot_encoded_features = one_hot_encoded_features or set()
    quadratic_results: list[dict[str, Any]] = []
    valid_columns: list[tuple[Any, pd.Series]] = []
    for name, values in numeric_predictors.items():
        paired = pd.concat(
            [values.reset_index(drop=True), target_values.reset_index(drop=True)],
            axis=1,
        ).dropna()
        x = paired.iloc[:, 0].to_numpy(dtype="float64")
        y = paired.iloc[:, 1].to_numpy(dtype="float64")
        result: dict[str, Any] = {
            "feature": str(name),
            "valid_paired_observation_count": int(len(paired)),
            "linear_r2": None,
            "quadratic_r2": None,
            "quadratic_r2_gain": None,
            "limitation": None,
        }
        if str(name) in one_hot_encoded_features:
            result["encoding"] = "one_hot_binary"
        if len(paired) < 5:
            result["limitation"] = "fewer_than_five_valid_pairs"
        elif np.unique(x).size < 2 or np.unique(y).size < 2:
            result["limitation"] = "insufficient_feature_or_target_variation"
        elif (
            str(name) in one_hot_encoded_features
            and np.isin(np.unique(x), [0.0, 1.0]).all()
        ):
            # For a 0/1 variable, x squared equals x, so the quadratic adds no
            # new information beyond the existing linear term.
            result["limitation"] = "quadratic_term_redundant_for_binary_feature"
        elif np.unique(x).size < 3:
            result["limitation"] = "insufficient_feature_or_target_variation"
        else:
            x_scaled, _ = _normalized(x)
            y_scaled, _ = _normalized(y)
            linear = LinearRegression().fit(x_scaled.reshape(-1, 1), y_scaled)
            quadratic_design = np.column_stack([x_scaled, x_scaled**2])
            quadratic = LinearRegression().fit(quadratic_design, y_scaled)
            linear_r2 = _finite_float(linear.score(x_scaled.reshape(-1, 1), y_scaled))
            quadratic_r2 = _finite_float(quadratic.score(quadratic_design, y_scaled))
            result.update(
                {
                    "linear_r2": linear_r2,
                    "quadratic_r2": quadratic_r2,
                    "quadratic_r2_gain": (
                        _finite_float(quadratic_r2 - linear_r2)
                        if linear_r2 is not None and quadratic_r2 is not None
                        else None
                    ),
                }
            )
        quadratic_results.append(result)
        valid_columns.append((name, values))

    interactions: list[dict[str, Any]] = []
    for (left_name, left_values), (right_name, right_values) in combinations(
        valid_columns, 2
    ):
        paired = pd.concat(
            [
                left_values.reset_index(drop=True),
                right_values.reset_index(drop=True),
                target_values.reset_index(drop=True),
            ],
            axis=1,
        ).dropna()
        left = paired.iloc[:, 0].to_numpy(dtype="float64")
        right = paired.iloc[:, 1].to_numpy(dtype="float64")
        y = paired.iloc[:, 2].to_numpy(dtype="float64")
        result = {
            "feature_1": str(left_name),
            "feature_2": str(right_name),
            "valid_observation_count": int(len(paired)),
            "additive_r2": None,
            "interaction_r2": None,
            "interaction_r2_gain": None,
            "limitation": None,
        }
        if len(paired) < 5:
            result["limitation"] = "fewer_than_five_valid_observations"
        elif (
            np.unique(left).size < 2
            or np.unique(right).size < 2
            or np.unique(y).size < 2
        ):
            result["limitation"] = "constant_feature_or_target"
        else:
            left_scaled, _ = _normalized(left)
            right_scaled, _ = _normalized(right)
            y_scaled, _ = _normalized(y)
            additive_design = np.column_stack([left_scaled, right_scaled])
            interaction_design = np.column_stack(
                [left_scaled, right_scaled, left_scaled * right_scaled]
            )
            additive = LinearRegression().fit(additive_design, y_scaled)
            interaction = LinearRegression().fit(interaction_design, y_scaled)
            additive_r2 = _finite_float(additive.score(additive_design, y_scaled))
            interaction_r2 = _finite_float(
                interaction.score(interaction_design, y_scaled)
            )
            result.update(
                {
                    "additive_r2": additive_r2,
                    "interaction_r2": interaction_r2,
                    "interaction_r2_gain": (
                        _finite_float(interaction_r2 - additive_r2)
                        if additive_r2 is not None and interaction_r2 is not None
                        else None
                    ),
                }
            )
        interactions.append(result)

    return {
        "method": (
            "In-sample R² gain from adding a univariate quadratic term or a "
            "pairwise product term to the corresponding linear/additive model. "
            "This is a screening heuristic, not a significance test."
        ),
        "quadratic_feature_screens": quadratic_results,
        "pairwise_interaction_screens": interactions,
    }


def _linear_baseline_residual_diagnostics(
    numeric_predictors: dict[Any, pd.Series],
    target_values: pd.Series,
) -> dict[str, Any]:
    """Fit a complete-case numeric baseline and summarize its residuals."""
    if not numeric_predictors:
        return {
            "available": False,
            "limitation": "no_numeric_predictors",
            "complete_case_sample_count": 0,
            "numeric_feature_count": 0,
        }

    x_frame = pd.DataFrame(
        {
            str(name): values.reset_index(drop=True)
            for name, values in numeric_predictors.items()
        }
    )
    y_series = target_values.reset_index(drop=True)
    complete = pd.concat([x_frame, y_series.rename("__target__")], axis=1).dropna()
    feature_count = len(numeric_predictors)
    sample_count = len(complete)
    base_report = {
        "available": False,
        "complete_case_sample_count": sample_count,
        "numeric_feature_count": feature_count,
    }
    if sample_count < 3:
        return {
            **base_report,
            "limitation": "fewer_than_three_complete_cases",
        }

    x = complete.iloc[:, :-1].to_numpy(dtype="float64")
    y = complete.iloc[:, -1].to_numpy(dtype="float64")
    if np.unique(y).size < 2:
        return {**base_report, "limitation": "constant_target"}

    x_scaled = np.column_stack([_normalized(x[:, index])[0] for index in range(x.shape[1])])
    y_scaled, y_scale = _normalized(y)
    model = LinearRegression().fit(x_scaled, y_scaled)
    predictions = model.predict(x_scaled)
    residuals = y_scaled - predictions
    residual_variance = float(np.var(residuals, ddof=0))
    fitted_variance = float(np.var(predictions, ddof=0))
    residual_skewness = None
    standardized_outlier_count = None
    if residual_variance > 0:
        residual_std = math.sqrt(residual_variance)
        standardized_residuals = residuals / residual_std
        standardized_outlier_count = int(
            np.count_nonzero(np.abs(standardized_residuals) > 3)
        )
        residual_skewness = _finite_float(
            float(
                np.mean(
                    ((residuals - float(np.mean(residuals))) / residual_std) ** 3
                )
            )
        )
    absolute_residual_correlation, paired_count, limitation = _correlation(
        pd.Series(np.abs(residuals)),
        pd.Series(predictions),
        "pearson",
    )
    return {
        **base_report,
        "available": True,
        "method": "LinearRegression fit on all complete numeric rows (in-sample)",
        "in_sample_r2": _finite_float(model.score(x_scaled, y_scaled)),
        "in_sample_mae": _finite_float(float(np.mean(np.abs(residuals))) * y_scale),
        "in_sample_rmse": _finite_float(math.sqrt(residual_variance) * y_scale),
        "residual_mean": _finite_float(float(np.mean(residuals)) * y_scale),
        "residual_standard_deviation": _finite_float(
            math.sqrt(residual_variance) * y_scale
        ),
        "residual_skewness": residual_skewness,
        "standardized_residual_outlier_count": standardized_outlier_count,
        "absolute_residual_fitted_correlation": absolute_residual_correlation,
        "absolute_residual_fitted_correlation_pair_count": paired_count,
        "absolute_residual_fitted_correlation_limitation": limitation,
        "residual_degrees_of_freedom": max(sample_count - feature_count - 1, 0),
        "residual_degrees_of_freedom_limitation": (
            None
            if sample_count > feature_count + 1
            else "no_positive_residual_degrees_of_freedom"
        ),
        "fitted_value_variance": _finite_float(fitted_variance * y_scale * y_scale),
    }


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
    one_hot_encoded_features: (
        list[str] | tuple[str, ...] | set[str] | frozenset[str] | None
    ) = None,
) -> str:
    """Return strict JSON with regression statistics and diagnostic screens.

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
    Added distribution, mutual-information, cardinality, scale, nonlinear,
    interaction, and residual sections supplement rather than replace the
    original correlation evidence. Nonlinear screens and residual diagnostics
    are in-sample heuristics, not significance tests or estimates of
    out-of-sample performance.

    ``one_hot_encoded_features`` lists predictor column names created by
    one-hot encoding. They remain included in valid association, mutual
    information, and pairwise correlation calculations. Verified 0/1 features
    are explicitly excluded from quadratic-term comparisons because squaring
    them duplicates their linear values.
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

    if one_hot_encoded_features is None:
        encoded_features: list[str] = []
    elif isinstance(one_hot_encoded_features, (set, frozenset)):
        encoded_features = sorted(one_hot_encoded_features)
    elif isinstance(one_hot_encoded_features, (list, tuple)):
        encoded_features = list(one_hot_encoded_features)
    else:
        raise TypeError(
            "one_hot_encoded_features must be a list, tuple, set, or None."
        )
    if not all(isinstance(name, str) for name in encoded_features):
        raise TypeError("Every one-hot encoded feature name must be a string.")
    if len(encoded_features) != len(set(encoded_features)):
        raise ValueError("one_hot_encoded_features must not contain duplicates.")
    missing_encoded_features = [
        name for name in encoded_features if name not in data.columns
    ]
    if missing_encoded_features:
        raise ValueError(
            "One-hot encoded feature columns not found in data: {!r}.".format(
                missing_encoded_features
            )
        )
    if target_column in encoded_features:
        raise ValueError("The target column cannot be a one-hot encoded feature.")
    encoded_feature_set = set(encoded_features)

    target_report, target_values = _target_report(data[target_column])
    features, numeric_predictors, feature_reports = _collect_features(
        data, target_column, encoded_feature_set
    )
    relationships = _feature_target_relationships(
        numeric_predictors, target_values, feature_reports
    )
    feature_feature_relationships, highly_correlated_pairs = (
        _feature_pair_relationships(numeric_predictors, threshold)
    )
    distributions = _feature_distribution_summaries(
        target_column, target_values, numeric_predictors
    )
    mutual_information = _mutual_information_summaries(
        numeric_predictors, target_values
    )
    cardinality, missingness = _feature_cardinality_and_missingness(
        data, target_column, target_values, numeric_predictors
    )
    scale_summaries = _feature_scale_summaries(
        numeric_predictors, target_column, target_values
    )
    nonlinear_screening = _nonlinear_relationship_screening(
        numeric_predictors, target_values, encoded_feature_set
    )
    residual_diagnostics = _linear_baseline_residual_diagnostics(
        numeric_predictors, target_values
    )
    warnings = _build_warnings(target_report, relationships)
    binary_validation = {
        name: bool(feature_reports[name]["encoding"]["binary_values_verified"])
        for name in encoded_features
    }
    for name, is_binary in binary_validation.items():
        if not is_binary:
            warnings.append(
                "Feature {!r} was declared one-hot encoded, but its available "
                "finite numeric values are not verified as binary 0/1; "
                "interpret binary-specific guidance cautiously.".format(name)
            )

    result = {
        "schema_version": 1,
        "encoding_metadata": {
            "one_hot_encoded_features": encoded_features,
            "binary_values_verified": binary_validation,
            "instruction": (
                "Treat these features as one-hot-encoded binary variables, "
                "not continuous numerical variables. Use their statistics "
                "only where statistically appropriate. Do not discard valid "
                "binary-feature statistics such as target associations, "
                "mutual information, or pairwise correlations. Ignore or "
                "qualify statistics whose assumptions are inappropriate for "
                "binary variables. Do not assume that every statistic "
                "calculated for these features is invalid."
            ),
        },
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
        "distribution_summaries": distributions,
        "feature_target_mutual_information": mutual_information,
        "feature_cardinality_and_effective_sample_ratio": cardinality,
        "feature_scale_summaries": scale_summaries,
        "missingness_patterns": missingness,
        "nonlinear_relationship_screening": nonlinear_screening,
        "linear_baseline_residual_diagnostics": residual_diagnostics,
        "diagnostic_interpretation": (
            "Mutual information, nonlinear/interaction screens, and residual "
            "diagnostics use numeric values only. The nonlinear, interaction, "
            "and residual diagnostics are in-sample heuristics and do not "
            "establish statistical significance or out-of-sample performance."
        ),
        "warnings": warnings,
        "interpretation": (
            "Correlations describe association, not causation, and do not "
            "guarantee predictive usefulness. This evidence neither identifies "
            "every important feature nor determines the best regression model."
        ),
    }
    return json.dumps(result, indent=2, allow_nan=False)
