"""Coefficient-ranked greedy feature selection using fixed CV evaluation."""

from collections.abc import Callable, Mapping
import math
from numbers import Real
from typing import Any

import numpy as np

R2_IMPROVEMENT_THRESHOLD = 0.005
CV_METRICS = ("r2", "adjusted_r2", "rmse", "mae")


def rank_features_by_coefficient(
    coefficients: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return finite coefficients in descending absolute-magnitude order.

    Ties retain mapping order. Zero coefficients remain eligible at the end
    of the ranking, while invalid values are reported separately and excluded.
    """
    if not isinstance(coefficients, Mapping):
        raise TypeError("coefficients must be a mapping of feature names to values.")

    ranked = []
    invalid = []
    for feature, coefficient in coefficients.items():
        if not isinstance(feature, str):
            raise TypeError("Coefficient feature names must be strings.")
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
            invalid.append(
                {
                    "feature": feature,
                    "coefficient": repr(coefficient),
                    "reason": "coefficient is not a real number",
                }
            )
            continue
        try:
            numeric_coefficient = float(coefficient)
        except (TypeError, ValueError, OverflowError):
            invalid.append(
                {
                    "feature": feature,
                    "coefficient": repr(coefficient),
                    "reason": "coefficient cannot be represented as a finite float",
                }
            )
            continue
        if not math.isfinite(numeric_coefficient):
            invalid.append(
                {
                    "feature": feature,
                    "coefficient": repr(coefficient),
                    "reason": "coefficient is not finite",
                }
            )
            continue
        ranked.append(
            {
                "feature": feature,
                "coefficient": numeric_coefficient,
                "absolute_coefficient": abs(numeric_coefficient),
            }
        )

    ranked.sort(key=lambda item: -item["absolute_coefficient"])
    return ranked, invalid


def _summarize_cv_metrics(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Validate and keep the requested aggregate cross-validation metrics."""
    if not isinstance(metrics, dict):
        raise ValueError("Cross-validation results must be a metrics dictionary.")

    summary: dict[str, dict[str, float | None]] = {}
    for metric in CV_METRICS:
        values = metrics.get(metric)
        if not isinstance(values, dict):
            raise ValueError(
                "Cross-validation results are missing {!r}.".format(metric)
            )
        mean = values.get("mean")
        std = values.get("std")
        if metric == "adjusted_r2" and mean is None and std is None:
            summary[metric] = {"mean": None, "std": None}
            continue
        for value, value_name in ((mean, "mean"), (std, "std")):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not _is_finite_real(value)
            ):
                raise ValueError(
                    "Cross-validation {} for {!r} must be finite.".format(
                        value_name, metric
                    )
                )
        summary[metric] = {
            "mean": float(mean),
            "std": float(std),
        }
    return summary


def _is_finite_real(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def greedy_coefficient_feature_selection(
    coefficients: Mapping[str, Any],
    evaluate_features: Callable[[list[str]], dict[str, dict[str, Any]]],
    r2_improvement_threshold: float = R2_IMPROVEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Greedily accept ranked features that improve mean CV R² enough."""
    if not callable(evaluate_features):
        raise TypeError("evaluate_features must be callable.")
    if (
        isinstance(r2_improvement_threshold, bool)
        or not isinstance(r2_improvement_threshold, Real)
        or not _is_finite_real(r2_improvement_threshold)
        or r2_improvement_threshold < 0
    ):
        raise ValueError("r2_improvement_threshold must be finite and non-negative.")

    ranked_features, invalid_coefficients = rank_features_by_coefficient(
        coefficients
    )
    accepted_features: list[str] = []
    accepted_metrics = None
    evaluations = []

    for item in ranked_features:
        feature = item["feature"]
        candidate_features = accepted_features + [feature]
        evaluation = {
            "candidate_features": candidate_features,
            "feature_count": len(candidate_features),
            "cross_validation": {
                metric: {"mean": None, "std": None}
                for metric in CV_METRICS
            },
            "r2_improvement": None,
            "decision": "rejected",
            "reason": "",
        }
        try:
            candidate_metrics = _summarize_cv_metrics(
                evaluate_features(candidate_features)
            )
        except (
            ValueError,
            TypeError,
            OverflowError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            evaluation["reason"] = "evaluation failed: {}".format(error)
            evaluations.append(evaluation)
            continue

        evaluation["cross_validation"] = candidate_metrics
        if accepted_metrics is None:
            accepted_features = candidate_features
            accepted_metrics = candidate_metrics
            evaluation["decision"] = "accepted"
            evaluation["reason"] = (
                "initial feature accepted because evaluation succeeded"
            )
        else:
            improvement = (
                candidate_metrics["r2"]["mean"]
                - accepted_metrics["r2"]["mean"]
            )
            evaluation["r2_improvement"] = improvement
            if improvement >= r2_improvement_threshold:
                accepted_features = candidate_features
                accepted_metrics = candidate_metrics
                evaluation["decision"] = "accepted"
                evaluation["reason"] = (
                    "R² improvement meets or exceeds the threshold"
                )
            else:
                evaluation["reason"] = (
                    "R² improvement is below the threshold"
                )
        evaluations.append(evaluation)

    return {
        "r2_improvement_threshold": float(r2_improvement_threshold),
        "ranked_features": ranked_features,
        "invalid_coefficients": invalid_coefficients,
        "evaluations": evaluations,
        "accepted_features": accepted_features,
        "accepted_cross_validation": accepted_metrics,
    }
