"""Select the preferred feature set from evaluated linear-regression reports."""

import math
from typing import Any


def format_best_model_configuration(report: dict[str, Any]) -> str:
    """Format the selected model and its recorded hyperparameters for output."""
    training = report["training"]
    model_names = {
        "lasso": "Lasso",
        "ridge": "Ridge",
        "elastic_net": "Elastic Net",
    }
    model_type = training["model_type"]
    if model_type not in model_names:
        raise ValueError("Report contains an unsupported model_type.")
    if training["model"] != model_names[model_type]:
        raise ValueError("Report model name does not match its model_type.")
    hyperparameters = training["hyperparameters"]
    lines = [
        "Best Model: {}".format(training["model"]),
        "Alpha: {}".format(hyperparameters["alpha"]),
    ]
    if model_type == "elastic_net":
        lines.append("L1 Ratio: {}".format(hyperparameters["l1_ratio"]))
    return "\n".join(lines)


def format_model_feature_weights(report: dict[str, Any]) -> str:
    """Format the selected fitted model's coefficients with feature names."""
    coefficients = report["training"]["coefficients"]
    if not isinstance(coefficients, dict):
        raise ValueError("Report feature coefficients must be an object.")
    return "\n".join(
        "{}: {}".format(feature, weight)
        for feature, weight in coefficients.items()
    )


def select_best_feature_report(
    reports: list[dict[str, Any]],
    r2_tolerance: float = 0.005,
) -> tuple[dict[str, Any], list[str]]:
    """Choose among fitted reports within the best mean CV R² tolerance.

    Adjusted R² is used only when it is defined for every R²-eligible report,
    so unavailable values are never treated as numeric rankings.
    """
    evaluated_reports = []
    for report in reports:
        features = report["features"]
        if not features:
            continue
        cross_validation = report["training"]["cross_validation"]
        r2_mean = cross_validation["r2"]["mean"]
        if (
            isinstance(r2_mean, bool)
            or not isinstance(r2_mean, (int, float))
            or not math.isfinite(r2_mean)
        ):
            continue
        evaluated_reports.append(report)

    if not evaluated_reports:
        raise ValueError("No fitted feature-set reports have a valid mean CV R².")

    best_r2 = max(
        report["training"]["cross_validation"]["r2"]["mean"]
        for report in evaluated_reports
    )
    candidates = [
        report
        for report in evaluated_reports
        if best_r2 - report["training"]["cross_validation"]["r2"]["mean"]
        <= r2_tolerance
    ]

    cross_validations = [
        report["training"]["cross_validation"] for report in candidates
    ]
    adjusted_r2_available = all(
        isinstance(metrics.get("adjusted_r2", {}).get("mean"), (int, float))
        and not isinstance(metrics["adjusted_r2"]["mean"], bool)
        and math.isfinite(metrics["adjusted_r2"]["mean"])
        for metrics in cross_validations
    )
    ranking_metrics = ["r2_tolerance"]
    metric_key = []
    if adjusted_r2_available:
        ranking_metrics.append("adjusted_r2")
        metric_key.append(
            lambda report: -report["training"]["cross_validation"][
                "adjusted_r2"
            ]["mean"]
        )

    for metric in ("rmse", "mae"):
        if all(
            isinstance(metrics.get(metric, {}).get("mean"), (int, float))
            and not isinstance(metrics[metric]["mean"], bool)
            and math.isfinite(metrics[metric]["mean"])
            for metrics in cross_validations
        ):
            metric_key.append(
                lambda report, metric=metric: report["training"][
                    "cross_validation"
                ][metric]["mean"]
            )
            ranking_metrics.append(metric)

    candidates.sort(
        key=lambda report: tuple(
            value(report) for value in metric_key
        )
        + (len(report["features"]),)
    )
    ranking_metrics.append("feature_count")
    return candidates[0], ranking_metrics
