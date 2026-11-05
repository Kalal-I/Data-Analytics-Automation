"""Compact reports for Random Forest hyperparameter tuning results."""
from collections import defaultdict
from math import isfinite

from .optimization import TUNING_PARAMETERS


def _trial_score(result):
    return result["mean_std"]["r2"]["mean"]


def _trial_sort_key(result):
    metrics = result["mean_std"]
    mean_mae = metrics.get("mae", {}).get("mean")
    return (
        -metrics["r2"]["mean"],
        mean_mae if mean_mae is not None else float("inf"),
        tuple(str(result["configuration"][name]) for name in TUNING_PARAMETERS),
    )


def _ranked_trials(tuning_results):
    results = tuning_results.get("results")
    if not isinstance(results, list):
        raise ValueError("tuning_results must contain a results list")

    successful = []
    failed = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(
            result.get("configuration"), dict
        ):
            raise ValueError("each tuning result must include a configuration")
        metrics = result.get("mean_std")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("r2"), dict):
            raise ValueError("each tuning result must include mean_std R2 metrics")
        mean_r2 = metrics["r2"].get("mean")
        if mean_r2 is None:
            failed.append(result)
            continue
        if not isfinite(mean_r2):
            raise ValueError("CV mean R2 must be finite or None")
        successful.append(result)

    successful.sort(key=_trial_sort_key)
    ranked = [
        {**result, "rank": rank}
        for rank, result in enumerate(successful, start=1)
    ]
    ranked.extend({**result, "rank": None} for result in failed)
    return ranked, successful, failed


def _compact_trial(result, rank):
    r2 = result["mean_std"]["r2"]
    return {
        "rank": rank,
        "configuration": dict(result["configuration"]),
        "cv_mean_r2": r2["mean"],
        "cv_std_r2": r2["std"],
        "fold_r2_scores": [
            fold["r2"] for fold in result.get("fold_metrics", [])
        ],
    }


def _parameter_observations(successful_trials):
    observations = []
    for parameter in TUNING_PARAMETERS:
        grouped = defaultdict(list)
        for result in successful_trials:
            configuration = result["configuration"]
            if parameter in configuration:
                grouped[configuration[parameter]].append(_trial_score(result))

        supported_values = [
            {
                "value": value,
                "trial_count": len(scores),
                "mean_cv_r2": round(sum(scores) / len(scores), 12),
            }
            for value, scores in grouped.items()
            if len(scores) >= 2
        ]
        if len(supported_values) < 2:
            continue
        supported_values.sort(
            key=lambda item: (-item["mean_cv_r2"], str(item["value"]))
        )
        observations.append(
            {
                "parameter": parameter,
                "observed_cv_r2_by_value": supported_values,
            }
        )
    return observations


def build_tuning_report(tuning_results, include_history=False):
    """Prepare a compact agent report, optionally including all ranked trials."""
    ranked_trials, successful_trials, failed_trials = _ranked_trials(
        tuning_results
    )
    top_trials = [
        _compact_trial(result, rank)
        for rank, result in enumerate(successful_trials[:10], start=1)
    ]
    best = top_trials[0] if top_trials else None
    successful_scores = [_trial_score(result) for result in successful_trials]
    report = {
        "selection_metric": tuning_results.get(
            "selection_metric", "mean_cv_r2"
        ),
        "best": (
            {
                "configuration": best["configuration"],
                "cv_mean_r2": best["cv_mean_r2"],
                "cv_std_r2": best["cv_std_r2"],
            }
            if best
            else None
        ),
        "total_configurations_evaluated": len(ranked_trials),
        "summary": {
            "successful_configurations": len(successful_trials),
            "failed_configurations": len(failed_trials),
            "mean_configuration_cv_r2": (
                round(sum(successful_scores) / len(successful_scores), 12)
                if successful_scores
                else None
            ),
        },
        "top_configurations": top_trials,
        "parameter_observations": _parameter_observations(successful_trials),
        "parameter_observation_note": (
            "Descriptive summaries only; trials may differ in multiple "
            "parameters, so these do not establish causal effects."
        ),
    }
    if include_history:
        report["history"] = [
            {
                "rank": result["rank"],
                "configuration": dict(result["configuration"]),
                "mean_std": result["mean_std"],
                "fold_metrics": result.get("fold_metrics", []),
                "failures": result.get("failures", []),
            }
            for result in ranked_trials
        ]
    return report
