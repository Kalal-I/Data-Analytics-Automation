"""Construct compact, agent-friendly reports from completed tuning results."""
import numpy as np


def create_tuning_report(tuning, include_history=False):
    """Summarize tuning without fitting additional models."""
    ordered = list(tuning.get("results", []))
    compact = ordered[:10]
    observations = []
    if compact:
        best = compact[0]
        observations.append({
            "parameter": "best_configuration",
            "observation": "The top configuration achieved mean CV score {!r}.".format(
                best.get("mean_test_score")
            ),
        })
        for parameter in ("n_estimators", "max_depth", "min_samples_split",
                          "min_samples_leaf", "max_features", "criterion"):
            values = {}
            for row in ordered:
                key = str(row["configuration"].get(parameter))
                values.setdefault(key, []).append(row["mean_test_score"])
            usable = {key: np.mean([score for score in scores if score is not None])
                       for key, scores in values.items()
                       if any(score is not None for score in scores)}
            if usable:
                winner = max(usable, key=usable.get)
                observations.append({
                    "parameter": parameter,
                    "observation": "Observed strongest mean CV score for value {!r}.".format(winner),
                })
    report = {
        "best_configuration": tuning.get("best_configuration"),
        "best_score": tuning.get("best_score"),
        "selection_metric": tuning.get("selection_metric"),
        "total_configurations": tuning.get("n_configurations"),
        "cv_folds": tuning.get("cv_folds"),
        "expected_model_fits": tuning.get("expected_model_fits"),
        "summary": "{} configurations evaluated across {} folds.".format(
            tuning.get("n_configurations"), tuning.get("cv_folds")
        ),
        "top_10": compact,
        "parameter_observations": observations,
    }
    if include_history:
        report["full_history"] = tuning.get("full_history", ordered)
    return report
