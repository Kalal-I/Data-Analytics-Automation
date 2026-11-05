from random_forest_regression.optimization import TUNING_PARAMETERS
from random_forest_regression.tuning_report import build_tuning_report


def _trial(index, score):
    configuration = {
        "n_estimators": 50 + 50 * (index % 4),
        "max_depth": [None, 5][index % 2],
        "min_samples_split": [2, 5, 10][index % 3],
        "min_samples_leaf": [1, 2, 4][index % 3],
        "max_features": ["sqrt", 0.8, 1.0][index % 3],
        "max_leaf_nodes": [None, 20, 50, 100][index % 4],
    }
    return {
        "configuration": configuration,
        "mean_std": {
            "r2": {"mean": score, "std": 0.01},
            "mae": {"mean": 1 - score, "std": 0.02},
            "rmse": {"mean": 2 - score, "std": 0.03},
        },
        "fold_metrics": [
            {"r2": score - 0.01, "mae": 1 - score, "rmse": 2 - score},
            {"r2": score + 0.01, "mae": 1 - score, "rmse": 2 - score},
        ],
        "failures": [],
    }


def _tuning_results(count=12):
    results = [_trial(index, 0.9 - index / 100) for index in range(count)]
    results.append(
        {
            "configuration": _trial(0, 0.0)["configuration"],
            "mean_std": {
                "r2": {"mean": None, "std": None},
                "mae": {"mean": None, "std": None},
                "rmse": {"mean": None, "std": None},
            },
            "fold_metrics": [],
            "failures": [{"fold": 1, "error": "failed"}],
        }
    )
    return {
        "results": results,
        "n_configurations": len(results),
        "selection_metric": "mean_cv_r2",
    }


def test_report_ranks_by_cv_score_and_limits_default_to_top_ten():
    report = build_tuning_report(_tuning_results())

    assert report["best"]["configuration"] == _trial(0, 0.9)["configuration"]
    assert report["best"]["cv_mean_r2"] == 0.9
    assert report["total_configurations_evaluated"] == 13
    assert report["summary"] == {
        "successful_configurations": 12,
        "failed_configurations": 1,
        "mean_configuration_cv_r2": 0.845,
    }
    assert len(report["top_configurations"]) == 10
    assert [trial["rank"] for trial in report["top_configurations"]] == list(
        range(1, 11)
    )
    assert report["top_configurations"][0]["fold_r2_scores"] == [0.89, 0.91]
    assert "history" not in report


def test_report_includes_ranked_complete_history_only_when_requested():
    report = build_tuning_report(_tuning_results(), include_history=True)
    history = report["history"]

    assert len(history) == 13
    assert [trial["rank"] for trial in history[:12]] == list(range(1, 13))
    assert history[-1]["rank"] is None
    assert set(history[0]["configuration"]) == set(TUNING_PARAMETERS)
    assert set(history[0]["mean_std"]) == {"r2", "mae", "rmse"}
    assert history[0]["fold_metrics"] == _trial(0, 0.9)["fold_metrics"]
    assert history[-1]["failures"]


def test_parameter_observations_require_repeated_trials_at_multiple_values():
    report = build_tuning_report(_tuning_results(count=1))
    assert report["parameter_observations"] == []

    report = build_tuning_report(_tuning_results())
    assert report["parameter_observations"]
    observation = report["parameter_observations"][0]
    assert observation["parameter"] in TUNING_PARAMETERS
    assert all(
        value["trial_count"] >= 2
        for value in observation["observed_cv_r2_by_value"]
    )
    assert "do not establish causal effects" in report[
        "parameter_observation_note"
    ]
