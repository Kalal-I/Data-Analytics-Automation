import copy
import json

import pandas as pd
import pytest

import main as pipeline


INITIAL_ALPHAS = [0.1, 0.5, 1.0, 10.0, 100.0]


def _model_report(experiment, target_column):
    alpha = experiment["alpha"]
    return {
        "schema_version": 1,
        "target_column": target_column,
        "features": ["feature"],
        "training": {
            "model": {
                "lasso": "Lasso",
                "ridge": "Ridge",
                "elastic_net": "Elastic Net",
            }[experiment["model_type"]],
            "model_type": experiment["model_type"],
            "hyperparameters": {
                key: value
                for key, value in experiment.items()
                if key in {"alpha", "l1_ratio"}
            },
            "cross_validation": {
                "r2": {"mean": 1 - abs(alpha - 1) / 100, "std": 0.01},
                "adjusted_r2": {
                    "mean": 1 - abs(alpha - 1) / 100,
                    "std": 0.01,
                },
                "rmse": {"mean": alpha, "std": 0.1},
                "mae": {"mean": alpha / 2, "std": 0.1},
            },
            "coefficients": {"feature": 1.0},
        },
    }


def _patch_evaluation(monkeypatch):
    executed = []

    def fake_train(data, target_column, model_type, alpha, **kwargs):
        experiment = {
            "model_type": model_type,
            "alpha": alpha,
            "target_column": target_column,
            "data_row_count": len(data),
            "folds_file": kwargs["folds_file"],
        }
        if "l1_ratio" in kwargs:
            experiment["l1_ratio"] = kwargs["l1_ratio"]
        executed.append(experiment)
        return experiment

    monkeypatch.setattr(pipeline, "create_folds_and_train", fake_train)
    monkeypatch.setattr(
        pipeline,
        "split_result_to_json",
        lambda result, target_column: json.dumps(
            _model_report(result, target_column)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "apply_deferred_encodings",
        lambda *args: args[0],
    )
    monkeypatch.setattr(pipeline, "_data_fingerprint", lambda *_: "test-data")
    return executed


@pytest.mark.parametrize(
    ("model", "model_type", "histories_key", "extra_hyperparameters"),
    [
        ("Ridge", "ridge", "ridge_alphas_used", {}),
        ("Lasso", "lasso", "lasso_alphas_used", {}),
        (
            "ElasticNet",
            "elastic_net",
            "elastic_net_alphas_used",
            {"l1_ratio": 0.37},
        ),
    ],
)
def test_initializes_five_alphas_and_passes_results_to_next_call(
    monkeypatch,
    model,
    model_type,
    histories_key,
    extra_hyperparameters,
):
    executed = _patch_evaluation(monkeypatch)
    calls = []
    recommendation = {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": model,
        "hyperparameters": {
            "alpha": 2.3,
            **extra_hyperparameters,
        },
    }

    def fake_agent(report, **kwargs):
        calls.append(
            {
                "report": copy.deepcopy(report),
                "kwargs": copy.deepcopy(kwargs),
            }
        )
        return recommendation if len(calls) == 1 else {}

    monkeypatch.setattr(
        pipeline, "feature_optimization_linear_model_agent", fake_agent
    )
    reports = pipeline._run_regularized_optimization(
        pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
        "target",
        {"target_column": "target"},
        {},
    )

    assert [experiment["alpha"] for experiment in executed] == INITIAL_ALPHAS
    assert len({experiment["folds_file"] for experiment in executed}) == 1
    assert len(calls) == 2
    assert [call["kwargs"]["iteration_number"] for call in calls] == [1, 2]
    assert len(reports) == 5
    assert calls[1]["kwargs"][histories_key] == INITIAL_ALPHAS
    assert len(calls[1]["kwargs"]["past_experiences"]) == 5
    for experience, alpha in zip(
        calls[1]["kwargs"]["past_experiences"], INITIAL_ALPHAS
    ):
        assert experience["status"] == "completed"
        assert experience["model_type"] == model_type
        assert experience["hyperparameters"]["alpha"] == alpha
        assert "cross_validation" in experience

    if model_type == "elastic_net":
        assert all(
            experiment["l1_ratio"] == 0.37 for experiment in executed
        )
        assert calls[1]["kwargs"]["elastic_net_combinations_used"] == [
            {"alpha": alpha, "l1_ratio": 0.37} for alpha in INITIAL_ALPHAS
        ]
    else:
        assert all("l1_ratio" not in experiment for experiment in executed)
        other_history_keys = {
            "ridge_alphas_used",
            "lasso_alphas_used",
            "elastic_net_alphas_used",
        } - {histories_key}
        assert all(calls[1]["kwargs"][key] == [] for key in other_history_keys)


def test_normal_optimization_resumes_and_repeated_alpha_remains_executable(
    monkeypatch,
):
    executed = _patch_evaluation(monkeypatch)
    calls = []
    recommendations = [
        {
            "model": "Ridge",
            "hyperparameters": {"alpha": 2.3},
        },
        {
            "model": "Ridge",
            "hyperparameters": {"alpha": 0.1},
        },
        {},
    ]

    def fake_agent(report, **kwargs):
        _ = report
        calls.append(copy.deepcopy(kwargs))
        return recommendations[len(calls) - 1]

    monkeypatch.setattr(
        pipeline, "feature_optimization_linear_model_agent", fake_agent
    )
    reports = pipeline._run_regularized_optimization(
        pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
        "target",
        {"target_column": "target"},
        {},
    )

    assert len(calls) == 3
    assert [call["iteration_number"] for call in calls] == [1, 2, 3]
    assert [item["alpha"] for item in executed] == INITIAL_ALPHAS + [0.1]
    assert len(reports) == 6
    assert calls[1]["ridge_alphas_used"] == INITIAL_ALPHAS
    assert len(calls[1]["past_experiences"]) == 5
    assert calls[2]["ridge_alphas_used"] == INITIAL_ALPHAS + [0.1]
    assert len(calls[2]["past_experiences"]) == 6


def test_alpha_histories_are_separate_and_update_after_each_experiment(
    monkeypatch,
):
    _patch_evaluation(monkeypatch)
    calls = []
    recommendations = [
        {"model": "Ridge", "hyperparameters": {"alpha": 2.3}},
        {"model": "Lasso", "hyperparameters": {"alpha": 0.3}},
        {
            "model": "ElasticNet",
            "hyperparameters": {"alpha": 0.4, "l1_ratio": 0.25},
        },
        {},
    ]

    def fake_agent(report, **kwargs):
        _ = report
        calls.append(copy.deepcopy(kwargs))
        return recommendations[len(calls) - 1]

    monkeypatch.setattr(
        pipeline, "feature_optimization_linear_model_agent", fake_agent
    )
    pipeline._run_regularized_optimization(
        pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
        "target",
        {"target_column": "target"},
        {},
    )

    assert calls[3]["ridge_alphas_used"] == INITIAL_ALPHAS
    assert calls[3]["lasso_alphas_used"] == [0.3]
    assert calls[3]["elastic_net_alphas_used"] == [0.4]
    assert calls[3]["elastic_net_combinations_used"] == [
        {"alpha": 0.4, "l1_ratio": 0.25}
    ]


def test_initialization_reports_keep_existing_model_ranking_behavior(monkeypatch):
    executed = _patch_evaluation(monkeypatch)

    def fake_agent(report, **kwargs):
        _ = report
        return (
            {"model": "Ridge", "hyperparameters": {"alpha": 2.3}}
            if kwargs["iteration_number"] == 1
            else {}
        )

    monkeypatch.setattr(
        pipeline, "feature_optimization_linear_model_agent", fake_agent
    )
    reports = pipeline._run_regularized_optimization(
        pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
        "target",
        {"target_column": "target"},
        {},
    )
    selected, metrics = pipeline.select_best_feature_report(reports)
    expected, expected_metrics = pipeline.select_best_feature_report(
        copy.deepcopy(reports)
    )

    assert len(executed) == 5
    assert selected == expected
    assert metrics == expected_metrics


def test_greedy_selection_uses_fixed_best_model_configuration_and_common_folds(
    monkeypatch,
    tmp_path,
    capsys,
):
    calls = []

    def fake_train(data, target_column, model_type, alpha, **kwargs):
        calls.append(
            {
                "data_columns": list(data.columns),
                "target_column": target_column,
                "model_type": model_type,
                "alpha": alpha,
                "l1_ratio": kwargs["l1_ratio"],
                "features": kwargs["features"],
                "folds_file": kwargs["folds_file"],
            }
        )
        return len(calls)

    def fake_serialize(result, target_column):
        scores = {1: 0.7, 2: 0.71}
        return json.dumps(
            {
                "target_column": target_column,
                "training": {
                    "cross_validation": {
                        "r2": {"mean": scores[result], "std": 0.01},
                        "adjusted_r2": {"mean": 0.6, "std": 0.02},
                        "rmse": {"mean": 1.0, "std": 0.03},
                        "mae": {"mean": 0.5, "std": 0.04},
                    }
                }
            }
        )

    monkeypatch.setattr(pipeline, "create_folds_and_train", fake_train)
    monkeypatch.setattr(pipeline, "split_result_to_json", fake_serialize)
    report = {
        "features": ["first", "second"],
        "training": {
            "model_type": "elastic_net",
            "hyperparameters": {"alpha": 0.4, "l1_ratio": 0.3},
            "coefficients": {"first": 2.0, "second": -1.0},
        },
    }
    original_report = copy.deepcopy(report)

    selection = pipeline._run_greedy_feature_selection(
        pd.DataFrame({"first": [1, 2], "second": [2, 3], "target": [3, 5]}),
        "target",
        report,
        tmp_path / "same-folds.json",
    )

    assert [call["features"] for call in calls] == [
        ["first"],
        ["first", "second"],
    ]
    assert all(call["model_type"] == "elastic_net" for call in calls)
    assert all(call["alpha"] == 0.4 for call in calls)
    assert all(call["l1_ratio"] == 0.3 for call in calls)
    assert all(call["target_column"] == "target" for call in calls)
    assert len({call["folds_file"] for call in calls}) == 1
    assert selection["accepted_features"] == ["first", "second"]
    assert report == original_report
    assert capsys.readouterr().out == ""


def test_main_prints_only_accepted_features_and_their_scores(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "data.csv")
    monkeypatch.setattr(
        pipeline,
        "read_csv_files",
        lambda _: pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
    )
    monkeypatch.setattr(
        pipeline,
        "preprocessing_agent",
        lambda _: {},
    )
    monkeypatch.setattr(
        pipeline,
        "generate_report",
        lambda *_: json.dumps(
            {"Reports": [{"Target column": {"Type": "numeric"}}]}
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_model_statistics",
        lambda *_: json.dumps({"statistics": {}}),
    )
    monkeypatch.setattr(
        pipeline,
        "linear_regression_agent",
        lambda **_: {"ranked_models": [{"model_name": "Lasso"}]},
    )
    model_report = {
        "target_column": "target",
        "features": ["feature"],
        "training": {
            "model": "Ridge",
            "model_type": "ridge",
            "hyperparameters": {"alpha": 0.1},
            "coefficients": {"feature": 1.0},
            "cross_validation": {
                "r2": {"mean": 0.8, "std": 0.01},
                "adjusted_r2": {"mean": 0.7, "std": 0.02},
                "rmse": {"mean": 0.4, "std": 0.03},
                "mae": {"mean": 0.3, "std": 0.04},
            },
        },
    }
    monkeypatch.setattr(
        pipeline,
        "_run_regularized_optimization",
        lambda *_: [model_report],
    )
    selection_result = {
        "accepted_features": ["feature"],
        "accepted_cross_validation": {
            "r2": {"mean": 0.8, "std": 0.01},
            "adjusted_r2": {"mean": 0.7, "std": 0.02},
            "rmse": {"mean": 0.4, "std": 0.03},
            "mae": {"mean": 0.3, "std": 0.04},
        },
        "ranked_features": [{"feature": "feature", "coefficient": 1.0}],
        "evaluations": [{"candidate_features": ["feature"], "decision": "accepted"}],
    }
    monkeypatch.setattr(
        pipeline,
        "_run_greedy_feature_selection",
        lambda *_: selection_result,
    )
    monkeypatch.setattr(pipeline, "_data_fingerprint", lambda *_: "test-data")

    result = pipeline.main()

    assert result is selection_result
    output = capsys.readouterr().out
    summary = {
        "accepted_features": ["feature"],
        "cross_validation": selection_result["accepted_cross_validation"],
    }
    assert "Final accepted features and scores:" in output
    assert json.dumps(summary, indent=2, allow_nan=False) in output
    assert "ranked_features" not in output
    assert "evaluations" not in output
    assert "Best regularized regression model:" not in output
    assert "Model weights by feature:" not in output
