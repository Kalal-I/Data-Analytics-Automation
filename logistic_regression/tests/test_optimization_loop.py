import copy

import pandas as pd

import logistic_regression.optimization_loop as optimization_loop


def _decision(model, C, l1_ratio=None, threshold=0.5):
    hyperparameters = {"C": C, "threshold": threshold}
    if l1_ratio is not None:
        hyperparameters["l1_ratio"] = l1_ratio
    return {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": model,
        "hyperparameters": hyperparameters,
        "reasoning": ["Test a supported regularization setting."],
        "hypothesis": "Compare cross-validation metrics.",
    }


def _patch_execution(monkeypatch):
    data = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": ["no", "yes"] * 10,
        }
    )
    train_X = data[["feature"]].iloc[:16].reset_index(drop=True)
    holdout_X = data[["feature"]].iloc[16:].reset_index(drop=True)
    train_y = data["target"].iloc[:16].reset_index(drop=True)
    holdout_y = data["target"].iloc[16:].reset_index(drop=True)
    folds = [object()]
    evaluations = []
    split_calls = []
    applied_encodings = []
    fitted = []

    monkeypatch.setattr(
        optimization_loop,
        "apply_deferred_encodings",
        lambda frame, operations, functions: (
            applied_encodings.append((operations, functions)) or frame
        ),
    )
    monkeypatch.setattr(
        optimization_loop,
        "validate_features",
        lambda frame, target: ["feature"],
    )

    def fake_split(frame, target, random_state):
        split_calls.append((frame, target, random_state))
        return train_X, holdout_X, train_y, holdout_y

    monkeypatch.setattr(
        optimization_loop,
        "train_holdout_split",
        fake_split,
    )
    monkeypatch.setattr(
        optimization_loop,
        "create_cv_folds",
        lambda X, y, random_state: folds,
    )

    def fake_evaluate(X, y, supplied_folds, configuration, **kwargs):
        evaluations.append(
            {
                "X": X,
                "y": y,
                "folds": supplied_folds,
                "configuration": copy.deepcopy(configuration),
                "kwargs": kwargs,
            }
        )
        return {
            "configuration": copy.deepcopy(configuration),
            "fold_metrics": [{"f1": 0.8}],
            "failures": [],
            "mean_std": {
                name: {"mean": score, "std": 0.01}
                for name, score in {
                    "f1": 0.8,
                    "accuracy": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "roc_auc": 0.8,
                }.items()
            },
        }

    monkeypatch.setattr(
        optimization_loop,
        "evaluate_configuration",
        fake_evaluate,
    )
    monkeypatch.setattr(
        optimization_loop,
        "fit_model",
        lambda X, y, **configuration: (
            fitted.append((X, y, configuration)) or "fitted-model"
        ),
    )
    monkeypatch.setattr(
        optimization_loop,
        "evaluate_holdout",
        lambda model, X, y, **kwargs: {
            "accuracy": 0.75,
            "model": model,
            "holdout_rows": len(X),
            "threshold": kwargs["threshold"],
        },
    )
    return {
        "data": data,
        "folds": folds,
        "evaluations": evaluations,
        "split_calls": split_calls,
        "applied_encodings": applied_encodings,
        "fitted": fitted,
    }


def test_initializes_c_values_and_reuses_split_folds_and_histories(monkeypatch):
    context = _patch_execution(monkeypatch)
    agent_calls = []

    def fake_agent(report, **kwargs):
        agent_calls.append(
            {"report": copy.deepcopy(report), "kwargs": copy.deepcopy(kwargs)}
        )
        return (
            _decision("Ridge", 2.0, threshold=0.6)
            if len(agent_calls) == 1
            else {}
        )

    monkeypatch.setattr(
        optimization_loop,
        "logistic_regression_agent",
        fake_agent,
    )

    result = optimization_loop.run_logistic_optimization(
        context["data"],
        "target",
        {"target_column": "target"},
        {"category": [{"function": "one_hot_encode"}]},
    )

    expected_C_values = [0.1, 0.5, 1.0, 10.0, 100.0]
    assert [item["configuration"]["C"] for item in context["evaluations"]] == (
        expected_C_values
    )
    assert all(
        item["configuration"]["penalty"] == "l2"
        for item in context["evaluations"]
    )
    assert all(
        item["configuration"]["threshold"] == 0.6
        for item in context["evaluations"]
    )
    assert all(item["folds"] is context["folds"] for item in context["evaluations"])
    assert len(context["split_calls"]) == 1
    assert context["split_calls"][0][2] == 42
    assert len(context["applied_encodings"]) == 1
    assert len(agent_calls) == 2
    assert agent_calls[1]["kwargs"]["ridge_c_values_used"] == expected_C_values
    assert agent_calls[1]["kwargs"]["lasso_c_values_used"] == []
    assert agent_calls[1]["kwargs"]["elastic_net_c_values_used"] == []
    assert agent_calls[1]["kwargs"]["threshold_values_used"] == [0.6] * 5
    assert len(agent_calls[1]["kwargs"]["past_experiences"]) == 5
    assert all(
        item["hyperparameters"]["threshold"] == 0.6
        for item in agent_calls[1]["kwargs"]["past_experiences"]
    )
    assert result["best_configuration"]["penalty"] == "l2"
    assert result["best_configuration"]["threshold"] == 0.6
    assert result["selected_features"] == ["feature"]
    assert result["holdout_metrics"]["accuracy"] == 0.75
    assert result["holdout_metrics"]["threshold"] == 0.6
    assert result["model"] == "fitted-model"


def test_agent_histories_are_separate_for_each_regularization_model(monkeypatch):
    context = _patch_execution(monkeypatch)
    recommendations = [
        _decision("Ridge", 1.0),
        _decision("Lasso", 2.0, threshold=0.4),
        _decision("ElasticNet", 3.0, l1_ratio=0.4, threshold=0.7),
        {},
    ]
    calls = []

    def fake_agent(report, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        return recommendations[len(calls) - 1]

    monkeypatch.setattr(
        optimization_loop,
        "logistic_regression_agent",
        fake_agent,
    )

    optimization_loop.run_logistic_optimization(
        context["data"],
        "target",
        {"target_column": "target"},
    )

    assert calls[-1]["ridge_c_values_used"] == [
        0.1,
        0.5,
        1.0,
        10.0,
        100.0,
    ]
    assert calls[-1]["lasso_c_values_used"] == [2.0]
    assert calls[-1]["elastic_net_c_values_used"] == [3.0]
    assert calls[-1]["elastic_net_combinations_used"] == [
        {"C": 3.0, "l1_ratio": 0.4}
    ]
    assert calls[-1]["threshold_values_used"] == [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.4,
        0.7,
    ]
    assert [
        item["configuration"]["penalty"]
        for item in context["evaluations"][-2:]
    ] == ["l1", "elasticnet"]
