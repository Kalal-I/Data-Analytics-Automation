import copy

import pandas as pd
import pytest

import main as pipeline
from agents_code import rfr_agent as rfr_module


BASE_CONFIGURATION = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_leaf_nodes": None,
}


def _cv_result(configuration, r2):
    return {
        "configuration": dict(configuration),
        "mean_std": {
            "r2": {"mean": r2, "std": 0.01},
            "rmse": {"mean": 1 - r2, "std": 0.02},
            "mae": {"mean": (1 - r2) / 2, "std": 0.01},
        },
        "fold_metrics": [],
        "failures": [],
    }


def _workflow_result(configuration, cv_results, best_configuration=None):
    return {
        "best_configuration": dict(best_configuration or configuration),
        "cross_validation": {
            "results": cv_results,
            "best_configuration": dict(best_configuration or configuration),
            "selection_metric": "mean_cv_r2",
        },
        "holdout_metrics": {"r2": 0.7, "rmse": 1.0, "mae": 0.5},
        "selected_features": ["feature"],
        "model": object(),
        "folds": [],
    }


def test_random_forest_optimization_runs_baseline_and_five_agent_experiments(
    monkeypatch,
    capsys,
):
    baseline_results = [
        _cv_result(BASE_CONFIGURATION, 0.8),
        _cv_result({**BASE_CONFIGURATION, "n_estimators": 50}, 0.7),
    ]
    workflow_calls = []

    def fake_workflow(_data, _target, **kwargs):
        _ = (_data, _target)
        workflow_calls.append(kwargs)
        if not kwargs:
            return _workflow_result(
                BASE_CONFIGURATION,
                baseline_results,
            )
        configuration = {
            name: values[0] for name, values in kwargs["search_space"].items()
        }
        configuration.setdefault(
            "n_estimators", BASE_CONFIGURATION["n_estimators"]
        )
        configuration.setdefault("max_depth", BASE_CONFIGURATION["max_depth"])
        configuration.setdefault(
            "min_samples_split", BASE_CONFIGURATION["min_samples_split"]
        )
        configuration.setdefault(
            "min_samples_leaf", BASE_CONFIGURATION["min_samples_leaf"]
        )
        configuration.setdefault(
            "max_features", BASE_CONFIGURATION["max_features"]
        )
        configuration.setdefault(
            "max_leaf_nodes", BASE_CONFIGURATION["max_leaf_nodes"]
        )
        r2 = 0.75 if len(workflow_calls) == 2 else 0.8 + len(workflow_calls) / 100
        result = _cv_result(configuration, r2)
        return _workflow_result(configuration, [result])

    recommendations = [
        {"max_depth": 5},
        {"min_samples_leaf": 2},
        {"max_features": "sqrt"},
        {"n_estimators": 150},
        {"max_leaf_nodes": 20},
    ]
    agent_calls = []

    def fake_agent(**kwargs):
        agent_calls.append(copy.deepcopy(kwargs))
        return recommendations[len(agent_calls) - 1]

    monkeypatch.setattr(pipeline, "run_random_forest_workflow", fake_workflow)
    monkeypatch.setattr(pipeline, "rfr_agent", fake_agent)

    result = pipeline._run_random_forest_optimization(
        pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
        "target",
        {"target_column": "target"},
    )

    assert workflow_calls[0] == {}
    assert len(workflow_calls) == 6
    assert all(call["n_iter"] == 1 for call in workflow_calls[1:])
    assert len(agent_calls) == 5
    assert [call["iteration_number"] for call in agent_calls] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert [len(call["past_experiences"]) for call in agent_calls] == [
        2,
        3,
        4,
        5,
        6,
    ]
    assert all(
        call["dataset_statistics"] == {"target_column": "target"}
        for call in agent_calls
    )
    assert agent_calls[1]["current_configuration"]["max_depth"] == 5
    assert agent_calls[1]["best_configuration"] == BASE_CONFIGURATION
    assert len(result["past_experiences"]) == 7
    assert result["agent_iterations"] == 5
    assert result["best_configuration"]["max_leaf_nodes"] == 20
    assert result["best_cv_r2"] == pytest.approx(0.86)
    output = capsys.readouterr().out
    assert output.count("Random forest iteration ") == 10
    assert output.count(" recommendation: ") == 5
    assert output.count(" result: parameters=") == 5
    assert output.count("CV mean R2=") == 5
    assert "rmse" not in output
    assert "mae" not in output


def test_random_forest_optimization_rejects_missing_iteration_recommendation(
    monkeypatch,
):
    baseline_results = [_cv_result(BASE_CONFIGURATION, 0.8)]
    workflow_calls = []
    agent_iterations = []

    def fake_workflow(_data, _target, **kwargs):
        _ = (_data, _target)
        workflow_calls.append(kwargs)
        return _workflow_result(BASE_CONFIGURATION, baseline_results)

    def fake_agent(**kwargs):
        agent_iterations.append(kwargs["iteration_number"])
        return {}

    monkeypatch.setattr(pipeline, "run_random_forest_workflow", fake_workflow)
    monkeypatch.setattr(pipeline, "rfr_agent", fake_agent)

    with pytest.raises(ValueError, match="all five iterations"):
        pipeline._run_random_forest_optimization(
            pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
            "target",
            {},
        )

    assert workflow_calls == [{}]
    assert agent_iterations == [1]


def test_random_forest_optimization_skips_duplicate_and_continues_iterations(
    monkeypatch,
    capsys,
):
    baseline_results = [_cv_result(BASE_CONFIGURATION, 0.8)]
    workflow_calls = []
    agent_calls = []

    def fake_workflow(_data, _target, **kwargs):
        _ = (_data, _target)
        workflow_calls.append(kwargs)
        if not kwargs:
            return _workflow_result(BASE_CONFIGURATION, baseline_results)
        configuration = {
            name: values[0] for name, values in kwargs["search_space"].items()
        }
        result = _cv_result(configuration, 0.81)
        return _workflow_result(configuration, [result])

    recommendations = [
        {"max_depth": 5},
        {"max_depth": 5},
        {"min_samples_leaf": 2},
        {"max_features": "sqrt"},
        {"n_estimators": 150},
    ]

    def fake_agent(**kwargs):
        agent_calls.append(copy.deepcopy(kwargs))
        return recommendations[len(agent_calls) - 1]

    monkeypatch.setattr(pipeline, "run_random_forest_workflow", fake_workflow)
    monkeypatch.setattr(pipeline, "rfr_agent", fake_agent)

    result = pipeline._run_random_forest_optimization(
        pd.DataFrame({"feature": [1, 2], "target": [2, 4]}),
        "target",
        {},
    )

    assert [call["iteration_number"] for call in agent_calls] == [1, 2, 3, 4, 5]
    assert len(workflow_calls) == 5
    assert result["agent_iterations"] == 5
    skipped = agent_calls[2]["past_experiences"][-1]
    assert skipped["iteration_number"] == 2
    assert skipped["status"] == "skipped_duplicate"
    assert skipped["parameters"]["max_depth"] == 5
    assert len(result["past_experiences"]) == 6
    output = capsys.readouterr().out
    assert "Random forest iteration 2 skipped: duplicate configuration." in output
    assert output.count(" result: parameters=") == 4
    assert output.count("CV mean R2=") == 4


def test_rfr_agent_sends_context_and_validates_json_response(monkeypatch):
    request = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": '{"max_depth": 10}'}

    def fake_post(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(rfr_module, "API_URL", "https://example.test/")
    monkeypatch.setattr(rfr_module, "API_TOKEN", "test-token")
    monkeypatch.setattr(rfr_module.requests, "post", fake_post)

    past_experiences = [{"parameters": BASE_CONFIGURATION, "metrics": {}}]
    recommendation = rfr_module.rfr_agent(
        dataset_statistics={"rows": 20},
        past_experiences=past_experiences,
        iteration_number=2,
        current_configuration=BASE_CONFIGURATION,
        best_configuration=BASE_CONFIGURATION,
    )

    body = request["json"]["prompt"]
    assert request["url"] == "https://example.test/generate"
    assert request["headers"]["Authorization"] == "Bearer test-token"
    assert '"rows": 20' in body
    assert '"iteration_number": 2' not in body
    assert "### Current iteration\n\n2" in body
    assert '"n_estimators": 100' in body
    assert '"parameters":' in body
    assert "NEVER recommend a complete configuration" in rfr_module.prompt
    assert "A duplicate recommendation will be skipped" in rfr_module.prompt
    assert recommendation == {"max_depth": 10}


@pytest.mark.parametrize(
    "response",
    [
        '{"criterion": "squared_error"}',
        '{"max_depth": 7}',
        '{"max_depth": true}',
        "not json",
    ],
)
def test_rfr_agent_rejects_invalid_recommendations(monkeypatch, response):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": response}

    monkeypatch.setattr(rfr_module, "API_URL", "https://example.test")
    monkeypatch.setattr(rfr_module, "API_TOKEN", "test-token")
    def fake_post(*args, **kwargs):
        _ = (args, kwargs)
        return FakeResponse()

    monkeypatch.setattr(rfr_module.requests, "post", fake_post)

    with pytest.raises(ValueError):
        rfr_module.rfr_agent({}, [], 1, BASE_CONFIGURATION, BASE_CONFIGURATION)
