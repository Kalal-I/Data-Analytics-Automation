import json

import pandas as pd
import pytest

from agents_code.feature_optimization_linear_model import clean_json_output
from linear_regression.model_selection import select_best_feature_report
from linear_regression.optimization_workflow import (
    INITIALIZATION_ALPHAS,
    run_model_optimization,
)


def _proposal(model="Ridge", alpha=2.5, l1_ratio=None):
    hyperparameters = {"alpha": alpha}
    if model == "ElasticNet":
        hyperparameters["l1_ratio"] = l1_ratio
    return {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": model,
        "hyperparameters": hyperparameters,
    }


def _report(target="target", model="Ridge", model_type="ridge", alpha=0.1,
            l1_ratio=None, r2=0.8):
    hyperparameters = {"alpha": alpha}
    if model_type == "elastic_net":
        hyperparameters["l1_ratio"] = l1_ratio
    return {
        "target_column": target,
        "features": ["x"],
        "training": {
            "model": model,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "cross_validation": {
                "r2": {"mean": r2, "std": 0.01},
                "adjusted_r2": {"mean": r2 - 0.01},
                "rmse": {"mean": 1.0},
                "mae": {"mean": 0.5},
            },
        },
    }


def _run(tmp_path, first_proposal=None, later_proposals=None, evaluate=None):
    calls = []
    responses = iter([first_proposal or _proposal(), *(later_proposals or [{}])])

    def agent(report, *, past_experiences, iteration_number):
        calls.append(
            {
                "report": report,
                "history": list(past_experiences),
                "iteration": iteration_number,
            }
        )
        return next(responses, {})

    evaluations = []

    def evaluator(data, target, model_type, hyperparameters):
        evaluations.append((model_type, dict(hyperparameters)))
        if evaluate:
            return evaluate(data, target, model_type, hyperparameters)
        model_name = {
            "ridge": "Ridge",
            "lasso": "Lasso",
            "elastic_net": "Elastic Net",
        }[model_type]
        return _report(
            target=target,
            model=model_name,
            model_type=model_type,
            alpha=hyperparameters["alpha"],
            l1_ratio=hyperparameters.get("l1_ratio"),
        )

    history_path = tmp_path / "experiments.json"
    results = run_model_optimization(
        data=pd.DataFrame({"x": [1.0, 2.0], "target": [2.0, 4.0]}),
        target_column="target",
        initial_model_report={"target_column": "target", "initial": True},
        history_path=history_path,
        data_fingerprint="fingerprint",
        agent=agent,
        evaluate=evaluator,
    )
    return results, calls, evaluations, history_path


@pytest.mark.parametrize(
    ("model", "model_type"),
    [("Ridge", "ridge"), ("Lasso", "lasso")],
)
def test_ridge_and_lasso_initialization_run_five_alphas(
    tmp_path, model, model_type
):
    results, calls, evaluations, history_path = _run(
        tmp_path,
        first_proposal=_proposal(model=model, alpha=3.7),
    )

    assert [item[1]["alpha"] for item in evaluations] == list(
        INITIALIZATION_ALPHAS
    )
    assert all(item[0] == model_type for item in evaluations)
    assert len(results) == 5
    assert [call["iteration"] for call in calls] == [1, 2]
    stored = json.loads(history_path.read_text(encoding="utf-8"))
    assert [item["agent_iteration"] for item in stored["experiments"]] == [1] * 5


def test_elastic_net_keeps_proposed_ratio_for_all_initialization_alphas(tmp_path):
    _, calls, evaluations, history_path = _run(
        tmp_path,
        first_proposal=_proposal("ElasticNet", alpha=4.2, l1_ratio=0.37),
    )

    assert [item[1] for item in evaluations] == [
        {"alpha": alpha, "l1_ratio": 0.37} for alpha in INITIALIZATION_ALPHAS
    ]
    assert calls[1]["history"]
    stored = json.loads(history_path.read_text(encoding="utf-8"))
    assert all(
        item["hyperparameters"]["l1_ratio"] == 0.37
        for item in stored["experiments"]
    )


def test_initial_alpha_is_overridden_and_results_reach_next_call(tmp_path):
    _, calls, evaluations, _ = _run(
        tmp_path,
        first_proposal=_proposal(alpha=25.0),
    )

    assert len(evaluations) == 5
    assert all(item[1]["alpha"] != 25.0 for item in evaluations)
    assert calls[1]["iteration"] == 2
    assert len(calls[1]["history"]) == 5
    assert all(item["status"] == "completed" for item in calls[1]["history"])
    assert {
        item["hyperparameters"]["alpha"] for item in calls[1]["history"]
    } == set(INITIALIZATION_ALPHAS)


def test_subsequent_proposal_runs_once_and_initialization_is_not_repeated(tmp_path):
    _, calls, evaluations, _ = _run(
        tmp_path,
        later_proposals=[_proposal(alpha=0.2), {}],
    )

    assert [call["iteration"] for call in calls] == [1, 2, 3]
    assert len(evaluations) == 6
    assert evaluations[-1][1]["alpha"] == 0.2


def test_completed_initialization_is_not_reexecuted_on_later_run(tmp_path):
    history_path = tmp_path / "completed.json"
    calls = []
    evaluations = []

    def agent(report, *, past_experiences, iteration_number):
        _ = (report, past_experiences)
        calls.append(iteration_number)
        return _proposal() if iteration_number == 1 else {}

    def evaluate(data, target, model_type, hyperparameters):
        _ = (data, target, model_type)
        evaluations.append(hyperparameters["alpha"])
        return _report(alpha=hyperparameters["alpha"])

    arguments = (
        pd.DataFrame({"x": [1.0], "target": [2.0]}),
        "target",
        {},
        history_path,
        "fingerprint",
        agent,
        evaluate,
    )
    run_model_optimization(*arguments)
    run_model_optimization(*arguments)

    assert evaluations == list(INITIALIZATION_ALPHAS)
    assert calls == [1, 2]


def test_duplicate_configuration_is_rejected_without_execution(tmp_path):
    calls = []
    evaluations = []
    proposals = iter([_proposal(alpha=2.0), _proposal(alpha=0.1)])

    def agent(report, *, past_experiences, iteration_number):
        _ = (report, past_experiences)
        calls.append(iteration_number)
        return next(proposals)

    def evaluate(data, target, model_type, hyperparameters):
        _ = (data, target, model_type)
        evaluations.append(hyperparameters["alpha"])
        return _report(alpha=hyperparameters["alpha"])

    with pytest.raises(ValueError, match="duplicate"):
        run_model_optimization(
            pd.DataFrame({"x": [1.0], "target": [2.0]}),
            "target",
            {},
            tmp_path / "duplicate.json",
            "fingerprint",
            agent,
            evaluate,
        )

    assert len(evaluations) == 5
    assert calls == [1, 2]


@pytest.mark.parametrize(
    ("model", "alpha", "l1_ratio", "message"),
    [
        ("Ridge", 1.25, None, "at most one decimal"),
        ("ElasticNet", 0.1, 0, "l1_ratio"),
        ("ElasticNet", 0.1, 1.1, "l1_ratio"),
    ],
)
def test_invalid_hyperparameters_are_rejected(
    tmp_path, model, alpha, l1_ratio, message
):
    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            first_proposal=_proposal(
                model=model,
                alpha=alpha,
                l1_ratio=l1_ratio,
            ),
        )


def test_failed_experiment_is_recorded_and_pending_initialization_resumes(tmp_path):
    history_path = tmp_path / "partial.json"
    calls = []
    attempts = []

    def agent(report, *, past_experiences, iteration_number):
        _ = report
        calls.append((iteration_number, list(past_experiences)))
        return _proposal() if iteration_number == 1 else {}

    def fail_second(data, target, model_type, hyperparameters):
        _ = (data, target, model_type)
        alpha = hyperparameters["alpha"]
        attempts.append(alpha)
        if alpha == 0.5:
            raise RuntimeError("training failed")
        return _report(alpha=alpha)

    with pytest.raises(RuntimeError, match="training failed"):
        run_model_optimization(
            pd.DataFrame({"x": [1.0], "target": [2.0]}),
            "target",
            {},
            history_path,
            "fingerprint",
            agent,
            fail_second,
        )

    initial_history = json.loads(history_path.read_text(encoding="utf-8"))
    failed = initial_history["experiments"][1]
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "training failed"
    assert failed["cross_validation"] is None
    assert failed["report"] is None

    def resume_evaluator(data, target, model_type, hyperparameters):
        _ = (data, target, model_type)
        attempts.append(hyperparameters["alpha"])
        return _report(alpha=hyperparameters["alpha"])

    run_model_optimization(
        pd.DataFrame({"x": [1.0], "target": [2.0]}),
        "target",
        {},
        history_path,
        "fingerprint",
        agent,
        resume_evaluator,
    )

    assert attempts == [0.1, 0.5, 1.0, 10.0, 100.0]
    assert len(calls) == 2
    assert len(calls[1][1]) == 5
    assert calls[1][1][1]["status"] == "failed"


@pytest.mark.parametrize("alpha", [0, -1, 1.25])
def test_agent_output_rejects_invalid_alpha_precision(alpha):
    response = json.dumps(
        {
            "schema_version": 1,
            "action": "evaluate_model",
            "model": "Ridge",
            "hyperparameters": {"alpha": alpha},
        }
    )
    with pytest.raises(ValueError):
        clean_json_output(response)


def test_existing_final_ranking_remains_deterministic():
    first = _report(alpha=0.1, r2=0.9)
    second = _report(alpha=0.5, r2=0.9)

    selected, metrics = select_best_feature_report([first, second])

    assert selected is first
    assert metrics == ["r2_tolerance", "adjusted_r2", "rmse", "mae", "feature_count"]
