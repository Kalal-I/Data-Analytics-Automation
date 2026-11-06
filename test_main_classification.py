import json

import pandas as pd
import pytest

import main as pipeline


@pytest.mark.parametrize(
    "label_type",
    ["Categorical", "Non-categorical string"],
)
def test_main_routes_string_target_to_classification_agent_and_prints_result(
    monkeypatch,
    capsys,
    label_type,
):
    dataset = pd.DataFrame({"feature": [1, 2], "target": ["a", "b"]})
    description = {
        "Reports": [
            {"Target column": {"Type": label_type}}
        ]
    }
    recommendation = {
        "schema_version": 1,
        "target_column": "target",
        "label_type": label_type,
        "ranked_models": [],
    }
    calls = []
    user_inputs = iter(["dataset.csv", "target"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: next(user_inputs),
    )
    monkeypatch.setattr(pipeline, "read_csv_files", lambda _path: dataset)
    monkeypatch.setattr(
        pipeline,
        "generate_report",
        lambda *_args: json.dumps(description),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_model_statistics_classification",
        lambda *_args: json.dumps({"classification_class_count": 2}),
    )
    monkeypatch.setattr(pipeline, "preprocessing_agent", lambda _report: {})
    monkeypatch.setattr(
        pipeline,
        "validate_preprocessing_calls",
        lambda operations, _report: operations,
    )
    monkeypatch.setattr(
        pipeline,
        "partition_deferred_encodings",
        lambda operations: (operations, {}),
    )
    monkeypatch.setattr(
        pipeline,
        "classification_agent",
        lambda **kwargs: calls.append(kwargs) or recommendation,
    )
    monkeypatch.setattr(
        pipeline,
        "linear_regression_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Regression agent must not run for string targets")
        ),
    )

    result = pipeline.main()

    assert result is recommendation
    assert calls == [
        {
            "dataset_description": description,
            "target_column": "target",
            "label_type": label_type,
            "statistics": {"classification_class_count": 2},
            "row_count": 2,
        }
    ]
    output = capsys.readouterr().out
    assert "Classification agent returned:" in output
    assert str(recommendation) in output


def test_main_runs_logistic_optimizer_for_first_ranked_logistic_model(
    monkeypatch,
):
    dataset = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "target": ["no", "yes", "no", "yes"]}
    )
    description = {
        "Reports": [
            {"Target column": {"Type": "Categorical"}}
        ]
    }
    recommendation = {
        "ranked_models": [
            {"rank": 1, "model_name": "LogisticRegression"}
        ]
    }
    optimizer_result = {
        "best_configuration": {"penalty": "l2"},
        "cross_validation": {"mean_std": {}},
        "holdout_metrics": {},
    }
    calls = []
    user_inputs = iter(["dataset.csv", "target"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: next(user_inputs),
    )
    monkeypatch.setattr(pipeline, "read_csv_files", lambda _path: dataset)
    monkeypatch.setattr(
        pipeline,
        "generate_report",
        lambda *_args: json.dumps(description),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_model_statistics_classification",
        lambda *_args: json.dumps({"class_count": 2}),
    )
    monkeypatch.setattr(pipeline, "preprocessing_agent", lambda _report: {})
    monkeypatch.setattr(
        pipeline,
        "validate_preprocessing_calls",
        lambda operations, _report: operations,
    )
    monkeypatch.setattr(
        pipeline,
        "partition_deferred_encodings",
        lambda operations: (operations, {}),
    )
    monkeypatch.setattr(
        pipeline,
        "classification_agent",
        lambda **_kwargs: recommendation,
    )
    monkeypatch.setattr(
        pipeline,
        "run_logistic_optimization",
        lambda *args: calls.append(args) or optimizer_result,
    )
    monkeypatch.setattr(
        pipeline,
        "linear_regression_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Regression agent must not run for string targets")
        ),
    )

    result = pipeline.main()

    assert result is optimizer_result
    assert len(calls) == 1
    assert calls[0][0] is dataset
    assert calls[0][1] == "target"
    assert calls[0][2]["initial_model_ranking"] is recommendation
    assert calls[0][2]["statistics"] == {"class_count": 2}
    assert calls[0][3] == {}
