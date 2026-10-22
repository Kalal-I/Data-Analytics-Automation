from typing import cast

import pytest

from agents_code import feature_optimization_linear_model as feature_agent


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_agent_serializes_dict_input_and_returns_decision(monkeypatch):
    captured = {}
    monkeypatch.setattr(feature_agent, "API_URL", "https://example.test/")
    monkeypatch.setattr(feature_agent, "API_TOKEN", "test-token")

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            {
                "response": (
                    '{"schema_version": 1, "action": "evaluate_model", '
                    '"model": "ElasticNet", "hyperparameters": '
                    '{"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10000}}'
                )
            }
        )

    monkeypatch.setattr(feature_agent.requests, "post", fake_post)
    result = feature_agent.feature_optimization_linear_model_agent(
        {
            "target_column": "target",
            "training": {
                "model": "Ridge",
                "model_type": "ridge",
                "hyperparameters": {"alpha": 1.0},
                "cross_validation": {"r2": {"mean": 0.8}},
            },
        },
        iteration_number=3,
        past_experiences=[
            {
                "target_column": "target",
                "training": {
                    "model": "Ridge",
                    "model_type": "ridge",
                    "hyperparameters": {"alpha": 1.0},
                    "cross_validation": {"r2": {"mean": 0.8}},
                },
            }
        ],
    )

    assert captured["url"] == "https://example.test/generate"
    prompt = captured["json"]["prompt"]
    assert "Current iteration number: `3`" in prompt
    assert '"target_column": "target"' in prompt
    assert '"features"' not in prompt
    assert '"feature_removal_tests"' not in prompt
    assert "<MODEL_OPTIMIZATION_REPORT>" not in prompt
    assert "<FEATURE_OPTIMIZATION_REPORT>" not in prompt
    assert "<PAST_EXPERIENCES>" not in prompt
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert result == {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": "ElasticNet",
        "hyperparameters": {
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "max_iter": 10000,
        },
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            '{"schema_version": 1, "action": "evaluate_model", '
            '"model": "Ridge", "hyperparameters": {"alpha": 0}}',
            "alpha must be a positive finite number",
        ),
        (
            '{"schema_version": 1, "action": "evaluate_model", '
            '"model": "ElasticNet", "hyperparameters": {"alpha": 1}}',
            "requires l1_ratio",
        ),
        (
            '{"schema_version": 1, "action": "evaluate_model", '
            '"model": "Lasso", "hyperparameters": {"alpha": 1, "l1_ratio": 0.5}}',
            "must not include l1_ratio",
        ),
    ],
)
def test_clean_json_output_rejects_invalid_model_configurations(response, message):
    with pytest.raises(ValueError, match=message):
        feature_agent.clean_json_output(response)


@pytest.mark.parametrize(
    ("iteration_number", "exception", "message"),
    [
        ("1", TypeError, "iteration_number must be an integer"),
        (0, ValueError, "iteration_number must be at least 1"),
    ],
)
def test_agent_rejects_invalid_iteration_number(
    monkeypatch,
    iteration_number,
    exception,
    message,
):
    monkeypatch.setattr(feature_agent, "API_URL", "https://example.test")
    monkeypatch.setattr(feature_agent, "API_TOKEN", "test-token")

    with pytest.raises(exception, match=message):
        feature_agent.feature_optimization_linear_model_agent(
            {},
            iteration_number=iteration_number,
        )


@pytest.mark.parametrize(
    "value",
    ['{"current_feature_set": {}}', None],
)
def test_agent_rejects_non_dictionary_input(monkeypatch, value):
    monkeypatch.setattr(feature_agent, "API_URL", "https://example.test")
    monkeypatch.setattr(feature_agent, "API_TOKEN", "test-token")

    with pytest.raises(TypeError, match="model_optimization_report"):
        feature_agent.feature_optimization_linear_model_agent(value)


def test_agent_rejects_non_finite_json_input(monkeypatch):
    monkeypatch.setattr(feature_agent, "API_URL", "https://example.test")
    monkeypatch.setattr(feature_agent, "API_TOKEN", "test-token")

    with pytest.raises(ValueError, match="Out of range float values"):
        feature_agent.feature_optimization_linear_model_agent({"metric": float("nan")})


def test_agent_rejects_invalid_past_experiences(monkeypatch):
    monkeypatch.setattr(feature_agent, "API_URL", "https://example.test")
    monkeypatch.setattr(feature_agent, "API_TOKEN", "test-token")

    with pytest.raises(TypeError, match="past_experiences"):
        feature_agent.feature_optimization_linear_model_agent(
            {},
            past_experiences=cast(list[dict] | None, ["not a report"]),
        )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (None, "no content"),
        ("", "empty JSON output"),
        ("not-json", "not valid JSON"),
        ("[]", "JSON object"),
    ],
)
def test_clean_json_output_rejects_invalid_response(response, message):
    with pytest.raises(ValueError, match=message):
        feature_agent.clean_json_output(response)
