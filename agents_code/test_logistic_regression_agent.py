import json

import pytest

from agents_code import logistic_regression_agent as logistic_agent_module


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "response": json.dumps(
                {
                    "schema_version": 1,
                    "action": "evaluate_model",
                    "model": "Ridge",
                    "hyperparameters": {"C": 1.0, "threshold": 0.5},
                }
            )
        }


@pytest.mark.parametrize(
    ("api_token", "expected_authorization"),
    [
        ("test-token", "Bearer test-token"),
        ("  bEaReR test-token  ", "bEaReR test-token"),
    ],
)
def test_agent_normalizes_bearer_authorization_header(
    monkeypatch,
    api_token,
    expected_authorization,
):
    captured = {}
    monkeypatch.setattr(
        logistic_agent_module,
        "API_URL",
        "https://example.test/",
    )
    monkeypatch.setattr(logistic_agent_module, "API_TOKEN", api_token)

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(logistic_agent_module.requests, "post", fake_post)

    result = logistic_agent_module.logistic_regression_agent(
        {"training": {}},
        threshold_values_used=[0.5],
    )

    assert captured["url"] == "https://example.test/generate"
    assert captured["headers"]["Authorization"] == expected_authorization
    assert "[0.5]" in captured["json"]["prompt"]
    assert "<THRESHOLD_VALUES_USED>" not in captured["json"]["prompt"]
    assert result["model"] == "Ridge"


@pytest.mark.parametrize("threshold", [0.29, 0.71, float("nan"), True])
def test_agent_rejects_threshold_outside_safe_range(threshold):
    decision = {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": "Ridge",
        "hyperparameters": {"C": 1.0, "threshold": threshold},
    }

    with pytest.raises(ValueError, match="threshold"):
        logistic_agent_module.clean_json_output(json.dumps(decision))


def test_decision_to_configuration_preserves_threshold():
    decision = {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": "Ridge",
        "hyperparameters": {"C": 1.0, "threshold": 0.6},
    }

    assert logistic_agent_module.decision_to_configuration(decision) == {
        "penalty": "l2",
        "C": 1.0,
        "threshold": 0.6,
    }


def test_agent_requires_threshold():
    decision = {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": "Ridge",
        "hyperparameters": {"C": 1.0},
    }

    with pytest.raises(ValueError, match="threshold"):
        logistic_agent_module.clean_json_output(json.dumps(decision))
