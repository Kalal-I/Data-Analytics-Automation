import pytest

from agents_code import logistic_regression_agent as logistic_agent_module


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_agent_returns_native_logistic_configuration_decision(monkeypatch):
    captured = {}
    monkeypatch.setattr(logistic_agent_module, "API_URL", "https://example.test/")
    monkeypatch.setattr(logistic_agent_module, "API_TOKEN", "test-token")

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({
            "response": (
                '{"schema_version": 1, "action": "evaluate_model", '
                '"model": "ElasticNet", "hyperparameters": '
                '{"C": 0.5, "l1_ratio": 0.25, "max_iter": 1000}}'
            )
        })

    monkeypatch.setattr(logistic_agent_module.requests, "post", fake_post)
    result = logistic_agent_module.logistic_regression_agent(
        {"cross_validation": {"f1": {"mean": 0.8}}},
        iteration_number=2,
        lasso_c_values_used=[0.1],
        ridge_c_values_used=[1.0],
        elastic_net_combinations_used=[{"C": 0.5, "l1_ratio": 0.25}],
    )

    prompt = captured["json"]["prompt"]
    assert captured["url"] == "https://example.test/generate"
    assert captured["headers"]["Authorization"] == "test-token"
    assert '"f1": {"mean": 0.8}' in prompt
    assert "Current iteration number: `2`" in prompt
    assert '"C": 0.5, "l1_ratio": 0.25' in prompt
    assert "<MODEL_OPTIMIZATION_REPORT>" not in prompt
    assert logistic_agent_module.decision_to_configuration(result) == {
        "penalty": "elasticnet",
        "C": 0.5,
        "l1_ratio": 0.25,
    }


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ('{"schema_version":1,"action":"evaluate_model","model":"Lasso","hyperparameters":{"C":0}}', "positive finite"),
        ('{"schema_version":1,"action":"evaluate_model","model":"Lasso","hyperparameters":{"C":0.25}}', "at most one decimal place"),
        ('{"schema_version":1,"action":"evaluate_model","model":"ElasticNet","hyperparameters":{"C":1}}', "requires l1_ratio"),
        ('{"schema_version":1,"action":"evaluate_model","model":"Ridge","hyperparameters":{"C":1,"l1_ratio":0.5}}', "must not include l1_ratio"),
        ('{"schema_version":1,"action":"evaluate_model","model":"Ridge","hyperparameters":{"C":1,"max_iter":0}}', "positive integer"),
    ],
)
def test_clean_json_output_rejects_invalid_configuration(response, message):
    with pytest.raises(ValueError, match=message):
        logistic_agent_module.clean_json_output(response)


@pytest.mark.parametrize("model, penalty", [("Lasso", "l1"), ("Ridge", "l2")])
def test_decision_to_configuration_supports_linear_penalties(model, penalty):
    decision = {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": model,
        "hyperparameters": {"C": 1.0, "max_iter": 1000},
    }
    assert logistic_agent_module.decision_to_configuration(decision) == {
        "penalty": penalty,
        "C": 1.0,
    }


def test_decision_to_configuration_preserves_nondefault_max_iter():
    decision = {
        "schema_version": 1,
        "action": "evaluate_model",
        "model": "Lasso",
        "hyperparameters": {"C": 1.0, "max_iter": 1500},
    }
    assert logistic_agent_module.decision_to_configuration(decision) == {
        "penalty": "l1",
        "C": 1.0,
        "max_iter": 1500,
    }
