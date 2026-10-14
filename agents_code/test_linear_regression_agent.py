import pytest

from agents_code import linear_regression_agent as regression_agent_module


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_agent_replaces_dict_context_and_returns_ranked_models(monkeypatch):
    captured = {}
    monkeypatch.setattr(regression_agent_module, "API_URL", "https://example.test/")
    monkeypatch.setattr(regression_agent_module, "API_TOKEN", "test-token")

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            {
                "response": (
                    '{"assessment": "suitable", "ranked_models": '
                    '[{"model_name": "LinearRegression"}]}'
                )
            }
        )

    monkeypatch.setattr(regression_agent_module.requests, "post", fake_post)
    result = regression_agent_module.linear_regression_agent(
        dataset_description={"row_count": 100},
        target_column="Price_USD",
        label_type="continuous numeric",
        statistics={"missing_values": 0, "target_mean": 250000},
        row_count=100,
    )

    prompt = captured["json"]["prompt"]
    assert captured["url"] == "https://example.test/generate"
    assert '"row_count": 100' in prompt
    assert "Price_USD" in prompt
    assert "continuous numeric" in prompt
    assert '"missing_values": 0' in prompt
    for model_name in (
        "LinearRegression",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "KNeighborsRegressor",
        "GradientBoostingRegressor",
        "Ridge",
        "Lasso",
    ):
        assert model_name in prompt
    assert "feature_columns" not in prompt
    assert "TRAINING SETTINGS" not in prompt
    assert "<DATASET_DESCRIPTION>" not in prompt
    assert "<STATISTICS>" not in prompt
    assert "<TARGET_COLUMN>" not in prompt
    assert result == {
        "assessment": "suitable",
        "ranked_models": [{"model_name": "LinearRegression"}],
    }


def test_agent_accepts_structured_dataset_description(monkeypatch):
    monkeypatch.setattr(regression_agent_module, "API_URL", "https://example.test")
    monkeypatch.setattr(regression_agent_module, "API_TOKEN", "test-token")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            {
                "response": (
                    '{"assessment": "insufficient_evidence", "ranked_models": []}'
                )
            }
        )

    monkeypatch.setattr(regression_agent_module.requests, "post", fake_post)
    result = regression_agent_module.linear_regression_agent(
        dataset_description={"columns": ["x", "target"]},
        target_column="target",
        label_type="numeric",
        statistics={"target": {"valid_numeric_count": 30}},
    )

    assert captured["url"] == "https://example.test/generate"
    assert '"columns": ["x", "target"]' in captured["json"]["prompt"]
    assert result["assessment"] == "insufficient_evidence"


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("dataset_description", '{"columns": []}', "dataset_description"),
        ("statistics", '{"count": 1}', "statistics"),
    ],
)
def test_agent_requires_dictionary_context(monkeypatch, argument, value, message):
    monkeypatch.setattr(regression_agent_module, "API_URL", "https://example.test")
    monkeypatch.setattr(regression_agent_module, "API_TOKEN", "test-token")
    with pytest.raises(TypeError, match=message):
        regression_agent_module.linear_regression_agent(
            dataset_description=(
                value if argument == "dataset_description" else {}
            ),
            target_column="target",
            label_type="numeric",
            statistics=value if argument == "statistics" else {},
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
def test_clean_json_output_rejects_invalid_responses(response, message):
    with pytest.raises(ValueError, match=message):
        regression_agent_module.clean_json_output(response)
