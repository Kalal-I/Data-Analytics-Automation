import json

import pytest

from agents_code import classification_agent as classification_agent_module


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _recommendation(target="label", label_type="Categorical"):
    return {
        "schema_version": 1,
        "target_column": target,
        "label_type": label_type,
        "assessment": "suitable",
        "reasons": ["The target has multiple represented classes."],
        "requirements_to_address": [],
        "ranked_models": [
            {
                "rank": 1,
                "model_name": "LogisticRegression",
                "reasoning": ["A linear baseline is worth evaluating."],
                "supporting_evidence": [
                    {
                        "measure": "class count",
                        "observation": "The dataset report lists two classes.",
                        "implication": "A binary classifier can be evaluated.",
                    }
                ],
                "risks": ["The decision boundary may not be linear."],
                "recommended_checks": ["Compare weighted F1 in cross-validation."],
                "confidence": "medium",
            }
        ],
        "overall_assessment": {
            "main_findings": ["The report supports initial classification."],
            "key_uncertainties": ["Comparative CV results are unavailable."],
            "evaluation_priority": "Compare candidates with common folds.",
        },
    }


def test_agent_sends_dataset_context_and_validates_recommendation(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        classification_agent_module,
        "API_URL",
        "https://example.test/",
    )
    monkeypatch.setattr(
        classification_agent_module,
        "API_TOKEN",
        "test-token",
    )

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            {"response": json.dumps(_recommendation())}
        )

    monkeypatch.setattr(
        classification_agent_module.requests,
        "post",
        fake_post,
    )

    result = classification_agent_module.classification_agent(
        dataset_description={"row_count": 20},
        target_column="label",
        label_type="Categorical",
        statistics={"class_count": 2},
        row_count=20,
    )

    assert captured["url"] == "https://example.test/generate"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["json"]["think"] is False
    assert '"row_count": 20' in captured["json"]["prompt"]
    assert '"class_count": 2' in captured["json"]["prompt"]
    assert "label" in captured["json"]["prompt"]
    assert "Categorical" in captured["json"]["prompt"]
    assert "<DATASET_DESCRIPTION>" not in captured["json"]["prompt"]
    assert "<STATISTICS>" not in captured["json"]["prompt"]
    assert result == _recommendation()


def test_agent_does_not_duplicate_bearer_prefix(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        classification_agent_module,
        "API_URL",
        "https://example.test",
    )
    monkeypatch.setattr(
        classification_agent_module,
        "API_TOKEN",
        " Bearer test-token ",
    )
    monkeypatch.setattr(
        classification_agent_module.requests,
        "post",
        lambda _url, **kwargs: (
            captured.update(kwargs) or _FakeResponse(
                {"response": json.dumps(_recommendation())}
            )
        ),
    )

    classification_agent_module.classification_agent(
        dataset_description={},
        target_column="label",
        label_type="Categorical",
        statistics={},
    )

    assert captured["headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda result: result.pop("assessment"), "missing required fields"),
        (lambda result: result.update(assessment="not_suitable"), "invalid assessment"),
        (
            lambda result: result["ranked_models"][0].update(rank=2),
            "consecutive",
        ),
        (
            lambda result: result["ranked_models"][0].update(
                model_name="UnsupportedClassifier"
            ),
            "unsupported model name",
        ),
        (
            lambda result: result["ranked_models"][0].update(confidence="certain"),
            "confidence",
        ),
        (
            lambda result: result["overall_assessment"].pop("main_findings"),
            "overall_assessment",
        ),
    ],
)
def test_agent_rejects_responses_that_violate_prompt_schema(
    monkeypatch,
    mutate,
    message,
):
    result = _recommendation()
    mutate(result)
    monkeypatch.setattr(
        classification_agent_module,
        "API_URL",
        "https://example.test",
    )
    monkeypatch.setattr(
        classification_agent_module,
        "API_TOKEN",
        "test-token",
    )
    monkeypatch.setattr(
        classification_agent_module.requests,
        "post",
        lambda *_args, **_kwargs: _FakeResponse(
            {"response": json.dumps(result)}
        ),
    )

    with pytest.raises(ValueError, match=message):
        classification_agent_module.classification_agent(
            dataset_description={},
            target_column="label",
            label_type="Categorical",
            statistics={},
        )


def test_agent_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        classification_agent_module.clean_json_output("not-json")
