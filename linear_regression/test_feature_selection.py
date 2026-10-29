import math

import pytest

from linear_regression.feature_selection import (
    R2_IMPROVEMENT_THRESHOLD,
    greedy_coefficient_feature_selection,
    rank_features_by_coefficient,
)


def _metrics(r2, adjusted_r2=None, rmse=1.0, mae=0.5):
    return {
        "r2": {"mean": r2, "std": 0.01},
        "adjusted_r2": {
            "mean": adjusted_r2,
            "std": 0.02 if adjusted_r2 is not None else None,
        },
        "rmse": {"mean": rmse, "std": 0.03},
        "mae": {"mean": mae, "std": 0.04},
    }


def test_ranks_coefficients_by_absolute_magnitude_and_reports_invalid_values():
    ranked, invalid = rank_features_by_coefficient(
        {
            "negative": -3.5,
            "zero": 0.0,
            "positive": 2.0,
            "nan": math.nan,
            "infinite": math.inf,
            "text": "not a coefficient",
        }
    )

    assert [item["feature"] for item in ranked] == [
        "negative",
        "positive",
        "zero",
    ]
    assert [item["absolute_coefficient"] for item in ranked] == [3.5, 2.0, 0.0]
    assert [item["feature"] for item in invalid] == [
        "nan",
        "infinite",
        "text",
    ]
    assert [item["reason"] for item in invalid] == [
        "coefficient is not finite",
        "coefficient is not finite",
        "coefficient is not a real number",
    ]


def test_greedy_selection_accepts_threshold_boundary_and_excludes_rejected_features():
    evaluated = []

    def evaluate(features):
        evaluated.append(features.copy())
        scores = {
            ("A",): 0.70,
            ("A", "B"): 0.705,
            ("A", "B", "C"): 0.709,
            ("A", "B", "D"): 0.72,
        }
        return _metrics(scores[tuple(features)], adjusted_r2=0.6)

    result = greedy_coefficient_feature_selection(
        {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        evaluate,
    )

    assert R2_IMPROVEMENT_THRESHOLD == 0.005
    assert evaluated == [
        ["A"],
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "D"],
    ]
    assert [item["decision"] for item in result["evaluations"]] == [
        "accepted",
        "accepted",
        "rejected",
        "accepted",
    ]
    assert result["evaluations"][1]["r2_improvement"] >= 0.005
    assert result["evaluations"][2]["candidate_features"] == ["A", "B", "C"]
    assert result["evaluations"][3]["candidate_features"] == ["A", "B", "D"]
    assert result["accepted_features"] == ["A", "B", "D"]
    assert result["accepted_cross_validation"]["r2"]["mean"] == 0.72


def test_greedy_selection_rejects_improvement_below_threshold():
    def evaluate(features):
        return _metrics(
            {"A": 0.8, "A,B": 0.8049}[",".join(features)],
        )

    result = greedy_coefficient_feature_selection(
        {"A": 2.0, "B": 1.0},
        evaluate,
    )

    assert result["evaluations"][1]["r2_improvement"] < 0.005
    assert result["evaluations"][1]["reason"] == (
        "R² improvement is below the threshold"
    )
    assert result["accepted_features"] == ["A"]


def test_failed_candidate_is_reported_and_next_feature_can_start_subset():
    attempted = []

    def evaluate(features):
        attempted.append(features.copy())
        if features == ["A"]:
            raise ValueError("candidate cannot be evaluated")
        return _metrics(0.6)

    result = greedy_coefficient_feature_selection(
        {"A": 2.0, "B": 1.0},
        evaluate,
    )

    assert attempted == [["A"], ["B"]]
    assert result["evaluations"][0]["decision"] == "rejected"
    assert "candidate cannot be evaluated" in result["evaluations"][0]["reason"]
    assert result["evaluations"][1]["decision"] == "accepted"
    assert result["evaluations"][1]["r2_improvement"] is None
    assert result["accepted_features"] == ["B"]


def test_greedy_selection_includes_all_required_metrics_and_reason():
    result = greedy_coefficient_feature_selection(
        {"feature": 1.0},
        lambda _: _metrics(0.75, adjusted_r2=0.7, rmse=0.8, mae=0.6),
    )

    candidate = result["evaluations"][0]
    assert candidate["feature_count"] == 1
    assert candidate["cross_validation"] == {
        "r2": {"mean": 0.75, "std": 0.01},
        "adjusted_r2": {"mean": 0.7, "std": 0.02},
        "rmse": {"mean": 0.8, "std": 0.03},
        "mae": {"mean": 0.6, "std": 0.04},
    }
    assert candidate["decision"] == "accepted"
    assert candidate["reason"] == (
        "initial feature accepted because evaluation succeeded"
    )


def test_greedy_selection_validates_threshold_and_coefficient_mapping():
    with pytest.raises(ValueError, match="finite and non-negative"):
        greedy_coefficient_feature_selection(
            {"feature": 1.0},
            lambda _: _metrics(0.5),
            r2_improvement_threshold=-0.01,
        )
    with pytest.raises(TypeError, match="coefficients must be a mapping"):
        rank_features_by_coefficient([("feature", 1.0)])
