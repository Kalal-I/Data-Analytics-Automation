import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import dataset_stats.model_statistics_classification as statistics_module


def _report(data, **kwargs):
    return json.loads(
        statistics_module.generate_model_statistics_classification(
            data,
            "target",
            **kwargs,
        )
    )


def test_finite_decimal_target_labels_are_accepted():
    report = _report(
    pd.DataFrame(
        {
            "target": [Decimal("1.0"), Decimal("2.0")] * 10,
        }
    )
    )

    assert report["target"]["class_count"] == 2
    assert sum(report["target"]["class_counts"].values()) == 20


def _balanced_labels(count):
    return ["no", "yes"] * (count // 2) + (["no"] if count % 2 else [])


def test_numeric_strings_are_converted_and_invalid_values_become_missing():
    count = 30
    numeric = [str(value) for value in range(count)]
    numeric[0] = "not-a-number"
    numeric[1] = "inf"
    numeric[2] = None
    report = _report(
        pd.DataFrame(
            {
                "numeric": numeric,
                "target": _balanced_labels(count),
            }
        )
    )

    feature = report["features"][0]
    assert feature["feature_type"] == "numeric"
    assert feature["mean"] == pytest.approx(
        np.mean([value for value in range(count) if value >= 3])
    )
    assert feature["invalid_count"] == 1
    assert feature["non_finite_count"] == 1
    assert feature["missing_count"] == 3
    assert report["baseline_model_evaluation"]["models"][
        "LogisticRegression"
    ]["status"] == "evaluated"


def test_numeric_string_features_use_numeric_values_for_pearson_correlation():
    values = np.arange(30)
    report = _report(
        pd.DataFrame(
            {
                "first": values.astype(str),
                "second": (values * 3 + 2).astype(str),
                "target": _balanced_labels(len(values)),
            }
        )
    )

    pair = report["feature_feature_associations"][0]
    assert pair["value"] == pytest.approx(1.0)
    assert pair["valid_observation_count"] == len(values)
    assert report["multicollinearity"]["highly_correlated_pairs"][0][
        "correlation"
    ] == pytest.approx(1.0)


def test_cv_results_survive_holdout_failure_and_holdout_split_is_shared(
    monkeypatch,
):
    split_calls = []

    def failing_holdout_split(*args, **kwargs):
        split_calls.append((args, kwargs))
        raise ValueError("holdout split failed")

    monkeypatch.setattr(
        statistics_module,
        "train_test_split",
        failing_holdout_split,
    )
    data = pd.DataFrame(
        {
            "feature": np.arange(30),
            "target": _balanced_labels(30),
        }
    )
    report = _report(data)

    assert len(split_calls) == 1
    for model in report["baseline_model_evaluation"]["models"].values():
        assert model["status"] == "evaluated"
        assert model["metrics"]["accuracy"]["mean"] is not None
        assert model["fold_metrics"]
        assert model["holdout_metrics"] is None
        assert "holdout split failed" in model["holdout_failure_reason"]
    assert report["baseline_model_evaluation"]["evaluation"][
        "outer_holdout"
    ]["status"] == "not_available"


def test_holdout_fit_failure_is_separate_from_successful_cv(monkeypatch):
    original_fit = statistics_module.fit_logistic_model
    call_count = 0

    def fail_on_holdout(X, y, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 6:
            raise RuntimeError("holdout fit failed")
        return original_fit(X, y, **kwargs)

    monkeypatch.setattr(
        statistics_module,
        "fit_logistic_model",
        fail_on_holdout,
    )
    report = _report(
        pd.DataFrame(
            {
                "feature": np.arange(30),
                "target": _balanced_labels(30),
            }
        )
    )
    logistic = report["baseline_model_evaluation"]["models"][
        "LogisticRegression"
    ]

    assert call_count == 6
    assert logistic["status"] == "evaluated"
    assert logistic["metrics"]["accuracy"]["mean"] is not None
    assert logistic["holdout_metrics"] is None
    assert "holdout fit failed" in logistic["holdout_failure_reason"]


def test_numeric_and_string_target_labels_remain_distinct():
    target = pd.Series([1, "1"] * 15, dtype=object, name="target")
    report = _report(pd.DataFrame({"target": target}))
    target_report = report["target"]

    assert target_report["class_count"] == 2
    assert len(target_report["class_counts"]) == 2
    assert sorted(target_report["class_counts"].values()) == [15, 15]
    assert sorted(target_report["class_percentages"].values()) == [
        50.0,
        50.0,
    ]
    assert target_report["imbalance_ratio"] == pytest.approx(1.0)
    assert target_report["entropy_bits"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "complex_value",
    [1 + 0j, complex(1, 2)],
)
def test_complex_target_values_are_rejected(complex_value):
    with pytest.raises(ValueError, match="target contains"):
        _report(
            pd.DataFrame(
                {
                    "feature": range(4),
                    "target": [0, 1, complex_value, 0],
                }
            )
        )


def test_target_only_data_reports_statistics_and_clear_baseline_failure():
    report = _report(
        pd.DataFrame({"target": _balanced_labels(20)})
    )

    assert report["dataset"]["predictor_feature_count"] == 0
    assert report["target"]["class_count"] == 2
    evaluation = report["baseline_model_evaluation"]
    assert evaluation["evaluation"]["status"] == "failed"
    assert evaluation["evaluation"]["reason"] == (
        "no predictor features are available"
    )
    assert all(
        model["status"] == "failed"
        and model["failure_reason"] == "no predictor features are available"
        for model in evaluation["models"].values()
    )


def test_mixed_categorical_values_are_encoded_safely():
    data = pd.DataFrame(
        {
            "category": [1, "1", "other", "text", None] * 6,
            "target": _balanced_labels(30),
        }
    )
    report = _report(data)

    assert report["features"][0]["feature_type"] == "categorical"
    assert report["feature_target_associations"][0]["method"] == (
        "chi-square test of independence"
    )
    assert all(
        model["status"] == "evaluated"
        for model in report["baseline_model_evaluation"]["models"].values()
    )


def test_missing_category_sentinel_cannot_collide_with_real_category():
    encoded = statistics_module._encode_categorical_matrix(
        np.array([["__classification_missing__"], [None]], dtype=object)
    )

    assert encoded[0, 0] != encoded[1, 0]
    assert encoded[1, 0] == statistics_module._MISSING_CATEGORY


def test_one_hot_encoder_ignores_categories_unseen_during_fit():
    preprocessor = statistics_module._numeric_pipeline(
        [],
        ["category"],
        scale_numeric=False,
    )
    training = pd.DataFrame({"category": [1, "known", None]})
    holdout = pd.DataFrame({"category": ["unseen"]})
    preprocessor.fit_transform(training)

    encoded = preprocessor.transform(holdout)

    assert encoded.shape[0] == 1


def test_default_logistic_and_random_forest_baselines_remain_evaluated():
    values = np.arange(30)
    report = _report(
        pd.DataFrame(
            {
                "feature": values,
                "category": ["A", "B"] * 15,
                "target": _balanced_labels(len(values)),
            }
        )
    )
    models = report["baseline_model_evaluation"]["models"]

    assert models["LogisticRegression"]["status"] == "evaluated"
    assert models["RandomForestClassifier"]["status"] == "evaluated"
    assert models["LogisticRegression"]["metrics"]["accuracy"]["mean"] is not None
    assert models["RandomForestClassifier"]["metrics"]["accuracy"]["mean"] is not None
    assert set(models["LogisticRegression"]["fold_metrics"][0]["labels"]) == {
        "no",
        "yes",
    }


def test_json_output_is_finite_and_valid_with_nonfinite_numeric_features():
    data = pd.DataFrame(
        {
            "feature": [str(value) for value in range(30)],
            "target": _balanced_labels(30),
        }
    )
    data.loc[0, "feature"] = "inf"
    data.loc[1, "feature"] = "nan"

    serialized = statistics_module.generate_model_statistics_classification(
        data,
        "target",
    )
    parsed = json.loads(
        serialized,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )

    assert parsed["features"][0]["non_finite_count"] >= 1
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
