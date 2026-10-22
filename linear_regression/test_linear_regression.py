"""Tests for the linear regression training and 80/20 split helpers."""

import json
import inspect

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from linear_regression.fold_manager import (
    create_folds_and_train as _create_folds_and_train,
    create_or_load_folds,
)
from linear_regression.linear_model import train_linear_regression
from linear_regression.model_selection import (
    format_best_model_configuration,
    format_model_feature_weights,
    select_best_feature_report,
)
from linear_regression.results_json import split_result_to_json
from linear_regression.split_data import (
    _cross_validation_metrics,
    split_and_train as _split_and_train,
)


def split_and_train(
    data,
    target_column,
    model_type="ridge",
    alpha=1e-12,
    l1_ratio=None,
    **kwargs,
):
    return _split_and_train(
        data,
        target_column,
        model_type=model_type,
        alpha=alpha,
        l1_ratio=l1_ratio,
        **kwargs,
    )


def create_folds_and_train(
    data,
    target_column,
    model_type="ridge",
    alpha=1e-12,
    l1_ratio=None,
    **kwargs,
):
    return _create_folds_and_train(
        data,
        target_column,
        model_type=model_type,
        alpha=alpha,
        l1_ratio=l1_ratio,
        **kwargs,
    )


def _fit_features(data, target_column, features=None, model_type="ridge", alpha=1e-12):
    fit_data = data
    if features is not None:
        feature_names = [features] if isinstance(features, str) else list(features)
        fit_data = data[feature_names + [target_column]]
    return train_linear_regression(
        fit_data,
        target_column,
        model_type=model_type,
        alpha=alpha,
    )


def _sample_data():
    x = np.arange(20, dtype=float)
    z = np.arange(20, dtype=float) * 0.5
    return pd.DataFrame({"x": x, "z": z, "target": 3 * x - 2 * z + 5})


def _feature_report(features, r2, adjusted_r2, rmse, mae):
    return {
        "features": features,
        "training": {
            "cross_validation": {
                "r2": {"mean": r2, "std": 0.001},
                "adjusted_r2": {"mean": adjusted_r2, "std": 0.001},
                "rmse": {"mean": rmse, "std": 0.01},
                "mae": {"mean": mae, "std": 0.01},
            }
        },
    }


def test_feature_selection_obeys_r2_tolerance_then_adjusted_r2_and_errors():
    best_r2 = _feature_report(["a", "b", "c"], 0.91, 0.89, 2.0, 1.5)
    better_adjusted = _feature_report(["a", "b"], 0.906, 0.90, 2.1, 1.6)
    outside_tolerance = _feature_report(["a"], 0.904, 0.95, 1.0, 0.5)

    selected, ranking_metrics = select_best_feature_report(
        [best_r2, better_adjusted, outside_tolerance]
    )

    assert selected is better_adjusted
    assert ranking_metrics == [
        "r2_tolerance",
        "adjusted_r2",
        "rmse",
        "mae",
        "feature_count",
    ]


def test_feature_selection_uses_error_and_complexity_when_adjusted_r2_unavailable():
    smaller = _feature_report(["a"], 0.91, None, 2.0, 1.5)
    larger = _feature_report(["a", "b"], 0.909, 0.90, 1.9, 1.4)
    simpler_tie = _feature_report(["c"], 0.909, None, 1.9, 1.4)

    selected, ranking_metrics = select_best_feature_report(
        [smaller, larger, simpler_tie]
    )

    assert selected is simpler_tie
    assert ranking_metrics == [
        "r2_tolerance",
        "rmse",
        "mae",
        "feature_count",
    ]


@pytest.mark.parametrize(
    ("model_type", "model_name", "hyperparameters", "expected"),
    [
        ("lasso", "Lasso", {"alpha": 0.1}, "Best Model: Lasso\nAlpha: 0.1"),
        ("ridge", "Ridge", {"alpha": 1.0}, "Best Model: Ridge\nAlpha: 1.0"),
        (
            "elastic_net",
            "Elastic Net",
            {"alpha": 0.1, "l1_ratio": 0.5},
            "Best Model: Elastic Net\nAlpha: 0.1\nL1 Ratio: 0.5",
        ),
    ],
)
def test_best_model_configuration_output(
    model_type, model_name, hyperparameters, expected
):
    report = {
        "training": {
            "model_type": model_type,
            "model": model_name,
            "hyperparameters": hyperparameters,
        }
    }
    assert format_best_model_configuration(report) == expected


def test_best_model_feature_weights_include_names_and_fitted_values():
    report = {
        "training": {
            "coefficients": {
                "square_feet": 0.75,
                "bedrooms": -1.25,
            }
        }
    }
    assert format_model_feature_weights(report) == (
        "square_feet: 0.75\nbedrooms: -1.25"
    )


def test_feature_selection_excludes_unfittable_and_undefined_r2_reports():
    no_features = _feature_report([], 1.0, 1.0, 0.0, 0.0)
    undefined_r2 = _feature_report(["a"], None, None, 0.0, 0.0)
    valid = _feature_report(["a", "b"], 0.9, None, 2.0, 1.0)

    selected, _ = select_best_feature_report([no_features, undefined_r2, valid])

    assert selected is valid


def test_training_defaults_to_all_predictors_and_returns_fitted_model():
    """Training uses every column other than the target as a predictor."""
    model = train_linear_regression(
        _sample_data(), "target", model_type="ridge", alpha=1e-12
    )

    assert isinstance(model, Ridge)
    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["x", "z"]
    assert model.predict(pd.DataFrame({"x": [2.0], "z": [1.0]}))[0] == pytest.approx(9)


def test_training_function_has_no_features_parameter():
    assert "features" not in inspect.signature(train_linear_regression).parameters


@pytest.mark.parametrize(
    ("model_type", "alpha", "l1_ratio", "expected_type"),
    [
        ("lasso", 0.1, None, Lasso),
        ("ridge", 0.1, None, Ridge),
        ("elastic_net", 0.1, 0.5, ElasticNet),
    ],
)
def test_training_fits_supported_regularized_models(
    model_type, alpha, l1_ratio, expected_type
):
    model = train_linear_regression(
        _sample_data(),
        "target",
        model_type=model_type,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    assert isinstance(model, expected_type)
    assert model.training_row_count_ == 20


@pytest.mark.parametrize("model_type", ["lasso", "ridge"])
def test_lasso_and_ridge_do_not_receive_l1_ratio(model_type):
    model = train_linear_regression(
        _sample_data(),
        "target",
        model_type=model_type,
        alpha=0.1,
        l1_ratio=0.5,
    )
    assert "l1_ratio" not in model.get_params()


def test_training_rejects_unsupported_model_and_invalid_regularization():
    data = _sample_data()
    with pytest.raises(ValueError, match="model_type must be"):
        train_linear_regression(data, "target", model_type="ordinary", alpha=1.0)
    for alpha in (0, -1):
        with pytest.raises(ValueError, match="alpha must be"):
            train_linear_regression(data, "target", model_type="ridge", alpha=alpha)
    with pytest.raises(ValueError, match="required for Elastic Net"):
        train_linear_regression(data, "target", model_type="elastic_net", alpha=1.0)
    for l1_ratio in (0, -0.1, 1.1):
        with pytest.raises(ValueError, match="l1_ratio"):
            train_linear_regression(
                data,
                "target",
                model_type="elastic_net",
                alpha=1.0,
                l1_ratio=l1_ratio,
            )


@pytest.mark.parametrize(
    ("model_type", "alpha", "l1_ratio", "expected_type", "expected_report"),
    [
        ("lasso", 0.1, None, Lasso, {"alpha": 0.1}),
        ("ridge", 0.25, None, Ridge, {"alpha": 0.25}),
        (
            "elastic_net",
            0.25,
            0.4,
            ElasticNet,
            {"alpha": 0.25, "l1_ratio": 0.4},
        ),
    ],
)
def test_split_forwards_model_configuration_to_fitted_model(
    model_type, alpha, l1_ratio, expected_type, expected_report
):
    result = split_and_train(
        _sample_data(),
        "target",
        model_type=model_type,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    model = result[0]
    assert isinstance(model, expected_type)
    assert model.alpha == pytest.approx(alpha)
    if model_type == "elastic_net":
        assert model.l1_ratio == pytest.approx(l1_ratio)

    report = json.loads(split_result_to_json(result, "target"))
    assert report["training"]["hyperparameters"] == expected_report
    assert all(
        test["hyperparameters"] == expected_report
        for test in report["feature_removal_tests"]
    )


def test_split_trains_on_eighty_percent_and_returns_twenty_percent_holdout():
    """Fit and CV use the 16-row training split; four rows remain for testing."""
    model, x_test, y_test, metrics = split_and_train(
        _sample_data(), "target"
    )

    assert isinstance(model, Ridge)
    assert model.n_features_in_ == 2
    assert len(x_test) == 4
    assert len(y_test) == 4
    assert x_test.index.equals(y_test.index)
    assert set(x_test.columns) == {"x", "z"}
    assert set(metrics) == {"r2", "adjusted_r2", "mae", "rmse"}
    for metric in metrics.values():
        assert len(metric["fold_scores"]) == 5
        if any(score is None for score in metric["fold_scores"]):
            assert metric["mean"] is None
            assert metric["std"] is None
            continue
        assert metric["mean"] == pytest.approx(
            np.mean(metric["fold_scores"])
        )
        assert metric["std"] == pytest.approx(
            np.std(metric["fold_scores"], ddof=0)
        )
    assert metrics["r2"]["mean"] == pytest.approx(1.0)
    assert metrics["adjusted_r2"]["mean"] is None
    assert metrics["adjusted_r2"]["std"] is None
    assert any(score is None for score in metrics["adjusted_r2"]["fold_scores"])
    assert metrics["mae"]["mean"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["rmse"]["mean"] == pytest.approx(0.0, abs=1e-10)


def test_split_uses_only_the_requested_feature_columns():
    """Feature selection is respected for training and holdout predictors."""
    model, x_test, _, _ = split_and_train(
        _sample_data(), "target", features=["z"]
    )

    assert list(model.feature_names_in_) == ["z"]
    assert list(x_test.columns) == ["z"]


def test_training_does_not_modify_input_dataframe():
    """Training preserves the caller's data and original index."""
    data = _sample_data()
    data.index = range(100, 120)
    original = data.copy(deep=True)

    train_linear_regression(data, "target", model_type="ridge", alpha=1e-12)

    assert_frame_equal(data, original)


@pytest.mark.parametrize(
    ("data", "target", "error", "message"),
    [
        ({"x": [1], "target": [2]}, "target", TypeError, "pandas DataFrame"),
        (pd.DataFrame({"x": [1]}), "target", ValueError, "not found"),
        (
            pd.DataFrame({"x": ["red", "blue"], "target": [1, 2]}),
            "target",
            ValueError,
            "numeric values",
        ),
        (
            pd.DataFrame({"x": [1, np.nan], "target": [2, 3]}),
            "target",
            ValueError,
            "missing or non-finite",
        ),
    ],
)
def test_training_rejects_invalid_inputs(data, target, error, message):
    """Invalid input shapes, columns, categories, and missing values fail clearly."""
    with pytest.raises(error, match=message):
        train_linear_regression(data, target, model_type="ridge", alpha=1e-12)


def test_training_rejects_target_as_predictor():
    """The target is excluded from the predictor columns."""
    model = train_linear_regression(
        _sample_data(), "target", model_type="ridge", alpha=1e-12
    )
    assert "target" not in model.feature_names_in_


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (
            pd.DataFrame({"x": [1 + 2j, 2 + 3j], "target": [2, 3]}),
            ValueError,
            "complex",
        ),
        (
            pd.DataFrame({"x": [1, np.inf], "target": [2, 3]}),
            ValueError,
            "missing or non-finite",
        ),
    ],
)
def test_training_rejects_invalid_numeric_values(data, error, message):
    with pytest.raises(error, match=message):
        train_linear_regression(data, "target", model_type="ridge", alpha=1e-12)


def test_training_rejects_duplicate_dataframe_columns():
    data = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "target"])

    with pytest.raises(ValueError, match="column names must be unique"):
        train_linear_regression(data, "target", model_type="ridge", alpha=1e-12)


def test_split_rejects_single_row_data():
    """An 80/20 split requires at least two input rows."""
    with pytest.raises(ValueError, match="At least two rows"):
        split_and_train(pd.DataFrame({"x": [1], "target": [2]}), "target")


def test_cross_validation_requires_two_observations_per_validation_fold():
    """Reject small training partitions where R^2 cannot be computed per fold."""
    data = pd.DataFrame(
        {"x": np.arange(10, dtype=float), "target": np.arange(10, dtype=float)}
    )

    with pytest.raises(ValueError, match="At least 10 training rows"):
        split_and_train(data, "target")


def test_cross_validation_rejects_a_constant_target_validation_fold():
    data = pd.DataFrame(
        {"x": np.arange(10, dtype=float), "target": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    )
    validation_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    folds = [
        {
            "train_indices": [
                index for index in range(len(data)) if index not in validation
            ],
            "validation_indices": validation,
        }
        for validation in validation_groups
    ]

    with pytest.raises(ValueError, match="undefined R\\^2.*constant target"):
        _cross_validation_metrics(
            data,
            "target",
            None,
            "ridge",
            1e-12,
            None,
            42,
            cv_folds=folds,
        )


def test_cross_validation_scores_are_deterministic():
    """A fixed split seed also makes the five CV folds and scores reproducible."""
    data = _sample_data()

    first = split_and_train(data, "target", random_state=7)[3]
    second = split_and_train(data, "target", random_state=7)[3]

    assert first == second


def test_split_does_not_modify_input_dataframe():
    """The split and cross-validation leave source data and index untouched."""
    data = _sample_data()
    data.index = range(100, 120)
    original = data.copy(deep=True)

    split_and_train(data, "target")

    assert_frame_equal(data, original)


def test_cv_and_final_fit_never_use_holdout_rows(monkeypatch):
    """Every CV/final model fit is restricted to the outer training rows."""
    from linear_regression import split_data

    data = _sample_data()
    data.index = pd.Index(range(100, 120), name="source_row")
    expected_test_index = data.iloc[
        split_data.train_test_split(
            np.arange(len(data)), test_size=0.2, random_state=42
        )[1]
    ].index
    original_fit = Ridge.fit
    fitted_row_indexes = []

    def recording_fit(model, predictors, target, *args, **kwargs):
        fitted_row_indexes.append(set(predictors.index))
        return original_fit(model, predictors, target, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", recording_fit)
    split_data.split_and_train(
        data, "target", model_type="ridge", alpha=1e-12
    )

    assert len(fitted_row_indexes) == 6 + (5 * 2)
    assert all(not set(expected_test_index).intersection(rows) for rows in fitted_row_indexes)


def test_saved_folds_are_reused_across_feature_selection_calls(tmp_path):
    """Different feature sets share identical persisted split and CV assignments."""
    data = _sample_data()
    fold_file = tmp_path / "stable_folds.json"

    first = create_folds_and_train(
        data, "target", features=["x"], folds_file=fold_file
    )
    first_manifest = create_or_load_folds(data, fold_file)
    second = create_folds_and_train(
        data, "target", features=["z"], random_state=999, folds_file=fold_file
    )
    second_manifest = create_or_load_folds(data, fold_file)

    assert fold_file.exists()
    assert first_manifest == second_manifest
    assert first_manifest["train_indices"] != first_manifest["test_indices"]
    assert len(first_manifest["cv_folds"]) == 5
    assert len(first[1]) == len(second[1]) == 4
    assert first[1].index.tolist() == second[1].index.tolist()
    assert list(first[0].feature_names_in_) == ["x"]
    assert list(second[0].feature_names_in_) == ["z"]


def test_saved_folds_reject_different_or_reordered_data(tmp_path):
    """Persisted assignments cannot silently be applied to another dataset."""
    data = _sample_data()
    fold_file = tmp_path / "stable_folds.json"
    create_or_load_folds(data, fold_file)

    changed_data = data.copy()
    changed_data.loc[0, "x"] = -100
    with pytest.raises(ValueError, match="different data or row ordering"):
        create_or_load_folds(changed_data, fold_file)

    with pytest.raises(ValueError, match="different data or row ordering"):
        split_and_train(data.iloc[::-1], "target", folds=create_or_load_folds(data, fold_file))


def test_saved_folds_fingerprint_includes_index_metadata(tmp_path):
    data = _sample_data()
    fold_file = tmp_path / "folds.json"
    create_or_load_folds(data, fold_file)

    renamed_index = data.copy()
    renamed_index.index.name = "source_row"

    with pytest.raises(ValueError, match="different data or row ordering"):
        create_or_load_folds(renamed_index, fold_file)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda folds: folds.update(schema_version=2), "schema version"),
        (lambda folds: folds.update(random_state=True), "random_state"),
        (
            lambda folds: folds["cv_folds"][0]["validation_indices"].append(16),
            "CV fold positions are invalid",
        ),
        (
            lambda folds: folds["train_indices"].pop(),
            "partition all rows",
        ),
    ],
)
def test_existing_fold_file_rejects_invalid_manifest(tmp_path, change, message):
    data = _sample_data()
    fold_file = tmp_path / "folds.json"
    folds = create_or_load_folds(data, fold_file)
    change(folds)
    fold_file.write_text(json.dumps(folds), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        create_or_load_folds(data, fold_file)


def test_malformed_existing_fold_file_is_not_replaced(tmp_path):
    data = _sample_data()
    fold_file = tmp_path / "folds.json"
    original = "{not valid JSON"
    fold_file.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read saved fold data"):
        create_or_load_folds(data, fold_file)

    assert fold_file.read_text(encoding="utf-8") == original


def test_existing_unreadable_fold_path_raises_clear_error(tmp_path):
    data = _sample_data()
    fold_path = tmp_path / "folds-directory"
    fold_path.mkdir()

    with pytest.raises(ValueError, match="Could not read saved fold data"):
        create_or_load_folds(data, fold_path)


def test_split_result_serializes_model_cv_and_holdout_metrics_as_strict_json(tmp_path):
    """JSON report includes selected predictors, fold scores, and holdout results."""
    data = _sample_data()
    fold_file = tmp_path / "folds.json"
    result = create_folds_and_train(
        data, "target", features=["x"], folds_file=fold_file
    )
    folds = create_or_load_folds(data, fold_file)
    serialized = split_result_to_json(
        result, "target", folds=folds, include_holdout_stats=True
    )

    report = json.loads(
        serialized,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert report["features"] == ["x"]
    assert report["target_column"] == "target"
    assert list(report)[-1] == "feature_removal_tests"
    assert report["data_split"]["training_row_count"] == 16
    assert report["data_split"]["holdout_row_positions"] == folds["test_indices"]
    assert len(report["data_split"]["cross_validation_folds"]) == 5
    assert report["training"]["model"] == "Ridge"
    assert report["training"]["model_type"] == "ridge"
    assert report["training"]["hyperparameters"] == {"alpha": 1e-12}
    assert len(report["training"]["cross_validation"]["r2"]["fold_scores"]) == 5
    assert len(report["feature_removal_tests"]) == 1
    removal_test = report["feature_removal_tests"][0]
    assert removal_test["removed_feature"] == "x"
    assert removal_test["remaining_features"] == []
    assert removal_test["cross_validation"] == {
        "r2": {"mean": None, "std": None},
        "adjusted_r2": {"mean": None, "std": None},
        "mae": {"mean": None, "std": None},
        "rmse": {"mean": None, "std": None},
    }
    assert removal_test["limitation"] == "cannot_fit_without_predictor_features"
    assert report["holdout"]["row_count"] == 4
    assert set(report["holdout"]["metrics"]) == {
        "r2",
        "adjusted_r2",
        "mae",
        "rmse",
    }
    assert len(report["holdout"]["rows"]) == 4
    predictions = result[0].predict(result[1])
    for row, index, actual, prediction in zip(
        report["holdout"]["rows"],
        result[2].index,
        result[2].to_numpy(),
        predictions,
    ):
        assert row["index"] == {"type": "int", "value": int(index)}
        assert row["actual"] == pytest.approx(actual)
        assert row["predicted"] == pytest.approx(prediction)
    assert "NaN" not in serialized
    assert "Infinity" not in serialized


def test_feature_removal_tests_evaluate_each_feature_on_identical_folds():
    """JSON contains each ablation and compares its metrics to the full model."""
    result = split_and_train(
        _sample_data(),
        "target",
        features=["x", "z"],
        random_state=23,
    )

    report = json.loads(split_result_to_json(result, "target"))
    current_metrics = report["training"]["cross_validation"]
    tests = report["feature_removal_tests"]
    assert len(tests) == 2
    assert [item["removed_feature"] for item in tests] == ["x", "z"]
    assert tests[0]["remaining_features"] == ["z"]
    assert tests[1]["remaining_features"] == ["x"]
    assert all(test["model"] == "ridge" for test in tests)
    assert all(test["hyperparameters"] == {"alpha": 1e-12} for test in tests)
    assert set(tests[0]["cross_validation"]) == {
        "r2",
        "adjusted_r2",
        "mae",
        "rmse",
    }

    for item in tests:
        for metric in ("r2", "adjusted_r2", "mae", "rmse"):
            if (
                item["cross_validation"][metric]["mean"] is None
                or current_metrics[metric]["mean"] is None
            ):
                assert item["comparison_to_current"][metric + "_change"] is None
                continue
            assert item["cross_validation"][metric]["mean"] is not None
            assert item["cross_validation"][metric]["std"] >= 0
            assert item["comparison_to_current"][metric + "_change"] == pytest.approx(
                item["cross_validation"][metric]["mean"]
                - current_metrics[metric]["mean"]
            )


def test_adjusted_r2_is_computed_for_cross_validation_and_holdout():
    data = _sample_data()
    result = split_and_train(data, "target", features=["x"])

    cv_adjusted_r2 = result[3]["adjusted_r2"]
    assert cv_adjusted_r2["mean"] == pytest.approx(1.0)
    assert cv_adjusted_r2["std"] == pytest.approx(0.0)
    assert cv_adjusted_r2["fold_scores"] == pytest.approx([1.0] * 5)

    report = json.loads(
        split_result_to_json(result, "target", include_holdout_stats=True)
    )
    assert report["holdout"]["metrics"]["adjusted_r2"] == pytest.approx(1.0)


def test_json_omits_holdout_statistics_by_default_and_includes_training_count():
    result = split_and_train(_sample_data(), "target", features="x")

    report = json.loads(split_result_to_json(result, "target"))

    assert "holdout" not in report
    assert report["data_split"]["training_row_count"] == 16
    assert report["data_split"]["holdout_row_count"] == 4


def test_json_preserves_typed_index_values_and_computes_single_row_errors():
    data = _sample_data()
    model = _fit_features(data, "target", features="x")
    x_test = pd.DataFrame({"x": [2.0]}, index=pd.Index([7], dtype="int64"))
    y_test = pd.Series([12.0], index=x_test.index, name="target")
    cv_metrics = {
        "r2": {"mean": 1.0, "std": 0.0, "fold_scores": [1.0] * 5},
        "mae": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
        "rmse": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
    }

    report = json.loads(
        split_result_to_json(
            (model, x_test, y_test, cv_metrics),
            "target",
            include_holdout_stats=True,
        )
    )

    assert report["holdout"]["metrics"]["r2"] is None
    assert report["holdout"]["metrics"]["adjusted_r2"] is None
    assert report["holdout"]["metrics"]["mae"] == pytest.approx(3.0)
    assert report["holdout"]["metrics"]["rmse"] == pytest.approx(3.0)
    assert report["holdout"]["rows"][0]["index"] == {"type": "int", "value": 7}


def test_json_reports_null_metrics_for_empty_holdout():
    data = _sample_data()
    model = _fit_features(data, "target", features="x")
    x_test = pd.DataFrame({"x": pd.Series(dtype=float)})
    y_test = pd.Series(dtype=float, name="target")
    cv_metrics = {
        "r2": {"mean": 1.0, "std": 0.0, "fold_scores": [1.0] * 5},
        "mae": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
        "rmse": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
    }

    report = json.loads(
        split_result_to_json(
            (model, x_test, y_test, cv_metrics),
            "target",
            include_holdout_stats=True,
        )
    )

    assert report["holdout"]["metrics"] == {
        "r2": None,
        "adjusted_r2": None,
        "mae": None,
        "rmse": None,
    }
    assert report["holdout"]["rows"] == []


def test_json_reports_undefined_r2_and_defined_errors_for_constant_holdout_target():
    data = _sample_data()
    model = _fit_features(data, "target", features="x")
    x_test = pd.DataFrame({"x": [2.0, 3.0]}, index=[4, 5])
    y_test = pd.Series([7.0, 7.0], index=x_test.index, name="target")
    cv_metrics = {
        "r2": {"mean": 1.0, "std": 0.0, "fold_scores": [1.0] * 5},
        "mae": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
        "rmse": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
    }

    report = json.loads(
        split_result_to_json(
            (model, x_test, y_test, cv_metrics),
            "target",
            include_holdout_stats=True,
        )
    )

    assert report["holdout"]["metrics"]["r2"] is None
    assert report["holdout"]["metrics"]["adjusted_r2"] is None
    assert report["holdout"]["metrics"]["mae"] == pytest.approx(3.0)
    assert report["holdout"]["metrics"]["rmse"] == pytest.approx(np.sqrt(10))


def test_json_preserves_multiindex_type_information_and_rejects_unsupported_index():
    data = _sample_data()
    model = _fit_features(data, "target", features="x")
    x_test = pd.DataFrame(
        {"x": [2.0, 3.0]},
        index=pd.MultiIndex.from_tuples([(1, "1"), ("1", 1)]),
    )
    y_test = pd.Series([11.0, 14.0], index=x_test.index, name="target")
    cv_metrics = {
        "r2": {"mean": 1.0, "std": 0.0, "fold_scores": [1.0] * 5},
        "mae": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
        "rmse": {"mean": 0.0, "std": 0.0, "fold_scores": [0.0] * 5},
    }

    report = json.loads(
        split_result_to_json(
            (model, x_test, y_test, cv_metrics),
            "target",
            include_holdout_stats=True,
        )
    )
    assert report["holdout"]["rows"][0]["index"] == {
        "type": "tuple",
        "value": [{"type": "int", "value": 1}, {"type": "str", "value": "1"}],
    }
    assert report["holdout"]["rows"][1]["index"] == {
        "type": "tuple",
        "value": [{"type": "str", "value": "1"}, {"type": "int", "value": 1}],
    }

    unsupported_x = pd.DataFrame({"x": [2.0]}, index=[object()])
    unsupported_y = pd.Series([11.0], index=unsupported_x.index, name="target")
    with pytest.raises(TypeError, match="Unsupported index label type"):
        split_result_to_json(
            (model, unsupported_x, unsupported_y, cv_metrics),
            "target",
            include_holdout_stats=True,
        )


def test_json_rejects_non_finite_cross_validation_metrics():
    data = _sample_data()
    result = split_and_train(data, "target", features="x")
    result[3]["r2"]["fold_scores"][0] = float("nan")

    with pytest.raises(ValueError, match="strict JSON"):
        split_result_to_json(result, "target")
