"""Tests for the linear regression training and 80/20 split helpers."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression

from linear_regression.fold_manager import (
    create_folds_and_train,
    create_or_load_folds,
)
from linear_regression.linear_model import train_linear_regression
from linear_regression.results_json import split_result_to_json
from linear_regression.split_data import _cross_validation_metrics, split_and_train


def _sample_data():
    x = np.arange(20, dtype=float)
    z = np.arange(20, dtype=float) * 0.5
    return pd.DataFrame({"x": x, "z": z, "target": 3 * x - 2 * z + 5})


def test_training_defaults_to_all_predictors_and_returns_fitted_model():
    """Omitting a feature list fits a multiple regression on every predictor."""
    model = train_linear_regression(_sample_data(), "target")

    assert isinstance(model, LinearRegression)
    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["x", "z"]
    assert model.predict(pd.DataFrame({"x": [2.0], "z": [1.0]}))[0] == pytest.approx(9)


def test_training_accepts_a_selected_feature_subset():
    """The requested columns alone are used to fit the model."""
    model = train_linear_regression(_sample_data(), "target", features=["x"])

    assert list(model.feature_names_in_) == ["x"]


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        ("z", ["z"]),
        (["z", "x"], ["z", "x"]),
        (("z", "x"), ["z", "x"]),
    ],
)
def test_training_preserves_string_list_and_tuple_feature_order(features, expected):
    model = train_linear_regression(_sample_data(), "target", features=features)

    assert list(model.feature_names_in_) == expected


def test_split_trains_on_eighty_percent_and_returns_twenty_percent_holdout():
    """Fit and CV use the 16-row training split; four rows remain for testing."""
    model, x_test, y_test, metrics = split_and_train(
        _sample_data(), "target"
    )

    assert isinstance(model, LinearRegression)
    assert model.n_features_in_ == 2
    assert len(x_test) == 4
    assert len(y_test) == 4
    assert x_test.index.equals(y_test.index)
    assert set(x_test.columns) == {"x", "z"}
    assert set(metrics) == {"r2", "mae", "rmse"}
    for metric in metrics.values():
        assert len(metric["fold_scores"]) == 5
        assert metric["mean"] == pytest.approx(
            np.mean(metric["fold_scores"])
        )
        assert metric["std"] == pytest.approx(
            np.std(metric["fold_scores"], ddof=0)
        )
    assert metrics["r2"]["mean"] == pytest.approx(1.0)
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

    train_linear_regression(data, "target")

    assert_frame_equal(data, original)


@pytest.mark.parametrize(
    ("data", "target", "features", "error", "message"),
    [
        ({"x": [1], "target": [2]}, "target", None, TypeError, "pandas DataFrame"),
        (pd.DataFrame({"x": [1]}), "target", None, ValueError, "not found"),
        (pd.DataFrame({"x": [1], "target": [2]}), "target", ["missing"], ValueError, "not found"),
        (pd.DataFrame({"x": ["red", "blue"], "target": [1, 2]}), "target", None, ValueError, "numeric values"),
        (pd.DataFrame({"x": [1, np.nan], "target": [2, 3]}), "target", None, ValueError, "missing or non-finite"),
    ],
)
def test_training_rejects_invalid_inputs(data, target, features, error, message):
    """Invalid input shapes, columns, categories, and missing values fail clearly."""
    with pytest.raises(error, match=message):
        train_linear_regression(data, target, features)


def test_training_rejects_target_as_predictor():
    """The target cannot be leaked into the predictor set."""
    with pytest.raises(ValueError, match="cannot also be used"):
        train_linear_regression(_sample_data(), "target", features=["target"])


@pytest.mark.parametrize(
    ("data", "features", "error", "message"),
    [
        (
            pd.DataFrame({"x": [1, 2], "target": [2, 3]}),
            [],
            ValueError,
            "cannot be empty",
        ),
        (
            pd.DataFrame({"x": [1, 2], "target": [2, 3]}),
            ["x", "x"],
            ValueError,
            "duplicate",
        ),
        (
            pd.DataFrame({"x": [1, 2], "target": [2, 3]}),
            [1],
            TypeError,
            "string",
        ),
        (
            pd.DataFrame({"x": [1 + 2j, 2 + 3j], "target": [2, 3]}),
            None,
            ValueError,
            "complex",
        ),
        (
            pd.DataFrame({"x": [1, np.inf], "target": [2, 3]}),
            None,
            ValueError,
            "missing or non-finite",
        ),
    ],
)
def test_training_rejects_invalid_feature_selection_and_values(
    data, features, error, message
):
    with pytest.raises(error, match=message):
        train_linear_regression(data, "target", features)


def test_training_rejects_duplicate_dataframe_columns():
    data = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "target"])

    with pytest.raises(ValueError, match="column names must be unique"):
        train_linear_regression(data, "target")


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
        _cross_validation_metrics(data, "target", None, 42, cv_folds=folds)


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
    original_fit = LinearRegression.fit
    fitted_row_indexes = []

    def recording_fit(model, predictors, target, *args, **kwargs):
        fitted_row_indexes.append(set(predictors.index))
        return original_fit(model, predictors, target, *args, **kwargs)

    monkeypatch.setattr(LinearRegression, "fit", recording_fit)
    split_data.split_and_train(data, "target")

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
    assert report["training"]["model"] == "LinearRegression"
    assert len(report["training"]["cross_validation"]["r2"]["fold_scores"]) == 5
    assert len(report["feature_removal_tests"]) == 1
    removal_test = report["feature_removal_tests"][0]
    assert removal_test["removed_feature"] == "x"
    assert removal_test["remaining_features"] == []
    assert removal_test["cross_validation"] == {
        "r2": {"mean": None, "std": None},
        "mae": {"mean": None, "std": None},
        "rmse": {"mean": None, "std": None},
    }
    assert removal_test["limitation"] == "cannot_fit_without_predictor_features"
    assert report["holdout"]["row_count"] == 4
    assert set(report["holdout"]["metrics"]) == {"r2", "mae", "rmse"}
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
    assert set(tests[0]["cross_validation"]) == {"r2", "mae", "rmse"}

    for item in tests:
        for metric in ("r2", "mae", "rmse"):
            assert item["cross_validation"][metric]["mean"] is not None
            assert item["cross_validation"][metric]["std"] >= 0
            assert item["comparison_to_current"][metric + "_change"] == pytest.approx(
                item["cross_validation"][metric]["mean"]
                - current_metrics[metric]["mean"]
            )


def test_json_omits_holdout_statistics_by_default_and_includes_training_count():
    result = split_and_train(_sample_data(), "target", features="x")

    report = json.loads(split_result_to_json(result, "target"))

    assert "holdout" not in report
    assert report["data_split"]["training_row_count"] == 16
    assert report["data_split"]["holdout_row_count"] == 4


def test_json_preserves_typed_index_values_and_computes_single_row_errors():
    data = _sample_data()
    model = train_linear_regression(data, "target", features="x")
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
    assert report["holdout"]["metrics"]["mae"] == pytest.approx(3.0)
    assert report["holdout"]["metrics"]["rmse"] == pytest.approx(3.0)
    assert report["holdout"]["rows"][0]["index"] == {"type": "int", "value": 7}


def test_json_reports_null_metrics_for_empty_holdout():
    data = _sample_data()
    model = train_linear_regression(data, "target", features="x")
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
        "mae": None,
        "rmse": None,
    }
    assert report["holdout"]["rows"] == []


def test_json_reports_undefined_r2_and_defined_errors_for_constant_holdout_target():
    data = _sample_data()
    model = train_linear_regression(data, "target", features="x")
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
    assert report["holdout"]["metrics"]["mae"] == pytest.approx(3.0)
    assert report["holdout"]["metrics"]["rmse"] == pytest.approx(np.sqrt(10))


def test_json_preserves_multiindex_type_information_and_rejects_unsupported_index():
    data = _sample_data()
    model = train_linear_regression(data, "target", features="x")
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
