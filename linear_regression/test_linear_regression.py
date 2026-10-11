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
from linear_regression.split_data import split_and_train


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


def test_split_result_serializes_model_cv_and_holdout_metrics_as_strict_json(tmp_path):
    """JSON report includes selected predictors, fold scores, and holdout results."""
    data = _sample_data()
    fold_file = tmp_path / "folds.json"
    result = create_folds_and_train(
        data, "target", features=["x"], folds_file=fold_file
    )
    folds = create_or_load_folds(data, fold_file)
    serialized = split_result_to_json(result, "target", folds=folds)

    report = json.loads(
        serialized,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert report["features"] == ["x"]
    assert report["target_column"] == "target"
    assert report["data_split"]["training_row_count"] == 16
    assert report["data_split"]["holdout_row_positions"] == folds["test_indices"]
    assert len(report["data_split"]["cross_validation_folds"]) == 5
    assert report["training"]["model"] == "LinearRegression"
    assert len(report["training"]["cross_validation"]["r2"]["fold_scores"]) == 5
    assert report["holdout"]["row_count"] == 4
    assert set(report["holdout"]["metrics"]) == {"r2", "mae", "rmse"}
    assert len(report["holdout"]["rows"]) == 4
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
