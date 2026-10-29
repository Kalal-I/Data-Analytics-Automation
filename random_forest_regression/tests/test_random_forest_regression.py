import numpy as np
import pandas as pd
import pytest

from random_forest_regression.data_split import train_holdout_split
from random_forest_regression.folds import create_cv_folds
from random_forest_regression.feature_selection import validate_features
from random_forest_regression.random_forest_model import create_random_forest_model
from random_forest_regression.optimization import optimize_hyperparameters, evaluate_holdout
from random_forest_regression.main_rf_workflow import run_random_forest_workflow


def sample_data(n=60):
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 2.5 * x1 - 0.8 * x2 + 0.1 * x3 + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "constant": 1.0, "y": y})


def test_reproducible_split_and_no_overlap():
    data = sample_data()
    first = train_holdout_split(data, "y")
    second = train_holdout_split(data, "y")
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])
    assert len(first[0]) == 48 and len(first[1]) == 12
    assert set(first[0].index).isdisjoint(first[1].index)


def test_folds_are_reproducible_complete_and_training_only():
    X = pd.DataFrame({"x": np.arange(48)})
    first = create_cv_folds(X)
    second = create_cv_folds(X)
    assert len(first) == 5
    assert all(np.array_equal(a.validation_indices, b.validation_indices)
               for a, b in zip(first, second))
    assert sorted(np.concatenate([f.validation_indices for f in first])) == list(range(48))
    assert all(set(f.train_indices).isdisjoint(f.validation_indices) for f in first)


def test_feature_selection_explicit_and_automatic():
    data = sample_data()
    assert validate_features(data, "y", ["x2", "x1"]) == ["x2", "x1"]
    assert validate_features(data, "y") == ["x1", "x2", "x3"]
    with pytest.raises(KeyError):
        validate_features(data, "y", ["missing"])
    with pytest.raises(ValueError):
        validate_features(data, "y", ["y"])
    with pytest.raises(ValueError):
        validate_features(data, "y", [])


def test_invalid_features_and_small_splits():
    data = sample_data()
    data.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError):
        validate_features(data, "y", ["x1"])
    with pytest.raises(ValueError):
        train_holdout_split(pd.DataFrame({"x": range(9), "y": range(9)}), "y")
    with pytest.raises(ValueError):
        create_random_forest_model(n_estimators=0)


def test_model_fit_predict_and_holdout_metrics():
    data = sample_data()
    X_train, X_holdout, y_train, y_holdout = train_holdout_split(data[["x1", "x2", "x3", "y"]], "y")
    model = create_random_forest_model(n_estimators=20, random_state=7).fit(X_train, y_train)
    assert model.predict(X_holdout).shape == (len(X_holdout),)
    metrics = evaluate_holdout(model, X_holdout, y_holdout)
    assert set(metrics) == {"r2", "mae", "rmse"}
    assert all(np.isfinite(value) for value in metrics.values())


def small_search():
    return {
        "n_estimators": [20, 30],
        "max_depth": [None, 5],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.8],
        "max_leaf_nodes": [None, 20],
    }


def test_optimization_is_bounded_and_reproducible():
    data = sample_data()
    X_train, _, y_train, _ = train_holdout_split(data[["x1", "x2", "x3", "y"]], "y")
    folds = create_cv_folds(X_train)
    first = optimize_hyperparameters(X_train, y_train, folds, small_search(), n_iter=4, random_state=3)
    second = optimize_hyperparameters(X_train, y_train, folds, small_search(), n_iter=4, random_state=3)
    assert first["n_configurations"] <= 4
    assert first["results"]
    assert first["best_configuration"] is not None
    assert first["results"] == second["results"]
    assert first["selection_metric"] == "mean_cv_r2"


def test_workflow_output_and_programmatic_invocation():
    result = run_random_forest_workflow(
        sample_data(), "y", features=["x1", "x2", "x3"],
        search_space=small_search(), n_iter=4, random_state=42,
    )
    assert set(result) == {"model", "selected_features", "best_configuration", "cross_validation", "holdout_metrics", "folds"}
    assert result["selected_features"] == ["x1", "x2", "x3"]
    assert set(result["holdout_metrics"]) == {"r2", "mae", "rmse"}
    assert len(result["folds"]) == 5
