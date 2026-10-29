import numpy as np
import pandas as pd
import pytest

from knn_regression.data_split import train_holdout_split
from knn_regression.folds import create_cv_folds
from knn_regression.feature_selection import validate_features
from knn_regression.knn_model import create_knn_pipeline, fit_knn_model
from knn_regression.optimization import optimize_hyperparameters, elbow_analysis, evaluate_holdout
from knn_regression.main_knn_workflow import run_knn_workflow


def sample_data(n=60):
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2 * x1 - 0.5 * x2 + rng.normal(scale=.1, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "constant": 1.0, "y": y})


def test_reproducible_split_and_no_overlap():
    data = sample_data()
    a = train_holdout_split(data, "y")
    b = train_holdout_split(data, "y")
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not set(a[0].index).intersection(a[1].index)
    assert len(a[0]) == 48 and len(a[1]) == 12


def test_cv_folds_are_complete_and_reproducible():
    X = pd.DataFrame({"x": np.arange(25)})
    a = create_cv_folds(X)
    b = create_cv_folds(X)
    validation = [fold.validation_indices for fold in a]
    assert all(np.array_equal(f.validation_indices, g.validation_indices) for f, g in zip(a, b))
    assert sorted(np.concatenate(validation).tolist()) == list(range(25))
    assert all(set(f.validation_indices).isdisjoint(f.train_indices) for f in a)


def test_manual_and_automatic_feature_selection():
    data = sample_data()
    assert validate_features(data, "y", ["x2", "x1"]) == ["x2", "x1"]
    assert validate_features(data, "y") == ["x1", "x2"]
    with pytest.raises(KeyError):
        validate_features(data, "y", ["missing"])
    with pytest.raises(ValueError):
        validate_features(data, "y", ["y"])
    data["category"] = "a"
    assert "category" not in validate_features(data, "y")


def test_missing_and_empty_features_are_rejected():
    data = sample_data()
    data.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError):
        validate_features(data, "y", ["x1"])
    with pytest.raises(ValueError):
        validate_features(data, "y", [])


def test_pipeline_fits_and_predicts_and_scaling_is_fold_local():
    data = sample_data()
    X_train, X_holdout, y_train, y_holdout = train_holdout_split(data[["x1", "x2", "y"]], "y")
    folds = create_cv_folds(X_train)
    fold = folds[0]
    model = create_knn_pipeline(n_neighbors=5)
    model.fit(X_train.iloc[fold.train_indices], y_train.iloc[fold.train_indices])
    predictions = model.predict(X_train.iloc[fold.validation_indices])
    assert predictions.shape == (len(fold.validation_indices),)
    scaler = model.named_steps["scaler"]
    assert np.isclose(scaler.mean_, X_train.iloc[fold.train_indices].mean().to_numpy()).all()


def test_grid_search_elbow_and_invalid_neighbors():
    data = sample_data()
    X_train, _, y_train, _ = train_holdout_split(data[["x1", "x2", "y"]], "y")
    folds = create_cv_folds(X_train)
    result = optimize_hyperparameters(X_train, y_train, folds, {
        "n_neighbors": [3, 5], "weights": ["uniform"], "p": [1, 2]
    })
    assert len(result["results"]) == 4
    assert result["best_configuration"] in [r["configuration"] for r in result["results"]]
    elbow = elbow_analysis(X_train, y_train, folds, [3, 5])
    assert [item["n_neighbors"] for item in elbow] == [3, 5]
    with pytest.raises(ValueError):
        optimize_hyperparameters(X_train, y_train, folds, {"n_neighbors": [0], "weights": ["uniform"] , "p": [2]})


def test_holdout_and_programmatic_workflow():
    data = sample_data()
    result = run_knn_workflow(data, "y", features=["x1", "x2"], interactive=False,
                              search_space={"n_neighbors": [3, 5], "weights": ["uniform"], "p": [2]})
    assert set(result["holdout_metrics"]) == {"r2", "mae", "rmse"}
    assert result["model"] is not None
    metrics = evaluate_holdout(result["model"], data[["x1", "x2"]], data["y"])
    assert set(metrics) == {"r2", "mae", "rmse"}
