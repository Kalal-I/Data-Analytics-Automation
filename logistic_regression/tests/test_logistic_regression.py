import numpy as np
import pandas as pd
import pytest

from logistic_regression.data_split import train_holdout_split
from logistic_regression.feature_selection import validate_features
from logistic_regression.folds import create_cv_folds
from logistic_regression.logistic_model import create_model, fit_model, predict_proba
from logistic_regression.main_logistic_workflow import run_logistic_workflow
from logistic_regression.optimization import classification_metrics, optimize_hyperparameters


def binary_data(n=60):
    rng = np.random.default_rng(123)
    x1 = np.linspace(-2, 2, n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.1 * x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "constant": 1.0, "target": y})


def multiclass_data(n=90):
    rng = np.random.default_rng(321)
    x1 = np.repeat(np.arange(3), n // 3)
    x2 = x1 + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": x1})


def test_split_is_reproducible_and_stratified():
    data = binary_data()
    a = train_holdout_split(data, "target")
    b = train_holdout_split(data, "target")
    assert all(x.equals(y) for x, y in zip(a, b))
    assert set(a[0].index).isdisjoint(a[1].index) if hasattr(a[0], "index") else True
    assert len(a[0]) == 48 and len(a[1]) == 12
    assert abs(a[1]["target"].mean() if False else a[3].mean() - data["target"].mean()) <= 0.1


def test_fixed_stratified_folds():
    data = binary_data()
    X, _, y, _ = train_holdout_split(data, "target")
    folds = create_cv_folds(X, y, 5, random_state=7)
    assert len(folds) == 5
    validation = np.concatenate([fold.validation_indices for fold in folds])
    assert sorted(validation.tolist()) == list(range(len(X)))


def test_feature_selection_modes_and_errors():
    data = binary_data()
    assert validate_features(data, "target") == ["x1", "x2"]
    assert validate_features(data, "target", ["x2", "x1"]) == ["x2", "x1"]
    with pytest.raises(KeyError):
        validate_features(data, "target", ["missing"])
    with pytest.raises(ValueError):
        validate_features(data, "target", [])


@pytest.mark.parametrize("penalty,solver", [
    ("l1", "saga"),
    ("l2", "lbfgs"),
    ("elasticnet", "saga"),
])
def test_model_regularizations(penalty, solver):
    params = {"penalty": penalty, "solver": solver}
    if penalty == "elasticnet":
        params["l1_ratio"] = 0.4
    model = create_model(**params)
    assert model.penalty == penalty


def test_solver_compatibility_and_invalid_parameters():
    with pytest.raises(ValueError):
        create_model(penalty="l1", solver="lbfgs")
    with pytest.raises(ValueError):
        create_model(penalty="elasticnet", l1_ratio=0)
    with pytest.raises(ValueError):
        create_model(C=0)


def test_binary_multiclass_and_predict_proba():
    binary = binary_data()
    X_train, _, y_train, _ = train_holdout_split(binary, "target")
    model = fit_model(X_train[["x1", "x2"]], y_train, penalty="l2", C=1.0, random_state=7)
    assert len(model.predict(X_train[["x1", "x2"]])) == len(y_train)
    assert predict_proba(model, X_train[["x1", "x2"]]).shape == (len(y_train), 2)

    multi = multiclass_data()
    X_train, _, y_train, _ = train_holdout_split(multi, "target")
    model = fit_model(X_train[["x1", "x2"]], y_train, penalty="l2", solver="lbfgs")
    assert predict_proba(model, X_train[["x1", "x2"]]).shape[1] == 3


def test_fixed_configuration_reused_and_reproducible():
    data = binary_data()
    X, _, y, _ = train_holdout_split(data, "target")
    X = X[["x1", "x2"]]
    folds = create_cv_folds(X, y, random_state=9)
    configurations = [
        {"penalty": "l1", "C": 0.1},
        {"penalty": "l2", "C": 1.0},
        {"penalty": "elasticnet", "C": 1.0, "l1_ratio": 0.5},
    ]
    a = optimize_hyperparameters(X, y, folds, configurations=configurations)
    b = optimize_hyperparameters(X, y, folds, configurations=configurations)
    assert len(a["results"]) == 3
    assert a["results"][0]["configuration"] == b["results"][0]["configuration"]


def test_metrics_and_roc_auc():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    metrics = classification_metrics(y, pred, proba)
    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] is not None
    assert classification_metrics([0, 1, 2], [0, 1, 2])["roc_auc"] is None
    assert classification_metrics(y, pred, proba, average="macro")["precision"] == pytest.approx(5 / 6)


def test_invalid_class_counts_and_workflow():
    with pytest.raises(ValueError):
        create_cv_folds(pd.DataFrame({"x": range(10)}), np.array([0] * 9 + [1]), n_splits=5)

    data = binary_data(100)
    result = run_logistic_workflow(
        data,
        "target",
        features=["x1", "x2"],
        configurations=[
            {"penalty": "l2", "C": 0.1},
            {"penalty": "l1", "C": 1.0},
        ],
    )
    assert set(result) == {
        "model", "selected_features", "best_regularization", "cross_validation",
        "holdout_metrics", "confusion_matrix", "folds",
    }
    assert result["selected_features"] == ["x1", "x2"]
