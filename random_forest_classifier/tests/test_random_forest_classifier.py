import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from random_forest_classifier.data_split import train_holdout_split
from random_forest_classifier.feature_selection import select_features
from random_forest_classifier.folds import create_cv_folds
from random_forest_classifier.random_forest_model import create_random_forest_model, fit_random_forest_model
from random_forest_classifier.optimization import (
    optimize_hyperparameters, evaluate_holdout, create_tuning_report,
)
from random_forest_classifier.workflow import run_random_forest_classifier_workflow


def classification_data(n=100):
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.4 * x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "constant": 1.0, "target": y})


SMALL_SEARCH = {
    "n_estimators": [20, 30],
    "max_depth": [None, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.8],
    "criterion": ["gini", "entropy"],
}


def test_stratified_split_is_reproducible():
    data = classification_data()
    first = train_holdout_split(data, "target")
    second = train_holdout_split(data, "target")
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])
    assert y_proportion(first[3]) == pytest.approx(y_proportion(data["target"]), abs=0.05)
    assert len(first[0]) == 80 and len(first[1]) == 20


def y_proportion(values):
    return values.value_counts(normalize=True)[1]


def test_stratified_folds_and_insufficient_counts():
    data = classification_data()
    X_train, _, y_train, _ = train_holdout_split(data, "target")
    folds = create_cv_folds(X_train, y_train)
    assert len(folds) == 5
    assert sorted(np.concatenate([fold.validation_indices for fold in folds])) == list(range(len(X_train)))
    with pytest.raises(ValueError):
        create_cv_folds(pd.DataFrame({"x": range(10)}), np.array([0] * 9 + [1]))


def test_automatic_feature_filtering():
    data = classification_data()
    data["bad"] = np.nan
    data["category"] = np.where(np.arange(len(data)) % 2, "b", "a")
    selected, exclusions = select_features(
        data, "target", return_exclusions=True
    )
    assert selected == ["x1", "x2", "category"]
    assert {row["column"] for row in exclusions} == {"target", "constant", "bad"}


def test_feature_filtering_keeps_missing_high_cardinality_and_invalid_numeric_for_review():
    data = classification_data(30)
    data["missing_some"] = np.nan
    data["high_cardinality"] = np.arange(len(data))
    data["invalid_numeric"] = np.arange(len(data), dtype=float)
    data.loc[2, "invalid_numeric"] = np.inf
    with pytest.raises(ValueError, match="positive or negative infinity"):
        select_features(data, "target")


def test_classifier_creation_and_fit():
    model = create_random_forest_model(n_estimators=20, criterion="entropy", random_state=7)
    assert isinstance(model, RandomForestClassifier)
    data = classification_data()
    fitted = fit_random_forest_model(data[["x1", "x2"]], data["target"], n_estimators=20)
    assert fitted.predict(data[["x1", "x2"]]).shape == (len(data),)


def test_randomized_search_fixed_folds_and_report():
    data = classification_data()
    X_train, _, y_train, _ = train_holdout_split(data, "target")
    X_train = X_train[["x1", "x2"]]
    folds = create_cv_folds(X_train, y_train)
    tuning = optimize_hyperparameters(
        X_train, y_train, folds, SMALL_SEARCH, n_iter=4, random_state=3
    )
    assert len(tuning["results"]) <= 4
    assert tuning["cv_folds"] == 5
    assert tuning["expected_model_fits"] == len(tuning["results"]) * 5
    assert tuning["best_configuration"] in [row["configuration"] for row in tuning["results"]]
    report = create_tuning_report(tuning)
    assert report["top_10"] == tuning["results"][:10]
    assert set(report) >= {
        "best_configuration", "best_score", "summary",
        "top_10", "parameter_observations",
    }


def test_holdout_metrics_binary_and_multiclass():
    data = classification_data()
    result = run_random_forest_classifier_workflow(
        data, "target", search_space=SMALL_SEARCH, n_iter=3, random_state=42
    )
    assert result["selected_features"] == ["x1", "x2"]
    assert set(result["holdout_metrics"]) >= {
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "confusion_matrix", "classification_report",
    }

    multi = pd.DataFrame({
        "x1": np.tile(np.arange(3), 20),
        "x2": np.tile(np.arange(3), 20) + np.linspace(-0.1, 0.1, 60),
        "target": np.tile([0, 1, 2], 20),
    })
    X_train, X_holdout, y_train, y_holdout = train_holdout_split(multi, "target")
    model = fit_random_forest_model(X_train[["x1", "x2"]], y_train, n_estimators=30)
    metrics = evaluate_holdout(model, X_holdout[["x1", "x2"]], y_holdout)
    assert metrics["roc_auc"] is not None


def test_workflow_fit_count_contract_and_invalid_class_counts():
    data = classification_data()
    result = run_random_forest_classifier_workflow(
        data, "target", search_space=SMALL_SEARCH, n_iter=4, random_state=42
    )
    assert set(result) == {
        "model", "selected_features", "best_configuration", "cross_validation",
        "holdout_metrics", "folds", "tuning_report",
    }
    with pytest.raises(ValueError):
        train_holdout_split(
            pd.DataFrame({"x": range(10), "target": [0] * 9 + [1]}),
            "target",
        )
