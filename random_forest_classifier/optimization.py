"""Classification tuning with fixed stratified folds and no automatic refit."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import RandomizedSearchCV

from .random_forest_model import create_random_forest_model
from .tuning_report import create_tuning_report


TUNING_PARAMETERS = (
    "n_estimators", "max_depth", "min_samples_split",
    "min_samples_leaf", "max_features", "criterion",
)
DEFAULT_SEARCH_SPACE = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
    "criterion": ["gini", "entropy", "log_loss"],
}


def _ordered_classes(y_true, model=None):
    if model is not None and hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    return np.unique(np.asarray(y_true))


def _roc_auc(y_true, probabilities, classes, binary_positive=None):
    if probabilities is None:
        return None
    y_array = np.asarray(y_true)
    classes = np.asarray(classes)
    try:
        if len(classes) == 2:
            if binary_positive is None:
                binary_positive = classes[1]
            column = int(np.flatnonzero(classes == binary_positive)[0])
            value = roc_auc_score(y_array == binary_positive, np.asarray(probabilities)[:, column])
        else:
            probability_array = np.asarray(probabilities)
            if probability_array.shape[1] != len(classes):
                return None
            value = roc_auc_score(y_array, probability_array, multi_class="ovr", average="macro")
        return None if not np.isfinite(value) else round(float(value), 12)
    except (ValueError, IndexError, TypeError):
        return None


def classification_metrics(y_true, y_pred, probabilities=None, model=None,
                           positive_class=None, average=None):
    """Calculate binary or multiclass classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = _ordered_classes(y_true, model)
    if len(classes) < 2:
        raise ValueError("classification metrics require at least two classes")
    binary = len(classes) == 2
    averaging = average or ("binary" if binary else "weighted")
    if binary:
        positive = positive_class if positive_class is not None else classes[1]
        if positive not in classes:
            raise ValueError("positive_class is not present in the target")
        truth = y_true == positive
        predictions = y_pred == positive
        precision = precision_score(truth, predictions, average="binary", zero_division=0)
        recall = recall_score(truth, predictions, average="binary", zero_division=0)
        f1 = f1_score(truth, predictions, average="binary", zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average=averaging, zero_division=0)
        recall = recall_score(y_true, y_pred, average=averaging, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=averaging, zero_division=0)

    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 12),
        "precision": round(float(precision), 12),
        "recall": round(float(recall), 12),
        "f1": round(float(f1), 12),
        "roc_auc": _roc_auc(y_true, probabilities, classes, positive_class),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": report,
        "labels": classes.tolist(),
    }


def evaluate_configuration(X, y, folds, configuration, fixed_params=None,
                           random_state=42, scoring="f1_weighted", average=None):
    """Evaluate one configuration on fixed folds (primarily for diagnostics)."""
    fold_metrics = []
    failures = []
    params = {"random_state": random_state, "n_jobs": -1}
    params.update(fixed_params or {})
    params.update(configuration)
    for fold_number, fold in enumerate(folds, start=1):
        try:
            model = create_random_forest_model(**params)
            X_train = X.iloc[fold.train_indices] if hasattr(X, "iloc") else np.asarray(X)[fold.train_indices]
            X_validation = X.iloc[fold.validation_indices] if hasattr(X, "iloc") else np.asarray(X)[fold.validation_indices]
            y_train = np.asarray(y)[fold.train_indices]
            y_validation = np.asarray(y)[fold.validation_indices]
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_validation)
            metrics = classification_metrics(
                y_validation, model.predict(X_validation), probabilities, model, average=average
            )
            metrics[scoring] = metrics["f1"] if scoring == "f1_weighted" else metrics[scoring]
            fold_metrics.append(metrics)
        except Exception as exc:
            failures.append({"fold": fold_number, "error": str(exc)})
    return {"configuration": dict(configuration), "fold_metrics": fold_metrics, "failures": failures}


def random_search(X, y, folds, search_space=None, n_iter=40, random_state=42,
                  scoring="f1_weighted", fixed_params=None):
    """Run reproducible RandomizedSearchCV with refit disabled."""
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    unknown = set(space) - set(TUNING_PARAMETERS)
    missing = set(TUNING_PARAMETERS) - set(space)
    if unknown:
        raise ValueError("unknown tuning parameters: {}".format(sorted(unknown)))
    if missing:
        raise ValueError("search space is missing tuning parameters: {}".format(sorted(missing)))

    cv = [(np.asarray(fold.train_indices, dtype=int), np.asarray(fold.validation_indices, dtype=int))
          for fold in folds]
    params = {"random_state": random_state, "n_jobs": -1}
    params.update(fixed_params or {})
    search = RandomizedSearchCV(
        estimator=create_random_forest_model(**params),
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        refit=False,
        random_state=random_state,
        n_jobs=-1,
        error_score=np.nan,
        return_train_score=False,
    )
    search.fit(X, y)

    records = []
    scores = np.asarray(search.cv_results_.get("mean_test_score", []))
    order = np.argsort(-scores, kind="mergesort")
    for position, index in enumerate(order, start=1):
        mean = search.cv_results_["mean_test_score"][index]
        std = search.cv_results_["std_test_score"][index]
        records.append({
            "rank": position,
            "configuration": {name: search.cv_results_["params"][index][name] for name in TUNING_PARAMETERS},
            "mean_test_score": None if np.isnan(mean) else round(float(mean), 12),
            "std_test_score": None if np.isnan(std) else round(float(std), 12),
            "fit_failed": not np.isfinite(mean),
        })

    best_index = search.best_index_
    best_configuration = None if best_index < 0 else {
        name: search.cv_results_["params"][best_index][name] for name in TUNING_PARAMETERS
    }
    tuning = {
        "results": records,
        "best_configuration": best_configuration,
        "best_score": None if best_index < 0 else round(float(search.best_score_), 12),
        "n_configurations": len(search.cv_results_["params"]),
        "cv_folds": len(cv),
        "random_state": random_state,
        "selection_metric": scoring,
        "expected_model_fits": len(search.cv_results_["params"]) * len(cv),
        "full_history": records,
    }
    tuning["report"] = create_tuning_report(tuning)
    return tuning


optimize_hyperparameters = random_search


def evaluate_holdout(model, X_holdout, y_holdout, positive_class=None, average=None):
    probabilities = model.predict_proba(X_holdout) if hasattr(model, "predict_proba") else None
    return classification_metrics(
        y_holdout,
        model.predict(X_holdout),
        probabilities,
        model,
        positive_class=positive_class,
        average=average,
    )
