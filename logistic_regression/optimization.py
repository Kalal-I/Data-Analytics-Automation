"""Cross-validation and classification metric utilities."""
from itertools import product
from numbers import Real

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .logistic_model import create_model, validate_model_params


DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0]


def _as_array(values):
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def normalize_configurations(configurations=None):
    if configurations is None:
        return [{"penalty": "l2", "C": value} for value in DEFAULT_C_VALUES]
    if isinstance(configurations, dict):
        if "penalty" in configurations:
            penalties = _as_array(configurations.get("penalty"))
            C_values = _as_array(configurations.get("C", DEFAULT_C_VALUES))
            ratios = configurations.get("l1_ratio")
            result = []
            for penalty in penalties:
                validate_penalty = penalty
                if validate_penalty == "elasticnet":
                    ratio_values = _as_array(ratios) if ratios is not None else None
                    if not ratio_values:
                        raise ValueError("l1_ratio is required for Elastic Net configurations")
                    for C_value, ratio in product(C_values, ratio_values):
                        result.append({"penalty": penalty, "C": C_value, "l1_ratio": ratio})
                else:
                    for C_value in C_values:
                        result.append({"penalty": penalty, "C": C_value})
            if not result:
                raise ValueError("candidate configurations cannot be empty")
            return result
        if "C" not in configurations:
            raise ValueError("configuration must specify penalty")
        configurations = [configurations]

    if isinstance(configurations, (list, tuple)):
        result = []
        for configuration in configurations:
            if not isinstance(configuration, dict) or "penalty" not in configuration:
                raise ValueError("each configuration must be a dictionary containing penalty")
            result.append(dict(configuration))
        if not result:
            raise ValueError("candidate configurations cannot be empty")
        return result
    raise TypeError("configurations must be a dictionary or a list of dictionaries")


def _average_type(y):
    classes = np.unique(np.asarray(y))
    return "binary" if len(classes) == 2 else "macro"


def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    positive_class=None,
    average=None,
    model_classes=None,
):
    """Calculate classification metrics; undefined ROC-AUC is returned as None."""
    classes = np.asarray(model_classes) if model_classes is not None else np.unique(np.asarray(y_true))
    if len(classes) < 2:
        raise ValueError("at least two classes are required to calculate classification metrics")
    binary = len(classes) == 2
    averaging = average if average else ("binary" if binary else "macro")
    if not binary and averaging == "binary":
        raise ValueError("multiclass classification requires macro, weighted, or micro averaging")
    if binary and averaging not in {"binary", "macro", "weighted", "micro"}:
        raise ValueError("unsupported average {!r}".format(averaging))

    metric_args = {"zero_division": 0}
    if binary and averaging == "binary":
        target_positive = positive_class if positive_class is not None else classes[1]
        if target_positive not in classes:
            raise ValueError("positive_class {!r} is not present in y_true".format(positive_class))
        truth = np.asarray(y_true) == target_positive
        pred = np.asarray(y_pred) == target_positive
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(truth, pred, average="binary", zero_division=0))
        recall = float(recall_score(truth, pred, average="binary", zero_division=0))
        f1 = float(f1_score(truth, pred, average="binary", zero_division=0))
    else:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average=averaging, **metric_args))
        recall = float(recall_score(y_true, y_pred, average=averaging, **metric_args))
        f1 = float(f1_score(y_true, y_pred, average=averaging, **metric_args))

    roc_auc = None
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            if binary:
                positive = positive_class if positive_class is not None else classes[1]
                if y_proba.ndim == 1:
                    positive_scores = y_proba
                elif y_proba.shape[1] == 2:
                    if model_classes is not None:
                        ordered_classes = np.asarray(model_classes)
                    else:
                        ordered_classes = np.asarray(classes)
                    positive_scores = y_proba[:, int(np.flatnonzero(ordered_classes == positive)[0])]
                else:
                    positive_scores = y_proba[:, 0]
                roc_auc = float(roc_auc_score(np.asarray(y_true) == positive, positive_scores))
            elif y_proba.ndim == 2 and y_proba.shape[1] == len(classes):
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except (ValueError, IndexError):
            roc_auc = None
        if roc_auc is not None and not np.isfinite(roc_auc):
            roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "labels": classes.tolist(),
    }


def evaluate_configuration(X, y, folds, configuration, metric="f1", average=None, positive_class=None):
    configuration = dict(configuration)
    fold_metrics = []
    failures = []
    for fold_number, fold in enumerate(folds, start=1):
        try:
            model = create_model(**configuration)
            X_train = X.iloc[fold.train_indices] if hasattr(X, "iloc") else np.asarray(X)[fold.train_indices]
            X_validation = X.iloc[fold.validation_indices] if hasattr(X, "iloc") else np.asarray(X)[fold.validation_indices]
            y_array = np.asarray(y)
            model.fit(X_train, y_array[fold.train_indices])
            predictions = model.predict(X_validation)
            probabilities = model.predict_proba(X_validation)
            scores = classification_metrics(
                y_array[fold.validation_indices],
                predictions,
                probabilities,
                positive_class=positive_class,
                average=average,
                model_classes=model.classes_,
            )
            scores["positive_class"] = positive_class
            fold_metrics.append(scores)
        except Exception as exc:
            failures.append({"fold": fold_number, "error": str(exc)})

    mean_std = {}
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = np.asarray([item[name] for item in fold_metrics if item[name] is not None], dtype=float)
        mean_std[name] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
        }
    return {
        "configuration": configuration,
        "fold_metrics": fold_metrics,
        "failures": failures,
        "mean_std": mean_std,
    }


def optimize_hyperparameters(
    X,
    y,
    folds,
    configurations=None,
    metric="f1",
    average=None,
    positive_class=None,
):
    """Evaluate every candidate on exactly the supplied folds and select best mean metric."""
    supported = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if metric not in supported:
        raise ValueError("metric must be one of {}".format(sorted(supported)))
    candidates = normalize_configurations(configurations)
    results = [
        evaluate_configuration(
            X, y, folds, configuration, metric=metric, average=average, positive_class=positive_class
        )
        for configuration in candidates
    ]
    for configuration in candidates:
        validate_model_params(configuration)
        if configuration["penalty"] == "elasticnet" and "l1_ratio" not in configuration:
            raise ValueError("Elastic Net requires l1_ratio")

    def ranking(result):
        mean = result["mean_std"][metric]["mean"]
        primary = mean if mean is not None else -np.inf
        configuration = result["configuration"]
        return (
            -primary,
            configuration.get("penalty", ""),
            float(configuration.get("C", 0)),
            float(configuration.get("l1_ratio", 0) or 0),
        )

    results.sort(key=ranking)
    best = results[0]["configuration"] if results and results[0]["mean_std"][metric]["mean"] is not None else None
    return {
        "results": results,
        "best_configuration": best,
        "selection_metric": metric,
    }


def evaluate_holdout(model, X_holdout, y_holdout, positive_class=None, average=None):
    predictions = model.predict(X_holdout)
    probabilities = model.predict_proba(X_holdout)
    metrics = classification_metrics(
        y_holdout,
        predictions,
        probabilities,
        positive_class=positive_class,
        average=average,
        model_classes=model.classes_,
    )
    metrics["positive_class"] = positive_class
    return metrics


grid_search = optimize_hyperparameters
