"""Standalone logistic regression classification workflow."""
from typing import Iterable, Optional

import pandas as pd

from .data_split import train_holdout_split
from .feature_selection import validate_features
from .folds import create_cv_folds
from .logistic_model import fit_model
from .optimization import evaluate_holdout, optimize_hyperparameters


def run_logistic_workflow(
    data,
    target,
    features: Optional[Iterable[str]] = None,
    configurations=None,
    selection_metric="f1",
    positive_class=None,
    average=None,
    random_state=42,
    max_iter=1000,
):
    """Run leakage-safe classification tuning and holdout evaluation."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(target, str) or target not in data.columns:
        raise KeyError("target column not found: {!r}".format(target))

    X_train_full, X_holdout, y_train, y_holdout = train_holdout_split(
        data, target, random_state=random_state
    )
    training_frame = X_train_full.assign(**{target: y_train.to_numpy()})
    selected_features = validate_features(training_frame, target, features)
    X_train = X_train_full[selected_features]
    X_holdout = X_holdout[selected_features]

    folds = create_cv_folds(X_train, y_train, n_splits=5, random_state=random_state)
    tuning = optimize_hyperparameters(
        X_train,
        y_train,
        folds,
        configurations=configurations,
        metric=selection_metric,
        average=average,
        positive_class=positive_class,
    )
    configuration = tuning["best_configuration"]
    if configuration is None:
        raise RuntimeError("cross-validation produced no usable configuration")

    final_params = {
        **configuration,
        "max_iter": max_iter,
        "random_state": random_state,
    }
    model = fit_model(X_train, y_train, **final_params)
    holdout_metrics = evaluate_holdout(
        model,
        X_holdout,
        y_holdout,
        positive_class=positive_class,
        average=average,
    )
    return {
        "model": model,
        "selected_features": selected_features,
        "best_regularization": configuration,
        "cross_validation": tuning,
        "holdout_metrics": holdout_metrics,
        "confusion_matrix": holdout_metrics["confusion_matrix"],
        "folds": folds,
    }


main = run_logistic_workflow
