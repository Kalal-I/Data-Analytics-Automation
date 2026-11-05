"""Programmatic Random Forest classification workflow."""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from .data_split import train_holdout_split
from .feature_selection import select_features
from .folds import create_cv_folds
from .optimization import evaluate_holdout, optimize_hyperparameters
from .random_forest_model import fit_random_forest_model
from .tuning_report import create_tuning_report


def run_random_forest_classifier_workflow(
    data, target, search_space=None, n_iter=40, scoring="f1_weighted",
    random_state=42, include_tuning_history=False,
):
    """Split, tune, fit once, and evaluate a Random Forest classifier."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    training, holdout, y_train, y_holdout = train_holdout_split(
        data, target, random_state=random_state
    )
    training_frame = training.copy()
    training_frame[target] = y_train.to_numpy()
    selected_features = select_features(training_frame, target)
    X_train = training[selected_features]
    X_holdout = holdout[selected_features]
    unsupported = []
    for column in selected_features:
        values = X_train[column]
        if values.isna().any():
            unsupported.append(
                "{} contains missing values; preprocess the feature before modeling".format(column)
            )
        if not is_numeric_dtype(values.dtype):
            unsupported.append(
                "{} is categorical/non-numeric; encode it before modeling".format(column)
            )
        elif not np.isfinite(values.to_numpy(dtype=float)).all():
            unsupported.append(
                "{} contains non-finite numeric values; preprocess the feature before modeling".format(column)
            )
    if unsupported:
        raise ValueError("; ".join(unsupported))

    folds = create_cv_folds(X_train, y_train, n_splits=5, random_state=random_state)
    tuning = optimize_hyperparameters(
        X_train, y_train, folds, search_space=search_space, n_iter=n_iter,
        random_state=random_state, scoring=scoring,
    )
    if tuning["best_configuration"] is None:
        raise RuntimeError("cross-validation produced no usable configuration")
    tuning["report"] = create_tuning_report(
        tuning, include_history=include_tuning_history
    )

    model = fit_random_forest_model(
        X_train, y_train, random_state=random_state, n_jobs=-1,
        **tuning["best_configuration"]
    )
    return {
        "model": model,
        "selected_features": selected_features,
        "best_configuration": tuning["best_configuration"],
        "cross_validation": tuning,
        "holdout_metrics": evaluate_holdout(model, X_holdout, y_holdout),
        "folds": folds,
        "tuning_report": tuning["report"],
    }


run_random_forest_workflow = run_random_forest_classifier_workflow
main = run_random_forest_classifier_workflow
