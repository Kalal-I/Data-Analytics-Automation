"""Programmatic workflow for Random Forest regression."""
from typing import Iterable, Optional
import pandas as pd
from .data_split import train_holdout_split
from .folds import create_cv_folds
from .feature_selection import validate_features
from .optimization import optimize_hyperparameters, evaluate_holdout
from .random_forest_model import fit_random_forest_model


def run_random_forest_workflow(data: pd.DataFrame, target: str,
                               features: Optional[Iterable[str]] = None,
                               search_space=None, n_iter: int = 40,
                               random_state: int = 42):
    """Run selection, tuning, final fit, and one holdout evaluation."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    training, holdout, y_train, y_holdout = train_holdout_split(
        data, target, random_state=random_state
    )
    training_frame = training.copy()
    training_frame[target] = y_train
    selected_features = validate_features(training_frame, target, features)

    X_train = training[selected_features]
    X_holdout = holdout[selected_features]
    folds = create_cv_folds(X_train, n_splits=5, random_state=random_state)
    fixed_params = {"random_state": random_state, "n_jobs": -1}
    tuning = optimize_hyperparameters(
        X_train, y_train, folds, search_space=search_space, n_iter=n_iter,
        random_state=random_state, fixed_params=fixed_params
    )
    if tuning["best_configuration"] is None:
        raise RuntimeError("cross-validation produced no usable configuration")

    model = fit_random_forest_model(
        X_train, y_train, **fixed_params, **tuning["best_configuration"]
    )
    return {
        "model": model,
        "selected_features": selected_features,
        "best_configuration": tuning["best_configuration"],
        "cross_validation": tuning,
        "holdout_metrics": evaluate_holdout(model, X_holdout, y_holdout),
        "folds": folds,
    }


run_rf_workflow = run_random_forest_workflow
main = run_random_forest_workflow
