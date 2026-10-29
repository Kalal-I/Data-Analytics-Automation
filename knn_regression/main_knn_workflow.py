"""Programmatic and interactive entry point for the KNN workflow."""
from typing import Iterable, Optional
import pandas as pd
from .data_split import train_holdout_split
from .folds import create_cv_folds
from .feature_selection import validate_features
from .optimization import optimize_hyperparameters, evaluate_holdout
from .knn_model import fit_knn_model


def run_knn_workflow(
    data,
    target,
    features: Optional[Iterable[str]] = None,
    interactive: bool = False,
    search_space=None,
    random_state: int = 42,
):
    """Run KNN training without prompts; set interactive=True for console feature prompts."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if interactive and features is None:
        answer = input("Specify feature names manually? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            entered = input("Enter comma-separated feature names: ").strip()
            features = [item.strip() for item in entered.split(",") if item.strip()]

    # Split before feature validation so holdout values cannot participate in
    # feature eligibility decisions.
    training, holdout, y_train, y_holdout = train_holdout_split(
        data, target, random_state=random_state
    )
    selected_features = validate_features(
        training.assign(**{target: y_train}), target, features
    )
    X_train = training[selected_features]
    X_holdout = holdout[selected_features]
    folds = create_cv_folds(X_train, n_splits=5, random_state=random_state)
    tuning = optimize_hyperparameters(X_train, y_train, folds, search_space=search_space)
    configuration = tuning["best_configuration"]
    if configuration is None:
        raise RuntimeError("cross-validation produced no usable configuration")

    model = fit_knn_model(X_train, y_train, **configuration)
    holdout_metrics = evaluate_holdout(model, X_holdout, y_holdout)
    return {
        "selected_features": selected_features,
        "hyperparameters": configuration,
        "cross_validation": tuning,
        "holdout_metrics": holdout_metrics,
        "model": model,
        "folds": folds,
    }


main = run_knn_workflow
