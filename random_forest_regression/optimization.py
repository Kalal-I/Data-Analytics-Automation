"""Bounded randomized cross-validation for Random Forest regression."""
import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score
from .random_forest_model import create_random_forest_model

TUNING_PARAMETERS = (
    "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
    "max_features", "max_leaf_nodes",
)
DEFAULT_SEARCH_SPACE = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
    "max_leaf_nodes": [None, 20, 50, 100],
}


def _metrics(y_true, y_pred):
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    metrics = {
        "r2": round(float(r2_score(y_true, y_pred)), 12),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 12),
        "rmse": round(float(np.sqrt(np.mean(residual ** 2))), 12),
    }
    if not all(np.isfinite(list(metrics.values()))):
        raise ValueError("evaluation produced non-finite metrics")
    return metrics


def evaluate_configuration(X, y, folds, configuration, fixed_params=None, random_state=42):
    fold_metrics = []
    failures = []
    model_params = dict(fixed_params or {})
    if "random_state" not in model_params:
        model_params["random_state"] = random_state
    model_params.update(configuration)
    for fold_number, fold in enumerate(folds, start=1):
        try:
            model = create_random_forest_model(**model_params)
            X_train = X.iloc[fold.train_indices] if hasattr(X, "iloc") else X[fold.train_indices]
            X_validation = X.iloc[fold.validation_indices] if hasattr(X, "iloc") else X[fold.validation_indices]
            y_train = y.iloc[fold.train_indices] if hasattr(y, "iloc") else y[fold.train_indices]
            y_validation = y.iloc[fold.validation_indices] if hasattr(y, "iloc") else y[fold.validation_indices]
            model.fit(X_train, y_train)
            fold_metrics.append(_metrics(y_validation, model.predict(X_validation)))
        except Exception as exc:
            failures.append({"fold": fold_number, "error": str(exc)})

    summary = {}
    for metric in ("r2", "mae", "rmse"):
        values = np.asarray([item[metric] for item in fold_metrics], dtype=float)
        summary[metric] = {
            "mean": round(float(values.mean()), 12) if values.size else None,
            "std": round(float(values.std(ddof=0)), 12) if values.size else None,
        }
    return {"configuration": dict(configuration), "fold_metrics": fold_metrics,
            "failures": failures, "mean_std": summary}


def _valid_configuration(configuration):
    if set(configuration) != set(TUNING_PARAMETERS):
        raise ValueError("each configuration must contain exactly these tuning parameters: {}".format(list(TUNING_PARAMETERS)))
    # Model validation supplies detailed checks; this call is deliberately not
    # used to fit because n_estimators=1 is enough for validation here.
    from .random_forest_model import validate_model_params
    validate_model_params(configuration)


def random_search(X, y, folds, search_space=None, n_iter=40, random_state=42,
                  fixed_params=None):
    """Sample a bounded, reproducible search and return CV results."""
    space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    unknown = set(space) - set(TUNING_PARAMETERS)
    missing = set(TUNING_PARAMETERS) - set(space)
    if unknown:
        raise ValueError("unknown tuning parameters: {}".format(sorted(unknown)))
    if missing:
        raise ValueError("search space is missing tuning parameters: {}".format(sorted(missing)))
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")

    sampled = list(ParameterSampler(space, n_iter=n_iter, random_state=random_state))
    unique = []
    seen = set()
    for configuration in sampled:
        key = tuple(configuration[name] for name in TUNING_PARAMETERS)
        if key in seen:
            continue
        seen.add(key)
        _valid_configuration(configuration)
        unique.append(configuration)

    results = [evaluate_configuration(X, y, folds, configuration, fixed_params, random_state) for configuration in unique]
    usable = [result for result in results if result["mean_std"]["r2"]["mean"] is not None]
    usable.sort(key=lambda result: (
        -result["mean_std"]["r2"]["mean"],
        result["mean_std"]["mae"]["mean"],
        tuple(str(result["configuration"][name]) for name in TUNING_PARAMETERS),
    ))
    return {
        "results": results,
        "best_configuration": dict(usable[0]["configuration"]) if usable else None,
        "n_configurations": len(unique),
        "random_state": random_state,
        "selection_metric": "mean_cv_r2",
    }


def optimize_hyperparameters(X, y, folds, search_space=None, n_iter=40,
                             random_state=42, fixed_params=None):
    return random_search(X, y, folds, search_space, n_iter, random_state, fixed_params)


def evaluate_holdout(model, X_holdout, y_holdout):
    return _metrics(y_holdout, model.predict(X_holdout))
