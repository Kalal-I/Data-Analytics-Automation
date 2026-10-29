"""Deterministic cross-validation for KNN hyperparameters."""
from itertools import product
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from .knn_model import create_knn_pipeline

DEFAULT_SEARCH_SPACE = {
    "n_neighbors": [3, 5, 7, 11, 15, 21, 31],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}


def _metrics(y_true, y_pred):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    mse = float(np.mean(residual ** 2))
    return {"r2": float(r2_score(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mse))}


def evaluate_configuration(X, y, folds, configuration):
    fold_metrics = []
    failures = []
    for fold_number, fold in enumerate(folds, start=1):
        train_count = len(fold.train_indices)
        k = int(configuration["n_neighbors"])
        if k > train_count:
            failures.append({"fold": fold_number, "error": "n_neighbors={} exceeds the fold training sample count {}".format(k, train_count)})
            continue
        try:
            pipeline = create_knn_pipeline(**configuration)
            pipeline.fit(X.iloc[fold.train_indices] if hasattr(X, "iloc") else X[fold.train_indices],
                          y.iloc[fold.train_indices] if hasattr(y, "iloc") else y[fold.train_indices])
            predictions = pipeline.predict(X.iloc[fold.validation_indices] if hasattr(X, "iloc") else X[fold.validation_indices])
            truth = y.iloc[fold.validation_indices] if hasattr(y, "iloc") else y[fold.validation_indices]
            scores = _metrics(truth, predictions)
            if not all(np.isfinite(list(scores.values()))):
                raise ValueError("evaluation produced non-finite metrics")
            fold_metrics.append(scores)
        except Exception as exc:
            failures.append({"fold": fold_number, "error": str(exc)})

    summary = {}
    for metric in ("r2", "mae", "rmse"):
        values = np.asarray([item[metric] for item in fold_metrics], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
        }
    return {"configuration": dict(configuration), "fold_metrics": fold_metrics, "failures": failures, "mean_std": summary}


def grid_search(X, y, folds, search_space=None):
    """Evaluate a bounded grid and return the ordered search report."""
    search_space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    required = set(search_space)
    if required - {"n_neighbors", "weights", "p"}:
        raise ValueError("search_space only supports n_neighbors, weights, and p")
    keys = [key for key in ("n_neighbors", "weights", "p") if key in search_space]
    if not keys:
        raise ValueError("search_space cannot be empty")
    configurations = [dict(zip(keys, values)) for values in product(*(search_space[key] for key in keys))]
    for configuration in configurations:
        configuration.update({key: DEFAULT_SEARCH_SPACE[key][0] for key in DEFAULT_SEARCH_SPACE if key not in configuration})
        neighbor_value = configuration["n_neighbors"]
        if isinstance(neighbor_value, bool) or not isinstance(neighbor_value, (int, np.integer)) or neighbor_value < 1:
            raise ValueError("n_neighbors values must be positive integers")
        if configuration["weights"] not in {"uniform", "distance"}:
            raise ValueError("unsupported weights value")
        if configuration["p"] <= 0:
            raise ValueError("p must be positive")

    results = [evaluate_configuration(X, y, folds, {**configuration, "metric": "minkowski", "algorithm": "auto"}) for configuration in configurations]
    def rank(result):
        mean = result["mean_std"]["r2"]["mean"]
        mae = result["mean_std"]["mae"]["mean"]
        return (-(mean if mean is not None else -np.inf), mae if mae is not None else np.inf,
                tuple(result["configuration"][key] for key in ("n_neighbors", "weights", "p") ))
    results.sort(key=rank)
    return {"results": results, "best_configuration": results[0]["configuration"] if results and results[0]["mean_std"]["r2"]["mean"] is not None else None}


def optimize_hyperparameters(X, y, folds, search_space=None):
    return grid_search(X, y, folds, search_space)


def elbow_analysis(X, y, folds, k_values, weights="uniform", p=2):
    results = []
    if not k_values:
        raise ValueError("k_values cannot be empty")
    for k in k_values:
        evaluation = evaluate_configuration(X, y, folds, {"n_neighbors": k, "weights": weights, "p": p, "metric": "minkowski"})
        results.append({"n_neighbors": k, "mean_error": evaluation["mean_std"]["mae"]["mean"], "std_error": evaluation["mean_std"]["mae"]["std"], "failures": evaluation["failures"]})
    return results


def evaluate_holdout(model, X_holdout, y_holdout):
    predictions = model.predict(X_holdout)
    return _metrics(y_holdout, predictions)
