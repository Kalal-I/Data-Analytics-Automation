"""Pipeline-based KNN regression model."""
from numbers import Integral
from typing import Any, Dict, Optional
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULTS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
    "n_jobs": -1,
}


def validate_knn_params(params: Dict[str, Any]) -> Dict[str, Any]:
    values = {**DEFAULTS, **params}
    if isinstance(values["n_neighbors"], bool) or not isinstance(values["n_neighbors"], Integral) or values["n_neighbors"] < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if values["weights"] not in {"uniform", "distance"}:
        raise ValueError("weights must be 'uniform' or 'distance'")
    if values["algorithm"] not in {"auto", "ball_tree", "kd_tree", "brute"}:
        raise ValueError("unsupported algorithm: {}".format(values["algorithm"]))
    if values["leaf_size"] <= 0:
        raise ValueError("leaf_size must be positive")
    if values["p"] <= 0:
        raise ValueError("p must be positive")
    if values["metric"] != "minkowski":
        # KNN supports other sklearn metrics, but the optimization contract
        # intentionally uses Minkowski distance.
        pass
    return {key: values[key] for key in DEFAULTS}


def create_knn_model(**params: Any) -> KNeighborsRegressor:
    values = validate_knn_params(params)
    return KNeighborsRegressor(**values)


def create_knn_pipeline(**params: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", create_knn_model(**params)),
    ])


def fit_knn_model(X, y, **params: Any):
    pipeline = create_knn_pipeline(**params)
    return pipeline.fit(X, y)


def predict_knn(model, X):
    return model.predict(X)


create_model = create_knn_model
fit_model = fit_knn_model
predict = predict_knn
