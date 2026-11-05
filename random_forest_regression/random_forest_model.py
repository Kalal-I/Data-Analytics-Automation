"""Creation, fitting, and prediction for Random Forest regression."""
import math
from numbers import Integral, Real
from sklearn.ensemble import RandomForestRegressor

DEFAULTS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_leaf_nodes": None,
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": -1,
}


def validate_model_params(params):
    values = {**DEFAULTS, **params}
    n_estimators = values["n_estimators"]
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, Integral)
        or n_estimators < 1
    ):
        raise ValueError("n_estimators must be a positive integer")
    max_depth = values["max_depth"]
    if max_depth is not None and (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, Integral)
        or max_depth < 1
    ):
        raise ValueError("max_depth must be None or a positive integer")
    min_samples_split = values["min_samples_split"]
    if isinstance(min_samples_split, bool):
        raise ValueError("min_samples_split must be an integer >= 2 or a float in (0, 1]")
    if isinstance(min_samples_split, Integral):
        if min_samples_split < 2:
            raise ValueError("min_samples_split integer values must be at least 2")
    elif (
        not isinstance(min_samples_split, Real)
        or not math.isfinite(min_samples_split)
        or not 0 < min_samples_split <= 1
    ):
        raise ValueError("min_samples_split must be an integer >= 2 or a float in (0, 1]")

    min_samples_leaf = values["min_samples_leaf"]
    if isinstance(min_samples_leaf, bool):
        raise ValueError("min_samples_leaf must be a positive integer or a float in (0, 0.5]")
    if isinstance(min_samples_leaf, Integral):
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf integer values must be positive")
    elif (
        not isinstance(min_samples_leaf, Real)
        or not math.isfinite(min_samples_leaf)
        or not 0 < min_samples_leaf <= 0.5
    ):
        raise ValueError("min_samples_leaf must be a positive integer or a float in (0, 0.5]")

    max_leaf_nodes = values["max_leaf_nodes"]
    if max_leaf_nodes is not None and (
        isinstance(max_leaf_nodes, bool)
        or not isinstance(max_leaf_nodes, Integral)
        or max_leaf_nodes < 2
    ):
        raise ValueError("max_leaf_nodes must be None or an integer of at least 2")
    max_features = values["max_features"]
    if max_features is not None and max_features not in ("sqrt", "log2"):
        if isinstance(max_features, bool):
            raise ValueError(
                "max_features must be 'sqrt', 'log2', None, a positive integer, or a float in (0, 1]"
            )
        if isinstance(max_features, Integral):
            if max_features < 1:
                raise ValueError("max_features integer values must be positive")
        elif (
            not isinstance(max_features, Real)
            or not math.isfinite(max_features)
            or not 0 < max_features <= 1
        ):
            raise ValueError(
                "max_features must be 'sqrt', 'log2', None, a positive integer, or a float in (0, 1]"
            )
    return values


def create_random_forest_model(**params):
    values = validate_model_params(params)
    return RandomForestRegressor(**values)


def fit_random_forest_model(X, y, **params):
    return create_random_forest_model(**params).fit(X, y)


def predict_random_forest(model, X):
    return model.predict(X)


create_model = create_random_forest_model
fit_model = fit_random_forest_model
predict = predict_random_forest
