"""Random Forest classification model creation and fitting."""
from numbers import Integral, Real

from sklearn.ensemble import RandomForestClassifier


DEFAULTS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "gini",
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": None,
}


def validate_model_params(params):
    values = {**DEFAULTS, **params}
    if isinstance(values["n_estimators"], bool) or not isinstance(values["n_estimators"], Integral) or values["n_estimators"] < 1:
        raise ValueError("n_estimators must be a positive integer")
    if values["max_depth"] is not None and (
        isinstance(values["max_depth"], bool) or not isinstance(values["max_depth"], Integral) or values["max_depth"] < 1
    ):
        raise ValueError("max_depth must be None or a positive integer")
    if isinstance(values["min_samples_split"], bool) or not isinstance(values["min_samples_split"], (Integral, Real)) or values["min_samples_split"] < 2:
        raise ValueError("min_samples_split must be at least 2")
    if isinstance(values["min_samples_leaf"], bool) or not isinstance(values["min_samples_leaf"], (Integral, Real)) or values["min_samples_leaf"] < 1:
        raise ValueError("min_samples_leaf must be positive")
    if values["criterion"] not in {"gini", "entropy", "log_loss"}:
        raise ValueError("criterion must be gini, entropy, or log_loss")
    max_features = values["max_features"]
    if max_features not in {"sqrt", "log2", None}:
        if isinstance(max_features, bool) or not isinstance(max_features, (Integral, Real)):
            raise ValueError("invalid max_features")
        if isinstance(max_features, Integral) and max_features < 1:
            raise ValueError("max_features integer must be positive")
        if isinstance(max_features, Real) and not isinstance(max_features, Integral) and not 0 < max_features <= 1:
            raise ValueError("max_features proportion must be in (0, 1]")
    return values


def create_random_forest_model(**params):
    return RandomForestClassifier(**validate_model_params(params))


def fit_random_forest_model(X, y, **params):
    return create_random_forest_model(**params).fit(X, y)


def predict_random_forest(model, X):
    return model.predict(X)


def predict_proba_random_forest(model, X):
    return model.predict_proba(X)


create_model = create_random_forest_model
fit_model = fit_random_forest_model
predict = predict_random_forest
