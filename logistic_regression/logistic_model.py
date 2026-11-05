"""Creation and fitting helpers for scikit-learn logistic regression."""
from numbers import Integral, Real
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


DEFAULTS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "l1_ratio": None,
    "max_iter": 1000,
    "random_state": 42,
}


def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    values = {**DEFAULTS, **params}
    if values["penalty"] not in {"l1", "l2", "elasticnet"}:
        raise ValueError("penalty must be 'l1', 'l2', or 'elasticnet'")
    if isinstance(values["C"], bool) or not isinstance(values["C"], Real) or not np.isfinite(values["C"]) or values["C"] <= 0:
        raise ValueError("C must be a positive finite number")
    if isinstance(values["max_iter"], bool) or not isinstance(values["max_iter"], Integral) or values["max_iter"] <= 0:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(values["random_state"], bool) or not isinstance(values["random_state"], Integral):
        raise ValueError("random_state must be an integer")

    allowed_solvers = {
        "l1": {"saga"},
        "l2": {"lbfgs", "saga"},
        "elasticnet": {"saga"},
    }
    values["solver"] = values["solver"].lower() if isinstance(values["solver"], str) else values["solver"]
    if values["solver"] == "lbfgs" and values["penalty"] in {"l1", "elasticnet"} and "solver" not in params:
        values["solver"] = "saga"
    if values["solver"] not in allowed_solvers[values["penalty"]]:
        raise ValueError("solver {!r} is incompatible with penalty {!r}".format(values["solver"], values["penalty"]))
    if values["penalty"] == "elasticnet":
        ratio = values["l1_ratio"]
        if isinstance(ratio, bool) or not isinstance(ratio, Real) or not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("elasticnet l1_ratio must be in the interval (0, 1]")
    elif values["l1_ratio"] is not None:
        raise ValueError("l1_ratio is only valid for elasticnet penalty")
    return values


def create_model(**params: Any) -> LogisticRegression:
    values = validate_model_params(params)
    kwargs = {
        "penalty": values["penalty"],
        "C": float(values["C"]),
        "solver": values["solver"],
        "max_iter": int(values["max_iter"]),
        "random_state": int(values["random_state"]),
    }
    if values["penalty"] == "elasticnet":
        kwargs["l1_ratio"] = float(values["l1_ratio"])
    return LogisticRegression(**kwargs)


def fit_model(X, y, **params: Any):
    return create_model(**params).fit(X, y)


def predict(model, X):
    return model.predict(X)


def predict_proba(model, X):
    if not hasattr(model, "predict_proba"):
        raise AttributeError("model does not support probability prediction")
    return model.predict_proba(X)


create_logistic_model = create_model
fit_logistic_model = fit_model
