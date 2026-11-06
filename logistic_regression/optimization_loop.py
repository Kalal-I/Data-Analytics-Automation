"""Agent-guided regularization optimization for logistic regression."""

from typing import Any

from agents_code.logistic_regression_agent import (
    decision_to_configuration,
    logistic_regression_agent,
)
from linear_regression.optimization_loop import PREPROCESSING_FUNCTIONS
from preproccessing.deferred_encoding import apply_deferred_encodings

from .data_split import train_holdout_split
from .feature_selection import validate_features
from .folds import create_cv_folds
from .logistic_model import fit_model
from .optimization import evaluate_configuration, evaluate_holdout


INITIAL_C_VALUES = (0.1, 0.5, 1.0, 10.0, 100.0)
MODEL_TYPES = {
    "Lasso": "lasso",
    "Ridge": "ridge",
    "ElasticNet": "elastic_net",
}


def _optimization_summary(report: dict[str, Any]) -> dict[str, Any]:
    training = report["training"]
    return {
        "target_column": report["target_column"],
        "model": training["model"],
        "model_type": training["model_type"],
        "hyperparameters": training["hyperparameters"],
        "cross_validation": training["cross_validation"],
        "status": "completed",
    }


def _performance_key(report: dict[str, Any]) -> tuple[float, ...] | None:
    metrics = report["training"]["cross_validation"]["mean_std"]
    names = ("f1", "accuracy", "precision", "recall")
    means = [metrics[name]["mean"] for name in names]
    if means[0] is None:
        return None
    score = tuple(
        value if value is not None else float("-inf")
        for value in means
    )
    f1_std = metrics["f1"]["std"]
    return score + (
        -(f1_std if f1_std is not None else float("inf")),
    )


def _best_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        (score, report)
        for report in reports
        if (score := _performance_key(report)) is not None
    ]
    if not usable:
        raise RuntimeError(
            "Cross-validation produced no usable Logistic Regression metrics."
        )
    return max(usable, key=lambda item: item[0])[1]


def run_logistic_optimization(
    data,
    target: str,
    current_model_report: dict[str, Any],
    deferred_encoding_operations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run up to five agent iterations on a shared split and CV fold set."""
    if not isinstance(current_model_report, dict):
        raise TypeError("current_model_report must be a dictionary")
    if deferred_encoding_operations is not None and not isinstance(
        deferred_encoding_operations, dict
    ):
        raise TypeError("deferred_encoding_operations must be a dictionary or None")

    encoded_data = apply_deferred_encodings(
        data.copy(),
        deferred_encoding_operations or {},
        PREPROCESSING_FUNCTIONS,
    )
    features = validate_features(encoded_data, target)
    X_train_full, X_holdout, y_train, y_holdout = train_holdout_split(
        encoded_data,
        target,
        random_state=42,
    )
    X_train = X_train_full[features]
    X_holdout = X_holdout[features]
    folds = create_cv_folds(X_train, y_train, random_state=42)

    reports: list[dict[str, Any]] = []
    experiences: list[dict[str, Any]] = []
    c_histories: dict[str, list[Any]] = {
        "ridge_c_values_used": [],
        "lasso_c_values_used": [],
        "elastic_net_c_values_used": [],
        "elastic_net_combinations_used": [],
        "threshold_values_used": [],
    }
    best_report = None
    agent_iterations = 0

    for iteration_number in range(1, 6):
        decision = logistic_regression_agent(
            current_model_report,
            past_experiences=experiences,
            iteration_number=iteration_number,
            **c_histories,
        )
        print(
            "Logistic Regression optimization agent returned "
            "(iteration {}):".format(iteration_number)
        )
        print(decision)
        agent_iterations += 1
        if not decision:
            break

        base_configuration = decision_to_configuration(decision)
        model_type = MODEL_TYPES[decision["model"]]
        if iteration_number == 1:
            configurations = []
            for C_value in INITIAL_C_VALUES:
                configuration = dict(base_configuration)
                configuration["C"] = C_value
                configurations.append(configuration)
        else:
            configurations = [base_configuration]

        for configuration in configurations:
            cross_validation = evaluate_configuration(
                X_train,
                y_train,
                folds,
                configuration,
                metric="f1",
            )
            hyperparameters = {
                "C": configuration["C"],
                "threshold": configuration["threshold"],
            }
            if model_type == "elastic_net":
                hyperparameters["l1_ratio"] = configuration["l1_ratio"]
            if configuration.get("max_iter", 1000) != 1000:
                hyperparameters["max_iter"] = configuration["max_iter"]

            report = {
                "schema_version": 1,
                "target_column": target,
                "features": list(features),
                "training": {
                    "model": decision["model"],
                    "model_type": model_type,
                    "hyperparameters": hyperparameters,
                    "cross_validation": cross_validation,
                    "holdout_metrics": None,
                },
            }
            reports.append(report)
            experiences.append(_optimization_summary(report))

            if model_type == "elastic_net":
                c_histories["elastic_net_c_values_used"].append(
                    configuration["C"]
                )
                c_histories["elastic_net_combinations_used"].append(
                    {
                        "C": configuration["C"],
                        "l1_ratio": configuration["l1_ratio"],
                    }
                )
            else:
                c_histories[
                    "{}_c_values_used".format(model_type)
                ].append(configuration["C"])
            c_histories["threshold_values_used"].append(
                configuration["threshold"]
            )

        best_report = _best_report(reports)
        current_model_report = _optimization_summary(best_report)

    if best_report is None:
        raise ValueError(
            "The Logistic Regression agent did not recommend a configuration "
            "to evaluate."
        )

    best_training = best_report["training"]
    best_configuration = dict(
        best_training["cross_validation"]["configuration"]
    )
    best_configuration["random_state"] = 42
    model = fit_model(
        X_train,
        y_train,
        **best_configuration,
    )
    holdout_metrics = evaluate_holdout(
        model,
        X_holdout,
        y_holdout,
        threshold=best_configuration["threshold"],
    )
    best_report["training"]["holdout_metrics"] = holdout_metrics

    return {
        "model": model,
        "selected_features": list(features),
        "best_configuration": {
            key: value
            for key, value in best_configuration.items()
            if key != "random_state"
        },
        "best_report": best_report,
        "model_reports": reports,
        "cross_validation": best_training["cross_validation"],
        "holdout_metrics": holdout_metrics,
        "agent_iterations": agent_iterations,
        "folds": folds,
    }
