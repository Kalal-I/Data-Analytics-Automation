import json
from pathlib import Path
from typing import Any

from agents_code.feature_optimization_linear_model import (
    feature_optimization_linear_model_agent,
)
from preproccessing.deferred_encoding import apply_deferred_encodings
from preproccessing.drop_rows_with_missing_values import remove_missing_values
from preproccessing.impute_missing_values import (
    fill_missing_values,
    fill_missing_values_by_group,
)
from preproccessing.label_encoding import label_encode
from preproccessing.min_max_scaler import min_max_scale
from preproccessing.one_hot_encoding import one_hot_encode
from preproccessing.ordinal_encoding import ordinal_encode
from preproccessing.outlier_detection import detect_outliers
from preproccessing.standard_scaler import standard_scale
from .feature_selection import greedy_coefficient_feature_selection
from .fold_manager import create_folds_and_train
from .model_selection import select_best_feature_report
from .results_json import split_result_to_json
from .split_data import _data_fingerprint

PREPROCESSING_FUNCTIONS = {
    "fill_missing_values": fill_missing_values,
    "fill_missing_values_by_group": fill_missing_values_by_group,
    "remove_missing_values": remove_missing_values,
    "one_hot_encode": one_hot_encode,
    "standard_scale": standard_scale,
    "label_encode": label_encode,
    "min_max_scale": min_max_scale,
    "ordinal_encode": ordinal_encode,
    "detect_outliers": detect_outliers,
}

INITIAL_ALPHA_VALUES = (0.1, 0.5, 1.0, 10.0, 100.0)
MODEL_TYPES = {
    "Lasso": "lasso",
    "Ridge": "ridge",
    "ElasticNet": "elastic_net",
}


def _model_optimization_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep experiment context focused on model settings and CV metrics."""
    training = report["training"]
    return {
        "target_column": report["target_column"],
        "model": training["model"],
        "model_type": training["model_type"],
        "hyperparameters": training["hyperparameters"],
        "cross_validation": training["cross_validation"],
        "status": "completed",
    }


def _run_regularized_optimization(
    df,
    target: str,
    current_model_report: dict[str, Any],
    deferred_encoding_operations: dict[str, Any],
) -> list[dict[str, Any]]:
    model_reports: list[dict[str, Any]] = []
    model_experiences: list[dict[str, Any]] = []
    alpha_histories = {
        "ridge_alphas_used": [],
        "lasso_alphas_used": [],
        "elastic_net_alphas_used": [],
        "elastic_net_combinations_used": [],
    }
    encodings_applied = False
    folds_file = None

    for optimization_iteration in range(5):
        model_recommendation = feature_optimization_linear_model_agent(
            current_model_report,
            past_experiences=model_experiences,
            iteration_number=optimization_iteration + 1,
            **alpha_histories,
        )
        print(
            "Model optimization agent returned "
            "(iteration {}):".format(optimization_iteration + 1)
        )
        print(model_recommendation)
        if not model_recommendation:
            break

        model_type = MODEL_TYPES[model_recommendation["model"]]
        if not encodings_applied:
            df = apply_deferred_encodings(
                df,
                deferred_encoding_operations,
                PREPROCESSING_FUNCTIONS,
            )
            print(
                "Standard scaling will be fit on each training partition "
                "and then applied to its validation or holdout data."
            )
            encodings_applied = True
            folds_file = (
                Path(__file__).resolve().parent
                / "folds_{}.json".format(_data_fingerprint(df))
            )

        if optimization_iteration == 0 and model_type in {
            "ridge",
            "lasso",
            "elastic_net",
        }:
            experiment_recommendations = []
            for alpha in INITIAL_ALPHA_VALUES:
                initial_recommendation = {
                    **model_recommendation,
                    "hyperparameters": {
                        **model_recommendation["hyperparameters"],
                        "alpha": alpha,
                    },
                }
                experiment_recommendations.append(initial_recommendation)
        else:
            experiment_recommendations = [model_recommendation]

        for experiment_recommendation in experiment_recommendations:
            hyperparameters = experiment_recommendation["hyperparameters"]
            alpha = float(hyperparameters["alpha"])
            if model_type == "elastic_net":
                model_result = create_folds_and_train(
                    df,
                    target_column=target,
                    model_type=model_type,
                    alpha=alpha,
                    l1_ratio=float(hyperparameters["l1_ratio"]),
                    folds_file=folds_file,
                )
                alpha_histories["elastic_net_combinations_used"].append(
                    {
                        "alpha": alpha,
                        "l1_ratio": float(hyperparameters["l1_ratio"]),
                    }
                )
                alpha_histories["elastic_net_alphas_used"].append(alpha)
            else:
                model_result = create_folds_and_train(
                    df,
                    target_column=target,
                    model_type=model_type,
                    alpha=alpha,
                    folds_file=folds_file,
                )
                alpha_histories["{}_alphas_used".format(model_type)].append(
                    alpha
                )
            model_report = json.loads(
                split_result_to_json(
                    model_result,
                    target_column=target,
                )
            )
            model_reports.append(model_report)
            experience = _model_optimization_summary(model_report)
            model_experiences.append(experience)

        best_report, _ = select_best_feature_report(model_reports)
        current_model_report = _model_optimization_summary(best_report)

    return model_reports


def _run_greedy_feature_selection(
    data,
    target: str,
    best_model_report: dict[str, Any],
    folds_file: Path,
) -> dict[str, Any]:
    """Evaluate coefficient-ranked feature subsets with the chosen configuration."""
    training = best_model_report["training"]
    model_type = training["model_type"]
    if model_type not in {"lasso", "ridge", "elastic_net"}:
        raise ValueError(
            "Coefficient-based selection only supports the fitted linear "
            "regression models in this optimization workflow."
        )
    hyperparameters = training["hyperparameters"]
    alpha = float(hyperparameters["alpha"])
    l1_ratio = (
        float(hyperparameters["l1_ratio"])
        if model_type == "elastic_net"
        else None
    )

    def evaluate_features(features: list[str]) -> dict[str, Any]:
        model_result = create_folds_and_train(
            data,
            target_column=target,
            model_type=model_type,
            alpha=alpha,
            l1_ratio=l1_ratio,
            features=features,
            folds_file=folds_file,
        )
        report = json.loads(
            split_result_to_json(
                model_result,
                target_column=target,
            )
        )
        return report["training"]["cross_validation"]

    selection = greedy_coefficient_feature_selection(
        training["coefficients"],
        evaluate_features,
    )
    return selection
