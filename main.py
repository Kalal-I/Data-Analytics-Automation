import json
from pathlib import Path
from typing import Any

from dataset_stats.description import generate_report
from dataset_stats.model_statistics import generate_model_statistics
from dataset_stats.open_file import read_csv_files
from agents_code.linear_regression_agent import linear_regression_agent
from agents_code.feature_optimization_linear_model import (
    feature_optimization_linear_model_agent,
)
from agents_code.preprocessing_agent import preprocessing_agent
from linear_regression import split_result_to_json
from linear_regression.fold_manager import create_folds_and_train
from linear_regression.model_selection import (
    select_best_feature_report,
)
from linear_regression.feature_selection import (
    greedy_coefficient_feature_selection,
)
from linear_regression.split_data import _data_fingerprint
from preproccessing.drop_rows_with_missing_values import remove_missing_values
from preproccessing.impute_missing_values import (
    fill_missing_values,
    fill_missing_values_by_group,
)
from preproccessing.one_hot_encoding import one_hot_encode
from preproccessing.standard_scaler import standard_scale
from preproccessing.label_encoding import label_encode
from preproccessing.min_max_scaler import min_max_scale
from preproccessing.ordinal_encoding import ordinal_encode
from preproccessing.outlier_detection import detect_outliers
from preproccessing.validation import validate_preprocessing_calls
from preproccessing.deferred_encoding import (
    apply_deferred_encodings,
    partition_deferred_encodings,
)

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
                / "linear_regression"
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


def main():
    #read the files
    path = input("Enter the path to the CSV file or directory: ")
    df = read_csv_files(path)
    if isinstance(df, list):
        raise ValueError("The main pipeline requires a single CSV DataFrame.")
    target = input("Enter the target column name: ")
    report = ""
    deferred_encoding_operations = {}
    '''if (isinstance(df,list)):
        print(df.size())
    else:
        report = generate_report(df,path,target)'''
    while True:
        report = generate_report(df,path,target)
        print(report)
        operation_to_do = preprocessing_agent(report)
        print("Initally:")
        print(operation_to_do)

        operation_to_do = validate_preprocessing_calls(operation_to_do, report)
        operation_to_do, deferred_operations = partition_deferred_encodings(
            operation_to_do
        )
        for column, operations in deferred_operations.items():
            deferred_encoding_operations.setdefault(column, operations)
        print("After validation:")
        print(operation_to_do)
        print("The following operations will be performed on the DataFrame:")
        if operation_to_do == {}:
            print(operation_to_do)
            break
        for column, operations in operation_to_do.items():
            for operation in operations:
                if operation["function"] == "standard_scale":
                    continue
                function = PREPROCESSING_FUNCTIONS[operation["function"]]
                df = function(df, **operation["inputs"])
    print("Preprocessing completed. The processed DataFrame is:")
    print(df)
    print(
        "Categorical encodings deferred until after model selection:",
        sorted(deferred_encoding_operations),
    )

    dataset_description = json.loads(generate_report(df, path, target))
    statistics = json.loads(generate_model_statistics(df, target))
    label_type = dataset_description["Reports"][0]["Target column"]["Type"]

    regression_recommendation = linear_regression_agent(
        dataset_description=dataset_description,
        target_column=target,
        label_type=label_type,
        statistics=statistics,
        row_count=len(df),
    )
    print("Linear regression agent returned:")
    print(regression_recommendation)

    ranked_models = regression_recommendation["ranked_models"]
    if ranked_models:
        current_model_report = {
            "target_column": target,
            "dataset_description": dataset_description,
            "statistics": statistics,
            "initial_model_ranking": regression_recommendation,
        }
        model_reports = _run_regularized_optimization(
            df,
            target,
            current_model_report,
            deferred_encoding_operations,
        )

        if not model_reports:
            raise ValueError(
                "The model optimization agent did not recommend a configuration "
                "to evaluate."
            )

        best_model_report, _ = select_best_feature_report(
            model_reports
        )
        selection_data = apply_deferred_encodings(
            df,
            deferred_encoding_operations,
            PREPROCESSING_FUNCTIONS,
        )
        selection_folds_file = (
            Path(__file__).resolve().parent
            / "linear_regression"
            / "folds_{}.json".format(_data_fingerprint(selection_data))
        )
        feature_selection_result = _run_greedy_feature_selection(
            selection_data,
            target,
            best_model_report,
            selection_folds_file,
        )
        final_selection_summary = {
            "accepted_features": feature_selection_result["accepted_features"],
            "cross_validation": feature_selection_result[
                "accepted_cross_validation"
            ],
        }
        print("Final accepted features and scores:")
        print(json.dumps(final_selection_summary, indent=2, allow_nan=False))
        return feature_selection_result


if __name__ == "__main__":
    main()
