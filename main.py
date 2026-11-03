import json
from pathlib import Path

from dataset_stats.description import generate_report
from dataset_stats.model_statistics import generate_model_statistics
from dataset_stats.open_file import read_csv_files
from agents_code.linear_regression_agent import linear_regression_agent
from agents_code.preprocessing_agent import preprocessing_agent
from knn_regression.main_knn_workflow import run_knn_workflow
from linear_regression.model_selection import (
    select_best_feature_report,
)
from linear_regression.optimization_loop import (
    PREPROCESSING_FUNCTIONS,
    _run_greedy_feature_selection,
    _run_regularized_optimization,
)
from linear_regression.split_data import _data_fingerprint
from preproccessing.validation import validate_preprocessing_calls
from preproccessing.deferred_encoding import (
    apply_deferred_encodings,
    partition_deferred_encodings,
)

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
    print(statistics)
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
    top_model = next(
        (model for model in ranked_models if model.get("rank") == 1),
        None,
    )
    if top_model and top_model.get("model_name") == "KNeighborsRegressor":
        knn_data = apply_deferred_encodings(
            df,
            deferred_encoding_operations,
            PREPROCESSING_FUNCTIONS,
        )
        knn_result = run_knn_workflow(
            knn_data,
            target,
            interactive=False,
        )
        knn_summary = {
            "selected_features": knn_result["selected_features"],
            "hyperparameters": knn_result["hyperparameters"],
            "cross_validation": knn_result["cross_validation"],
            "holdout_metrics": knn_result["holdout_metrics"],
        }
        print("KNN regression results:")
        print(json.dumps(knn_summary, indent=2, allow_nan=False))
        return knn_result
    elif top_model and top_model.get("model_name") == "LinearRegressions": #must be a typo, should be "LinearRegression"
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
