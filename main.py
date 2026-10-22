import json
from pathlib import Path

from dataset_stats.description import generate_report
from dataset_stats.model_statistics import generate_model_statistics
from dataset_stats.open_file import read_csv_files
from agents_code.linear_regression_agent import linear_regression_agent
from agents_code.feature_optimization_linear_model import (
    feature_optimization_linear_model_agent,
)
from agents_code.preprocessing_agent import preprocessing_agent
from linear_regression import create_folds_and_train, split_result_to_json
from linear_regression.model_selection import select_best_feature_report
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

#read the files
path = input("Enter the path to the CSV file or directory: ")
df = read_csv_files(path)
target = input("Enter the target column name: ")
report = ""
encoded_columns = set()
one_hot_encoded_features = set()
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
    operation_to_do = {
        column: operations
        for column, operations in operation_to_do.items()
        if column not in encoded_columns
    }
    print("After validation:")
    print(operation_to_do)
    print("The following operations will be performed on the DataFrame:")
    if(operation_to_do=={}):
        print(operation_to_do)
        break
    for column, operations in operation_to_do.items():
        for operation in operations:
            function = PREPROCESSING_FUNCTIONS[operation["function"]]
            columns_before = set(df.columns)
            df = function(df, **operation["inputs"])
            if operation["function"] == "one_hot_encode":
                one_hot_encoded_features.update(set(df.columns) - columns_before)
            if operation["function"] in {
                "one_hot_encode",
                "label_encode",
                "ordinal_encode",
            }:
                encoded_columns.update(set(df.columns) - columns_before)
                if column in df.columns:
                    encoded_columns.add(column)
print("Preprocessing completed. The processed DataFrame is:")
print(df)
print(encoded_columns)

dataset_description = json.loads(generate_report(df, path, target))
statistics = json.loads(
    generate_model_statistics(
        df,
        target,
        one_hot_encoded_features=sorted(one_hot_encoded_features),
    )
)
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
if ranked_models and ranked_models[0]["model_name"] == "LinearRegression":
    folds_file = (
        Path(__file__).resolve().parent
        / "linear_regression"
        / "folds_{}.json".format(_data_fingerprint(df))
    )
    linear_regression_result = create_folds_and_train(
        df,
        target_column=target,
        folds_file=folds_file,
    )
    linear_regression_report = split_result_to_json(
        linear_regression_result,
        target_column=target,
    )
    print("LinearRegression training and cross-validation metrics:")
    print(linear_regression_report)
    linear_regression_reports = [json.loads(linear_regression_report)]

    for optimization_iteration in range(5):
        current_linear_regression_report = linear_regression_reports[-1]
        feature_optimization_recommendation = (
            feature_optimization_linear_model_agent(
                current_linear_regression_report,
                past_experiences=linear_regression_reports[:-1],
                iteration_number=optimization_iteration + 1,
            )
        )
        print(
            "Feature optimization agent returned "
            "(iteration {}):".format(optimization_iteration + 1)
        )
        print(feature_optimization_recommendation)

        if not feature_optimization_recommendation:
            break
        if feature_optimization_recommendation.get("action") != "remove_features":
            raise ValueError(
                "Feature optimization agent returned an unsupported action."
            )

        remaining_features = feature_optimization_recommendation[
            "remaining_features"
        ]
        optimized_linear_regression_result = create_folds_and_train(
            df,
            target_column=target,
            features=remaining_features,
            folds_file=folds_file,
        )
        optimized_linear_regression_report = split_result_to_json(
            optimized_linear_regression_result,
            target_column=target,
        )
        linear_regression_reports.append(
            json.loads(optimized_linear_regression_report)
        )
        print("LinearRegression metrics with optimized features:")
        #print(optimized_linear_regression_report)

    last_linear_regression_report = linear_regression_reports[-1]
    print("Last LinearRegression iteration result:")
    print(json.dumps(last_linear_regression_report, indent=2))

    best_linear_regression_report, ranking_metrics = select_best_feature_report(
        linear_regression_reports
    )
    best_cross_validation = best_linear_regression_report["training"][
        "cross_validation"
    ]
    print("Preferred LinearRegression feature set:")
    print("Ranking criteria:", ", ".join(ranking_metrics))
    if "adjusted_r2" not in ranking_metrics:
        print(
            "Adjusted R² was not used for ranking because it is unavailable "
            "for at least one eligible feature set."
        )
    for metric in ("r2", "adjusted_r2", "rmse", "mae"):
        metric_result = best_cross_validation.get(metric)
        if metric_result is None or metric_result.get("mean") is None:
            print("{}: unavailable".format(metric))
            continue
        print(
            "{}: {} (std: {})".format(
                metric,
                metric_result["mean"],
                metric_result["std"],
            )
        )
    print("Features:", best_linear_regression_report["features"])
    print("Feature count:", len(best_linear_regression_report["features"]))
    print("Model:", best_linear_regression_report["training"]["model"])
