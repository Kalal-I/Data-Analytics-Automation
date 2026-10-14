import json

from dataset_stats.description import generate_report
from dataset_stats.model_statistics import generate_model_statistics
from dataset_stats.open_file import read_csv_files
from agents_code.linear_regression_agent import linear_regression_agent
from agents_code.preprocessing_agent import preprocessing_agent
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
'''if (isinstance(df,list)):
    print(df.size())
else:
    report = generate_report(df,path,target)'''
while True:
    report = generate_report(df,path,target)
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
