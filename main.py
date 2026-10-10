from dataset_stats.description import generate_report
from dataset_stats.open_file import read_csv_files
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
if (isinstance(df,list)):
    print(df.size())
else:
    report = generate_report(df,path,target)

operation_to_do = preprocessing_agent(report)
print("Initally:")
print(operation_to_do)
operation_to_do = validate_preprocessing_calls(operation_to_do, report)
print("After validation:")
print(operation_to_do)
print("The following operations will be performed on the DataFrame:")
for operations in operation_to_do.values():
    for operation in operations:
        function = PREPROCESSING_FUNCTIONS[operation["function"]]
        df = function(df, **operation["inputs"])
print("Preprocessing completed. The processed DataFrame is:")
print(df.head())