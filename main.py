from dataset_stats.description import generate_report
from dataset_stats.open_file import read_csv_files
from agents_code.preprocessing_agent import preprocessing_agent
from preproccessing.impute_missing_values import fill_missing_values
from preproccessing.one_hot_encoding import one_hot_encode
from preproccessing.standard_scaler import standard_scale
from preproccessing.label_encoding import label_encode
from preproccessing.min_max_scaler import min_max_scale
from preproccessing.ordinal_encoding import ordinal_encode
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
print("The following operations will be performed on the DataFrame:")
print(operation_to_do)
for i in operation_to_do:
    lst = operation_to_do[i]
    col = i
    for j in range(len(lst)):
        d = lst[j]
        if (d["function"] == "fill_missing_values"):
            df = fill_missing_values(df,d["inputs"]["strategy"],d["inputs"]["columns"])
        elif (d["function"] == "one_hot_encode"):
            df = one_hot_encode(df,d["inputs"]["columns"])
        elif (d["function"] == "standard_scale"):
            df = standard_scale(df,d["inputs"]["columns"])
        elif (d["function"] == "label_encode"):
            df = label_encode(df,d["inputs"]["columns"])
        elif (d["function"] == "min_max_scale"):
            df = min_max_scale(df,d["inputs"]["columns"])
        elif (d["function"] == "ordinal_encode"):
            df = ordinal_encode(df,d["inputs"]["columns"])
print("Preprocessing completed. The processed DataFrame is:")
print(df.head())