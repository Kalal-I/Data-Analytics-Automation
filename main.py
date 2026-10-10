from dataset_stats.description import generate_report
from dataset_stats.open_file import read_csv_files

#read the files
df = read_csv_files()
if (isinstance(df,list)):
    print(df.size())
else:
    print(df.head())
    report = generate_report(df,"student_data.csv","Class")
    print(report)
