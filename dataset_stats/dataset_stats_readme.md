# Dataset Stats

This folder contains two Python files for loading CSV data and generating a
JSON data-quality report.

## Files

### `open_file.py`

Provides `read_csv_files()`, which prompts for a CSV file or folder path.

- For a CSV file, it returns one pandas DataFrame.
- For a folder, it reads the CSV files directly inside that folder. It returns
  a DataFrame if there is one CSV, or a list of DataFrames if there are
  multiple CSVs.
- It raises an error if the path does not exist, the selected file is not a
  CSV, or the folder has no CSV files.

Subfolders are not searched. The function returns DataFrames only; it does not
return the source filenames.

### `description.py`

Provides `generate_report(data, file_names, target_column)`, which returns a
formatted JSON string. `data` can be one pandas DataFrame or a list/tuple of
DataFrames. Supply one filename for each DataFrame and a target-column name
that exists in every DataFrame.

Each dataset report contains:

- The source filename's basename.
- Overall row and column counts and the percentage of duplicate rows.
- Per-column statistics:
  - Numerical columns: minimum, maximum, mean, median, missing percentage,
    and count of non-numeric values where applicable.
  - Categorical columns: number of unique values, up to 10 category examples,
    and missing percentage.
  - Non-categorical string columns: missing percentage and duplicate-value
    count.
- Target-column name, detected type, and the same statistics in a separate
  section. The target is excluded from the other column sections.

String-like columns are classified by converting any numeric-looking values
first. Otherwise, a column is considered non-categorical when the values left
after removing duplicate rows and nulls are all unique; it is categorical
when those values include repeats.

## Example usage

```python
import pandas as pd

from description import generate_report

file_paths = [
    r"C:\data\customers.csv",
    r"C:\data\orders.csv"
]
dataframes = [pd.read_csv(path) for path in file_paths]

report_json = generate_report(dataframes, file_paths, "target")
print(report_json)
```

For a single DataFrame, pass its filename as a string:

```python
report_json = generate_report(dataframe, r"C:\data\customers.csv", "target")
```

The target column must exist in each DataFrame, and the number of filenames
must match the number of DataFrames.

`open_file.read_csv_files()` and `description.generate_report()` are separate
helpers. Since the loader does not return filenames, provide the corresponding
filenames separately when creating a report.

## Requirements

- Python
- pandas
