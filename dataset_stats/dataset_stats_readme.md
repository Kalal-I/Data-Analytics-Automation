# Dataset Stats

This folder contains helpers for loading CSV data, generating a general
data-quality report, and producing regression-specific statistical evidence.

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

Column type detection is shared with `column_analysis.py` so the general and
regression reports use one numeric-conversion policy. Native numeric columns
are numerical. An object/string column is numerical only when strictly more
than half of its non-missing values parse as numeric (including non-finite
numeric values); a tie does not qualify. Unsupported cell objects count as
invalid rather than being hashed or treated as categories. Otherwise, a column
with repeated non-missing values is categorical, and a column with unique
non-missing values is reported as a non-categorical string. This classification
uses the column's values, not values remaining after duplicate rows are removed.

### `model_statistics.py`

Provides `generate_model_statistics(data, target_column, correlation_threshold=0.8, one_hot_encoded_features=None)`,
which returns strict JSON containing regression-specific evidence rather than
repeating the general dataset profile above. Alongside the existing fields, it
reports target numeric validity (without claiming target-only evidence can
establish suitability), feature-target Pearson and Spearman correlations,
paired observation counts, and numeric-predictor pairs whose absolute Pearson
correlation meets the configurable heuristic threshold. The original fields
and correlation metrics remain present; added metrics are separate top-level
sections:

- `distribution_summaries`: skewness, selected quantiles, Tukey 1.5-IQR
  outlier fences/counts, and population variance for the target and usable
  numeric predictors.
- `feature_target_mutual_information`: deterministic k-nearest-neighbor
  estimates for numeric feature-target pairs. Estimates are non-negative and
  sample-size dependent; categorical predictors are not encoded or included.
- `feature_cardinality_and_effective_sample_ratio`: non-missing cardinality
  for every predictor, plus the ratio of rows complete across the target and
  all numerically detected predictors to the number of numeric predictors.
- `feature_scale_summaries`: minimum, maximum, mean, standard deviation, and
  range for numeric target/predictor values.
- `missingness_patterns`: per-column missing counts/percentages and the count
  of rows for each combination of columns missing values.
- `nonlinear_relationship_screening`: in-sample R² gain from univariate
  quadratic terms and pairwise product terms, with incomplete/constant input
  limitations reported explicitly.
- `linear_baseline_residual_diagnostics`: a linear baseline fit using all
  complete rows of the target and numerically detected predictors, with
  in-sample fit and residual summaries. This is diagnostic evidence, not a
  holdout evaluation; standardized residual outliers use an absolute
  three-standard-deviation cutoff.

Distribution and relationship calculations use the existing safe numeric
conversion policy and exclude missing, invalid, non-finite, and
precision-loss values. The nonlinear/interaction screens and baseline
residuals are exploratory in-sample heuristics; they are not significance
tests and do not estimate out-of-sample performance. Missing values are
reported, not imputed, and categorical features remain visible in cardinality
and missingness summaries without being treated as numeric predictors.

Native numeric predictors are treated as numerical. A string/object predictor
is treated as numerical only when strictly more than half of its non-missing
values parse as numeric; a tie is not enough. Non-finite numeric strings count
as numeric for detection but are excluded from calculations. Unsupported cell
values (such as lists, mappings, arrays, or custom objects) count as invalid;
columns that are not numerically detected and contain such values are
classified as `unsupported` without hashing or comparing those cells.
Integer values that cannot be represented exactly in float64 are counted as
precision-loss values and excluded from calculations. Complex scalar cells
count as numeric-type evidence, but nonzero imaginary parts are invalid real
observations. The target report separates numeric-data validity from full
regression suitability, which cannot be determined from target statistics alone.
Undefined Pearson and Spearman correlations are JSON `null` and have independent
limitation fields. Constant, all-missing, categorical, and non-categorical
string features are distinguished where relevant to correlation eligibility.
Pearson marks inputs near-constant only when their spread is within one float64
ULP at their magnitude; this representational test cannot determine whether
small variations are meaningful for a particular domain or instrument. Target
and feature counts separately report missing, invalid, non-finite, and
precision-loss cells.

Pass names of one-hot encoded columns through `one_hot_encoded_features`.
The output includes them and binary-feature interpretation guidance in
`encoding_metadata`, and marks those feature reports accordingly. Binary
target associations, mutual information, and feature-feature correlations
remain available. For verified 0/1 features, the quadratic screen is qualified
because squaring the feature adds no information. Declared columns are checked
against finite 0/1 values; non-binary values produce an explicit warning.

The function does not modify the input DataFrame, compare models, or select
features. Correlation is association, not causation or a guarantee of
predictive value.

```python
from dataset_stats.model_statistics import generate_model_statistics

statistics_json = generate_model_statistics(
    dataframe,
    "target",
    one_hot_encoded_features=["Property_Type_House", "Renovated_Yes"],
)
```

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
