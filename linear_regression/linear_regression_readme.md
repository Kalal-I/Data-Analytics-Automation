# Linear Regression Package

This folder provides utilities for fitting and evaluating scikit-learn
`LinearRegression` models using pandas `DataFrame` inputs. Its public
functions support model fitting, a reproducible holdout split, five-fold
cross-validation, reusable saved split assignments, and JSON serialization of
the results.

The package does **not** clean data, encode categorical variables, impute
missing values, scale features, select features, or choose a model
automatically. Prepare the input DataFrame before calling the training
functions. The selected predictors and target must be finite real numeric
values.

## Files

### `__init__.py`

Re-exports the supported package-level API:

- `train_linear_regression`
- `split_and_train`
- `create_or_load_folds`
- `create_folds_and_train`
- `split_result_to_json`

### `linear_model.py`

Implements numeric validation and `train_linear_regression()`. The fitted
model also carries `training_row_count_`, the number of rows used to fit it;
the JSON serializer uses this to report the training sample size.

### `split_data.py`

Implements the reproducible holdout split, cross-validation metrics, and
validation for saved fold assignments.

### `fold_manager.py`

Implements creation, atomic persistence, validation, and reuse of fixed
holdout/CV assignments. The default fold file path is
`stored_folds.json` in this folder. It is created when a call first needs to
persist an assignment; it is not required to exist beforehand.

### `results_json.py`

Implements strict JSON serialization of a training result. Holdout metrics
and row-level predictions are omitted unless explicitly requested.

### `test_linear_regression.py`

Tests training and validation behavior, deterministic splits, CV, fold
persistence/reuse, JSON output, index handling, and holdout-report options.

## Public API

### `train_linear_regression`

```python
train_linear_regression(
    training_data: pd.DataFrame,
    target_column: str,
    features: Sequence[str] | str | None = None,
) -> sklearn.linear_model.LinearRegression
```

Fits and returns a model using all rows in `training_data`.

- `training_data`: DataFrame containing the target and predictors.
- `target_column`: name of the numeric target column.
- `features`: one feature name, or a list/tuple of feature names. If omitted,
  every column other than the target is used. Explicit feature order is
  preserved.
- Returns a fitted `LinearRegression` model. The model's
  `training_row_count_` attribute is the number of rows used for fitting.

Raises a clear `TypeError` or `ValueError` for invalid inputs, including a
non-DataFrame or empty input, missing target/features, duplicate DataFrame
column names, invalid feature selection, non-numeric or complex values, and
missing or non-finite numbers. The function does not encode categories or
impute missing values and does not modify the caller's DataFrame.

### `split_and_train`

```python
split_and_train(
    data: pd.DataFrame,
    target_column: str,
    features: Sequence[str] | str | None = None,
    random_state: int = 42,
    folds: dict[str, Any] | None = None,
) -> tuple[
    LinearRegression,
    pd.DataFrame,
    pd.Series,
    dict[str, dict[str, Any]],
]
```

Splits the input into an 80% training partition and a 20% holdout partition.
Five-fold cross-validation is run only within the training partition. The
final model is fit on that full training partition; the holdout is returned
for optional evaluation and is not used in fitting or CV.

- `data`: prepared numeric DataFrame.
- `target_column`: target column name.
- `features`: optional predictor selection, following the same rules as
  `train_linear_regression`.
- `random_state`: integer seed used for reproducible splitting and CV when
  folds are not supplied.
- `folds`: optional saved fold assignment, normally returned by
  `create_or_load_folds()`.
- Returns exactly `(model, X_test, y_test, cross_validation_metrics)`.

`X_test` contains only the selected predictors. `X_test` and `y_test` retain
their matching original DataFrame indexes. Each metric in
`cross_validation_metrics` (`r2`, `mae`, and `rmse`) contains:

```json
{
  "mean": 0.0,
  "std": 0.0,
  "fold_scores": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

The standard deviation is the population standard deviation (`ddof=0`).
At least 10 rows must be available in the training partition. Undefined or
non-finite CV results raise an error instead of being returned as usable
metrics.

### `create_or_load_folds`

```python
create_or_load_folds(
    data: pd.DataFrame,
    folds_file: str | Path = DEFAULT_FOLDS_FILE,
    random_state: int = 42,
) -> dict[str, Any]
```

Returns a reproducible holdout/CV assignment for reuse across training calls.
If `folds_file` does not exist, the function creates an 80/20 split and five
CV folds, then persists the assignment. Parent directories are created when
needed, and the file is saved using a temporary file followed by replacement.

The returned JSON-compatible dictionary contains:

- `schema_version`: fold format version.
- `row_count`: number of rows in the source DataFrame.
- `data_fingerprint`: hash identifying the data values, row order, index,
  columns, and dtypes.
- `random_state`: seed used to create assignments.
- `train_indices` and `test_indices`: positional row indices into the original
  DataFrame, not index labels.
- `cv_folds`: five dictionaries, each containing `train_indices` and
  `validation_indices` relative to the outer training partition.

An existing fold file is reused only for matching data and only after its
schema, partition and CV membership have been validated. Unreadable,
malformed, incompatible, or stale files raise an error and are not silently
replaced. To start a new assignment, supply a different path or deliberately
remove the old file.

### `create_folds_and_train`

```python
create_folds_and_train(
    data: pd.DataFrame,
    target_column: str,
    features: list[str] | tuple[str, ...] | str | None = None,
    random_state: int = 42,
    folds_file: str | Path = DEFAULT_FOLDS_FILE,
) -> tuple[
    LinearRegression,
    pd.DataFrame,
    pd.Series,
    dict[str, dict[str, Any]],
]
```

Validates the selected target/features, obtains the saved assignment from
`create_or_load_folds()`, then calls the split/training logic with those folds.
Returns the same four-item tuple as `split_and_train()`. Calls using different
feature selections on the same DataFrame and fold file reuse the same holdout
and CV membership. The return annotation is the four-item tuple documented
for `split_and_train()`.

### `split_result_to_json`

```python
split_result_to_json(
    split_result: tuple[Any, pd.DataFrame, pd.Series, dict[str, Any]],
    target_column: str,
    folds: dict[str, Any] | None = None,
    include_holdout_stats: bool = False,
) -> str
```

Serializes the four-item result from `split_and_train()` as indented strict
JSON (`allow_nan=False`).

- `split_result`: `(model, X_test, y_test, cross_validation_metrics)`.
- `target_column`: name of the modeled target.
- `folds`: optional saved assignment. When supplied, saved split positions,
  random seed, fingerprint, and CV fold definitions are added to
  `data_split`.
- `include_holdout_stats`: must be a boolean. If `False` (the default), no
  `holdout` section is returned and the test set is not predicted for the
  report. If `True`, the JSON includes holdout metrics and a row entry for
  every holdout observation.

The JSON always includes the schema version, target, feature names, split
summary (including `training_row_count`), model name, fitted intercept,
coefficients, and cross-validation metrics. When requested, the holdout
section contains its row count, R²/MAE/RMSE, and row-level actual and
predicted values. R² is `null` for fewer than two observations or a constant
holdout target; MAE and RMSE are still provided whenever defined.

Holdout row indexes are encoded as tagged JSON values to retain identity and
type. Supported labels are null, booleans, integers, finite floats, strings,
dates, datetimes, and tuples made from supported labels. Unsupported index
types, non-finite predictions/targets/model values, and non-serializable
metrics are rejected rather than silently converted.

## Example

The example assumes `data` has already been cleaned and any categorical
predictors have already been encoded into finite numeric columns.

```python
import pandas as pd

from linear_regression import split_and_train, split_result_to_json

data = pd.read_csv("prepared_properties.csv")

result = split_and_train(
    data,
    target_column="Price_USD",
    features=None,  # Use every column except Price_USD
    random_state=42,
)

# Model, CV metrics, and training-row count only:
summary_json = split_result_to_json(result, target_column="Price_USD")

# Also include holdout metrics and each holdout prediction:
full_json = split_result_to_json(
    result,
    target_column="Price_USD",
    include_holdout_stats=True,
)

print(summary_json)
```

To use a persisted assignment for repeatable feature comparisons:

```python
from linear_regression import create_folds_and_train, split_result_to_json

result = create_folds_and_train(
    data,
    target_column="Price_USD",
    features=["Square_Feet", "Bedrooms", "Bathrooms"],
    random_state=42,
    folds_file="experiments/property_folds.json",
)

report_json = split_result_to_json(
    result,
    target_column="Price_USD",
    include_holdout_stats=True,
)
```

The fold file is tied to the complete DataFrame, including its columns and
index. Feature selection changes do not require a new fold file; changes to
the underlying data or row ordering do.
