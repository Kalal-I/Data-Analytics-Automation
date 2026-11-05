# Random Forest Classifier

This package provides a programmatic Random Forest classification workflow with
training/holdout splitting, automatic feature filtering, fixed stratified
cross-validation, hyperparameter tuning, final fitting, and holdout metrics.

## Directory structure

| Path | Purpose |
|---|---|
| `workflow.py` | Runs the complete workflow. |
| `data_split.py` | Creates the stratified train/holdout split. |
| `feature_selection.py` | Selects usable predictors and reports automatic exclusions. |
| `folds.py` | Creates reusable stratified cross-validation folds. |
| `random_forest_model.py` | Creates and fits scikit-learn random forest models. |
| `optimization.py` | Runs tuning, scoring, and holdout evaluation. |
| `tuning_report.py` | Builds compact and optional complete tuning reports. |
| `tests/` | Unit and workflow tests. |

## Installation and dependencies

Install the project dependencies from the repository root:

```text
python -m pip install -r requirements.txt
python -m pytest random_forest_classifier/tests -q
```

The implementation uses pandas, NumPy, and scikit-learn, following the
repository's existing Python conventions. Random forests do not require feature
scaling.

## Workflow example

```python
import pandas as pd
from random_forest_classifier import run_random_forest_classifier_workflow

data = pd.read_csv("classification.csv")
result = run_random_forest_classifier_workflow(
    data,
    target="class",
    n_iter=40,
    scoring="f1_weighted",
    random_state=42,
)

print(result["selected_features"])
print(result["best_configuration"])
print(result["cross_validation"]["expected_model_fits"])
print(result["holdout_metrics"])
print(result["tuning_report"]["top_10"])
```

Feature filtering is performed on the training partition only. The holdout
partition is never used to select features or tune parameters.

## Automatic feature filtering

`select_features(data, target)` automatically excludes:

- the target column;
- columns containing only missing values;
- columns whose non-null values are all identical;
- columns for which no usable values remain.

It does **not** exclude columns solely because they:

- have many missing values;
- contain a few missing values;
- are categorical;
- contain invalid numeric entries;
- have high cardinality;
- look like identifiers;
- are correlated with another feature.

Missing-value detection is consistent across pandas missing values, and
positive and negative infinity are detected in numeric columns. The workflow
does not silently discard a feature because some rows are invalid. Instead, it
raises an actionable error directing the user to preprocess numeric missing
or non-finite values, or to encode categorical features. Constant and
all-missing features are reported through `select_features(...,
return_exclusions=True)`, which returns `(selected_features, exclusion_report)`.

## Splitting and cross-validation

By default, the workflow uses an 80/20 stratified train/holdout split. Every
class represented in the data must have enough rows for both partitions.

On the training partition, it creates five fixed `StratifiedKFold` assignments.
The same folds and the same configurations are used for every tuning candidate.
Each class needs at least five training samples for five-fold stratification.

## Hyperparameter tuning

Tuning uses `RandomizedSearchCV` with `refit=False`. The fixed random seed makes
the default run reproducible. The default search space is:

- `n_estimators`: 50, 100, 150, 200
- `max_depth`: None, 5, 10, 20
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4
- `max_features`: sqrt, log2, 0.5, 1.0
- `criterion`: gini, entropy, log_loss

The default tuning calls for 40 configurations, evaluated across 5 folds. That
is 200 cross-validation fits. After tuning, the workflow fits the selected
configuration once on all training rows, so the complete operation uses 201
model fits. A custom `n_iter` changes the number of configurations and expected
fits. `RandomizedSearchCV` can also evaluate fewer unique configurations than
requested when the configured values contain fewer unique candidates.

## Classification metrics

The holdout metrics dictionary contains:

- accuracy;
- binary precision, recall, and F1, or weighted multiclass precision, recall,
  and F1;
- ROC-AUC, where supported;
- confusion matrix;
- classification report;
- class labels.

The ROC-AUC is binary for two-class targets and one-vs-rest macro ROC-AUC for
multiclass targets. It is `None` when probabilities or a valid ROC calculation
are unavailable.

## Tuning report

Each tuning result contains its rank, parameter configuration, mean and
standard-deviation test scores, and a failed-fit indicator. The tuning report
contains:

- best configuration and score;
- selection metric;
- configuration and fold counts;
- expected model fit count;
- summary text;
- top 10 configurations;
- parameter observations derived from the completed results.

The complete result is available from
`result["cross_validation"]["full_history"]`. It can also be included in
`result["tuning_report"]` by passing `include_tuning_history=True`. The report
defaults to its top 10 configurations, while `full_history` retains every
evaluated configuration.

## Limitations and errors

- Raw strings and other categorical pandas dtypes are not passed directly to
  `RandomForestClassifier`; encode them before calling the workflow.
- Missing numeric values and non-finite values are not silently ignored and
  must be imputed, cleaned, or encoded with the project's supported
  preprocessing utilities first.
- The target must be non-missing and suitable for stratified classification.
- Each class needs sufficient rows for the holdout split and requested number
  of folds.
- Duplicate DataFrame column names and target columns missing from the input
  are rejected.
- Tuning failures can produce failed configurations; the workflow raises when
  no usable cross-validation configuration remains.
