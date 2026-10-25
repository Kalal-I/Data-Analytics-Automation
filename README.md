# Data-Analytics-Automation

An interactive, report-driven CSV preprocessing tool. It profiles a dataset,
asks a configured language-model API for preprocessing recommendations,
validates those recommendations against the available operations, applies
them, and prints the processed DataFrame.

## Requirements

- Python 3.10 or newer
- The Python dependencies in `requirements.txt`
- Access to an HTTP API endpoint compatible with the request/response contract
  described below

No Ollama server or RAG/knowledge-index setup is needed to run the main
preprocessing application.

## Setup

Run these commands from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env_demo .env
```

Open `.env` and replace both placeholder values:

```dotenv
API_URL=https://your-api-base-url
API_TOKEN=your-api-token
```

`API_URL` is the base URL. The preprocessing agent appends `/generate` and
sends the token as a bearer token in the `Authorization` header. Keep the
private `.env` file out of version control; `.env` is ignored by `.gitignore`.
Do not commit or share your API token.

### API contract

The current preprocessing agent expects the configured service to accept a
`POST` request to `<API_URL>/generate`, authenticated with a bearer token. It
sends JSON with:

```json
{
  "prompt": "The dataset description and preprocessing instructions",
  "think": false
}
```

The response must be JSON with a string-valued `response` field containing the
agent's JSON preprocessing decisions. The project does not assume a particular
provider's URL or credentials. Before using a Kaggle-hosted service, confirm
that its endpoint supports this request format and response shape; adapt the
request/response handling in `agents_code/preprocessing_agent.py` if that
service uses a different API contract.

## Run

From the project directory, run:

```powershell
python main.py
```

When prompted:

1. Enter the path to one CSV file.
2. Enter the name of the target column.

The program generates a data-quality report, requests preprocessing decisions,
filters the decisions through the validator, and applies valid non-encoding
operations. It computes dataset and model statistics while categorical
features still contain their original categories. Recommended encodings are
applied only after a regularized linear-regression configuration is selected.
After the train/test split, `StandardScaler` is fit on the training predictors
and target only; the fitted scalers transform the holdout data. For
cross-validation, the scalers are fit separately on each fold's training rows
and applied to that fold's validation rows. This prevents holdout and
validation statistics from leaking into training. The regression model and
reported target metrics are in standardized units; the fitted model exposes
`feature_scaler_` and `target_scaler_` for transforming inputs or converting
predictions back to the original units. The processed DataFrame is printed to
the console; the program does not currently save it to a new CSV file.

Use a single CSV file for the preprocessing run. The CSV loader can read
directories, but the main preprocessing loop expects one DataFrame and is not
set up to process a list of multiple CSV DataFrames as one run.

## Regularized regression model selection and feature-removal evaluation

`linear_regression.create_folds_and_train()` uses a persisted 80/20 split and
five-fold cross-validation assignment to train a selected Lasso, Ridge, or
Elastic Net configuration. Its serialized report includes the model type and
hyperparameters, plus `feature_removal_tests`, which fits an additional
five-fold evaluation for each selected feature set with that one feature
removed. Every reduced model uses the same folds and model configuration as
the full model.

Each result reports cross-validation mean and population standard deviation
for R², adjusted R², MAE, and RMSE. Adjusted R² is `null` if the fold has no
more observations than predictors plus one. `comparison_to_current` is the
reduced model's mean minus the full model's mean: positive R² change is better,
while negative MAE or RMSE change is better. If removing the only feature
leaves no predictors,
the report includes null metrics and a limitation instead of attempting an
invalid fit. A `limitation` value of `null` means the reduced model was fit and
scored normally; a non-null value explains why it could not be evaluated.
`feature_removal_tests` is the final top-level JSON section, after the split,
training, and optional holdout statistics.

The optimizer evaluates model configurations using the same feature set and
fold assignments. At the end, it selects a report within `0.005` mean
cross-validation R² of the best evaluated configuration. Among those
candidates, selection prioritizes higher adjusted R² when it is available for
all candidates, then lower RMSE, lower MAE, and fewer features. If adjusted R²
is unavailable for any eligible candidate, selection falls back to the other
available metrics. The selected model and its recorded alpha (and Elastic Net
l1_ratio) are printed with its metric means and standard deviations.

## Supported preprocessing operations

The agent can recommend only these operations:

- **Missing values:** `fill_missing_values`,
  `fill_missing_values_by_group`, `remove_missing_values`
- **Categorical encoding:** `one_hot_encode`, `label_encode`, `ordinal_encode`
- **Numeric scaling:** `standard_scale`, `min_max_scale`
- **Outlier detection:** `detect_outliers`

The validator checks the agent's proposed operations against the generated
dataset report, supported function parameters, column types, and target-column
restrictions. Outlier detection removes rows identified as outliers; it is not
merely a report-only check.

Implementation details for the helper functions are in
[`preproccessing/preproccesing_README.md`](preproccessing/preproccesing_README.md).
Dataset loading and report generation are described in
[`dataset_stats/dataset_stats_readme.md`](dataset_stats/dataset_stats_readme.md).

## Tests

The included tests use `pytest`. Install it if you want to run the test suite:

```powershell
python -m pip install pytest
python -m pytest
```

## Project layout

```text
main.py                         Interactive application entry point
agents_code/preprocessing_agent.py
                                Builds the prompt and calls the configured API
preproccessing/                 Preprocessing functions and decision validation
dataset_stats/                  CSV loading and dataset reports
requirements.txt                Runtime Python dependencies
.env_demo                       Safe template for local API configuration
```
