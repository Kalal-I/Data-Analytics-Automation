'''You are working on the linear_regression package in this Python project. First inspect every file in the package and use the existing code and tests as the source of truth. The package currently contains:

- __init__.py: exposes the supported public functions.
- linear_model.py: validates data and fits a scikit-learn LinearRegression model.
- split_data.py: performs the holdout split, cross-validation, and model training.
- fold_manager.py: creates, saves, validates, and reuses fixed split/fold assignments.
- results_json.py: serializes training and evaluation results into strict JSON.
- test_linear_regression.py: documents and tests the expected behavior.

END DELIVERABLE

The end deliverable is a reliable, reusable Python module that lets a caller:
1. Fit a linear regression model to a DataFrame, optionally selecting predictor columns.
2. Evaluate it using a reproducible 80/20 training/holdout split, with five-fold cross-validation performed only on the training partition.
3. Reuse the exact same split and CV fold assignments across different feature-selection runs by saving and loading fold data.
4. Produce a detailed, valid JSON report with model parameters, CV results, and holdout predictions and metrics.

This is a library API, not a web application, CLI, notebook, or automated preprocessing system. Do not add those surfaces unless explicitly requested.

PUBLIC API AND BEHAVIOR

- train_linear_regression(training_data, target_column, features=None)
  - Accepts a pandas DataFrame and a target column name.
  - If features is omitted, use every column except the target.
  - If features is a string, use that one column; if it is a list or tuple, use those columns in the given order.
  - Return a fitted scikit-learn LinearRegression model.
  - Reject invalid input clearly: non-DataFrame input, empty data, missing target or feature columns, duplicate DataFrame column names, duplicate or empty feature selections, the target selected as a predictor, non-string feature names, categorical/non-numeric data, complex values, missing values, and non-finite values.
  - Do not silently encode categories, impute missing values, or modify the caller’s DataFrame.

- split_and_train(data, target_column, features=None, random_state=42, folds=None)
  - Use an 80/20 split. If saved folds are not supplied, make the split reproducible from random_state.
  - Calculate five-fold CV metrics using only the 80% training partition; never use the holdout partition for CV or fitting.
  - Return exactly:
    (fitted_model, X_test, y_test, cross_validation_metrics)
  - X_test must contain only the selected feature columns, and X_test/y_test must retain corresponding original row indexes.
  - For each of r2, mae, and rmse, return the five individual fold scores, their mean, and their population standard deviation (ddof=0).
  - Require at least 10 rows in the training partition so each validation fold has at least two observations for R².
  - Reject undefined or non-finite CV scores rather than emitting invalid results.

- create_or_load_folds(data, folds_file=DEFAULT_FOLDS_FILE, random_state=42)
  - Reuse an existing saved split only if it belongs to the exact same data and row ordering.
  - Identify data using a fingerprint derived from DataFrame values, index, columns, and dtypes.
  - If an existing file is unreadable or belongs to different data, raise a clear error. Never silently replace it with a newly randomized split.
  - If no saved file exists, create a deterministic 80/20 split and five-fold CV assignment, then persist it.
  - Create the destination directory if needed and save using a temporary file followed by replacement, as the current implementation does.
  - The persisted data includes schema_version, row_count, data_fingerprint, random_state, train_indices, test_indices, and cv_folds.
  - CV fold positions are relative to the training partition (not the original full DataFrame positions).

- create_folds_and_train(data, target_column, features=None, random_state=42, folds_file=DEFAULT_FOLDS_FILE)
  - Validate the target and selected features, obtain the compatible persisted fold assignment, and train/evaluate using that assignment.
  - Different feature selections over the same data should use the same holdout and CV membership, including when an existing valid fold file is reused.

- split_result_to_json(split_result, target_column, folds=None)
  - Accept the four-item result returned by split_and_train.
  - Return an indented JSON string, serialized with allow_nan=False.
  - Include schema_version, target_column, selected features, split details, model type, intercept, per-feature coefficients, CV metrics, and holdout results.
  - Include holdout row count, test metrics (R², MAE, RMSE), and each holdout row’s index, actual target, and predicted target.
  - When folds are provided, include the split metadata and saved row positions/fold definitions.
  - Reject values that cannot be represented as finite JSON numbers; do not output NaN or Infinity.
  - For a holdout set with fewer than two rows, report test metrics as null rather than computing undefined R².

QUALITY AND ACCEPTANCE CRITERIA

- Preserve the public API and its output shapes unless a requirement explicitly calls for a change.
- Keep responsibilities aligned with the existing module structure; reuse existing helpers rather than duplicating validation or metric logic.
- Errors for invalid input, stale fold files, and serialization problems must remain explicit; do not hide them with broad exception handling or successful-looking defaults.
- Do not change the source DataFrame or its index.
- Keep seeded split/fold behavior deterministic and keep holdout data isolated from training and CV.
- Update tests for any changed behavior. The existing test suite covers model fitting, selected features, invalid inputs, input immutability, deterministic CV, fixed-fold reuse/staleness, and strict JSON output.
- Run the focused tests for this package after making changes and report the result.
- Avoid unrelated refactoring and do not add dependencies or new user-facing interfaces without a clear requirement.

Before editing, compare the requested work with the current implementation and tests. If the end deliverable is already satisfied, explain that and make no unnecessary changes. If changes are needed, implement them surgically and verify them.'''