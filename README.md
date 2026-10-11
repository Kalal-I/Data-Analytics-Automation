# Data-Analytics-Automation

## Linear regression

`linear_regression.create_folds_and_train()` creates or reuses a persisted
80/20 split and five-fold cross-validation assignment, then trains the
requested feature set against those same rows:

```python
from linear_regression import (
    create_folds_and_train,
    create_or_load_folds,
    split_result_to_json,
)

fold_file = "linear_regression/stored_folds.json"
result = create_folds_and_train(
    data,
    target_column="target",
    features=["feature_a", "feature_b"],
    folds_file=fold_file,
)
folds = create_or_load_folds(data, fold_file)
report_json = split_result_to_json(result, "target", folds=folds)
```

The saved folds are tied to the DataFrame values, columns, dtypes, index, and
row order. Reusing the file with different data raises an error rather than
silently changing the evaluation rows. Choose a new `folds_file` to start a
separate experiment. The JSON report includes the selected features, fitted
coefficients, cross-validation fold metrics, saved split metadata when passed,
and held-out predictions and metrics.