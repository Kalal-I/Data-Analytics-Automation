# Preprocessing Helpers

These helpers accept a pandas `DataFrame` and return a transformed copy; they do not modify the input frame. Column arguments accept a single column name or a list/tuple of column names unless stated otherwise.

Dependencies: `pandas` and `scikit-learn`.

## Missing Values

### `drop_rows_with_missing_values.py`

`remove_missing_values(data, columns=None, missing_values=None)` removes rows that contain null values in the selected columns.

- `data`: pandas `DataFrame`.
- `columns`: optional column name or list/tuple of names. If omitted, checks every column.
- `missing_values`: optional scalar marker or list/tuple/set of markers to treat as missing, such as `"?"` or `"-"`. Native pandas nulls are always removed.

### `impute_missing_values.py`

`fill_missing_values(data, strategy="mean", columns=None, missing_values=None)` fills missing values using `"mean"`, `"median"`, or `"mode"`.

- `data`: pandas `DataFrame`.
- `strategy`: one of `"mean"`, `"median"`, or `"mode"`; defaults to `"mean"`. Mean and median require numeric values; mode works with categorical values.
- `columns`: optional column name or list/tuple of names. If omitted, processes every column containing missing values.
- `missing_values`: optional scalar marker or list/tuple/set of markers to treat as missing before filling.

The same module also provides `fill_missing_values_by_group(data, target_column, group_column, missing_values=None)`. It fills a numeric target column with the mean calculated within each group. Both column names are required; optional markers are treated as missing. It raises an error if any missing target has no available group mean.

## Encoding

### `one_hot_encoding.py`

`one_hot_encode(data, columns=None)` replaces selected columns with one-hot indicator columns.

- `data`: pandas `DataFrame`.
- `columns`: optional column name or list/tuple of names. If omitted, encodes every column with fewer than 10 distinct non-null values.

### `label_encoding.py`

`label_encode(data, columns=None)` replaces each selected category with an integer code and preserves nulls.

- `data`: pandas `DataFrame`.
- `columns`: optional column name or list/tuple of names. If omitted, selects string and categorical columns.

Label codes represent categories, not numeric magnitude or order.

### `ordinal_encoding.py`

`ordinal_encode(data, columns=None, categories=None)` replaces selected categories with integer codes and preserves nulls.

- `data`: pandas `DataFrame`.
- `columns`: optional column name or list/tuple of names. If omitted, selects string and categorical columns.
- `categories`: optional dictionary mapping selected column names to an ordered list/tuple of category values, for example `{"size": ["small", "medium", "large"]}`. Values receive codes in the supplied order. Values not in a supplied order receive `-1`. Without an explicit order, the encoder determines category order from the data.

## Scaling

### `standard_scaler.py`

`standard_scale(data, columns=None)` applies standard scaling (centered at mean zero and scaled to unit variance).

- `data`: pandas `DataFrame`.
- `columns`: optional numeric column name or list/tuple of names. If omitted, scales all numeric columns. Selected columns must be numeric.

### `min_max_scaler.py`

`min_max_scale(data, columns=None)` scales selected numeric columns to the range `[0, 1]`.

- `data`: pandas `DataFrame`.
- `columns`: optional numeric column name or list/tuple of names. If omitted, scales all numeric columns. Selected columns must be numeric.

## Outlier Detection

### `outlier_detection.py`

`detect_outliers(data, method="zscore", columns=None, threshold=3.0, multiplier=1.5)` removes rows identified as outliers.

- `data`: pandas `DataFrame`.
- `method`: `"zscore"` or `"iqr"`; defaults to `"zscore"`.
- `columns`: optional numeric column name or list/tuple of names. If omitted, checks all numeric columns. Selected columns must be numeric.
- `threshold`: positive Z-score cutoff; used only with `method="zscore"`.
- `multiplier`: positive IQR multiplier for the lower and upper bounds; used only with `method="iqr"`.

A row is removed if any selected column marks it as an outlier. Null values are ignored by the outlier checks and do not by themselves cause a row to be removed.