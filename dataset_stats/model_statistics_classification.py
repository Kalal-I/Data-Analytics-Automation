"""Classification dataset statistics and default-model baselines."""

import json
import math
import numbers
from decimal import Decimal
from itertools import combinations
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from dataset_stats.column_analysis import analyze_column
from logistic_regression.logistic_model import DEFAULTS as LOGISTIC_DEFAULTS
from logistic_regression.logistic_model import fit_model as fit_logistic_model
from random_forest_classifier.random_forest_model import DEFAULTS as RANDOM_FOREST_DEFAULTS
from random_forest_classifier.random_forest_model import fit_model as fit_random_forest_model


SCHEMA_VERSION = 1
DEFAULT_CORRELATION_THRESHOLD = 0.8


def _json_value(value: Any) -> Any:
    """Convert NumPy and non-finite values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return str(value)


def _is_supported_target_value(value: Any) -> bool:
    if isinstance(value, (list, dict, tuple, set, np.ndarray)):
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(value, (complex, np.complexfloating)):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, numbers.Number):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return isinstance(value, (str, bytes, numbers.Number, np.generic))


def _feature_type(values: pd.Series) -> tuple[str, Any]:
    analysis = analyze_column(values)
    if analysis.detected_type in {"constant", "numerical"} and analysis.numeric_candidate:
        return "numeric", analysis
    if analysis.detected_type == "all_missing":
        return "categorical", analysis
    return "categorical", analysis


def _target_report(values: pd.Series) -> dict[str, Any]:
    missing = int(values.isna().sum())
    supported = values.map(_is_supported_target_value)
    invalid = int((~supported).sum()) - missing
    non_finite = 0
    for value in values[~values.isna()]:
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                non_finite += 1
        elif isinstance(value, (complex, np.complexfloating)):
            non_finite += 1

    usable = values[supported & values.notna()].copy()
    class_codes, class_values = _class_codes(usable)
    counts_array = np.bincount(class_codes, minlength=len(class_values))
    class_count = len(class_values)
    class_keys = _class_output_keys(class_values)
    counts = {
        key: int(count)
        for key, count in zip(class_keys, counts_array)
    }
    probabilities = counts_array.astype(float) / max(len(usable), 1)
    majority_ratio = float(counts_array.max() / len(usable)) if len(usable) else None
    minority_ratio = float(counts_array.min() / len(usable)) if len(usable) else None
    imbalance_ratio = float(counts_array.max() / counts_array.min()) if class_count and counts_array.min() else None
    entropy_value = float(entropy(probabilities, base=2)) if len(probabilities) else None
    warnings = []
    suitable_for_cv = class_count >= 2 and len(usable) >= 10 and counts_array.min() >= 2
    if class_count < 2:
        warnings.append("Target has fewer than two usable classes.")
    if counts_array.size and counts_array.min() < 2:
        warnings.append("At least one class has fewer than two usable samples.")

    return {
        "name": str(values.name),
        "dtype": str(values.dtype),
        "sample_count": int(len(values)),
        "usable_sample_count": int(len(usable)),
        "missing_count": missing,
        "invalid_value_count": max(invalid, 0),
        "non_finite_count": non_finite,
        "class_count": class_count,
        "classification_type": "binary" if class_count == 2 else "multiclass" if class_count > 2 else "constant_or_invalid",
        "class_counts": counts,
        "class_percentages": {
            key: float(count / len(usable) * 100) if len(usable) else 0.0
            for key, count in counts.items()
        },
        "majority_class_ratio": majority_ratio,
        "minority_class_ratio": minority_ratio,
        "imbalance_ratio": imbalance_ratio,
        "minimum_class_count": int(counts_array.min()) if counts_array.size else 0,
        "maximum_class_count": int(counts_array.max()) if counts_array.size else 0,
        "entropy_bits": entropy_value,
        "is_constant": class_count == 1,
        "suitable_for_stratified_cv": suitable_for_cv,
        "warnings": warnings,
    }


def _class_identity(value: Any) -> str:
    value_type = type(value)
    return "{}.{}:{}".format(
        value_type.__module__,
        value_type.__qualname__,
        repr(value),
    )


def _class_codes(values: pd.Series | np.ndarray) -> tuple[np.ndarray, list[Any]]:
    codes = []
    classes = []
    class_indexes = {}
    for value in values:
        identity = _class_identity(value)
        if identity not in class_indexes:
            class_indexes[identity] = len(classes)
            classes.append(value)
        codes.append(class_indexes[identity])
    return np.asarray(codes, dtype=int), classes


def _class_output_keys(classes: list[Any]) -> list[str]:
    plain_keys = [str(value) for value in classes]
    if len(plain_keys) == len(set(plain_keys)):
        return plain_keys
    return [_class_identity(value) for value in classes]


_MISSING_CATEGORY = "__classification_missing__"


def _category_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return "value:{}:{}:{!r}".format(
        type(value).__module__,
        type(value).__qualname__,
        value,
    )


def _encode_categorical_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=object)
    encoded = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        encoded[index] = (
            _MISSING_CATEGORY
            if pd.isna(value)
            else _category_value(value)
        )
    return encoded


def _feature_reports(data: pd.DataFrame, target_column: str) -> tuple[list[dict[str, Any]], dict[str, str], list[str], list[str]]:
    reports = []
    types = {}
    numeric_features = []
    categorical_features = []
    for column in data.columns:
        if column == target_column:
            continue
        values = data[column]
        kind, analysis = _feature_type(values)
        types[column] = kind
        missing_count = int(values.isna().sum())
        non_missing = values.dropna()
        if kind == "numeric":
            numeric_features.append(column)
            numeric = pd.to_numeric(values, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            missing_count = int(numeric.isna().sum())
            finite = numeric.dropna()
            std = _finite(float(finite.std()) if finite.size else None)
            report = {
                "name": column,
                "feature_type": kind,
                "detected_type": analysis.detected_type,
                "missing_count": missing_count,
                "missing_percentage": float(missing_count / len(values) * 100) if len(values) else 0.0,
                "invalid_count": analysis.invalid_count,
                "non_finite_count": analysis.non_finite_count,
                "precision_loss_count": analysis.precision_loss_count,
                "unique_value_count": int(numeric.nunique()),
                "is_constant": bool(finite.size and finite.nunique() == 1),
                "mean": _finite(float(finite.mean()) if finite.size else None),
                "standard_deviation": std,
                "minimum": _finite(float(finite.min()) if finite.size else None),
                "maximum": _finite(float(finite.max()) if finite.size else None),
                "skewness": _finite(float(finite.skew()) if finite.size else None),
            }
            report["has_outlier_indication"] = bool(finite.size and _outlier_mask(finite).any())
        else:
            categorical_features.append(column)
            unique_count = int(non_missing.nunique())
            report = {
                "name": column,
                "feature_type": kind,
                "detected_type": analysis.detected_type,
                "missing_count": missing_count,
                "missing_percentage": float(missing_count / len(values) * 100) if len(values) else 0.0,
                "unique_category_count": unique_count,
                "cardinality_ratio": float(unique_count / len(non_missing)) if len(non_missing) else None,
                "is_constant": unique_count == 1 and len(non_missing) > 0,
                "high_cardinality": bool(len(non_missing) and unique_count / len(non_missing) > 0.5 and unique_count > 20),
                "invalid_count": analysis.invalid_count,
                "non_finite_count": analysis.non_finite_count,
            }
        reports.append(report)
    return reports, types, numeric_features, categorical_features


def _outlier_mask(values: pd.Series) -> pd.Series:
    if len(values) < 4:
        return pd.Series(False, index=values.index)
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return pd.Series(False, index=values.index)
    return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _clean_target(data: pd.DataFrame, target_column: str) -> pd.Series:
    values = data[target_column]
    usable = values.map(_is_supported_target_value)
    if int(usable.sum()) != len(values):
        raise ValueError("target contains missing, invalid, or non-finite values; only finite, supported labels are usable")
    if values.nunique(dropna=False) < 2:
        raise ValueError("target must contain at least two classes for classification")
    return values


def _associations(
    data: pd.DataFrame,
    target: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    threshold: float,
) -> dict[str, Any]:
    usable_rows = target.notna()
    target_codes, _ = _class_codes(target[usable_rows])
    y = target_codes
    associations = []
    numeric_encoded = []
    numeric_names = []
    for column in numeric_features:
        values = pd.to_numeric(data.loc[usable_rows, column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.notna()
        if int(finite.sum()) < 2 or values[finite].nunique() < 2:
            associations.append({"feature": column, "method": "ANOVA F-test", "value": None, "limitation": "constant_or_insufficient_usable_values"})
        else:
            try:
                paired_y = y[finite.to_numpy()]
                score = f_classif(values[finite].to_numpy().reshape(-1, 1), np.asarray(paired_y))[0][0]
                associations.append({"feature": column, "method": "ANOVA F-test", "value": _finite(score), "limitation": None})
            except (ValueError, FloatingPointError) as error:
                associations.append({"feature": column, "method": "ANOVA F-test", "value": None, "limitation": str(error)})
        try:
            medians_array = SimpleImputer(strategy="median").fit_transform(
                values.to_numpy().reshape(-1, 1)
            ).ravel()
        except ValueError:
            medians_array = np.full(len(values), np.nan)
        medians = pd.Series(medians_array, index=values.index)
        numeric_encoded.append(medians.to_numpy(dtype=float))
        numeric_names.append(column)

    for column in categorical_features:
        values = pd.Series(
            _encode_categorical_matrix(
                data.loc[usable_rows, column].to_numpy().reshape(-1, 1)
            ).ravel(),
            index=data.index[usable_rows],
        )
        if values.nunique() < 2:
            associations.append({"feature": column, "method": "chi-square test of independence", "value": None, "limitation": "constant_or_insufficient_usable_values"})
            continue
        table = pd.crosstab(
            pd.Series(values.to_numpy()),
            pd.Series(y),
        )
        try:
            from scipy.stats import chi2_contingency
            statistic, _, _, _ = chi2_contingency(table.to_numpy(), correction=False)
            score = _finite(statistic)
            limitation = None
        except (ValueError, FloatingPointError) as score_error:
            limitation = str(score_error)
            score = None
        associations.append({"feature": column, "method": "chi-square test of independence", "value": score, "limitation": limitation if score is None else None})

    mi_values = []
    all_names = numeric_features + categorical_features
    encoded = []
    if numeric_names:
        for values in numeric_encoded:
            encoded.append(values)
        encoded_matrix = np.column_stack(encoded)
        try:
            scores = mutual_info_classif(encoded_matrix, y, random_state=42)
            mi_values.extend({"feature": name, "value": _finite(score), "method": "mutual information"} for name, score in zip(numeric_names, scores))
        except ValueError as error:
            mi_values.extend({"feature": name, "value": None, "method": "mutual information", "limitation": str(error)} for name in numeric_names)
    for column in categorical_features:
        values = _encode_categorical_matrix(
            data.loc[usable_rows, column].to_numpy().reshape(-1, 1)
        ).ravel()
        codes = pd.Categorical(values).codes.reshape(-1, 1)
        try:
            score = mutual_info_classif(codes, y, discrete_features=True, random_state=42)[0]
            mi_values.append({"feature": column, "value": _finite(score), "method": "mutual information"})
        except ValueError as error:
            mi_values.append({"feature": column, "value": None, "method": "mutual information", "limitation": str(error)})

    correlations = []
    highly_correlated = []
    for left, right in combinations(numeric_features, 2):
        paired = pd.DataFrame(
            {
                left: pd.to_numeric(data[left], errors="coerce"),
                right: pd.to_numeric(data[right], errors="coerce"),
            },
            index=data.index,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            value = None
        else:
            value = _finite(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="pearson"))
        correlations.append({"feature_1": left, "feature_2": right, "method": "Pearson correlation", "value": value, "valid_observation_count": len(paired)})
        if value is not None and abs(value) >= threshold:
            highly_correlated.append({"feature_1": left, "feature_2": right, "absolute_correlation": abs(value), "correlation": value})

    return {
        "feature_target": associations,
        "mutual_information": mi_values,
        "methods": {
            "numeric": "ANOVA F-test and mutual information",
            "categorical": "Chi-square test of independence and mutual information",
        },
        "caution": "Association does not imply causation and is not automatic evidence for feature removal.",
        "feature_feature": correlations,
        "highly_correlated_pairs": highly_correlated,
    }


def _suitability(data: pd.DataFrame, target_report: dict[str, Any], numeric_features: list[str], categorical_features: list[str], associations: dict[str, Any]) -> dict[str, Any]:
    sample_count = len(data)
    feature_count = len(numeric_features) + len(categorical_features)
    correlations = [abs(pair["correlation"]) for pair in associations["highly_correlated_pairs"] if pair.get("correlation") is not None]
    return {
        "sample_count": sample_count,
        "feature_count": feature_count,
        "class_count": target_report["class_count"],
        "samples_per_feature_ratio": float(sample_count / feature_count) if feature_count else None,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "imbalance_ratio": target_report["imbalance_ratio"],
        "multicollinearity_indicator": bool(correlations),
        "high_dimensionality_indicator": feature_count > sample_count,
        "small_sample_indicator": sample_count < 100,
        "sparse_feature_indicator": any(
                float(data[column].isna().mean()) > 0.8
                for column in numeric_features + categorical_features
            ),
        "computational_cost_indicator": "high" if feature_count * sample_count > 1000000 else "standard",
        "stratified_cv_suitable": target_report["suitable_for_stratified_cv"],
    }


def _numeric_pipeline(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    transformers = []
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "canonicalize",
                    FunctionTransformer(
                        _encode_categorical_matrix,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _metric_summary(values: list[float | None]) -> dict[str, float | None]:
    usable = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "mean": float(np.mean(usable)) if usable else None,
        "std": float(np.std(usable, ddof=0)) if usable else None,
    }


def _classification_metrics(
    y_true,
    y_pred,
    probabilities=None,
    classes=None,
    positive_class=None,
    class_labels=None,
) -> dict[str, Any]:
    classes = np.unique(np.asarray(y_true)) if classes is None else np.asarray(classes)
    binary = len(classes) == 2
    weighted_kwargs = {"zero_division": 0}
    if binary:
        positive = positive_class if positive_class is not None else classes[-1]
        truth = np.asarray(y_true) == positive
        predicted = np.asarray(y_pred) == positive
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="weighted", **weighted_kwargs)
        recall = recall_score(y_true, y_pred, average="weighted", **weighted_kwargs)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", **weighted_kwargs)
    roc_auc = None
    try:
        probabilities = np.asarray(probabilities) if probabilities is not None else None
        if binary and probabilities is not None:
            positive = positive_class if positive_class is not None else classes[-1]
            if probabilities.ndim == 1:
                positive_scores = probabilities
            else:
                positive_scores = probabilities[:, int(np.flatnonzero(classes == positive)[0])]
            roc_auc = float(roc_auc_score(np.asarray(y_true) == positive, positive_scores))
        elif not binary and probabilities is not None:
            roc_auc = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except (ValueError, IndexError, TypeError):
        roc_auc = None
    labels = (
        classes.astype(str).tolist()
        if class_labels is None
        else [class_labels[int(class_code)] for class_code in classes]
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    if class_labels is not None:
        report_keys = {
            str(class_code): class_labels[class_code]
            for class_code in range(len(class_labels))
        }
        report = {
            report_keys.get(key, key): value
            for key, value in report.items()
        }
    return {
        "accuracy": _finite(accuracy_score(y_true, y_pred)),
        "weighted_precision": _finite(precision),
        "weighted_recall": _finite(recall),
        "weighted_f1": _finite(weighted_f1),
        "macro_f1": _finite(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": _finite(roc_auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": report,
        "labels": labels,
    }


def _baseline_evaluation(
    data: pd.DataFrame,
    target_column: str,
    target: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> dict[str, Any]:
    y, class_values = _class_codes(target)
    class_labels = _class_output_keys(class_values)
    class_counts = np.bincount(y, minlength=len(class_values))
    n_splits = min(5, int(class_counts.min())) if class_counts.size else 0
    specifications = [
        ("LogisticRegression", fit_logistic_model, LOGISTIC_DEFAULTS, True),
        ("RandomForestClassifier", fit_random_forest_model, RANDOM_FOREST_DEFAULTS, False),
    ]
    evaluation = {
        "method": "stratified_cross_validation",
        "n_splits": n_splits,
        "random_state": random_state,
    }

    def failed_models(reason: str) -> dict[str, Any]:
        return {
            model_name: {
                "model_name": model_name,
                "status": "failed",
                "configuration": _json_value(defaults),
                "configuration_source": "existing_project_default",
                "evaluation_strategy": {
                    "method": "stratified_cross_validation",
                    "n_splits": n_splits,
                    "random_state": random_state,
                    "preprocessing_fit_on_training_partition_only": True,
                },
                "metrics": None,
                "fold_metrics": [],
                "training_metrics": None,
                "holdout_metrics": None,
                "holdout_failure_reason": None,
                "warnings": [],
                "failure_reason": reason,
            }
            for model_name, _, defaults, _ in specifications
        }

    if not numeric_features and not categorical_features:
        reason = "no predictor features are available"
        evaluation.update({"status": "failed", "reason": reason})
        evaluation["outer_holdout"] = {
            "status": "not_available",
            "failure_reason": reason,
        }
        return {"models": failed_models(reason), "evaluation": evaluation}

    if n_splits < 2:
        reason = "each class must contain at least two samples"
        evaluation.update({"status": "failed", "reason": reason})
        return {
            "models": failed_models(reason),
            "evaluation": evaluation,
        }

    X = data[numeric_features + categorical_features].copy()
    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    usable_index = np.arange(len(data))

    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X, y))
    except ValueError as error:
        evaluation.update({"status": "failed", "reason": str(error)})
        return {
            "models": failed_models(str(error)),
            "evaluation": evaluation,
        }

    holdout_split = None
    holdout_failure_reason = None
    if len(data) >= max(10, len(class_values) * 2):
        try:
            holdout_split = train_test_split(
                usable_index,
                test_size=0.2,
                random_state=random_state,
                shuffle=True,
                stratify=y,
            )
        except ValueError as error:
            holdout_failure_reason = "{}: {}".format(type(error).__name__, error)
    else:
        holdout_failure_reason = (
            "insufficient samples for stratified holdout evaluation"
        )

    models = {}
    for model_name, fitter, defaults, scale_numeric in specifications:
        fold_metrics = []
        try:
            for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
                preprocessor = _numeric_pipeline(numeric_features, categorical_features, scale_numeric)
                X_train = preprocessor.fit_transform(X.iloc[train_index])
                X_validation = preprocessor.transform(X.iloc[validation_index])
                y_train = y[train_index]
                y_validation = y[validation_index]
                params = {key: value for key, value in defaults.items() if key not in {"random_state", "n_jobs"}}
                params.update({"random_state": random_state})
                model = fitter(X_train, y_train, **params)
                prediction = model.predict(X_validation)
                probabilities = model.predict_proba(X_validation) if hasattr(model, "predict_proba") else None
                metrics = _classification_metrics(
                    y_validation,
                    prediction,
                    probabilities,
                    classes=model.classes_,
                    class_labels=class_labels,
                )
                fold_metrics.append({"fold": fold_number, **metrics})
            metric_names = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_f1", "roc_auc"]
            aggregate = {
                name: _metric_summary([fold[name] for fold in fold_metrics])
                for name in metric_names
            }
        except Exception as error:
            failure_reason = "{}: {}".format(type(error).__name__, str(error))
            models[model_name] = {
                "model_name": model_name,
                "status": "failed",
                "configuration": _json_value(defaults),
                "configuration_source": "existing_project_default",
                "evaluation_strategy": {
                    "method": "stratified_cross_validation",
                    "n_splits": n_splits,
                    "random_state": random_state,
                    "preprocessing_fit_on_training_partition_only": True,
                },
                "metrics": None,
                "fold_metrics": fold_metrics,
                "training_metrics": None,
                "holdout_metrics": None,
                "holdout_failure_reason": None,
                "warnings": [],
                "failure_reason": failure_reason,
            }
            continue

        training_metrics = None
        holdout_metrics = None
        model_holdout_failure = holdout_failure_reason
        if holdout_split is not None:
            try:
                train_index, holdout_index = holdout_split
                holdout_preprocessor = _numeric_pipeline(
                    numeric_features, categorical_features, scale_numeric
                )
                X_train = holdout_preprocessor.fit_transform(X.iloc[train_index])
                X_holdout = holdout_preprocessor.transform(X.iloc[holdout_index])
                params = {key: value for key, value in defaults.items() if key not in {"random_state", "n_jobs"}}
                params["random_state"] = random_state
                holdout_model = fitter(X_train, y[train_index], **params)
                train_prediction = holdout_model.predict(X_train)
                holdout_prediction = holdout_model.predict(X_holdout)
                probabilities_train = (
                    holdout_model.predict_proba(X_train)
                    if hasattr(holdout_model, "predict_proba") else None
                )
                probabilities_holdout = (
                    holdout_model.predict_proba(X_holdout)
                    if hasattr(holdout_model, "predict_proba") else None
                )
                training_metrics = _classification_metrics(
                    y[train_index], train_prediction, probabilities_train,
                    classes=holdout_model.classes_,
                    class_labels=class_labels,
                )
                holdout_metrics = _classification_metrics(
                    y[holdout_index], holdout_prediction, probabilities_holdout,
                    classes=holdout_model.classes_,
                    class_labels=class_labels,
                )
            except Exception as error:
                model_holdout_failure = "{}: {}".format(
                    type(error).__name__, str(error)
                )
        models[model_name] = {
            "model_name": model_name,
            "status": "evaluated",
            "configuration": _json_value(defaults),
            "configuration_source": "existing_project_default",
            "evaluation_strategy": {
                "method": "stratified_cross_validation",
                "n_splits": n_splits,
                "random_state": random_state,
                "preprocessing_fit_on_training_partition_only": True,
            },
            "metrics": aggregate,
            "fold_metrics": fold_metrics,
            "training_metrics": training_metrics,
            "holdout_metrics": holdout_metrics,
            "holdout_failure_reason": model_holdout_failure,
            "warnings": [],
            "failure_reason": None,
        }

    holdout = {
        "status": (
            "available_for_independent_holdout"
            if holdout_split is not None
            else "not_available"
        ),
        "random_state": random_state,
    }
    if holdout_split is None:
        holdout["failure_reason"] = holdout_failure_reason
    else:
        train_index, holdout_index = holdout_split
        holdout["train_row_count"] = len(train_index)
        holdout["holdout_row_count"] = len(holdout_index)
    evaluation["outer_holdout"] = holdout

    return {
        "evaluation": evaluation,
        "models": models,
    }


def generate_model_statistics_classification(
    data: pd.DataFrame,
    target_column: str,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    random_state: int = 42,
) -> str:
    """Generate JSON classification statistics and default baseline results."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string.")
    if data.columns.has_duplicates:
        raise ValueError("DataFrame column names must be unique.")
    if target_column not in data.columns:
        raise ValueError("Target column {!r} was not found.".format(target_column))
    if data.empty:
        raise ValueError("data must contain at least one row.")
    if isinstance(correlation_threshold, bool) or not isinstance(correlation_threshold, Real):
        raise TypeError("correlation_threshold must be a real number.")
    threshold = float(correlation_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("correlation_threshold must be finite and between 0 and 1.")
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer.")

    target = _clean_target(data, target_column)
    target_report = _target_report(target)
    features, feature_types, numeric_features, categorical_features = _feature_reports(data, target_column)
    analysis_data = data.copy()
    for column in numeric_features:
        analysis_data[column] = pd.to_numeric(
            analysis_data[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    associations = _associations(analysis_data, target, numeric_features, categorical_features, threshold)
    suitability = _suitability(analysis_data, target_report, numeric_features, categorical_features, associations)
    baselines = _baseline_evaluation(
        analysis_data, target_column, target, numeric_features, categorical_features, int(random_state)
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "sample_count": len(data),
            "predictor_feature_count": len(features),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
        },
        "target": target_report,
        "features": features,
        "feature_target_associations": associations["feature_target"],
        "feature_target_mutual_information": associations["mutual_information"],
        "association_methods": associations["methods"],
        "association_caution": associations["caution"],
        "feature_feature_associations": associations["feature_feature"],
        "multicollinearity": {
            "method": "Pearson correlation with pairwise complete finite observations",
            "absolute_correlation_threshold": threshold,
            "threshold_is_heuristic": True,
            "highly_correlated_pairs": associations["highly_correlated_pairs"],
        },
        "model_suitability": suitability,
        "baseline_model_evaluation": baselines,
        "warnings": target_report["warnings"],
        "interpretation": (
            "Classification associations describe observed relationships and do not imply causation. "
            "Baseline metrics are descriptive and should not be interpreted as a test that the best model is optimal."
        ),
    }
    return json.dumps(_json_value(result), indent=2, allow_nan=False)


generate_model_statistics = generate_model_statistics_classification
