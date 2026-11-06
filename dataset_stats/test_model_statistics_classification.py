import json

import numpy as np
import pandas as pd
import pytest

from dataset_stats.model_statistics_classification import (
    generate_model_statistics_classification,
)


def _report(data, target="target", **kwargs):
    return json.loads(generate_model_statistics_classification(data, target, **kwargs))


def test_binary_statistics_and_default_baselines():
    rows = 40
    data = pd.DataFrame({
        "signal": list(range(rows)),
        "constant": [1] * rows,
        "category": ["A", "B"] * (rows // 2),
        "target": [0] * 20 + [1] * 20,
    })

    report = _report(data)

    assert report["schema_version"] == 1
    assert report["dataset"]["sample_count"] == rows
    assert report["target"]["classification_type"] == "binary"
    assert report["target"]["class_counts"] == {"0": 20, "1": 20}
    baselines = report["baseline_model_evaluation"]["models"]
    assert baselines["LogisticRegression"]["status"] == "evaluated"
    assert baselines["RandomForestClassifier"]["status"] == "evaluated"
    assert "weighted_f1" in baselines["LogisticRegression"]["metrics"]
    assert baselines["LogisticRegression"]["configuration"]["C"] == 1.0
    assert baselines["RandomForestClassifier"]["configuration"]["n_estimators"] == 100


def test_multiclass_missing_categorical_and_associations():
    data = pd.DataFrame({
        "numeric": [0, 1, 2, None, 4, 5, 6, 7, 8, 9, 10, 11] * 2,
        "category": ["A", "B", "A"] * 8,
        "target": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 2,
    })
    report = _report(data)
    assert report["target"]["classification_type"] == "binary"
    assert any(feature["name"] == "numeric" and feature["missing_count"] == 2 for feature in report["features"])

    multiclass = data.assign(target=[0, 1, 2] * 8)
    result = _report(multiclass)
    assert result["target"]["classification_type"] == "multiclass"
    assert "macro_f1" in result["baseline_model_evaluation"]["models"]["LogisticRegression"]["metrics"]


def test_invalid_and_constant_targets():
    with pytest.raises(ValueError, match="at least two classes"):
        generate_model_statistics_classification(
            pd.DataFrame({"x": [1, 2], "target": [1, 1]}), "target"
        )
    with pytest.raises(ValueError, match="missing, invalid, or non-finite"):
        generate_model_statistics_classification(
            pd.DataFrame({"x": [1, 2], "target": [0, np.nan]}), "target"
        )


def test_missing_target_and_empty_data_errors():
    with pytest.raises(ValueError, match="not found"):
        generate_model_statistics_classification(pd.DataFrame({"x": [1]}), "target")
    with pytest.raises(ValueError, match="at least one row"):
        generate_model_statistics_classification(
            pd.DataFrame(columns=["x", "target"]), "target"
        )
