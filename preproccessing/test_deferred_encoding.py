import pandas as pd
from pandas.testing import assert_frame_equal

from preproccessing.deferred_encoding import (
    apply_deferred_encodings,
    partition_deferred_encodings,
)
from preproccessing.one_hot_encoding import one_hot_encode


def test_categorical_encoding_is_staged_and_applied_on_request():
    """Encoding plans leave categories intact until explicitly applied."""
    data = pd.DataFrame(
        {"city": ["Paris", "Rome", "Paris"], "target": [1, 2, 3]}
    )
    operations = {
        "city": [
            {
                "function": "one_hot_encode",
                "inputs": {"columns": ["city"]},
            },
            {
                "function": "fill_missing_values",
                "inputs": {"strategy": "mode", "columns": ["city"]},
            },
        ]
    }

    immediate, deferred = partition_deferred_encodings(operations)

    assert immediate == {
        "city": [
            {
                "function": "fill_missing_values",
                "inputs": {"strategy": "mode", "columns": ["city"]},
            }
        ]
    }
    assert deferred == {
        "city": [
            {
                "function": "one_hot_encode",
                "inputs": {"columns": ["city"]},
            }
        ]
    }
    assert_frame_equal(
        data,
        pd.DataFrame({"city": ["Paris", "Rome", "Paris"], "target": [1, 2, 3]}),
    )

    encoded = apply_deferred_encodings(
        data,
        deferred,
        {"one_hot_encode": one_hot_encode},
    )

    assert list(encoded.columns) == ["target", "city_Paris", "city_Rome"]
    assert list(data.columns) == ["city", "target"]
