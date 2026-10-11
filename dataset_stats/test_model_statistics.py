import json
import math
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dataset_stats.model_statistics import generate_model_statistics


def _report(data, target="target", **kwargs):
    return json.loads(generate_model_statistics(data, target, **kwargs))


def test_normal_regression_reports_expected_correlations():
    """Numeric predictors return Pearson/Spearman coefficients and pair counts."""
    data = pd.DataFrame(
        {"signal": [1, 2, 3, 4], "noise": [4, 1, 3, 2], "target": [2, 4, 6, 8]}
    )

    report = _report(data)

    relationship = report["feature_target_relationships"][0]
    assert relationship["feature"] == "signal"
    assert relationship["pearson_correlation"] == pytest.approx(1.0)
    assert relationship["spearman_correlation"] == pytest.approx(1.0)
    assert relationship["valid_paired_observation_count"] == 4
    assert report["target"]["numeric_data_valid"] is True
    assert report["target"]["suitable_for_regression"] is None
    assert report["target"]["regression_suitability_assessment"] == (
        "undetermined_from_target_alone"
    )
    assert "row_count" not in report
    assert "mean" not in report["target"]


def test_missing_target_column_raises_clear_error():
    """A missing requested target raises ValueError naming the target."""
    with pytest.raises(ValueError, match="Target column 'missing' was not found"):
        generate_model_statistics(pd.DataFrame({"x": [1]}), "missing")


def test_non_dataframe_input_raises_type_error():
    """Inputs other than DataFrames are rejected."""
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        generate_model_statistics({"target": [1]}, "target")


def test_empty_dataframe_raises_clear_error():
    """A zero-row DataFrame cannot produce regression evidence."""
    with pytest.raises(ValueError, match="at least one row"):
        generate_model_statistics(pd.DataFrame(columns=["target"]), "target")


def test_missing_target_values_are_counted_and_excluded_pairwise():
    """Missing targets are explicit; correlations use only complete pairs."""
    report = _report(
        pd.DataFrame({"x": [1, 2, 3], "target": [2.0, None, 6.0]})
    )

    assert report["target"]["missing_count"] == 1
    relationship = report["feature_target_relationships"][0]
    assert relationship["valid_paired_observation_count"] == 2
    assert relationship["pearson_correlation"] == pytest.approx(1.0)
    assert any("Missing target values" in warning for warning in report["warnings"])


def test_invalid_non_numeric_target_values_are_counted_not_silently_hidden():
    """Unparseable target data is counted, excluded, and produces a warning."""
    report = _report(
        pd.DataFrame({"x": [1, 2, 3], "target": ["2", "invalid", "6"]})
    )

    assert report["target"]["valid_numeric_count"] == 2
    assert report["target"]["invalid_non_numeric_count"] == 1
    assert report["feature_target_relationships"][0][
        "valid_paired_observation_count"
    ] == 2
    assert any("Invalid or non-finite target values" in item for item in report["warnings"])


def test_constant_target_is_unsuitable_for_regression():
    """A finite constant target is identified as unsuitable."""
    report = _report(pd.DataFrame({"x": [1, 2, 3], "target": [7, 7, 7]}))

    assert report["target"]["is_constant"] is True
    assert report["target"]["suitable_for_regression"] is False
    assert report["target"]["numeric_data_valid"] is False
    assert report["target"]["regression_suitability_assessment"] == "unsuitable"
    assert report["target"]["unsuitable_reason"] == (
        "A constant target cannot support regression."
    )
    assert report["feature_target_relationships"][0]["pearson_correlation"] is None


def test_two_distinct_target_values_do_not_establish_regression_suitability():
    """Numeric variability is valid evidence, not a complete suitability verdict."""
    report = _report(
        pd.DataFrame({"x": [1, 2], "target": [100, 200]})
    )

    assert report["target"]["numeric_data_valid"] is True
    assert report["target"]["suitable_for_regression"] is None
    assert report["target"]["regression_suitability_assessment"] == (
        "undetermined_from_target_alone"
    )
    assert any("cannot be determined from target statistics alone" in warning
               for warning in report["warnings"])


def test_constant_predictor_correlation_is_undefined():
    """Constant features stay visible but are not treated as usable predictors."""
    report = _report(
        pd.DataFrame({"constant": [5, 5, 5], "target": [1, 2, 3]})
    )

    assert report["features"][0]["detected_type"] == "constant"
    assert report["features"][0]["usable_for_correlation"] is False
    assert report["feature_target_relationships"][0]["pearson_correlation"] is None
    assert report["feature_target_relationships"][0]["limitation"] == (
        "constant_feature_or_target_in_paired_data"
    )


def test_all_missing_predictor_is_not_reported_as_constant():
    """An all-missing feature is distinguished from a constant feature."""
    report = _report(
        pd.DataFrame({"empty": [None, None, None], "target": [1, 2, 3]})
    )

    assert report["features"][0]["detected_type"] == "all_missing"
    assert report["features"][0]["usable_for_correlation"] is False
    assert report["feature_target_relationships"] == []


def test_numeric_strings_are_detected_and_invalid_values_counted():
    """A strict majority of convertible cells classifies strings as numerical."""
    report = _report(
        pd.DataFrame(
            {"measure": ["1", "2", "bad", "4"], "target": [1, 2, 3, 4]}
        )
    )

    feature = report["features"][0]
    assert feature["detected_type"] == "numerical"
    assert feature["valid_numeric_count"] == 3
    assert feature["invalid_non_numeric_count"] == 1
    assert report["feature_target_relationships"][0][
        "valid_paired_observation_count"
    ] == 3


def test_mixed_text_numeric_feature_requires_a_strict_numeric_majority():
    """One numeric string among two text labels does not imply a numeric feature."""
    report = _report(
        pd.DataFrame(
            {"mixed": ["abc", "def", "3"], "target": [1, 2, 3]}
        )
    )

    assert report["features"][0]["detected_type"] == "non_categorical_string"
    assert report["features"][0]["usable_for_correlation"] is False
    assert report["feature_target_relationships"] == []


def test_numeric_string_tie_is_not_classified_as_numeric():
    """Exactly half numeric and half invalid values does not pass the threshold."""
    report = _report(
        pd.DataFrame({"mixed": ["1", "bad"], "target": [1, 2]})
    )

    assert report["features"][0]["detected_type"] == "non_categorical_string"
    assert report["features"][0]["valid_numeric_count"] == 1
    assert report["features"][0]["invalid_non_numeric_count"] == 1
    assert report["feature_target_relationships"] == []


def test_partial_invalid_numeric_predictor_remains_unusable_if_valid_part_is_constant():
    """Invalid cells do not create numeric variation or make a constant usable."""
    report = _report(
        pd.DataFrame(
            {"measure": ["1", "1", "bad"], "target": [2, 3, 4]}
        )
    )

    feature = report["features"][0]
    assert feature["detected_type"] == "constant"
    assert feature["valid_numeric_count"] == 2
    assert feature["invalid_non_numeric_count"] == 1
    assert feature["usable_for_correlation"] is False
    assert report["feature_target_relationships"][0]["pearson_limitation"] == (
        "constant_feature_or_target_in_paired_data"
    )


def test_categorical_and_non_categorical_string_predictors_are_excluded():
    """Text categories and unique labels are not falsely numeric."""
    data = pd.DataFrame(
        {
            "category": ["red", "blue", "red"],
            "label": ["alpha", "beta", "gamma"],
            "target": [1, 2, 3],
        }
    )

    report = _report(data)

    assert [feature["detected_type"] for feature in report["features"]] == [
        "categorical",
        "non_categorical_string",
    ]
    assert report["feature_target_relationships"] == []


def test_unhashable_cells_are_classified_and_counted_without_crashing():
    """Lists, mappings, arrays, and custom objects are safely marked unsupported."""
    class NoFloatConversion:
        def __float__(self):
            raise RuntimeError("conversion must not be called")

    values = pd.Series(
        [[1, 2], {"value": 2}, np.array([3]), NoFloatConversion()],
        dtype=object,
    )
    report = _report(
        pd.DataFrame({"odd": values, "target": [1, 2, 3, 4]})
    )

    feature = report["features"][0]
    assert feature["detected_type"] == "unsupported"
    assert feature["valid_numeric_count"] == 0
    assert feature["invalid_non_numeric_count"] == 4
    assert feature["usable_for_correlation"] is False
    assert feature["correlation_limitation"] == "not_a_numeric_predictor"
    assert report["feature_target_relationships"] == []


def test_unhashable_target_values_are_counted_as_invalid():
    """Unsupported target cells do not trigger array truth or hashing errors."""
    target = pd.Series([[1], {"value": 2}, np.array([3]), 4], dtype=object)
    report = _report(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": target})
    )

    assert report["target"]["valid_numeric_count"] == 1
    assert report["target"]["invalid_non_numeric_count"] == 3
    assert report["target"]["missing_count"] == 0
    assert report["feature_target_relationships"][0][
        "valid_paired_observation_count"
    ] == 1
    assert report["feature_target_relationships"][0]["pearson_limitation"] == (
        "fewer_than_two_valid_pairs"
    )


def test_numeric_majority_with_unsupported_cells_preserves_row_alignment():
    """Unsupported entries count invalid without shifting later paired values."""
    data = pd.DataFrame(
        {
            "x": pd.Series([1, [2], 3, 4], dtype=object),
            "target": [10, 20, 30, 40],
        }
    )

    report = _report(data)
    feature = report["features"][0]
    relationship = report["feature_target_relationships"][0]

    assert feature["detected_type"] == "numerical"
    assert feature["valid_numeric_count"] == 3
    assert feature["invalid_non_numeric_count"] == 1
    assert relationship["valid_paired_observation_count"] == 3
    assert relationship["pearson_correlation"] == pytest.approx(1.0)


def test_infinite_values_and_invalid_numeric_strings_are_counted():
    """Non-finite numbers and non-convertible entries are excluded separately."""
    report = _report(
        pd.DataFrame(
            {
                "x": [1.0, float("inf"), 3.0, "bad"],
                "target": [1, 2, 3, 4],
            }
        )
    )

    feature = report["features"][0]
    assert feature["valid_numeric_count"] == 2
    assert feature["non_finite_count"] == 1
    assert feature["invalid_non_numeric_count"] == 1
    assert report["feature_target_relationships"][0][
        "valid_paired_observation_count"
    ] == 2


def test_complex_values_with_nonzero_imaginary_parts_are_never_real_observations():
    """Complex components are counted invalid instead of silently discarded."""
    report = _report(
        pd.DataFrame(
            {
                "x": [1 + 2j, 2 + 0j, 3 + 0j, 4 + 0j],
                "target": [1 + 1j, 2 + 0j, 3 + 0j, 4 + 0j],
            }
        )
    )

    feature = report["features"][0]
    assert feature["valid_numeric_count"] == 3
    assert feature["invalid_non_numeric_count"] == 1
    assert feature["non_finite_count"] == 0
    assert report["target"]["valid_numeric_count"] == 3
    assert report["target"]["invalid_non_numeric_count"] == 1
    assert report["feature_target_relationships"][0][
        "valid_paired_observation_count"
    ] == 3


def test_complex_feature_with_no_finite_real_observations_has_specific_limitation():
    """Object complex columns remain numeric candidates but never become usable."""
    feature = pd.Series([1 + 2j, 2 + 3j, 3 + 4j], dtype=object)
    report = _report(
        pd.DataFrame({"x": feature, "target": [1, 2, 3]})
    )

    assert report["features"][0]["detected_type"] == "numerical"
    assert report["features"][0]["valid_numeric_count"] == 0
    assert report["features"][0]["invalid_non_numeric_count"] == 3
    assert report["features"][0]["usable_for_correlation"] is False
    assert report["features"][0]["correlation_limitation"] == (
        "fewer_than_two_valid_pairs"
    )
    relationship = report["feature_target_relationships"][0]
    assert relationship["valid_paired_observation_count"] == 0
    assert relationship["pearson_limitation"] == "fewer_than_two_valid_pairs"


def test_extremely_large_finite_values_produce_finite_correlations_and_json():
    """Large representable values are scaled before Pearson calculation."""
    report = _report(
        pd.DataFrame(
            {
                "x": [1e308, 1.1e308, 1.2e308],
                "target": [1e308, 1.1e308, 1.2e308],
            }
        )
    )

    assert report["feature_target_relationships"][0][
        "pearson_correlation"
    ] == pytest.approx(1.0)
    result = generate_model_statistics(
        pd.DataFrame(
            {
                "x": [1e308, 1.1e308, 1.2e308],
                "target": [1e308, 1.1e308, 1.2e308],
            }
        ),
        "target",
    )
    json.loads(
        result,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert "Infinity" not in result and "NaN" not in result


def test_integer_values_overflowing_float64_are_counted_non_finite():
    """Huge Python integers are excluded safely instead of aborting conversion."""
    huge = 10**400
    report = _report(
        pd.DataFrame(
            {"x": [huge, huge + 1], "target": [huge, huge + 1]},
            dtype=object,
        )
    )

    assert report["features"][0]["valid_numeric_count"] == 0
    assert report["features"][0]["non_finite_count"] == 2
    assert report["target"]["valid_numeric_count"] == 0
    assert report["target"]["non_finite_count"] == 2


def test_perfect_positive_and_negative_correlations():
    """Pearson and Spearman identify perfect direct and inverse relationships."""
    report = _report(
        pd.DataFrame(
            {
                "positive": [1, 2, 3, 4],
                "negative": [4, 3, 2, 1],
                "target": [10, 20, 30, 40],
            }
        )
    )

    positive, negative = report["feature_target_relationships"]
    assert positive["pearson_correlation"] == pytest.approx(1.0)
    assert positive["spearman_correlation"] == pytest.approx(1.0)
    assert negative["pearson_correlation"] == pytest.approx(-1.0)
    assert negative["spearman_correlation"] == pytest.approx(-1.0)


def test_highly_correlated_predictor_pair_obeys_configured_threshold():
    """Feature pairs at or above the configured absolute threshold are listed."""
    report = _report(
        pd.DataFrame(
            {
                "first": [1, 2, 3, 4],
                "second": [2, 4, 6, 8],
                "target": [1, 4, 2, 3],
            }
        ),
        correlation_threshold=0.95,
    )

    pair = report["multicollinearity"]["highly_correlated_pairs"][0]
    assert (pair["feature_1"], pair["feature_2"]) == ("first", "second")
    assert pair["pearson_correlation"] == pytest.approx(1.0)
    assert pair["valid_paired_observation_count"] == 4
    assert pair["threshold"] == pytest.approx(0.95)
    assert report["multicollinearity"]["threshold_is_heuristic"] is True


def test_multicollinearity_threshold_is_inclusive():
    """A perfect pair meets the inclusive threshold at exactly one."""
    report = _report(
        pd.DataFrame(
            {"first": [1, 2, 3], "second": [3, 2, 1], "target": [1, 3, 2]}
        ),
        correlation_threshold=1.0,
    )

    pairs = report["multicollinearity"]["highly_correlated_pairs"]
    assert len(pairs) == 1
    assert abs(pairs[0]["pearson_correlation"]) == pytest.approx(1.0)
    assert pairs[0]["threshold"] == 1.0


def test_feature_pair_with_no_overlapping_finite_values_has_limitation():
    """Every numeric pair is represented even when no complete pair exists."""
    report = _report(
        pd.DataFrame(
            {
                "first": [1, 2, None, None],
                "second": [None, None, 3, 4],
                "target": [10, 20, 30, 40],
            }
        )
    )

    pair = report["feature_feature_relationships"][0]
    assert pair["valid_paired_observation_count"] == 0
    assert pair["pearson_correlation"] is None
    assert pair["pearson_limitation"] == "fewer_than_two_valid_pairs"
    assert report["multicollinearity"]["highly_correlated_pairs"] == []


def test_insufficient_paired_observations_return_null_with_limitation():
    """Fewer than two complete pairs yield null correlations and a reason."""
    report = _report(
        pd.DataFrame({"x": [1, None, 3], "target": [1, 2, None]})
    )

    relationship = report["feature_target_relationships"][0]
    assert relationship["valid_paired_observation_count"] == 1
    assert relationship["pearson_correlation"] is None
    assert relationship["spearman_correlation"] is None
    assert relationship["limitation"] == "fewer_than_two_valid_pairs"


def test_predictor_constant_on_valid_target_rows_is_not_usable():
    """Global variation does not imply variation among usable feature-target pairs."""
    report = _report(
        pd.DataFrame(
            {"x": [1, 1, 2, 3], "target": [1, 2, None, None]}
        )
    )

    assert report["features"][0]["valid_numeric_count"] == 4
    assert report["features"][0]["usable_for_correlation"] is False
    relationship = report["feature_target_relationships"][0]
    assert relationship["valid_paired_observation_count"] == 2
    assert relationship["pearson_correlation"] is None
    assert relationship["spearman_correlation"] is None
    assert relationship["pearson_limitation"] == (
        "constant_feature_or_target_in_paired_data"
    )
    assert relationship["spearman_limitation"] == (
        "constant_feature_or_target_in_paired_data"
    )


def test_pearson_and_spearman_limitations_are_reported_independently():
    """Small but representable variation at a large offset remains usable."""
    report = _report(
        pd.DataFrame(
            {
                "x": [1e16, 1e16 + 4, 1e16 + 8],
                "target": [1, 2, 3],
            }
        )
    )

    relationship = report["feature_target_relationships"][0]
    assert relationship["pearson_correlation"] == pytest.approx(1.0)
    assert relationship["pearson_limitation"] is None
    assert relationship["spearman_correlation"] == pytest.approx(1.0)
    assert relationship["spearman_limitation"] is None
    assert relationship["limitation"] is None


def test_one_ulp_variation_is_flagged_for_pearson_but_not_spearman():
    """Variations at float64 representational resolution get Pearson limitation."""
    base = np.float64(1e16)
    next_value = np.nextafter(base, np.inf)
    report = _report(
        pd.DataFrame(
            {"x": [base, next_value, next_value], "target": [1, 2, 3]}
        )
    )

    relationship = report["feature_target_relationships"][0]
    assert relationship["pearson_correlation"] is None
    assert relationship["pearson_limitation"] == (
        "near_constant_values_pearson_unreliable"
    )
    assert relationship["spearman_correlation"] is not None
    assert relationship["spearman_limitation"] is None


def test_ordinary_small_magnitude_variation_is_not_near_constant():
    """Small values around zero are not rejected based only on magnitude."""
    report = _report(
        pd.DataFrame(
            {"x": [1e-200, 2e-200, 3e-200], "target": [1, 2, 3]}
        )
    )

    relationship = report["feature_target_relationships"][0]
    assert relationship["pearson_correlation"] == pytest.approx(1.0)
    assert relationship["pearson_limitation"] is None


def test_values_collapsed_at_float_precision_are_reported_constant():
    """Distinct source expressions rounded to equal floats cannot fake variation."""
    base = np.float64(1e16)
    report = _report(
        pd.DataFrame(
            {"x": [base, base + 1, base + 1], "target": [1, 2, 3]}
        )
    )

    relationship = report["feature_target_relationships"][0]
    assert report["features"][0]["valid_numeric_count"] == 3
    assert report["features"][0]["usable_for_correlation"] is False
    assert relationship["pearson_limitation"] == (
        "constant_feature_or_target_in_paired_data"
    )


def test_decimal_and_numpy_numeric_scalars_are_parsed_safely():
    """Decimal, NumPy scalar, and non-finite values receive consistent counts."""
    report = _report(
        pd.DataFrame(
            {
                "x": pd.Series(
                    [Decimal("1.5"), np.int64(2), Decimal("Infinity"), Decimal("NaN")],
                    dtype=object,
                ),
                "target": [1, 2, 3, 4],
            }
        )
    )

    feature = report["features"][0]
    assert feature["valid_numeric_count"] == 2
    assert feature["non_finite_count"] == 1
    assert feature["valid_numeric_count"] == 2
    assert feature["invalid_non_numeric_count"] == 0


def test_numeric_strings_representing_infinity_count_as_non_finite():
    """Numeric non-finite strings qualify for detection but not usable counts."""
    report = _report(
        pd.DataFrame(
            {"x": ["Infinity", "-inf", "bad"], "target": [1, 2, 3]}
        )
    )

    feature = report["features"][0]
    assert feature["detected_type"] == "numerical"
    assert feature["valid_numeric_count"] == 0
    assert feature["non_finite_count"] == 2
    assert feature["invalid_non_numeric_count"] == 1
    assert feature["correlation_limitation"] == "fewer_than_two_valid_pairs"


def test_integer_values_losing_float64_precision_are_excluded_and_counted():
    """Adjacent integers beyond exact float64 precision do not fake correlation."""
    start = 2**53
    report = _report(
        pd.DataFrame(
            {
                "x": [start, start + 1, start + 2],
                "target": [1, 2, 3],
            }
        )
    )

    feature = report["features"][0]
    relationship = report["feature_target_relationships"][0]
    assert feature["valid_numeric_count"] == 2
    assert feature["precision_loss_count"] == 1
    assert relationship["valid_paired_observation_count"] == 2
    assert relationship["pearson_correlation"] is None
    assert relationship["spearman_correlation"] == pytest.approx(1.0)
    assert relationship["pearson_limitation"] == (
        "near_constant_values_pearson_unreliable"
    )


def test_all_integer_values_losing_float64_precision_are_unusable():
    """Precision-lost values receive an explicit limitation, not null correlation silently."""
    start = 2**53 + 1
    report = _report(
        pd.DataFrame(
            {"x": [start, start + 2], "target": [1, 2]}
        )
    )

    feature = report["features"][0]
    relationship = report["feature_target_relationships"][0]
    assert feature["detected_type"] == "numerical"
    assert feature["valid_numeric_count"] == 0
    assert feature["precision_loss_count"] == 2
    assert feature["usable_for_correlation"] is False
    assert feature["correlation_limitation"] == (
        "no_precision_safe_numeric_observations"
    )
    assert relationship["pearson_limitation"] == (
        "no_precision_safe_numeric_observations"
    )


def test_decimal_values_underflowing_to_zero_are_precision_loss():
    """Finite Decimal magnitudes below float64 range are not treated as real zeroes."""
    report = _report(
        pd.DataFrame(
            {
                "x": pd.Series(
                    [Decimal("1e-10000"), Decimal("2e-10000")],
                    dtype=object,
                ),
                "target": [1, 2],
            }
        )
    )

    feature = report["features"][0]
    assert feature["valid_numeric_count"] == 0
    assert feature["precision_loss_count"] == 2
    assert feature["correlation_limitation"] == (
        "no_precision_safe_numeric_observations"
    )


def test_pearson_handles_values_near_both_float64_extremes():
    """Centering falls back safely when subtracting opposite extremes overflows."""
    report = _report(
        pd.DataFrame(
            {
                "x": [-1.7e308, 1.7e308, -1.0e308, 1.0e308],
                "target": [-2, 2, -1, 1],
            }
        )
    )

    correlation = report["feature_target_relationships"][0][
        "pearson_correlation"
    ]
    assert correlation is not None
    assert math.isfinite(correlation)
    assert correlation == pytest.approx(0.9976836926815398)


def test_input_with_unhashable_values_and_duplicate_index_is_unchanged():
    """Unusual cells and duplicate index labels remain intact after analysis."""
    array_value = np.array([2, 3])
    data = pd.DataFrame(
        {
            "x": [None, None, None],
            "target": [1, 2, 3],
        },
        index=[4, 4, 9],
    )
    data.iat[0, 0] = array_value
    data.iat[1, 0] = [2]
    original_index = data.index.copy()
    original_first_cell = data.iloc[0, 0]

    generate_model_statistics(data, "target")

    assert data.index.equals(original_index)
    assert data.iloc[0, 0] is original_first_cell
    assert data.iloc[0, 0] is array_value


def test_duplicate_column_names_are_rejected():
    """Duplicate labels are rejected because target/predictor identity is ambiguous."""
    data = pd.DataFrame([[1, 2], [3, 4]], columns=["target", "target"])

    with pytest.raises(ValueError, match="column names must be unique"):
        generate_model_statistics(data, "target")


def test_no_usable_numeric_predictors_is_reported_as_warning():
    """A text-only/constant/all-missing predictor set yields an explicit warning."""
    report = _report(
        pd.DataFrame(
            {
                "category": ["a", "b", "a"],
                "constant": [9, 9, 9],
                "missing": [None, None, None],
                "target": [1, 2, 3],
            }
        )
    )

    assert len(report["feature_target_relationships"]) == 1
    constant_relationship = report["feature_target_relationships"][0]
    assert constant_relationship["feature"] == "constant"
    assert constant_relationship["pearson_correlation"] is None
    assert constant_relationship["spearman_correlation"] is None
    assert constant_relationship["valid_paired_observation_count"] == 3
    assert constant_relationship["limitation"] == (
        "constant_feature_or_target_in_paired_data"
    )
    assert any(
        "No feature-target correlations could be computed" in item
        for item in report["warnings"]
    )


def test_target_with_no_usable_numeric_values_is_reported_unsuitable():
    """A present but entirely invalid target is described, not silently dropped."""
    report = _report(
        pd.DataFrame({"x": [1, 2], "target": ["bad", "also bad"]})
    )

    assert report["target"]["valid_numeric_count"] == 0
    assert report["target"]["invalid_non_numeric_count"] == 2
    assert report["target"]["suitable_for_regression"] is False


def test_non_finite_target_values_are_counted_separately():
    """Infinite targets are not valid numbers and remain visible in target counts."""
    report = _report(
        pd.DataFrame({"x": [1, 2, 3], "target": [1.0, float("inf"), 3.0]})
    )

    assert report["target"]["valid_numeric_count"] == 2
    assert report["target"]["non_finite_count"] == 1
    assert report["feature_target_relationships"][0][
        "valid_paired_observation_count"
    ] == 2


def test_strict_json_has_no_nan_or_infinity_tokens():
    """Undefined and infinite calculations serialize as JSON null, never extensions."""
    result = generate_model_statistics(
        pd.DataFrame(
            {
                "x": [1.0, float("inf"), 3.0],
                "target": [1.0, float("nan"), 3.0],
            }
        ),
        "target",
    )
    parsed = json.loads(
        result,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )

    assert parsed["features"][0]["non_finite_count"] == 1
    assert not re.search(r"(?<![A-Za-z])(?:NaN|Infinity)(?![A-Za-z])", result)


def test_original_dataframe_is_not_modified():
    """Analysis leaves both values and original index untouched."""
    data = pd.DataFrame(
        {"x": ["1", "bad", None], "target": [2.0, 4.0, 6.0]},
        index=[8, 8, 3],
    )
    original = data.copy(deep=True)

    generate_model_statistics(data, "target")

    assert_frame_equal(data, original)


def test_correlation_threshold_must_be_finite_and_in_range():
    """Threshold validation prevents nonsensical multicollinearity criteria."""
    data = pd.DataFrame({"target": [1, 2]})

    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_model_statistics(data, "target", correlation_threshold=1.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_model_statistics(data, "target", correlation_threshold=-0.1)
    with pytest.raises(ValueError, match="finite"):
        generate_model_statistics(data, "target", correlation_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        generate_model_statistics(
            data, "target", correlation_threshold=float("inf")
        )


@pytest.mark.parametrize("threshold", [True, "0.8", None])
def test_correlation_threshold_rejects_non_real_types(threshold):
    """Boolean, string, and null thresholds are rejected as invalid types."""
    with pytest.raises(TypeError, match="real number"):
        generate_model_statistics(
            pd.DataFrame({"target": [1, 2]}),
            "target",
            correlation_threshold=threshold,
        )
