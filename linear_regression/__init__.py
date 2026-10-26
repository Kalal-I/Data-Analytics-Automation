"""Simple linear regression training utilities."""

from linear_regression.fold_manager import (
    create_folds_and_train,
    create_or_load_folds,
)
from linear_regression.feature_selection import (
    R2_IMPROVEMENT_THRESHOLD,
    greedy_coefficient_feature_selection,
    rank_features_by_coefficient,
)
from linear_regression.linear_model import train_linear_regression
from linear_regression.results_json import split_result_to_json
from linear_regression.split_data import split_and_train

__all__ = [
    "create_folds_and_train",
    "create_or_load_folds",
    "R2_IMPROVEMENT_THRESHOLD",
    "greedy_coefficient_feature_selection",
    "rank_features_by_coefficient",
    "split_and_train",
    "split_result_to_json",
    "train_linear_regression",
]
