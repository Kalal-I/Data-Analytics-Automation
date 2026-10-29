"""A standalone, leakage-safe KNN regression package."""
from .data_split import train_holdout_split, split_data
from .folds import create_cv_folds, create_folds, FoldAssignment
from .feature_selection import validate_features, select_features
from .knn_model import (
    create_knn_model, create_knn_pipeline, fit_knn_model, predict_knn,
    create_model, fit_model, predict,
)
from .optimization import (
    grid_search, optimize_hyperparameters, elbow_analysis, evaluate_holdout,
    evaluate_configuration,
)
from .main_knn_workflow import run_knn_workflow

__all__ = [
    "train_holdout_split", "split_data", "create_cv_folds", "create_folds",
    "FoldAssignment", "validate_features", "select_features", "create_knn_model",
    "create_knn_pipeline", "fit_knn_model", "predict_knn", "grid_search",
    "optimize_hyperparameters", "elbow_analysis", "evaluate_holdout",
    "evaluate_configuration", "run_knn_workflow",
]
