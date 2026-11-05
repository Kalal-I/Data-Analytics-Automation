import json
from pathlib import Path

from dataset_stats.description import generate_report
from dataset_stats.model_statistics import generate_model_statistics
from dataset_stats.open_file import read_csv_files
from agents_code.linear_regression_agent import linear_regression_agent
from agents_code.preprocessing_agent import preprocessing_agent
from agents_code.rfr_agent import rfr_agent
from knn_regression.main_knn_workflow import run_knn_workflow
from linear_regression.model_selection import (
    select_best_feature_report,
)
from linear_regression.optimization_loop import (
    PREPROCESSING_FUNCTIONS,
    _run_greedy_feature_selection,
    _run_regularized_optimization,
)
from linear_regression.split_data import _data_fingerprint
from preproccessing.validation import validate_preprocessing_calls
from preproccessing.deferred_encoding import (
    apply_deferred_encodings,
    partition_deferred_encodings,
)
from random_forest_regression.main_rf_workflow import (
    run_random_forest_workflow,
)
from random_forest_regression.optimization import (
    TUNING_PARAMETERS,
)
from random_forest_regression.random_forest_model import (
    DEFAULTS as RF_DEFAULTS,
    validate_model_params,
)
from random_forest_regression.tuning_report import build_tuning_report


def _rf_experience(iteration_number, result, reasoning=None):
    experience = {
        "iteration_number": iteration_number,
        "status": (
            "completed"
            if result["mean_std"]["r2"]["mean"] is not None
            else "failed"
        ),
        "parameters": dict(result["configuration"]),
        "metrics": {"cross_validation": result["mean_std"]},
    }
    if result.get("failures"):
        experience["failures"] = result["failures"]
    if reasoning is not None:
        experience["reasoning"] = reasoning
    return experience


def _rf_performance_key(metrics):
    r2 = metrics["r2"]["mean"]
    rmse = metrics["rmse"]["mean"]
    mae = metrics["mae"]["mean"]
    r2_std = metrics["r2"]["std"]
    if None in (r2, rmse, mae, r2_std):
        return None
    return r2, -rmse, -mae, -r2_std


def _run_random_forest_optimization(data, target, dataset_statistics):
    """Run the default search, then evaluate up to five agent proposals."""
    baseline_result = run_random_forest_workflow(data, target)
    baseline_search = baseline_result["cross_validation"]["results"]
    past_experiences = [
        _rf_experience(0, result) for result in baseline_search
    ]
    tuning_report = baseline_result.get("tuning_report")
    if tuning_report is None:
        tuning_report = build_tuning_report(
            baseline_result["cross_validation"]
        )
    agent_experiences = [
        {
            "iteration_number": 0,
            "status": "tuning_summary",
            "tuning_report": tuning_report,
        }
    ]

    default_configuration = {
        name: RF_DEFAULTS[name] for name in TUNING_PARAMETERS
    }
    default_search_result = next(
        (
            result
            for result in baseline_search
            if result["configuration"] == default_configuration
        ),
        None,
    )
    best_configuration = dict(baseline_result["best_configuration"])
    best_result = baseline_result
    best_search_result = next(
        (
            result
            for result in baseline_search
            if result["configuration"] == best_configuration
        ),
        None,
    )
    if best_search_result is None:
        raise RuntimeError(
            "The default random-forest search did not return its best configuration."
        )
    best_performance = _rf_performance_key(best_search_result["mean_std"])
    if best_performance is None:
        raise RuntimeError(
            "The default random-forest search returned unusable cross-validation metrics."
        )

    default_result = (
        baseline_result
        if best_configuration == default_configuration
        else None
    )
    if default_search_result is None:
        default_result = run_random_forest_workflow(
            data,
            target,
            search_space={
                name: [default_configuration[name]]
                for name in TUNING_PARAMETERS
            },
            n_iter=1,
        )
        default_search = default_result["cross_validation"]["results"]
        if (
            len(default_search) != 1
            or default_search[0]["configuration"] != default_configuration
        ):
            raise RuntimeError(
                "The default random-forest configuration was not evaluated exactly once."
            )
        default_search_result = default_search[0]
        past_experiences.append(_rf_experience(0, default_search_result))
    agent_experiences.append(
        _rf_experience(0, default_search_result)
    )

    default_performance = _rf_performance_key(
        default_search_result["mean_std"]
    )
    if default_performance is None:
        raise RuntimeError(
            "The default random-forest configuration returned unusable cross-validation metrics."
        )
    if default_performance > best_performance:
        if default_result is None:
            default_result = run_random_forest_workflow(
                data,
                target,
                search_space={
                    name: [default_configuration[name]]
                    for name in TUNING_PARAMETERS
                },
                n_iter=1,
            )
            default_search = default_result["cross_validation"]["results"]
            if (
                len(default_search) != 1
                or default_search[0]["configuration"] != default_configuration
            ):
                raise RuntimeError(
                    "The default random-forest configuration was not evaluated exactly once."
                )
        best_configuration = dict(default_configuration)
        best_result = default_result
        best_performance = default_performance

    current_configuration = dict(best_configuration)
    evaluated_configurations = {
        tuple(result["configuration"][name] for name in TUNING_PARAMETERS)
        for result in baseline_search
    }
    evaluated_configuration_history = []
    for result in baseline_search:
        configuration = {
            name: result["configuration"][name] for name in TUNING_PARAMETERS
        }
        if configuration not in evaluated_configuration_history:
            evaluated_configuration_history.append(configuration)
    default_key = tuple(
        default_configuration[name] for name in TUNING_PARAMETERS
    )
    if default_key not in evaluated_configurations:
        evaluated_configuration_history.append(default_configuration)
    evaluated_configurations.add(default_key)
    agent_iterations = 0

    for iteration_number in range(1, 6):
        recommendation = rfr_agent(
            dataset_statistics=dataset_statistics,
            past_experiences=agent_experiences,
            iteration_number=iteration_number,
            current_configuration=current_configuration,
            best_configuration=best_configuration,
            evaluated_configurations=evaluated_configuration_history,
        )
        agent_iterations += 1
        if not isinstance(recommendation, dict):
            raise TypeError("Random forest agent must return a JSON object.")
        reasoning = recommendation.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(
                "Random forest agent must return a non-empty reasoning string."
            )
        returned_parameters = set(recommendation) - {"reasoning"}
        missing_parameters = set(TUNING_PARAMETERS) - returned_parameters
        unsupported_fields = set(recommendation) - set(TUNING_PARAMETERS) - {
            "reasoning"
        }
        if missing_parameters:
            raise ValueError(
                "Random forest agent must return all tuning parameters: {}".format(
                    sorted(missing_parameters)
                )
            )
        if unsupported_fields:
            raise ValueError(
                "Random forest agent returned unsupported fields: {}".format(
                    sorted(unsupported_fields)
                )
            )
        print(
            "Random forest iteration {} recommendation: {}".format(
                iteration_number,
                json.dumps(recommendation, allow_nan=False),
            )
        )
        candidate_configuration = {
            name: recommendation[name] for name in TUNING_PARAMETERS
        }
        try:
            validate_model_params(candidate_configuration)
        except ValueError as exc:
            raise ValueError(
                "Random forest agent returned an invalid configuration: {}".format(
                    exc
                )
            ) from exc
        candidate_key = tuple(
            candidate_configuration[name] for name in TUNING_PARAMETERS
        )
        if candidate_key in evaluated_configurations:
            duplicate_experience = {
                "iteration_number": iteration_number,
                "status": "skipped_duplicate",
                "parameters": candidate_configuration,
                "reasoning": reasoning,
                "reason": (
                    "This configuration was already evaluated; no model "
                    "run was performed."
                ),
            }
            past_experiences.append(duplicate_experience)
            agent_experiences.append(duplicate_experience)
            print(
                "Random forest iteration {} skipped: duplicate configuration.".format(
                    iteration_number
                )
            )
            continue
        candidate_result = run_random_forest_workflow(
            data,
            target,
            search_space={
                name: [candidate_configuration[name]]
                for name in TUNING_PARAMETERS
            },
            n_iter=1,
        )
        candidate_search = candidate_result["cross_validation"]["results"]
        if len(candidate_search) != 1:
            raise RuntimeError(
                "A random-forest agent recommendation did not evaluate exactly one configuration."
            )
        evaluated_result = candidate_search[0]
        if evaluated_result["configuration"] != candidate_configuration:
            raise RuntimeError(
                "Random-forest evaluation did not use the agent-recommended configuration."
            )
        evaluated_configurations.add(candidate_key)
        evaluated_configuration_history.append(
            dict(candidate_configuration)
        )
        print(
            "Random forest iteration {} result: parameters={}, CV mean R2={}".format(
                iteration_number,
                json.dumps(candidate_configuration, allow_nan=False),
                evaluated_result["mean_std"]["r2"]["mean"],
            )
        )
        evaluated_experience = _rf_experience(
            iteration_number, evaluated_result, reasoning
        )
        past_experiences.append(evaluated_experience)
        agent_experiences.append(evaluated_experience)

        current_configuration = dict(candidate_configuration)
        candidate_performance = _rf_performance_key(
            evaluated_result["mean_std"]
        )
        if (
            candidate_performance is not None
            and candidate_performance > best_performance
        ):
            best_configuration = dict(candidate_configuration)
            best_result = candidate_result
            best_performance = candidate_performance

    return {
        **best_result,
        "past_experiences": past_experiences,
        "agent_iterations": agent_iterations,
        "best_cv_r2": best_performance[0],
    }

def main():
    #read the files
    path = input("Enter the path to the CSV file or directory: ")
    df = read_csv_files(path)
    if isinstance(df, list):
        raise ValueError("The main pipeline requires a single CSV DataFrame.")
    target = input("Enter the target column name: ")
    report = ""
    deferred_encoding_operations = {}
    '''if (isinstance(df,list)):
        print(df.size())
    else:
        report = generate_report(df,path,target)'''
    while True:
        report = generate_report(df,path,target)
        print(report)
        operation_to_do = preprocessing_agent(report)
        print("Initally:")
        print(operation_to_do)

        operation_to_do = validate_preprocessing_calls(operation_to_do, report)
        operation_to_do, deferred_operations = partition_deferred_encodings(
            operation_to_do
        )
        for column, operations in deferred_operations.items():
            deferred_encoding_operations.setdefault(column, operations)
        print("After validation:")
        print(operation_to_do)
        print("The following operations will be performed on the DataFrame:")
        if operation_to_do == {}:
            print(operation_to_do)
            break
        for column, operations in operation_to_do.items():
            for operation in operations:
                if operation["function"] == "standard_scale":
                    continue
                function = PREPROCESSING_FUNCTIONS[operation["function"]]
                df = function(df, **operation["inputs"])
    print("Preprocessing completed. The processed DataFrame is:")
    print(df)
    print(
        "Categorical encodings deferred until after model selection:",
        sorted(deferred_encoding_operations),
    )

    dataset_description = json.loads(generate_report(df, path, target))
    statistics = json.loads(generate_model_statistics(df, target))
    label_type = dataset_description["Reports"][0]["Target column"]["Type"]
    print(statistics)
    regression_recommendation = linear_regression_agent(
        dataset_description=dataset_description,
        target_column=target,
        label_type=label_type,
        statistics=statistics,
        row_count=len(df),
    )
    print("Linear regression agent returned:")
    print(regression_recommendation)

    ranked_models = regression_recommendation["ranked_models"]
    top_model = next(
        (model for model in ranked_models if model.get("rank") == 1),
        None,
    )
    if top_model and top_model.get("model_name") == "KNeighborsRegressor":
        knn_data = apply_deferred_encodings(
            df,
            deferred_encoding_operations,
            PREPROCESSING_FUNCTIONS,
        )
        knn_result = run_knn_workflow(
            knn_data,
            target,
            interactive=False,
        )
        knn_summary = {
            "selected_features": knn_result["selected_features"],
            "hyperparameters": knn_result["hyperparameters"],
            "cross_validation": knn_result["cross_validation"],
            "holdout_metrics": knn_result["holdout_metrics"],
        }
        print("KNN regression results:")
        print(json.dumps(knn_summary, indent=2, allow_nan=False))
        return knn_result
    elif top_model and top_model.get("model_name") == "RandomForestRegressor":
        random_forest_data = apply_deferred_encodings(
            df,
            deferred_encoding_operations,
            PREPROCESSING_FUNCTIONS,
        )
        random_forest_result = _run_random_forest_optimization(
            random_forest_data,
            target,
            {
                "dataset_description": dataset_description,
                "model_statistics": statistics,
                "target_column": target,
            },
        )
        random_forest_summary = {
            "model": "RandomForestRegressor",
            "parameters": random_forest_result["best_configuration"],
            "cv_mean_r2": random_forest_result["best_cv_r2"],
        }
        print("Best random forest model:")
        print(json.dumps(random_forest_summary, indent=2, allow_nan=False))
        return random_forest_result
    elif top_model and top_model.get("model_name") == "LinearRegressions": #must be a typo, should be "LinearRegression"
        current_model_report = {
            "target_column": target,
            "dataset_description": dataset_description,
            "statistics": statistics,
            "initial_model_ranking": regression_recommendation,
        }
        model_reports = _run_regularized_optimization(
            df,
            target,
            current_model_report,
            deferred_encoding_operations,
        )

        if not model_reports:
            raise ValueError(
                "The model optimization agent did not recommend a configuration "
                "to evaluate."
            )

        best_model_report, _ = select_best_feature_report(
            model_reports
        )
        selection_data = apply_deferred_encodings(
            df,
            deferred_encoding_operations,
            PREPROCESSING_FUNCTIONS,
        )
        selection_folds_file = (
            Path(__file__).resolve().parent
            / "linear_regression"
            / "folds_{}.json".format(_data_fingerprint(selection_data))
        )
        feature_selection_result = _run_greedy_feature_selection(
            selection_data,
            target,
            best_model_report,
            selection_folds_file,
        )
        final_selection_summary = {
            "accepted_features": feature_selection_result["accepted_features"],
            "cross_validation": feature_selection_result[
                "accepted_cross_validation"
            ],
        }
        print("Final accepted features and scores:")
        print(json.dumps(final_selection_summary, indent=2, allow_nan=False))
        return feature_selection_result


if __name__ == "__main__":
    main()
