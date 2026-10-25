"""Run regularized-model experiments with resumable experiment history."""

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

import pandas as pd

INITIALIZATION_ALPHAS = (0.1, 0.5, 1.0, 10.0, 100.0)
MAX_AGENT_ITERATIONS = 5
MODEL_TYPES = {
    "Lasso": "lasso",
    "Ridge": "ridge",
    "ElasticNet": "elastic_net",
}


def experiment_history_path(
    directory: str | Path,
    data_fingerprint: str,
    target_column: str,
) -> Path:
    """Return a stable history path scoped to the exact data and target."""
    target_hash = hashlib.sha256(target_column.encode("utf-8")).hexdigest()[:12]
    return Path(directory) / "experiments_{}_{}.json".format(
        data_fingerprint,
        target_hash,
    )


class ExperimentHistory:
    """Atomically persist each configuration and its execution outcome."""

    def __init__(
        self,
        path: str | Path,
        data_fingerprint: str,
        target_column: str,
    ) -> None:
        self.path = Path(path)
        self.data_fingerprint = data_fingerprint
        self.target_column = target_column
        self.experiments: list[dict[str, Any]] = []
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as file:
                    stored = json.load(file)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Could not read experiment history from {!s}.".format(
                        self.path
                    )
                ) from error
            stored_experiments = (
                stored.get("experiments") if isinstance(stored, dict) else None
            )
            stopped = stored.get("stopped", False) if isinstance(stored, dict) else None
            required_fields = {
                "experiment_id",
                "agent_iteration",
                "initialization",
                "model",
                "model_type",
                "hyperparameters",
                "status",
                "cross_validation",
                "error",
                "report",
            }
            if (
                not isinstance(stored, dict)
                or stored.get("schema_version") != 1
                or stored.get("data_fingerprint") != data_fingerprint
                or stored.get("target_column") != target_column
                or not isinstance(stored_experiments, list)
                or not all(
                    isinstance(item, dict)
                    and required_fields.issubset(item)
                    and item["status"]
                    in {"pending", "completed", "failed", "skipped"}
                    and isinstance(item["hyperparameters"], dict)
                    for item in stored_experiments
                )
                or not isinstance(stopped, bool)
            ):
                raise ValueError(
                    "Experiment history is incompatible or malformed: {!s}."
                    .format(self.path)
                )
            self.experiments = stored_experiments
            self.stopped = stopped
        else:
            self.stopped = False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        contents = {
            "schema_version": 1,
            "data_fingerprint": self.data_fingerprint,
            "target_column": self.target_column,
            "experiments": self.experiments,
            "stopped": self.stopped,
        }
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(contents, file, indent=2, allow_nan=False)
            temporary_path.replace(self.path)
        except OSError as error:
            raise OSError(
                "Could not save experiment history to {!s}.".format(self.path)
            ) from error

    def append(self, experiment: dict[str, Any]) -> None:
        self.experiments.append(experiment)
        self.save()

    def append_many(self, experiments: list[dict[str, Any]]) -> None:
        self.experiments.extend(experiments)
        self.save()

    def completed_reports(self) -> list[dict[str, Any]]:
        return [
            experiment["report"]
            for experiment in self.experiments
            if experiment.get("status") == "completed"
            and isinstance(experiment.get("report"), dict)
        ]

    def agent_history(self) -> list[dict[str, Any]]:
        history = []
        for experiment in self.experiments:
            entry = {
                "experiment_id": experiment["experiment_id"],
                "agent_iteration": experiment["agent_iteration"],
                "status": experiment["status"],
                "model": experiment["model"],
                "model_type": experiment["model_type"],
                "hyperparameters": experiment["hyperparameters"],
                "initialization": experiment["initialization"],
            }
            if experiment.get("cross_validation") is not None:
                entry["cross_validation"] = experiment["cross_validation"]
            if experiment.get("error") is not None:
                entry["error"] = experiment["error"]
            history.append(entry)
        return history

    def has_configuration(
        self,
        model_type: str,
        hyperparameters: dict[str, Any],
    ) -> bool:
        for experiment in self.experiments:
            if experiment["model_type"] != model_type:
                continue
            previous = experiment["hyperparameters"]
            if float(previous["alpha"]) != float(hyperparameters["alpha"]):
                continue
            if model_type != "elastic_net" or float(
                previous["l1_ratio"]
            ) == float(hyperparameters["l1_ratio"]):
                return True
        return False

    def next_agent_iteration(self) -> int:
        if not self.experiments:
            return 1
        return max(item["agent_iteration"] for item in self.experiments) + 1


def _configuration(
    proposed_hyperparameters: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    hyperparameters = dict(proposed_hyperparameters)
    hyperparameters["alpha"] = alpha
    return hyperparameters


def _validate_configuration(
    model_type: str,
    hyperparameters: dict[str, Any],
) -> None:
    alpha = hyperparameters.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be positive with at most one decimal place.")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "alpha must be positive with at most one decimal place."
        ) from error
    if not math.isfinite(alpha_value) or alpha_value <= 0:
        raise ValueError("alpha must be positive with at most one decimal place.")
    try:
        decimal_alpha = Decimal(str(alpha)).normalize()
    except InvalidOperation as error:
        raise ValueError(
            "alpha must be positive with at most one decimal place."
        ) from error
    if decimal_alpha.as_tuple().exponent < -1:
        raise ValueError("alpha must be positive with at most one decimal place.")
    if model_type == "elastic_net":
        l1_ratio = hyperparameters.get("l1_ratio")
        if isinstance(l1_ratio, bool) or not isinstance(
            l1_ratio, (int, float)
        ):
            raise ValueError("l1_ratio must satisfy 0 < l1_ratio <= 1.")
        try:
            l1_ratio_value = float(l1_ratio)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "l1_ratio must satisfy 0 < l1_ratio <= 1."
            ) from error
        if not math.isfinite(l1_ratio_value) or not 0 < l1_ratio_value <= 1:
            raise ValueError("l1_ratio must satisfy 0 < l1_ratio <= 1.")
    elif "l1_ratio" in hyperparameters:
        raise ValueError("Lasso and Ridge must not include l1_ratio.")


def _new_experiment(
    experiment_id: str,
    agent_iteration: int,
    model_name: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    initialization: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "agent_iteration": agent_iteration,
        "initialization": initialization,
        "model": model_name,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "status": "pending",
        "cross_validation": None,
        "error": None,
        "report": None,
    }


def _run_experiment(
    data: pd.DataFrame,
    target_column: str,
    experiment: dict[str, Any],
    history: ExperimentHistory,
    evaluate: Callable[[pd.DataFrame, str, str, dict[str, Any]], dict[str, Any]],
) -> None:
    try:
        report = evaluate(
            data,
            target_column,
            experiment["model_type"],
            experiment["hyperparameters"],
        )
    except Exception as error:
        experiment["status"] = "failed"
        experiment["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        history.save()
        raise

    experiment["status"] = "completed"
    experiment["report"] = report
    experiment["cross_validation"] = report["training"]["cross_validation"]
    history.save()


def _optimization_summary(report: dict[str, Any]) -> dict[str, Any]:
    training = report["training"]
    return {
        "target_column": report["target_column"],
        "model": training["model"],
        "model_type": training["model_type"],
        "hyperparameters": training["hyperparameters"],
        "cross_validation": training["cross_validation"],
    }


def run_model_optimization(
    data: pd.DataFrame,
    target_column: str,
    initial_model_report: dict[str, Any],
    history_path: str | Path,
    data_fingerprint: str,
    agent: Callable[..., dict[str, Any]],
    evaluate: Callable[[pd.DataFrame, str, str, dict[str, Any]], dict[str, Any]],
    iteration_limit: int = MAX_AGENT_ITERATIONS,
) -> list[dict[str, Any]]:
    """Initialize a selected regularizer once, then continue agent iterations."""
    history = ExperimentHistory(history_path, data_fingerprint, target_column)
    if history.stopped:
        return history.completed_reports()
    current_model_report = initial_model_report
    initialization = [
        item for item in history.experiments if item.get("initialization")
    ]

    if not history.experiments:
        proposal = agent(
            initial_model_report,
            past_experiences=[],
            iteration_number=1,
        )
        if proposal:
            model_name = proposal["model"]
            model_type = MODEL_TYPES[model_name]
            if model_type in {"lasso", "ridge", "elastic_net"}:
                initial_hyperparameters = dict(proposal["hyperparameters"])
                _validate_configuration(model_type, initial_hyperparameters)
                initialization_experiments = []
                for alpha in INITIALIZATION_ALPHAS:
                    hyperparameters = _configuration(
                        initial_hyperparameters,
                        alpha,
                    )
                    _validate_configuration(model_type, hyperparameters)
                    initialization_experiments.append(_new_experiment(
                        "initialization-{}-alpha-{}".format(model_type, alpha),
                        1,
                        model_name,
                        model_type,
                        hyperparameters,
                        True,
                    ))
                history.append_many(initialization_experiments)
                initialization = [
                    item for item in history.experiments if item["initialization"]
                ]
            else:
                experiment = _new_experiment(
                    "iteration-1",
                    1,
                    model_name,
                    model_type,
                    dict(proposal["hyperparameters"]),
                    False,
                )
                _validate_configuration(model_type, experiment["hyperparameters"])
                if history.has_configuration(
                    model_type, experiment["hyperparameters"]
                ):
                    raise ValueError("The agent proposed a duplicate configuration.")
                history.append(experiment)
                _run_experiment(
                    data, target_column, experiment, history, evaluate
                )
        else:
            history.stopped = True
            history.save()

    for experiment in initialization:
        if experiment["status"] == "pending":
            _run_experiment(data, target_column, experiment, history, evaluate)

    completed = history.completed_reports()
    if completed:
        current_model_report = _optimization_summary(completed[-1])

    for experiment in history.experiments:
        if not experiment["initialization"] and experiment["status"] == "pending":
            _run_experiment(data, target_column, experiment, history, evaluate)
            current_model_report = _optimization_summary(experiment["report"])

    if history.stopped:
        return history.completed_reports()

    while history.next_agent_iteration() <= iteration_limit:
        iteration_number = history.next_agent_iteration()
        if iteration_number == 1:
            break
        proposal = agent(
            current_model_report,
            past_experiences=history.agent_history(),
            iteration_number=iteration_number,
        )
        if not proposal:
            history.stopped = True
            history.save()
            break
        model_name = proposal["model"]
        model_type = MODEL_TYPES[model_name]
        hyperparameters = dict(proposal["hyperparameters"])
        _validate_configuration(model_type, hyperparameters)
        if history.has_configuration(model_type, hyperparameters):
            raise ValueError("The agent proposed a duplicate configuration.")
        experiment = _new_experiment(
            "iteration-{}".format(iteration_number),
            iteration_number,
            model_name,
            model_type,
            hyperparameters,
            False,
        )
        history.append(experiment)
        _run_experiment(data, target_column, experiment, history, evaluate)
        current_model_report = _optimization_summary(experiment["report"])

    return history.completed_reports()
