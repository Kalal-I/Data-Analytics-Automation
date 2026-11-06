import json
import math
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.environ.get("API_URL")
API_TOKEN = os.environ.get("API_TOKEN")


system_prompt = """
# Role: Regularized Logistic Regression Model Selection Agent

## 1. Objective

You are a model-selection agent in an automated machine learning pipeline. Your responsibility is to select the most appropriate regularized logistic regression model and recommend its hyperparameters based on the supplied dataset description and experimental evidence.

You must choose exactly one of the following models:

1. `Lasso`
2. `Ridge`
3. `ElasticNet`

You will be called for multiple iterations. Each invocation provides the current model's evaluation report, previous model experiment reports, and the current iteration number.

Current iteration number: `<ITERATION_NUMBER>`

Current model report:

`<MODEL_OPTIMIZATION_REPORT>`

Past experiences:

`<PAST_EXPERIENCES>`

## 2. Previously Used C Values

The following C values have already been tested during this optimization run:

- **Ridge:** `<RIDGE_C_VALUES_USED>`
- **Lasso:** `<LASSO_C_VALUES_USED>`
- **Elastic Net:** `<ELASTIC_NET_C_VALUES_USED>`
- **Elastic Net (C, l1_ratio) combinations:** `<ELASTIC_NET_COMBINATIONS_USED>`

Before proposing a configuration:

1. Inspect previously used C values for the selected model.
2. Avoid intentionally proposing an already-tested C value.
3. Prefer reasonable, untested C values informed by evaluation results.
4. Do not blindly repeat initialization values.
5. For Elastic Net, consider previously tested `(C, l1_ratio)` combinations.
6. Treat these lists as guidance rather than absolute restrictions.

Use past experiments to understand how model choice and hyperparameters affect performance.

Your responsibilities are limited to:

- Selecting one of the three supported classification models.
- Recommending required hyperparameters.
- Explaining the decision using supplied evidence.
- Proposing a meaningful next experiment.

You must not modify data, preprocessing, data splits, or evaluation configuration.

## 3. Model Selection Guidelines

### Lasso

Lasso applies L1 regularization and can shrink some coefficients to zero.

Consider Lasso when:

- Feature sparsity is desirable.
- Some predictors may contribute little predictive value.
- A simpler model is preferred.

Required hyperparameter:

- `C`: positive regularization inverse strength.

### Ridge

Ridge applies L2 regularization and shrinks coefficients without generally setting them exactly to zero.

Consider Ridge when:

- Predictors are strongly correlated.
- Coefficient instability may be a concern.
- Retaining all predictors is acceptable.

Required hyperparameter:

- `C`: positive regularization inverse strength.

### Elastic Net

Elastic Net combines L1 and L2 regularization.

- Specify both `C` and `l1_ratio`.
- The agent may recommend any `l1_ratio` satisfying `0 < l1_ratio <= 1`.
- Select `l1_ratio` using dataset characteristics, past experiments, and validation performance.
- Ratios close to 1 indicate Lasso-like behavior.
- Ratios close to 0 indicate Ridge-like behavior.
- Treat these as exploration signals, not automatic model switches.
- Do not impose a predefined list of allowed ratios.

## 4. Hyperparameter Selection Rules

### C Precision Constraint

- `C` must be positive and have at most one decimal place.
- Valid examples: `0.1`, `0.2`, `0.5`, `1`, `1.5`, `2`, `10`.
- Invalid examples: `0.15`, `0.25`, `0.37`, `1.25`.
- Any positive value with at most one decimal place is allowed.
- Python must independently validate this constraint.

### Other Parameters

Use the fixed settings established by the implementation unless the report explicitly establishes a different required configuration.

Do not tune unsupported parameters.

The optional `max_iter` setting may be a positive integer. If omitted, the
implementation uses `1000`.

## 5. Experiment Diversity and Exploration

1. Review experiment history before recommending configurations.
2. Prefer untested configurations.
3. Explore Lasso, Ridge, and Elastic Net fairly.
4. Test Lasso and Ridge independently rather than assuming Elastic Net is superior.
5. Use Elastic Net's `l1_ratio` to guide exploration of Lasso and Ridge.
6. Explore meaningfully different C values based on previous results.
7. Do not repeatedly tune one model while ignoring alternatives.
8. Consider convergence warnings and failed experiments when available.
9. Return `{}` only when no worthwhile experiment remains within the experiment budget.

## 6. Metric Interpretation

Use supplied cross-validation metrics to compare completed experiments.

Prioritize the established classification ranking criteria:

- Weighted F1: higher is better.
- Accuracy: higher is better.
- Weighted precision: higher is better.
- Weighted recall: higher is better.

Follow the implementation's actual ranking criteria if they differ.

Consider:

- Class imbalance.
- Cross-validation variability.
- Whether improvements are consistent.
- Whether minority-class performance deteriorates.
- Convergence warnings or unstable results.

Do not select a model based on training performance alone.

Do not assume a model is better merely because it uses more regularization.

If metrics are missing, do not fabricate them.

Do not treat tiny differences as decisive evidence.

## 7. Decision-Making Procedure

### Step 1: Analyze the current report

Identify:

- Target column and classification task.
- Current model and hyperparameters.
- Cross-validation classification metrics.
- Class distribution and imbalance information, if supplied.
- Convergence warnings or fitting issues, if supplied.

Do not infer class imbalance or feature importance without evidence.

### Step 2: Analyze previous experiments

For each experiment:

- Identify model and hyperparameters.
- Compare cross-validation classification metrics.
- Identify previously tested configurations.
- Determine whether performance improved, deteriorated, or remained comparable.
- Account for variability and class imbalance.

Do not repeat identical configurations without justification.

### Step 3: Select model and hyperparameters

Choose exactly one model from `Lasso`, `Ridge`, and `ElasticNet`.

Return:

- Model name.
- Required hyperparameters.
- Fixed parameters required by the implementation.
- Concise evidence-based reasoning.
- Hypothesis for the next experiment.

If evidence is insufficient, select a reasonable candidate for evaluation rather than claiming certainty.

### Step 4: Decide whether to stop

If all reasonable model/configuration choices have been evaluated and no further experiment is justified, return exactly `{}`.

Otherwise, recommend one next experiment.

## 8. Output Format

If another experiment is justified, return valid JSON only.

### Lasso

{
  "schema_version": 1,
  "action": "evaluate_model",
  "model": "Lasso",
  "hyperparameters": {
    "C": 1.0
  },
  "reasoning": [
    "Evidence supporting the selected model",
    "Reason for the proposed C"
  ],
  "hypothesis": "Expected effect to evaluate"
}

### Ridge

{
  "schema_version": 1,
  "action": "evaluate_model",
  "model": "Ridge",
  "hyperparameters": {
    "C": 1.0
  },
  "reasoning": [
    "Evidence supporting the selected model",
    "Reason for the proposed C"
  ],
  "hypothesis": "Expected effect to evaluate"
}

### Elastic Net

{
  "schema_version": 1,
  "action": "evaluate_model",
  "model": "ElasticNet",
  "hyperparameters": {
    "C": 1.0,
    "l1_ratio": 0.5
  },
  "reasoning": [
    "Evidence supporting the selected model",
    "Reason for the proposed C and l1_ratio"
  ],
  "hypothesis": "Expected effect to evaluate"
}

Numerical values are illustrative, not universal recommendations.

## 9. Strict Output Constraints

- Return JSON only, without Markdown code fences.
- Select exactly one supported model per invocation.
- Use only hyperparameters appropriate to that model.
- `C` must be positive.
- Elastic Net requires `0 < l1_ratio <= 1`.
- Do not change data, preprocessing, splits, or evaluation configuration.
- Do not fabricate experimental results.
- Do not claim a configuration is optimal before evaluation.
- Return exactly `{}` when no further experiment is justified.


"""


def _is_positive_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric_value) and numeric_value > 0


def _is_valid_l1_ratio(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric_value) and 0 < numeric_value <= 1


def clean_json_output(raw_output):
    if raw_output is None:
        raise ValueError("Model returned no content")

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("Model returned empty JSON output")

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Model output is not valid JSON: {}".format(exc)) from exc

    if not isinstance(result, dict):
        raise ValueError("Model output must be a JSON object")
    if not result:
        return result
    if result.get("schema_version") != 1:
        raise ValueError("Model output schema_version must be 1.")
    if result.get("action") != "evaluate_model":
        raise ValueError("Model output action must be 'evaluate_model'.")

    model = result.get("model")
    if model not in {"Lasso", "Ridge", "ElasticNet"}:
        raise ValueError("Model output must select Lasso, Ridge, or ElasticNet.")

    hyperparameters = result.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError("Model output hyperparameters must be an object.")

    c_value = hyperparameters.get("C")
    if not _is_positive_finite_number(c_value):
        raise ValueError("Model output C must be a positive finite number.")
    try:
        decimal_places = Decimal(str(c_value)).as_tuple().exponent
    except (InvalidOperation, ValueError):
        raise ValueError(
            "Model output C must have at most one decimal place."
        )
    if decimal_places < -1:
        raise ValueError(
            "Model output C must have at most one decimal place."
        )

    max_iter = hyperparameters.get("max_iter", 1000)
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, int)
        or max_iter <= 0
    ):
        raise ValueError("Model output max_iter must be a positive integer.")

    if model == "ElasticNet":
        l1_ratio = hyperparameters.get("l1_ratio")
        if not _is_valid_l1_ratio(l1_ratio):
            raise ValueError(
                "ElasticNet output requires l1_ratio satisfying 0 < l1_ratio <= 1."
            )
    elif "l1_ratio" in hyperparameters:
        raise ValueError("Lasso and Ridge output must not include l1_ratio.")

    return result


def decision_to_configuration(decision):
    """Convert an agent decision into a Logistic Regression workflow configuration."""
    if not isinstance(decision, dict):
        raise TypeError("decision must be a dictionary")
    if not decision:
        return None

    clean_json_output(json.dumps(decision))
    hyperparameters = dict(decision["hyperparameters"])
    penalty_by_model = {
        "Lasso": "l1",
        "Ridge": "l2",
        "ElasticNet": "elasticnet",
    }
    configuration = {
        "penalty": penalty_by_model[decision["model"]],
        "C": hyperparameters["C"],
    }
    if decision["model"] == "ElasticNet":
        configuration["l1_ratio"] = hyperparameters["l1_ratio"]
    if hyperparameters.get("max_iter", 1000) != 1000:
        configuration["max_iter"] = hyperparameters["max_iter"]
    return configuration


def logistic_regression_agent(
    model_optimization_report: dict,
    past_experiences: list[dict] | None = None,
    iteration_number: int = 1,
    ridge_c_values_used: list[float] | None = None,
    lasso_c_values_used: list[float] | None = None,
    elastic_net_c_values_used: list[float] | None = None,
    elastic_net_combinations_used: list[dict] | None = None,
):
    if not API_URL or not API_TOKEN:
        raise ValueError(
            "API_URL and API_TOKEN must be set in the project .env file"
        )
    if not isinstance(model_optimization_report, dict):
        raise TypeError("model_optimization_report must be a dictionary")
    if not isinstance(iteration_number, int) or isinstance(iteration_number, bool):
        raise TypeError("iteration_number must be an integer")
    if iteration_number < 1:
        raise ValueError("iteration_number must be at least 1")
    if past_experiences is not None and (
        not isinstance(past_experiences, list)
        or not all(isinstance(experience, dict) for experience in past_experiences)
    ):
        raise TypeError("past_experiences must be a list of dictionaries or None")

    c_histories = {
        "lasso_c_values_used": [] if lasso_c_values_used is None else lasso_c_values_used,
        "ridge_c_values_used": [] if ridge_c_values_used is None else ridge_c_values_used,
        "elastic_net_c_values_used": (
            [] if elastic_net_c_values_used is None else elastic_net_c_values_used
        ),
        "elastic_net_combinations_used": (
            []
            if elastic_net_combinations_used is None
            else elastic_net_combinations_used
        ),
    }
    if any(not isinstance(values, list) for values in c_histories.values()):
        raise TypeError("C value histories must be lists or None")

    report_json = json.dumps(
        model_optimization_report,
        ensure_ascii=False,
        allow_nan=False,
    )
    experiences_json = json.dumps(
        past_experiences if past_experiences is not None else [],
        ensure_ascii=False,
        allow_nan=False,
    )
    prompt = system_prompt.replace("<ITERATION_NUMBER>", str(iteration_number))
    prompt = prompt.replace("<MODEL_OPTIMIZATION_REPORT>", report_json)
    prompt = prompt.replace("<PAST_EXPERIENCES>", experiences_json)
    for placeholder, values in c_histories.items():
        prompt = prompt.replace(
            "<{}>".format(placeholder.upper()),
            json.dumps(values, ensure_ascii=False, allow_nan=False),
        )

    response = requests.post(
        API_URL.rstrip("/") + "/generate",
        headers={"Authorization": API_TOKEN},
        json={
            "prompt": (
                prompt
                + "\n\nAnalyze the supplied Logistic Regression report and "
                "return exactly one JSON decision."
            ),
            "think": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return clean_json_output(response.json()["response"])
