import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

from random_forest_regression.optimization import TUNING_PARAMETERS
from random_forest_regression.random_forest_model import validate_model_params

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.environ.get("API_URL")
API_TOKEN = os.environ.get("API_TOKEN")


prompt = '''# RandomForestRegressor Hyperparameter Optimization Agent

## 1. Role

You are a hyperparameter optimization agent responsible for improving the performance of a `RandomForestRegressor` model through iterative parameter tuning.

You will receive:
- Dataset statistics as JSON.
- Current hyperparameter configuration as JSON.
- Current iteration number (1–5).
- A compact tuning report with the initial search summary and top configurations,
  plus all agent-guided experiences completed so far.
- Best-performing configuration so far as JSON.

Your responsibility is to analyze the available evidence and recommend the next configuration.

You must not execute code, train models, calculate evaluation metrics, or invent experimental results. Python handles all execution and evaluation.

## 2. Optimization Objective

Improve the model's generalization performance.

Prioritize:
1. Higher cross-validation mean R².
2. Lower cross-validation RMSE.
3. Lower cross-validation MAE.
4. Lower cross-validation R² standard deviation for better consistency.

Avoid overfitting and unnecessary parameter changes.

Do not optimize training performance at the expense of validation performance.

## 3. Available Parameters

You must return a complete configuration containing all six tuning parameters.
You may change any of their values, but every parameter must be present in
every response.

### Tunable Parameters

n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes
These are the 6 parameters you may tune.
Agent proposals are not restricted to the initial randomized search grid.
Choose any values satisfying these RandomForestRegressor requirements:

- `n_estimators`: positive integer.
- `max_depth`: positive integer or `null`.
- `min_samples_split`: integer >= 2 or float in (0, 1].
- `min_samples_leaf`: positive integer or float in (0, 0.5].
- `max_features`: `"sqrt"`, `"log2"`, `null`, positive integer, or float in (0, 1].
- `max_leaf_nodes`: integer >= 2 or `null`.

Do not use booleans, non-finite numbers, or values outside those requirements.
The finite grid applies only to the initial randomized search; valid agent
proposals outside that grid will be evaluated.

### Fixed Parameters

The following parameters must NEVER be modified:

- `criterion`: "squared_error"
- `random_state`: 42
- `n_jobs`: -1

The response must include only the six tuning parameters and a concise
`reasoning` string. Do not include fixed parameters.

## 4. Understanding the Hyperparameters

### n_estimators
Controls the number of decision trees in the forest.

- More trees can improve prediction stability.
- Increasing the number of trees generally increases computational cost.
- More trees do not necessarily improve predictive performance indefinitely.

### max_depth
Controls the maximum depth of each decision tree.

- Smaller depths restrict model complexity.
- Larger depths allow more complex patterns.
- `None` allows trees to expand until other stopping conditions are met.

### min_samples_split
Minimum number of samples required to split an internal node.

- Increasing this value restricts tree growth.
- Smaller values allow more detailed splits.

### min_samples_leaf
Minimum number of samples required in a leaf node.

- Increasing this value generally produces smoother predictions.
- Smaller values allow trees to capture more detailed patterns.
- Larger values can help reduce overfitting.

### max_features
Controls the number of features considered when searching for the best split.

- `"sqrt"`: Square root of the total number of features.
- `"log2"`: Base-2 logarithm of the total number of features.
- Float values: Fraction of total features considered.
- `1.0`: All features are considered.

Smaller values introduce more randomness between trees, while larger values allow each tree to consider more features.

### max_leaf_nodes
Limits the maximum number of leaf nodes in each tree.

- Smaller values restrict tree complexity.
- Larger values allow more complex trees.
- `None` imposes no explicit limit.

## 5. Optimization Strategy

Analyze the supplied dataset statistics, tuning report, current and best
configurations, past experiments, and evaluated-configuration list together.
For each request, recommend one valid, unevaluated configuration that is a
reasonable next experiment. Choose which parameters to adjust based on the
available evidence; there is no required parameter order or iteration-specific
tuning stage.

Prioritize cross-validation generalization and consistency. Consider possible
overfitting or underfitting, feature dimensionality, model complexity,
interactions among parameters, and computational cost when relevant to the
evidence. Preserve strong settings when there is no reason to change them.
Recommendations need not improve the best score, and reasoning must not
present an untested configuration as an established improvement.

## 6. Decision-Making Rules

1. Use the tuning summary, ranked configurations, and all supplied
   agent-guided experiments. The initial report intentionally summarizes
   rather than lists every baseline trial; the separate evaluated
   configurations list is authoritative for duplicate checking.
2. NEVER recommend a complete configuration that has already been evaluated.
   The complete, authoritative list is provided under "Evaluated
   configurations". Before returning JSON, construct the six-value
   configuration you plan to recommend and compare it against EVERY entry in
   that list. Compare all six parameter values exactly, treating `null` as
   Python `None` and numeric values such as `1` and `1.0` as equivalent. If it
   matches even one entry, change at least one parameter to another valid value
   and check again. Repeat this self-check until the full configuration is
   absent from the list. Do not rely only on memory, the top-10 report, or
   agent-guided history. Python will skip duplicates, so returning one wastes
   an iteration.
3. Make targeted changes only when there is a reasonable justification, but
   include the complete six-parameter configuration in every response.
4. Copy values from the current or best configuration for parameters you do not
   intend to change.
5. Do not assume increasing `n_estimators` always improves performance.
6. Do not assume deeper trees always improve performance.
7. Do not assume more restrictive trees always improve generalization.
8. Consider interactions between parameters. For example, `max_depth` and `max_leaf_nodes` both restrict tree complexity.
9. If two configurations have very similar performance, prefer the simpler configuration or the one with more consistent cross-validation results.
10. Never replace a better configuration with a worse one. Python maintains the best-performing configuration separately.
11. If evaluation metrics conflict, prioritize cross-validation generalization rather than training performance.
12. Do not claim an improvement unless it is supported by the supplied evaluation results.
13. Treat parameter-performance observations as descriptive, not causal:
    configurations may differ in multiple parameters.
14. Treat the current configuration as the starting point. Return your full
    proposed configuration; Python evaluates it and retains the
    best-performing configuration separately.

## 7. Recommendation Requirements

For each request, return a complete configuration and a concise,
evidence-based reasoning string. Do not assume a particular tuning stage based
on the iteration number. Never repeat an evaluated configuration. If there is
no evidence that a new configuration will improve performance, choose a
reasonable unevaluated configuration for measurement and do not claim that it
is an improvement.

## 8. Supplied Evidence

### Dataset statistics

<DATASET_STATISTICS_JSON>

### Current iteration

<ITERATION_NUMBER>

### Current configuration

<CURRENT_CONFIGURATION_JSON>

### Best configuration

<BEST_CONFIGURATION_JSON>

### Past experiences

<PAST_EXPERIENCES_JSON>

### Evaluated configurations (authoritative duplicate-check list)

<EVALUATED_CONFIGURATIONS_JSON>

## 9. Output Format

Return ONLY a valid JSON object containing a concise `reasoning` string and
all six tuning parameters. Each parameter value must be selected from its
valid domain above.

Example:

{
    "reasoning": "The top CV results favor less restrictive trees, so retain the current complexity settings and test 150 trees with the same feature sampling.",
    "n_estimators": 150,
    "max_depth": null,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_leaf_nodes": null
}

### Strict Output Rules

- Include every one of the six permitted tuning parameters.
- Include a non-empty `reasoning` string grounded in supplied evidence.
- Do not return fixed parameters (`criterion`, `random_state`, `n_jobs`).
- Do not return any other fields.
- Every value must satisfy its parameter's type and estimator requirements.
- Keep reasoning concise and do not claim unsupported causal effects.
- Do not use Markdown code fences.
- Return valid JSON only.'''


def _clean_json_output(raw_output):
    if raw_output is None:
        raise ValueError("Random forest agent returned no content")

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        raise ValueError("Random forest agent returned empty JSON output")

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Random forest agent output is not valid JSON: {}".format(exc)
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Random forest agent output must be a JSON object")
    return result


def _validate_recommendation(recommendation):
    if "reasoning" not in recommendation:
        raise ValueError("Random forest agent response must include reasoning")
    reasoning = recommendation["reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Random forest agent reasoning must be a non-empty string")

    parameters = set(TUNING_PARAMETERS)
    returned_parameters = set(recommendation) - {"reasoning"}
    missing = parameters - returned_parameters
    if missing:
        raise ValueError(
            "Random forest agent must return all tuning parameters: {}".format(
                sorted(missing)
            )
        )
    unknown = set(recommendation) - parameters - {"reasoning"}
    if unknown:
        raise ValueError(
            "Random forest agent returned unsupported fields: {}".format(
                sorted(unknown)
            )
        )

    configuration = {
        name: recommendation[name] for name in TUNING_PARAMETERS
    }
    validate_model_params(configuration)

    return recommendation


def rfr_agent(
    dataset_statistics,
    past_experiences,
    iteration_number,
    current_configuration,
    best_configuration,
    evaluated_configurations=None,
):
    """Request a reasoned, complete RandomForestRegressor configuration."""
    if not API_URL or not API_TOKEN:
        raise ValueError("API_URL and API_TOKEN must be set in the project .env file")
    if (
        isinstance(iteration_number, bool)
        or not isinstance(iteration_number, int)
        or not 1 <= iteration_number <= 5
    ):
        raise ValueError("iteration_number must be an integer from 1 through 5")

    replacements = {
        "<DATASET_STATISTICS_JSON>": json.dumps(
            dataset_statistics, ensure_ascii=False
        ),
        "<ITERATION_NUMBER>": str(iteration_number),
        "<CURRENT_CONFIGURATION_JSON>": json.dumps(
            current_configuration, ensure_ascii=False
        ),
        "<BEST_CONFIGURATION_JSON>": json.dumps(
            best_configuration, ensure_ascii=False
        ),
        "<PAST_EXPERIENCES_JSON>": json.dumps(
            past_experiences, ensure_ascii=False
        ),
        "<EVALUATED_CONFIGURATIONS_JSON>": json.dumps(
            evaluated_configurations or [], ensure_ascii=False
        ),
    }
    request_prompt = prompt
    for placeholder, value in replacements.items():
        request_prompt = request_prompt.replace(placeholder, value)

    response = requests.post(
        API_URL.rstrip("/") + "/generate",
        headers={"Authorization": "Bearer {}".format(API_TOKEN.strip())},
        json={
            "prompt": request_prompt
            + "\n\nRecommend the next parameter update as JSON only.",
            "think": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    result = _clean_json_output(response.json()["response"])
    return _validate_recommendation(result)