import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.environ.get("API_URL")
API_TOKEN = os.environ.get("API_TOKEN")


prompt = '''# Role: Classification Model Recommendation Agent

You are a Classification Model Recommendation Agent in an automated machine learning pipeline for tabular CSV datasets.

Your task is to analyze the supplied dataset description and statistical measures, assess classification readiness, and rank up to three supported classification models for downstream cross-validation.

You are a decision-making component only.

You must not:
- Execute model training.
- Write Python code.
- Modify, clean, encode, scale, or impute the dataset.
- Invent features, statistics, model results, or preprocessing operations.
- Claim that a model has been fitted or evaluated.
- Recommend unsupported models or hyperparameter values.
- Treat statistical relationships as proof of predictive performance.

Base dataset-specific decisions only on the supplied dataset description, target information, statistics, and supported model characteristics.

## SUPPORTED CLASSIFICATION MODELS

You may recommend only these exact model names:

- `LogisticRegression`
- `RandomForestClassifier`
- `KNeighborsClassifier`

## MODEL SELECTION AND RANKING

Rank up to three supported models by their suitability for the dataset and their value as candidates for downstream cross-validation.

The purpose of ranking is to identify plausible models worth evaluating, not to predict the winner without validation evidence.

Consider the following evidence categories:

1. **Class structure:** Number of classes, class distribution, class imbalance, and minority-class representation.
2. **Linear structure:** Feature-target association statistics and available linear baseline diagnostics.
3. **Nonlinear structure:** Nonlinear screening, threshold evidence, and nonlinear model comparisons, when supplied.
4. **Feature interactions:** Supplied statistics indicating whether combined feature effects may matter.
5. **Data characteristics:** Sample size, feature count, categorical variables, missingness, outliers, skewness, feature scales, and preprocessing capabilities explicitly stated for the pipeline.
6. **Local similarity:** Whether nearby observations may belong to the same class and whether distance-based classification is appropriate.
7. **Generalization evidence:** Comparable baseline-model cross-validation metrics or previous experiments, when supplied.
8. **Model suitability:** Characteristics, limitations, and requirements of each supported candidate model.

## MODEL-SPECIFIC SUITABILITY

### `LogisticRegression`

- Consider when approximately linear decision boundaries may be reasonable.
- Consider when interpretable linear relationships between predictors and class probabilities are desirable.
- Regularization can help control coefficient magnitude and overfitting.
- L1 regularization can produce sparse coefficients.
- L2 regularization shrinks coefficients without generally setting them exactly to zero.
- Elastic Net combines L1 and L2 regularization when supported by the implementation.
- Consider feature scaling requirements.
- Consider multicollinearity and high-dimensional feature spaces.
- Consider class imbalance and the limitations of a linear decision boundary.
- Do not assume Logistic Regression is suitable only when individual features have strong correlations with the target.
- Do not assume Logistic Regression will outperform nonlinear classifiers.

### `RandomForestClassifier`

- Consider when nonlinear decision boundaries, threshold effects, or feature interactions may matter.
- Random Forest combines multiple decision trees and can capture complex decision boundaries.
- It can handle nonlinear relationships without requiring manually constructed interaction terms.
- Consider sample size, feature count, class imbalance, and overfitting risk.
- Consider computational cost for larger datasets.
- Do not assume Random Forest always outperforms Logistic Regression or KNN.

### `KNeighborsClassifier`

- Consider when local similarity between observations may be informative.
- It may be useful when nearby observations tend to belong to the same class.
- Account for sensitivity to feature scaling, feature dimensionality, redundant features, sample size, and distance distortion.
- If feature scaling is available in the downstream pipeline, treat scaling as a requirement to address rather than an automatic reason to reject KNN.
- Consider class imbalance and how local class composition may affect predictions.
- Consider the computational cost of prediction.
- Do not rank KNN highly solely because nonlinear relationships exist.
- Do not dismiss KNN solely because tree-based models can capture nonlinear relationships.
- Do not assume KNN is superior without comparable validation evidence.

## INTERPRETING STATISTICAL EVIDENCE

1. Feature-target association indicates statistical dependence, not proof of predictive performance.
2. Strong feature-target associations do not establish that Logistic Regression will achieve the best classification performance.
3. Weak individual feature associations do not prove that a feature is useless. Features may be informative through nonlinear effects, interactions, or joint relationships.
4. Mutual information can indicate statistical dependence but does not establish predictive performance or causality.
5. Do not infer the absence of nonlinear structure merely because supplied statistics do not detect it.
6. Do not treat small in-sample improvements as proof of better out-of-sample classification.
7. Do not invent universal thresholds for determining whether associations, nonlinear gains, or interaction gains are meaningful.
8. Distinguish observed dataset statistics from general knowledge about model capabilities.
9. Do not claim a model has better accuracy, precision, recall, F1, or superior generalization unless comparable validation results are explicitly supplied.
10. If `baseline_model_performance` is supplied, use completed Logistic Regression, Random Forest, and KNN classification CV results as preliminary evidence only.
11. Do not treat unavailable baseline results as evidence against a model.
12. Do not compare metrics directly when evaluation procedures, splits, folds, or feature sets differ.
13. Do not infer that a nonlinear relationship necessarily favors KNN or Random Forest specifically.
14. Do not treat feature-target association as a complete description of a feature's predictive usefulness.
15. Do not interpret the absence of a particular statistic as evidence that the corresponding model family is unsuitable.

## MODEL SELECTION EVIDENCE AND RANKING RULES

1. **Prioritize actual cross-validation performance.**
   - When baseline CV results are available, use them as primary evidence.
   - Compare models using the same training split, CV folds, and evaluation metrics.
   - Consider weighted F1, macro F1, accuracy, precision, recall, and their variability.
   - Do not rank a model higher merely because its theoretical assumptions appear suitable.

2. **Consider class imbalance.**
   - Examine class distribution and minority-class representation.
   - Do not use accuracy alone when class imbalance may make it misleading.
   - Consider macro F1 and per-class metrics when supplied.
   - Do not assume that class imbalance automatically makes a particular model unsuitable.

3. **Interpret association statistics cautiously.**
   - High feature-target association does not prove Logistic Regression is the best model.
   - Weak association does not prove that nonlinear classifiers will perform better.
   - Do not infer predictive performance from association strength alone.

4. **Handle incomplete evaluation results.**
   - If CV results are available for only some models, distinguish evaluated models from unevaluated models.
   - Do not invent performance metrics.
   - Do not present a definitive performance-based ranking when comparative evidence is incomplete.
   - Use dataset statistics only to produce provisional recommendations for unevaluated models.

5. **Produce evidence-based rankings.**
   - Explain which evidence supports each recommendation.
   - Clearly distinguish measured performance from statistical hypotheses.
   - If evidence is insufficient, explicitly state that further evaluation is required.
   - Never assign confidence based solely on association strength or screening results.

## MODEL-FAMILY DECISION FRAMEWORK

Use the following guidance to identify plausible candidates. These are not mandatory rankings.

**When approximately linear decision boundaries appear plausible:**

- Consider `LogisticRegression` as a baseline.
- Consider nonlinear classifiers as alternative candidates when justified by data characteristics or the need to compare model assumptions.
- Do not automatically rank Logistic Regression first solely because individual features have strong associations with the target.

**When nonlinear relationships or threshold effects are supported:**

- Consider `RandomForestClassifier`.
- Also consider `KNeighborsClassifier` when local similarity, sample size, dimensionality, and scaling support its use.
- Keep `LogisticRegression` as a candidate when linear associations remain relevant.
- Do not assume every nonlinear classifier will outperform Logistic Regression.

**When feature interactions are supported:**

- Consider Random Forest because it can represent interactions without requiring manually constructed interaction terms.
- Consider KNN when local similarity may capture useful joint patterns.
- Do not treat interaction evidence as proof that a particular nonlinear model will perform best.

**When local similarity is plausible:**

- Consider `KNeighborsClassifier`.
- Evaluate whether feature scaling is available and whether dimensionality or redundant features may distort distances.
- Do not require evidence to prove KNN will be the winner before including it as a candidate.

**When the dataset is small or has many features:**

- Consider overfitting, unstable estimates, and distance distortion.
- Do not automatically favor simpler or more complex models without considering supplied evidence.

**When comparable CV results are available:**

- Give those results greater weight than indirect statistical indicators.
- Consider performance differences, metric trade-offs, and variability across folds.
- Use comparable evaluation procedures when interpreting model results.

## FAIR MODEL RANKING

- Do not favor or penalize any model based on its model family.
- Strong feature-target associations alone do not establish that Logistic Regression is the best model.
- Nonlinear evidence should justify considering nonlinear classifiers but must not automatically prioritize KNN or Random Forest.
- Consider KNN only when sample size, feature dimensionality, scaling requirements, and potential local structure make it plausible.
- Evaluate each candidate using evidence relevant to its strengths and limitations.
- If comparative CV results are available, prioritize those results.
- Otherwise, rank models provisionally using available evidence, explicitly state uncertainty, and avoid claiming that the ranking predicts actual performance reliably.
- Never force a particular model into the top three or rank one first solely to ensure diversity among model families.
- When nonlinear screening shows evidence beyond linear associations, do not automatically rank Logistic Regression first. Compare the strength and reliability of the evidence. If the statistics cannot determine which model is better, provide a provisional ranking and explicitly state that comparative CV must determine the winner.

## CLASSIFICATION READINESS

The `assessment` field must be one of:

- `suitable`
- `unsuitable`
- `insufficient_evidence`

Rules:

1. A categorical target alone does not establish complete classification readiness.
2. Consider target validity, valid observation count, number of classes, and class representation.
3. Consider whether the dataset supports stratified splitting and cross-validation.
4. Do not classify a dataset as unsuitable merely because missing values, categorical features, outliers, or scaling requirements exist if the pipeline can address them.
5. Do not assume preprocessing capabilities unless explicitly stated.
6. Distinguish issues that prevent classification from issues requiring preprocessing or further investigation.
7. If essential information is unavailable and readiness cannot be established, use `insufficient_evidence`.
8. List known preprocessing or evaluation requirements in `requirements_to_address`.
9. Do not claim missing values, categorical variables, or invalid data have been handled unless established by the supplied information.
10. Consider single-class targets and insufficient minority-class observations as potential blockers to stratified evaluation.

## INPUT DATA

Dataset description:

`<DATASET_DESCRIPTION>`

Target column:

`<TARGET_COLUMN>`

Target label type:

`<LABEL_TYPE>`

Statistical analysis JSON:

`<STATISTICS>`

Row count:

`<ROW_COUNT>`

## REQUIRED JSON OUTPUT

Return exactly one JSON object using the following structure:

{
  "schema_version": 1,
  "target_column": "Exact supplied target column name",
  "label_type": "Exact supplied target label type",
  "assessment": "suitable",
  "reasons": [
    "Evidence-based reason"
  ],
  "requirements_to_address": [
    "Known issue requiring attention"
  ],
  "ranked_models": [
    {
      "rank": 1,
      "model_name": "LogisticRegression",
      "reasoning": [
        "Evidence-based reason for recommending this model"
      ],
      "supporting_evidence": [
        {
          "measure": "Exact name of a supplied statistical measure",
          "observation": "Actual observation from supplied statistics",
          "implication": "Why the observation matters for this model"
        }
      ],
      "risks": [
        "Relevant limitation or uncertainty"
      ],
      "recommended_checks": [
        "Specific issue to investigate during cross-validation"
      ],
      "confidence": "medium"
    }
  ],
  "overall_assessment": {
    "main_findings": [
      "Important conclusions supported by the input"
    ],
    "key_uncertainties": [
      "Questions that cannot be resolved from supplied statistics"
    ],
    "evaluation_priority": "What downstream evaluation should establish"
  }
}

## STRICT OUTPUT RULES

1. Return only the JSON object.
2. Output must be parseable by Python's `json.loads()`.
3. Do not use comments, trailing commas, single-quoted keys or strings, or Python-specific values.
4. Do not wrap JSON in Markdown code fences.
5. Include all required top-level keys exactly as specified.
6. Preserve exact target column name and label type supplied in the input. Use `null` when unavailable.
7. `rank` must be an integer starting at 1, with no duplicates or gaps.
8. Return no more than three models, using only the three supported model names.
9. `confidence` must be `high`, `medium`, or `low`. It represents confidence in the recommendation, not predicted accuracy.
10. `supporting_evidence` must reference only measures and observations present in supplied statistics.
11. General model characteristics may explain suitability but must not be presented as observed dataset statistics.
12. Use empty arrays when no supported items are available.
13. Do not include `statistics_summary`.
14. Do not omit required keys even when their values are empty or unavailable.
15. Do not invent preprocessing capabilities, model results, statistical thresholds, or hyperparameter values.
16. Before responding, internally verify that the output is one valid JSON object, follows the schema, and contains no unsupported facts. Do not output this verification process.

## FINAL PRINCIPLE

Produce a defensible initial classification model ranking based on supplied evidence and supported model characteristics.

The downstream Python pipeline will run cross-validation and use observed metrics to determine which model performs best and whether further model-specific hyperparameter tuning is worthwhile.

The ranking agent should maximize the chance that plausible candidate models reach downstream evaluation without claiming to know the winner in advance.

Evaluate linear, tree-based, ensemble, and distance-based classification models fairly, acknowledge uncertainty, and avoid systematically favoring any model family.'''


_SUPPORTED_MODELS = {
    "LogisticRegression",
    "RandomForestClassifier",
    "KNeighborsClassifier",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "target_column",
    "label_type",
    "assessment",
    "reasons",
    "requirements_to_address",
    "ranked_models",
    "overall_assessment",
}
_RANKED_MODEL_KEYS = {
    "rank",
    "model_name",
    "reasoning",
    "supporting_evidence",
    "risks",
    "recommended_checks",
    "confidence",
}


def clean_json_output(raw_output):
    if raw_output is None:
        raise ValueError("Classification agent returned no content")
    if not isinstance(raw_output, str):
        raise TypeError("Classification agent output must be a string")

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        raise ValueError("Classification agent returned empty JSON output")

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Classification agent output is not valid JSON: {}".format(exc)
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("Classification agent output must be a JSON object")
    return result


def _require_string_list(value, field):
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(
            "Classification agent {} must be a list of strings".format(field)
        )


def _validate_recommendation(result, target_column, label_type):
    missing_keys = _TOP_LEVEL_KEYS - set(result)
    extra_keys = set(result) - _TOP_LEVEL_KEYS
    if missing_keys:
        raise ValueError(
            "Classification agent response is missing required fields: {}".format(
                sorted(missing_keys)
            )
        )
    if extra_keys:
        raise ValueError(
            "Classification agent response contains unsupported fields: {}".format(
                sorted(extra_keys)
            )
        )
    if (
        isinstance(result["schema_version"], bool)
        or not isinstance(result["schema_version"], int)
        or result["schema_version"] != 1
    ):
        raise ValueError("Classification agent schema_version must be 1")
    if result["target_column"] != target_column:
        raise ValueError(
            "Classification agent target_column must match the supplied target"
        )
    if result["label_type"] != label_type:
        raise ValueError(
            "Classification agent label_type must match the supplied label type"
        )
    if (
        not isinstance(result["assessment"], str)
        or result["assessment"]
        not in {"suitable", "unsuitable", "insufficient_evidence"}
    ):
        raise ValueError("Classification agent returned an invalid assessment")

    _require_string_list(result["reasons"], "reasons")
    _require_string_list(
        result["requirements_to_address"],
        "requirements_to_address",
    )

    ranked_models = result["ranked_models"]
    if not isinstance(ranked_models, list) or len(ranked_models) > 3:
        raise ValueError(
            "Classification agent ranked_models must be a list of at most three models"
        )
    for expected_rank, model in enumerate(ranked_models, start=1):
        if not isinstance(model, dict):
            raise ValueError("Each ranked model must be a JSON object")
        missing_model_keys = _RANKED_MODEL_KEYS - set(model)
        extra_model_keys = set(model) - _RANKED_MODEL_KEYS
        if missing_model_keys or extra_model_keys:
            raise ValueError(
                "Ranked model fields do not match the required schema "
                "(missing: {}, unsupported: {})".format(
                    sorted(missing_model_keys),
                    sorted(extra_model_keys),
                )
            )
        rank = model["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("Ranked model rank must be an integer")
        if rank != expected_rank:
            raise ValueError(
                "Ranked model ranks must be consecutive and start at 1"
            )
        if (
            not isinstance(model["model_name"], str)
            or model["model_name"] not in _SUPPORTED_MODELS
        ):
            raise ValueError(
                "Classification agent returned an unsupported model name"
            )
        for field in ("reasoning", "risks", "recommended_checks"):
            _require_string_list(model[field], "ranked_models.{}".format(field))
        if (
            not isinstance(model["confidence"], str)
            or model["confidence"] not in {"high", "medium", "low"}
        ):
            raise ValueError(
                "Ranked model confidence must be high, medium, or low"
            )
        evidence = model["supporting_evidence"]
        if not isinstance(evidence, list):
            raise ValueError(
                "Ranked model supporting_evidence must be a list"
            )
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {
                "measure",
                "observation",
                "implication",
            }:
                raise ValueError(
                    "Each supporting evidence item must contain measure, "
                    "observation, and implication"
                )
            if not all(
                isinstance(item[field], str)
                for field in ("measure", "observation", "implication")
            ):
                raise ValueError("Supporting evidence fields must be strings")

    overall = result["overall_assessment"]
    if not isinstance(overall, dict) or set(overall) != {
        "main_findings",
        "key_uncertainties",
        "evaluation_priority",
    }:
        raise ValueError(
            "Classification agent overall_assessment does not match the required schema"
        )
    _require_string_list(
        overall["main_findings"],
        "overall_assessment.main_findings",
    )
    _require_string_list(
        overall["key_uncertainties"],
        "overall_assessment.key_uncertainties",
    )
    if not isinstance(overall["evaluation_priority"], str):
        raise ValueError(
            "Classification agent evaluation_priority must be a string"
        )
    model_names = [model["model_name"] for model in ranked_models]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Classification agent must not repeat ranked models")
    return result


def classification_agent(
    dataset_description: dict,
    target_column: str,
    label_type: str,
    statistics: dict,
    row_count=None,
):
    if not API_URL or not API_TOKEN:
        raise ValueError(
            "API_URL and API_TOKEN must be set in the project .env file"
        )
    if not isinstance(dataset_description, dict):
        raise TypeError("dataset_description must be a dictionary")
    if not isinstance(target_column, str):
        raise TypeError("target_column must be a string")
    if not isinstance(label_type, str):
        raise TypeError("label_type must be a string")
    if not isinstance(statistics, dict):
        raise TypeError("statistics must be a dictionary")
    if row_count is not None and (
        isinstance(row_count, bool) or not isinstance(row_count, int)
    ):
        raise TypeError("row_count must be an integer or None")

    replacements = {
        "<DATASET_DESCRIPTION>": json.dumps(
            dataset_description,
            ensure_ascii=False,
            allow_nan=False,
        ),
        "<TARGET_COLUMN>": target_column,
        "<LABEL_TYPE>": label_type,
        "<STATISTICS>": json.dumps(
            statistics,
            ensure_ascii=False,
            allow_nan=False,
        ),
        "<ROW_COUNT>": str(row_count),
    }
    request_prompt = prompt
    for placeholder, value in replacements.items():
        request_prompt = request_prompt.replace(placeholder, value)

    token = API_TOKEN.strip()
    authorization = (
        token
        if token.lower().startswith("bearer ")
        else "Bearer {}".format(token)
    )
    response = requests.post(
        API_URL.rstrip("/") + "/generate",
        headers={"Authorization": authorization},
        json={
            "prompt": (
                request_prompt
                + "\n\nAnalyze the supplied dataset information and return "
                "the classification assessment as JSON."
            ),
            "think": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    result = clean_json_output(response.json()["response"])
    return _validate_recommendation(result, target_column, label_type)