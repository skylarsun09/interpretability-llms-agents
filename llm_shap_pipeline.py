# ============================================================
# Section 7: LLM Summary + Faithful Evaluation Pipeline
# ------------------------------------------------------------
# Depends on variables already defined in the notebook:
#   model_reduced, model_wrapper_reduced,
#   explainer_reduced, shap_values_reduced,
#   X_test_sample_reduced, remaining_features
# ============================================================

import json
import anthropic
import numpy as np

# ---- Shared Anthropic client (reads ANTHROPIC_API_KEY from env) ----
client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-20250514"
TOP_K  = 5   # number of top SHAP features to consider

base_value = float(explainer_reduced.expected_value)


# ============================================================
# PART A — LLM SUMMARY
# ============================================================

# ------------------------------------------------------------------
# A1. Global Summary
# ------------------------------------------------------------------

def build_global_prompt(shap_values, feature_names, base_val, top_k=TOP_K):
    """Construct a zero-shot prompt for the global SHAP summary."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_dir = shap_values.mean(axis=0)
    ranked   = np.argsort(mean_abs)[::-1]

    rows = []
    for rank, i in enumerate(ranked[:top_k], 1):
        direction = "positive" if mean_dir[i] > 0 else "negative"
        rows.append(
            f"  {rank}. {feature_names[i]}: "
            f"mean |SHAP| = {mean_abs[i]:.4f}, direction = {direction}"
        )
    block = "\n".join(rows)

    prompt = f"""You are an ML interpretability expert reviewing a credit card default \
prediction model (3-layer MLP, binary classification, KernelSHAP, {len(shap_values)} test samples).

Top {top_k} globally important features (mean |SHAP| over test set):
{block}

Base value (expected model output): {base_val:.4f}

Write a concise technical global summary (3–5 sentences). Cover:
1. Which features dominate and why this aligns with credit-risk intuition.
2. The direction of influence for each top feature.
3. Any notable patterns (e.g., repayment history vs. credit limit trade-offs).
"""
    return prompt, ranked[:top_k]


def get_global_summary(shap_values, feature_names):
    prompt, top_idx = build_global_prompt(shap_values, feature_names, base_value)
    resp    = client.messages.create(
        model=MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text, top_idx


# ------------------------------------------------------------------
# A2. Local Summary (per-sample) with ICL
# ------------------------------------------------------------------

# Curated few-shot reference examples
ICL_EXAMPLES = """
--- Reference Example 1 ---
Input: PAY_0=+0.18, LIMIT_BAL=-0.09, PAY_2=+0.07, AGE=-0.03, EDUCATION=-0.02
Prediction: 0.72 (high default risk). Base: 0.22.
Explanation: The primary driver of this high-risk prediction (0.72) is PAY_0 (+0.18),
reflecting a significant payment delay in September 2005. PAY_2 (+0.07) reinforces a
pattern of consecutive late payments. LIMIT_BAL (-0.09) and AGE (-0.03) provide a mild
mitigating effect but are insufficient to offset the repayment delay signals.

--- Reference Example 2 ---
Input: PAY_0=-0.15, LIMIT_BAL=+0.11, PAY_2=-0.08, AGE=+0.04, EDUCATION=+0.02
Prediction: 0.08 (low default risk). Base: 0.22.
Explanation: This sample is classified as low-risk (0.08). Timely recent repayment
(PAY_0=-0.15) is the strongest protective factor, substantially reducing default
probability below the base rate. A high credit limit (LIMIT_BAL=+0.11) and on-time
August payment (PAY_2=-0.08) further consolidate a stable repayment profile.
"""

def build_local_prompt(idx, shap_values, feature_names, X_sample, base_val, top_k=TOP_K):
    """Construct an ICL prompt for a single-sample local SHAP explanation."""
    sv      = shap_values[idx]
    fv      = X_sample[idx]
    pred    = float(base_val + sv.sum())
    ranked  = np.argsort(np.abs(sv))[::-1][:top_k]

    rows = [
        f"  {feature_names[i]}: SHAP={sv[i]:+.4f}, feature value={fv[i]:.4f}"
        for i in ranked
    ]
    block = "\n".join(rows)

    prompt = f"""You are an ML interpretability expert.

Use these reference explanations as a style guide:
{ICL_EXAMPLES}

Now produce a local explanation for the sample below.

Sample index  : {idx}
Base value    : {base_val:.4f}
Prediction    : {pred:.4f}  ({'high' if pred >= 0.5 else 'low'} default risk)
Top {top_k} SHAP contributors:
{block}

Write a concise technical explanation (3–4 sentences). Reference each feature's SHAP
value and direction; explain what drives the final prediction.
"""
    return prompt, ranked, sv, pred


def get_local_summary(idx, shap_values, feature_names, X_sample):
    prompt, top_idx, sv, pred = build_local_prompt(
        idx, shap_values, feature_names, X_sample, base_value
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text, top_idx, sv, pred


# ============================================================
# PART B — FAITHFULNESS EVALUATION
# ============================================================

# ------------------------------------------------------------------
# B1. Automated Scoring
# ------------------------------------------------------------------

def automated_score(explanation_text, sv_row, feature_names, top_k=TOP_K):
    """
    Returns:
      feature_overlap_rate  – fraction of top-K SHAP features mentioned in text
      direction_accuracy    – fraction of mentioned features with correct sign
    """
    ranked      = np.argsort(np.abs(sv_row))[::-1][:top_k]
    top_feats   = [feature_names[i] for i in ranked]
    top_dirs    = {feature_names[i]: ("positive" if sv_row[i] > 0 else "negative")
                   for i in ranked}

    text = explanation_text.lower()
    mentioned = [f for f in top_feats if f.lower() in text]
    overlap   = len(mentioned) / top_k

    POS_WORDS = ["positive", "increase", "higher", "raises", "pushes up", "+"]
    NEG_WORDS = ["negative", "decrease", "lower", "reduces", "pushes down", "mitigat"]

    hits = 0
    for feat in mentioned:
        pos = text.find(feat.lower())
        ctx = text[max(0, pos - 60): pos + 80]
        correct = top_dirs[feat]
        if correct == "positive" and any(w in ctx for w in POS_WORDS):
            hits += 1
        elif correct == "negative" and any(w in ctx for w in NEG_WORDS):
            hits += 1

    dir_acc = hits / len(mentioned) if mentioned else 0.0

    return {
        "top_k_features"      : top_feats,
        "mentioned_features"  : mentioned,
        "feature_overlap_rate": round(overlap, 3),
        "direction_accuracy"  : round(dir_acc, 3),
    }


# ------------------------------------------------------------------
# B2. Perturbation Consistency
# ------------------------------------------------------------------

def perturbation_test(idx, X_sample, shap_values, feature_names,
                      explainer, model_wrapper, top_k=TOP_K):
    """
    Perturbs the top-SHAP feature of a sample (by ±2 standardised units,
    opposing its current SHAP direction) and checks whether:
      - the model prediction changes (pred_changed)
      - the feature's SHAP magnitude decreases after perturbation (shap_consistent)
    """
    sv                = shap_values[idx]
    top_feat_idx      = int(np.argmax(np.abs(sv)))
    top_feat_name     = feature_names[top_feat_idx]
    original_val      = X_sample[idx, top_feat_idx]

    # Perturb opposite to SHAP direction to reduce its contribution
    delta = -2.0 if sv[top_feat_idx] > 0 else +2.0

    X_pert                            = X_sample.copy()
    X_pert[idx, top_feat_idx]        += delta

    orig_pred = float(model_wrapper(X_sample[idx: idx + 1]))
    pert_pred = float(model_wrapper(X_pert[idx: idx + 1]))

    shap_pert = explainer.shap_values(X_pert[idx: idx + 1])
    shap_pert = np.array(shap_pert).squeeze()

    return {
        "sample_idx"           : idx,
        "perturbed_feature"    : top_feat_name,
        "original_feature_val" : round(float(original_val), 4),
        "perturbation_delta"   : delta,
        "original_prediction"  : round(orig_pred, 4),
        "perturbed_prediction" : round(pert_pred, 4),
        "original_shap"        : round(float(sv[top_feat_idx]), 4),
        "perturbed_shap"       : round(float(shap_pert[top_feat_idx]), 4),
        "prediction_changed"   : abs(pert_pred - orig_pred) > 0.02,
        "shap_consistent"      : abs(shap_pert[top_feat_idx]) < abs(sv[top_feat_idx]),
    }


# ------------------------------------------------------------------
# B3. LLM-as-Judge
# ------------------------------------------------------------------

RUBRIC = """
Score faithfulness on a scale of 1–5:
  5 – All top SHAP features mentioned with correct directions; no hallucinations.
  4 – Most features mentioned correctly; minor omissions or one direction error.
  3 – Some features mentioned; notable omissions or one direction error.
  2 – Few features mentioned; multiple direction errors or hallucinations.
  1 – Explanation does not reflect the SHAP values.
"""

def llm_judge(explanation_text, sv_row, feature_names, base_val, top_k=TOP_K):
    """Ask Claude to score the faithfulness of an explanation against ground-truth SHAP."""
    pred    = float(base_val + sv_row.sum())
    ranked  = np.argsort(np.abs(sv_row))[::-1][:top_k]
    gt_rows = "\n".join([
        f"  {feature_names[i]}: SHAP={sv_row[i]:+.4f} "
        f"({'positive' if sv_row[i]>0 else 'negative'})"
        for i in ranked
    ])

    prompt = f"""You are a strict faithfulness evaluator for AI model explanations.

Ground-truth top {top_k} SHAP values:
{gt_rows}

Base value  : {base_val:.4f}
Prediction  : {pred:.4f}

Explanation to evaluate:
\"\"\"{explanation_text}\"\"\"

Scoring rubric:
{RUBRIC}

Reply ONLY with valid JSON (no markdown fences) matching this schema exactly:
{{
  "score"                : <int 1-5>,
  "reasoning"            : "<1-2 sentences>",
  "hallucinated_features": [<features mentioned but NOT in top SHAP>],
  "missing_features"     : [<top SHAP features NOT mentioned>]
}}
"""
    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ============================================================
# PART C — RUN FULL PIPELINE
# ============================================================

DIVIDER = "=" * 65

print(DIVIDER)
print("GLOBAL SUMMARY")
print(DIVIDER)
global_summary, global_top_idx = get_global_summary(
    shap_values_reduced, remaining_features
)
print(global_summary)

# Evaluate global summary against mean SHAP vector
mean_sv      = shap_values_reduced.mean(axis=0)
global_auto  = automated_score(global_summary, mean_sv, remaining_features)
global_judge = llm_judge(global_summary, mean_sv, remaining_features, base_value)

print("\n[Auto Score – Global]")
print(json.dumps(global_auto, indent=2))
print("\n[LLM Judge – Global]")
print(json.dumps(global_judge, indent=2))

# ------------------------------------------------------------------
# Local summaries for samples 0, 1, 2
# ------------------------------------------------------------------
all_results = []

for idx in range(3):
    print(f"\n{DIVIDER}")
    print(f"LOCAL SUMMARY — Sample {idx}")
    print(DIVIDER)

    local_summary, top_idx, sv, pred = get_local_summary(
        idx, shap_values_reduced, remaining_features, X_test_sample_reduced
    )
    print(local_summary)

    auto  = automated_score(local_summary, sv, remaining_features)
    judge = llm_judge(local_summary, sv, remaining_features, base_value)
    pert  = perturbation_test(
        idx,
        X_test_sample_reduced, shap_values_reduced, remaining_features,
        explainer_reduced, model_wrapper_reduced
    )

    print("\n[Auto Score]")
    print(json.dumps(auto, indent=2))
    print("\n[LLM Judge]")
    print(json.dumps(judge, indent=2))
    print("\n[Perturbation Consistency]")
    print(json.dumps(pert, indent=2))

    all_results.append({
        "sample_idx"   : idx,
        "prediction"   : round(pred, 4),
        "local_summary": local_summary,
        "auto_score"   : auto,
        "llm_judge"    : judge,
        "perturbation" : pert,
    })

print(f"\n{DIVIDER}")
print("PIPELINE COMPLETE")
print(DIVIDER)
