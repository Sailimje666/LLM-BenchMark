"""Root-cause diagnostic engine for LLM failures and error taxonomy classification."""

import re
import json
from typing import Optional, List, Dict, Any
from schema import TestCase, JudgeOutput, RootCauseDiagnosis


PRIMARY_CATEGORIES = [
    "Success",
    "Instruction Ignored",
    "Hallucination / Fact Drift",
    "Arithmetic / Logic Failure",
    "Truncation / Length Error",
    "Formatting Mismatch",
    "Knowledge Gap",
    "Tool / Execution Failure",
    "Provider / API Error",
    "Unclassified"
]

RECOMMENDED_ACTIONS = {
    "Success": "No corrective action required.",
    "Provider / API Error": "Check API keys, increase timeout/backoff settings, or verify provider rate limits.",
    "Truncation / Length Error": "Increase max_completion_tokens or instruct the model to provide concise responses.",
    "Formatting Mismatch": "Enable strict JSON schema mode, add few-shot formatting examples, or provide explicit output delimiters.",
    "Instruction Ignored": "Reinforce negative constraints in system prompt, position constraints at the end of the prompt, or use structured prompt tags.",
    "Arithmetic / Logic Failure": "Prompt for step-by-step chain-of-thought scratchpad or integrate a code interpreter/calculator tool.",
    "Hallucination / Fact Drift": "Incorporate retrieval-augmented grounding (RAG), lower temperature to 0.0, or ask the model to cite sources.",
    "Knowledge Gap": "Supplement model context with domain documentation or upgrade to a more capable frontier model.",
    "Tool / Execution Failure": "Verify tool argument formatting, sandbox environment health, and tool execution timeout.",
    "Unclassified": "Review prompt clarity and add explicit evaluation rubrics or unit tests."
}


def diagnose_failure(
    test_case: TestCase,
    response_text: str,
    is_correct: bool,
    accuracy_score: float,
    deterministic_reasoning: Optional[str] = None,
    constraint_violations: Optional[List[str]] = None,
    judge_output: Optional[JudgeOutput] = None,
    provider_error: Optional[str] = None,
    finish_reason: Optional[str] = None
) -> RootCauseDiagnosis:
    """Classify the root cause of an LLM response failure using transparent precedence rules."""

    violations = constraint_violations or []

    # Rule 1: Provider / API Error
    if provider_error:
        return RootCauseDiagnosis(
            primary_category="Provider / API Error",
            secondary_category=None,
            confidence_score=1.0,
            evidence=f"Provider call failed: {provider_error}",
            recommended_action=RECOMMENDED_ACTIONS["Provider / API Error"]
        )

    # Rule 2: Complete Success
    if is_correct and accuracy_score >= 0.99 and not violations:
        return RootCauseDiagnosis(
            primary_category="Success",
            secondary_category=None,
            confidence_score=1.0,
            evidence="Response satisfies all correctness rubrics and constraints.",
            recommended_action=RECOMMENDED_ACTIONS["Success"]
        )

    clean_resp = response_text.strip()

    # Rule 3: Truncation / Length Error
    is_truncated = False
    trunc_evidence = []
    if finish_reason in ["length", "max_tokens"]:
        is_truncated = True
        trunc_evidence.append(f"Provider reported finish_reason='{finish_reason}'.")
    if judge_output and judge_output.truncation_detected:
        is_truncated = True
        trunc_evidence.append("LLM Judge detected truncated response text.")
    if clean_resp.endswith(("...", "and", "the", "for", "with", "{", "[", "(")) and len(clean_resp) > 200:
        is_truncated = True
        trunc_evidence.append("Response ends abruptly mid-sentence or open syntax delimiter.")

    if is_truncated:
        return RootCauseDiagnosis(
            primary_category="Truncation / Length Error",
            secondary_category="Formatting Mismatch" if (clean_resp.startswith("{") and not clean_resp.endswith("}")) else None,
            confidence_score=0.95,
            evidence=" ".join(trunc_evidence),
            recommended_action=RECOMMENDED_ACTIONS["Truncation / Length Error"]
        )

    # Rule 4: Formatting Mismatch
    format_mismatch = False
    format_evidence = []
    if test_case.evaluation_method == "json_schema":
        try:
            # Check if JSON fails
            from diagnostics.deterministic import clean_json_string
            json.loads(clean_json_string(clean_resp))
        except Exception as e:
            format_mismatch = True
            format_evidence.append(f"Failed JSON syntax validation: {str(e)}.")
    
    if test_case.expected_format and "json" in test_case.expected_format.lower() and not ("{" in clean_resp or "[" in clean_resp):
        format_mismatch = True
        format_evidence.append(f"Expected JSON format, received plain text without structure.")

    if judge_output and judge_output.formatting_error:
        format_mismatch = True
        format_evidence.append("LLM Judge flagged formatting error.")

    if format_mismatch:
        return RootCauseDiagnosis(
            primary_category="Formatting Mismatch",
            secondary_category="Instruction Ignored" if violations else None,
            confidence_score=0.90,
            evidence=" ".join(format_evidence),
            recommended_action=RECOMMENDED_ACTIONS["Formatting Mismatch"]
        )

    # Rule 5: Instruction Ignored
    if violations or (judge_output and judge_output.instruction_violations):
        all_viol = list(violations)
        if judge_output:
            all_viol.extend(judge_output.instruction_violations)
        return RootCauseDiagnosis(
            primary_category="Instruction Ignored",
            secondary_category="Formatting Mismatch" if "format" in str(all_viol).lower() else "Arithmetic / Logic Failure",
            confidence_score=0.88,
            evidence=f"Model violated explicit constraints: {'; '.join(all_viol[:3])}",
            recommended_action=RECOMMENDED_ACTIONS["Instruction Ignored"]
        )

    # Rule 6: Knowledge Gap
    refusal_phrases = [
        "i do not know", "i cannot answer", "as an ai language model",
        "i don't have access", "not enough information", "cannot fulfill"
    ]
    if any(p in clean_resp.lower() for p in refusal_phrases):
        return RootCauseDiagnosis(
            primary_category="Knowledge Gap",
            secondary_category=None,
            confidence_score=0.85,
            evidence=f"Response indicates lack of knowledge or capability: '{clean_resp[:100]}...'",
            recommended_action=RECOMMENDED_ACTIONS["Knowledge Gap"]
        )

    # Rule 7: Arithmetic / Logic Failure
    if test_case.domain in ["Math", "Logic"] or (judge_output and judge_output.reasoning_errors):
        reasoning_ev = deterministic_reasoning or (judge_output.reasoning if judge_output else "Divergence in numeric or logical proof.")
        return RootCauseDiagnosis(
            primary_category="Arithmetic / Logic Failure",
            secondary_category="Hallucination / Fact Drift" if test_case.domain == "Logic" else None,
            confidence_score=0.90,
            evidence=f"Calculation or deduction error: {reasoning_ev}",
            recommended_action=RECOMMENDED_ACTIONS["Arithmetic / Logic Failure"]
        )

    # Rule 8: Hallucination / Fact Drift
    if test_case.domain in ["Text Extraction", "Factual"] or (judge_output and judge_output.factual_errors):
        ev = ("; ".join(judge_output.factual_errors) if (judge_output and judge_output.factual_errors) 
              else f"Extracted value failed ground truth match. Reason: {deterministic_reasoning or 'Wrong entity/value'}")
        return RootCauseDiagnosis(
            primary_category="Hallucination / Fact Drift",
            secondary_category="Instruction Ignored" if violations else None,
            confidence_score=0.85,
            evidence=ev,
            recommended_action=RECOMMENDED_ACTIONS["Hallucination / Fact Drift"]
        )

    # Rule 9: Partial score without specific error
    if 0.0 < accuracy_score < 1.0:
        return RootCauseDiagnosis(
            primary_category="Instruction Ignored",
            secondary_category="Unclassified",
            confidence_score=0.70,
            evidence=f"Response partially correct (score={accuracy_score:.2f}). Reason: {deterministic_reasoning or 'Incomplete match'}",
            recommended_action=RECOMMENDED_ACTIONS["Instruction Ignored"]
        )

    # Fallback: Unclassified
    return RootCauseDiagnosis(
        primary_category="Unclassified",
        secondary_category=None,
        confidence_score=0.50,
        evidence=deterministic_reasoning or "Evaluation failed without matching a specific diagnostic pattern.",
        recommended_action=RECOMMENDED_ACTIONS["Unclassified"]
    )
