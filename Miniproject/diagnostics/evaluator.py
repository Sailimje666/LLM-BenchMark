"""Unified evaluation coordinator combining deterministic checks, LLM-as-a-Judge, and root cause diagnostics."""

import time
import logging
from typing import Optional, Any
from datetime import datetime, timezone

from schema import TestCase, EvaluationResult, JudgeOutput
from diagnostics.deterministic import (
    evaluate_exact_match,
    evaluate_numeric_tolerance,
    evaluate_regex,
    evaluate_structured_json
)
from diagnostics.semantic_judge import evaluate_with_judge, evaluate_with_mock_judge
from diagnostics.root_cause import diagnose_failure

logger = logging.getLogger(__name__)


async def evaluate_response(
    test_case: TestCase,
    response_text: str,
    run_id: str,
    provider: str,
    model_name: str,
    execution_time_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    provider_error: Optional[str] = None,
    finish_reason: Optional[str] = None,
    enable_semantic_judge: bool = True,
    judge_adapter: Optional[Any] = None,
    judge_model: str = "gpt-4o-mini"
) -> EvaluationResult:
    """Evaluate an LLM response and return a fully detailed EvaluationResult."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Case 1: Provider / Network Failure
    if provider_error:
        diagnosis = diagnose_failure(
            test_case=test_case,
            response_text="",
            is_correct=False,
            accuracy_score=0.0,
            provider_error=provider_error,
            finish_reason=finish_reason
        )
        return EvaluationResult(
            run_id=run_id,
            test_case_id=test_case.id,
            provider=provider,
            model_name=model_name,
            response_text="",
            accuracy_score=0.0,
            is_correct=False,
            evaluation_method="provider_check",
            execution_time_ms=execution_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
            error_category=diagnosis.primary_category,
            error_subcategory=diagnosis.secondary_category,
            deterministic_reasoning=None,
            judge_reasoning=None,
            constraint_violations=[],
            provider_error=provider_error,
            status="provider_error",
            timestamp=timestamp
        )

    # Determine evaluation method
    eval_method = test_case.evaluation_method.lower().replace(" ", "_")
    if eval_method == "auto":
        if test_case.domain == "Math":
            eval_method = "numeric_tolerance"
        elif test_case.regex_pattern:
            eval_method = "regex"
        elif test_case.expected_format and "json" in test_case.expected_format.lower():
            eval_method = "json_schema"
        elif test_case.domain == "Coding":
            eval_method = "semantic"
        else:
            eval_method = "exact_match"

    # Layer 1: Deterministic Evaluation
    det_is_correct = False
    det_score = 0.0
    det_reasoning = ""
    det_violations = []

    if eval_method == "exact_match":
        det_is_correct, det_score, det_reasoning, det_violations = evaluate_exact_match(response_text, test_case)
    elif eval_method == "numeric_tolerance":
        det_is_correct, det_score, det_reasoning, det_violations = evaluate_numeric_tolerance(response_text, test_case)
    elif eval_method == "regex":
        det_is_correct, det_score, det_reasoning, det_violations = evaluate_regex(response_text, test_case)
    elif eval_method == "json_schema":
        det_is_correct, det_score, det_reasoning, det_violations = evaluate_structured_json(response_text, test_case)

    # Layer 2: Semantic Judge (if method is semantic, or auto-fallback, or explicitly enabled)
    judge_out: Optional[JudgeOutput] = None
    if eval_method == "semantic" or (not det_is_correct and enable_semantic_judge and test_case.domain in ["Coding", "Logic"]):
        try:
            judge_out = await evaluate_with_judge(
                test_case=test_case,
                response_text=response_text,
                provider_adapter=judge_adapter,
                judge_model=judge_model
            )
        except Exception as e:
            logger.warning("Judge call exception: %s", str(e))
            judge_out = evaluate_with_mock_judge(test_case, response_text)

    # Resolve final score and correctness according to policy
    if eval_method == "semantic" and judge_out is not None:
        final_is_correct = judge_out.is_correct
        final_score = judge_out.score
        method_used = "semantic_judge"
    else:
        final_is_correct = det_is_correct
        final_score = det_score
        method_used = f"deterministic_{eval_method}"

    # Diagnose root cause of failure or confirm success
    diagnosis = diagnose_failure(
        test_case=test_case,
        response_text=response_text,
        is_correct=final_is_correct,
        accuracy_score=final_score,
        deterministic_reasoning=det_reasoning,
        constraint_violations=det_violations,
        judge_output=judge_out,
        finish_reason=finish_reason
    )

    return EvaluationResult(
        run_id=run_id,
        test_case_id=test_case.id,
        provider=provider,
        model_name=model_name,
        response_text=response_text,
        accuracy_score=final_score,
        is_correct=final_is_correct,
        evaluation_method=method_used,
        execution_time_ms=execution_time_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
        error_category=diagnosis.primary_category,
        error_subcategory=diagnosis.secondary_category,
        judge_reasoning=judge_out.reasoning if judge_out else None,
        deterministic_reasoning=det_reasoning if det_reasoning else None,
        constraint_violations=det_violations,
        provider_error=None,
        status="success",
        timestamp=timestamp
    )
