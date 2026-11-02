"""Universal Question Evaluator engine.

Executes arbitrary user prompts across multiple LLMs concurrently with fault isolation,
measures real latency, token telemetry, and estimated costs, and provides honest evaluation
modes: Reference-based, Rubric-based, and Unreferenced (model agreement & quality heuristics).
"""

import re
import time
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable

from schema import TestCase, UniversalComparisonResult, UniversalComparison
import config
from config import (
    get_provider_status,
    get_token_pricing,
    get_api_key,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_COMPLETION_TOKENS,
    OLLAMA_BASE_URL
)
from engine.model_registry import list_registered_models, get_adapter, REGISTRY
from engine.telemetry import calculate_cost
from diagnostics.deterministic import (
    evaluate_exact_match,
    evaluate_numeric_tolerance,
    evaluate_regex,
    evaluate_structured_json,
    normalize_text,
    extract_numbers
)
from diagnostics.semantic_judge import evaluate_with_judge, evaluate_with_mock_judge

logger = logging.getLogger(__name__)

# Linguistic uncertainty / hedging cues
UNCERTAINTY_PATTERNS = [
    r"\bi am not (?:entirely |completely )?sure\b",
    r"\bi cannot (?:guarantee|confirm|verify)\b",
    r"\bit is (?:possible|unclear|uncertain|debatable) that\b",
    r"\bas of my (?:knowledge |training )?cutoff\b",
    r"\bi don't have access to\b",
    r"\bmight (?:or might not|possibly) be\b",
    r"\bto the best of my knowledge\b",
    r"\bapproximately|roughly|presumably\b"
]


def sanitize_text(text: Optional[str]) -> str:
    """Sanitize any accidental API keys or authorization secrets from strings."""
    if not text:
        return ""
    # Redact OpenAI, DeepSeek, Groq, xAI, Google, NVIDIA keys
    sanitized = re.sub(r"nvapi-[a-zA-Z0-9_\-]{15,}", "[REDACTED_API_KEY]", text)
    sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"gsk_[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"AIza[a-zA-Z0-9_\-]{30,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"xai-[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", "Bearer [REDACTED_TOKEN]", sanitized)
    return sanitized


def validate_question(question: str) -> str:
    """Validate question input, rejecting empty or whitespace-only inputs."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty. Please enter a valid question or prompt.")
    return question.strip()


async def execute_single_model_question(
    model_id: str,
    question: str,
    system_prompt: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    timeout: int = 60
) -> UniversalComparisonResult:
    """Execute a single model request with wall-clock latency, telemetry, and strict error isolation."""
    timestamp = datetime.now(timezone.utc).isoformat()
    meta = REGISTRY.get(model_id, {})
    provider = meta.get("provider", "mock")
    display_name = meta.get("display_name", model_id)
    pricing = get_token_pricing(model_id)
    pricing_str = f"Input: ${pricing['input']:.2f}/1M, Output: ${pricing['output']:.2f}/1M"
    is_simulated = (provider == "mock" or "mock" in model_id.lower())

    # Strict check: Never silently substitute mock for an unconfigured real provider!
    statuses = get_provider_status()
    prov_info = statuses.get(provider, {})
    if provider != "mock" and not prov_info.get("configured", False):
        err_msg = f"Provider '{provider}' is unconfigured. Set {provider.upper()}_API_KEY in environment or .env."
        return UniversalComparisonResult(
            model_id=model_id,
            provider=provider,
            display_name=display_name,
            is_simulated=is_simulated,
            status="unconfigured",
            response_text="",
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            pricing_assumptions=pricing_str,
            finish_reason=None,
            error_message=err_msg,
            sanitized_error=sanitize_text(err_msg),
            timestamp=timestamp
        )

    # Pre-flight check for local Ollama server connectivity
    if provider == "ollama":
        from engine.provider_adapter import OllamaAdapter
        is_alive, conn_msg = await OllamaAdapter.check_connection_static(config.OLLAMA_BASE_URL, timeout=2.0)
        if not is_alive:
            err_msg = f"Ollama local server is unavailable at {config.OLLAMA_BASE_URL}. Ensure the Ollama daemon is running ('ollama serve') before requesting local models."
            return UniversalComparisonResult(
                model_id=model_id,
                provider=provider,
                display_name=display_name,
                is_simulated=is_simulated,
                status="provider_error",
                response_text="",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                pricing_assumptions=pricing_str,
                finish_reason=None,
                error_message=err_msg,
                sanitized_error=sanitize_text(err_msg),
                timestamp=timestamp
            )

    # Instantiate provider adapter
    try:
        adapter = get_adapter(provider, model_id)
    except Exception as e:
        err_msg = f"Failed to initialize adapter for {model_id}: {str(e)}"
        return UniversalComparisonResult(
            model_id=model_id,
            provider=provider,
            display_name=display_name,
            is_simulated=is_simulated,
            status="provider_error",
            response_text="",
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            pricing_assumptions=pricing_str,
            finish_reason=None,
            error_message=err_msg,
            sanitized_error=sanitize_text(err_msg),
            timestamp=timestamp
        )

    # Execute generation with wall-clock timing
    start_time = time.perf_counter()
    try:
        gen_res = await adapter.generate(
            prompt=question,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        raw_text = gen_res.get("text", "") or ""
        p_tokens = int(gen_res.get("prompt_tokens", 0) or 0)
        c_tokens = int(gen_res.get("completion_tokens", 0) or 0)
        fin_reason = gen_res.get("finish_reason", "stop")
        prov_err = gen_res.get("error")

        # Fallback latency if adapter reported latency_ms
        if elapsed_ms <= 0.0 and "latency_ms" in gen_res:
            elapsed_ms = round(float(gen_res["latency_ms"]), 2)

        cost_usd = calculate_cost(p_tokens, c_tokens, pricing["input"], pricing["output"])

        if prov_err:
            return UniversalComparisonResult(
                model_id=model_id,
                provider=provider,
                display_name=display_name,
                is_simulated=is_simulated,
                status="provider_error",
                response_text=raw_text,
                latency_ms=elapsed_ms,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
                estimated_cost_usd=cost_usd,
                pricing_assumptions=pricing_str,
                finish_reason=fin_reason,
                error_message=str(prov_err),
                sanitized_error=sanitize_text(str(prov_err)),
                timestamp=timestamp
            )

        return UniversalComparisonResult(
            model_id=model_id,
            provider=provider,
            display_name=display_name,
            is_simulated=is_simulated,
            status="success",
            response_text=raw_text,
            latency_ms=elapsed_ms,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            estimated_cost_usd=cost_usd,
            pricing_assumptions=pricing_str,
            finish_reason=fin_reason,
            error_message=None,
            sanitized_error=None,
            timestamp=timestamp
        )

    except asyncio.TimeoutError:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        err = f"Request timed out after {timeout} seconds."
        return UniversalComparisonResult(
            model_id=model_id,
            provider=provider,
            display_name=display_name,
            is_simulated=is_simulated,
            status="timeout",
            response_text="",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            pricing_assumptions=pricing_str,
            finish_reason="timeout",
            error_message=err,
            sanitized_error=sanitize_text(err),
            timestamp=timestamp
        )
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        err = f"API error ({type(e).__name__}): {str(e)}"
        return UniversalComparisonResult(
            model_id=model_id,
            provider=provider,
            display_name=display_name,
            is_simulated=is_simulated,
            status="provider_error",
            response_text="",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            pricing_assumptions=pricing_str,
            finish_reason="error",
            error_message=err,
            sanitized_error=sanitize_text(err),
            timestamp=timestamp
        )


# =====================================================================
# EVALUATION MODES
# =====================================================================

def parse_rubric_criteria(rubric_text: str) -> List[str]:
    """Parse rubric text into distinct evaluation criteria or key points."""
    if not rubric_text:
        return []
    lines = rubric_text.strip().split("\n")
    criteria = []
    for line in lines:
        cleaned = line.strip()
        # Strip bullet points or numbers (e.g., "- ", "* ", "1. ", "1)")
        cleaned = re.sub(r"^[-*•\d\.\)]+\s*", "", cleaned).strip()
        if cleaned:
            criteria.append(cleaned)
    return criteria


def evaluate_response_rubric(
    response_text: str,
    criteria: List[str]
) -> Tuple[float, Dict[str, float], List[str], List[str], str]:
    """Evaluate candidate response against rubric criteria deterministically with lexical/semantic heuristics."""
    if not response_text.strip():
        return 0.0, {c: 0.0 for c in criteria}, criteria, ["Empty response provided."], "Response is empty."

    norm_resp = normalize_text(response_text)
    resp_tokens = set(re.findall(r"\w+", norm_resp))

    criteria_scores: Dict[str, float] = {}
    missing_points: List[str] = []
    unsupported_claims: List[str] = []

    for crit in criteria:
        norm_crit = normalize_text(crit)
        crit_tokens = set(re.findall(r"\w+", norm_crit))
        # Filter common stopwords for better overlap calculation
        stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "for", "with", "that", "this", "it"}
        content_crit_tokens = {t for t in crit_tokens if t not in stopwords and len(t) > 2}

        if not content_crit_tokens:
            content_crit_tokens = crit_tokens

        if not content_crit_tokens:
            criteria_scores[crit] = 1.0
            continue

        overlap = len(content_crit_tokens.intersection(resp_tokens)) / max(1, len(content_crit_tokens))
        
        # Check phrase match boost and semantic overlap
        if norm_crit in norm_resp:
            score = 1.0
        elif overlap >= 0.6:
            score = round(min(1.0, 0.85 + (overlap - 0.6) * 0.35), 2)
        elif overlap >= 0.4:
            score = round(min(1.0, 0.60 + (overlap - 0.4) * 1.0), 2)
        elif overlap > 0.1:
            score = round(overlap * 0.8, 2)
        else:
            score = 0.0

        criteria_scores[crit] = score
        if score < 0.5:
            missing_points.append(crit)

    # Compute overall score
    if criteria_scores:
        overall_score = round(sum(criteria_scores.values()) / len(criteria_scores), 2)
    else:
        overall_score = 1.0

    # Look for unsupported or contradictory markers
    if "not possible" in norm_resp and any("can" in c.lower() for c in criteria):
        unsupported_claims.append("Response claims impossibility contrary to expected criteria.")

    explanation = (
        f"Evaluated against {len(criteria)} rubric criteria. "
        f"Satisfied {len(criteria) - len(missing_points)}/{len(criteria)} key points "
        f"with an average criterion coverage of {overall_score * 100:.0f}%."
    )

    return overall_score, criteria_scores, missing_points, unsupported_claims, explanation


def evaluate_unreferenced_quality(
    question: str,
    response_text: str,
    system_prompt: Optional[str] = None
) -> Tuple[float, float, float, bool, List[str]]:
    """Compute response completeness, question relevance, instruction compliance, and uncertainty indicators."""
    if not response_text.strip():
        return 0.0, 0.0, 0.0, False, ["Empty response"]

    # 1. Completeness score
    word_count = len(response_text.split())
    # Substantive structure: sentences, punctuation, formatting
    has_sentence = bool(re.search(r"[.!?]\s+[A-Z]", response_text)) or "\n" in response_text
    if word_count < 5:
        completeness = 0.3
    elif word_count < 25:
        completeness = 0.6 if has_sentence else 0.5
    elif word_count < 200:
        completeness = 0.9 if has_sentence else 0.75
    else:
        completeness = 1.0

    # 2. Relevance score (lexical overlap with question content words)
    q_norm = normalize_text(question)
    stopwords = {"what", "how", "why", "who", "where", "when", "which", "is", "are", "the", "a", "an", "and", "or", "in", "of", "to"}
    q_tokens = {t for t in re.findall(r"\w+", q_norm) if t not in stopwords and len(t) > 2}
    r_tokens = set(re.findall(r"\w+", normalize_text(response_text)))

    if q_tokens:
        overlap = len(q_tokens.intersection(r_tokens)) / max(1, len(q_tokens))
        relevance = round(min(1.0, max(0.2, overlap * 1.2)), 2)
    else:
        relevance = 0.85

    # 3. Instruction following score
    instruction_score = 1.0
    if system_prompt:
        sys_lower = system_prompt.lower()
        # Check formatting constraints in system prompt
        if "json" in sys_lower:
            try:
                # strip fences if needed
                block = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_text).strip()
                json.loads(block)
                instruction_score = 1.0
            except Exception:
                instruction_score = 0.3
        elif "bullet" in sys_lower:
            has_bullets = bool(re.search(r"^\s*[-*•\d\.]\s+", response_text, re.MULTILINE))
            instruction_score = 1.0 if has_bullets else 0.5
        elif "single sentence" in sys_lower:
            sentences = re.split(r"[.!?]+", response_text.strip())
            sentences = [s for s in sentences if s.strip()]
            instruction_score = 1.0 if len(sentences) <= 1 else 0.5

    # 4. Uncertainty indicators
    uncertainty_detected = False
    indicators = []
    resp_lower = response_text.lower()
    for pat in UNCERTAINTY_PATTERNS:
        match = re.search(pat, resp_lower)
        if match:
            uncertainty_detected = True
            indicators.append(f"Hedging/uncertainty detected: '{match.group(0)}'")

    return completeness, relevance, instruction_score, uncertainty_detected, indicators


async def evaluate_single_result(
    result: UniversalComparisonResult,
    question: str,
    system_prompt: Optional[str] = None,
    reference_answer: Optional[str] = None,
    rubric: Optional[str] = None,
    eval_method: str = "auto",
    enable_llm_judge: bool = False,
    judge_adapter: Optional[Any] = None,
    judge_model: str = "mock-judge-v1"
) -> UniversalComparisonResult:
    """Evaluate one model's result honestly according to the requested evaluation mode."""
    # If model failed or produced no output, set failure metrics
    if result.status != "success" or not result.response_text:
        result.evaluation_mode = "reference" if reference_answer else ("rubric" if rubric else "unreferenced")
        result.evaluation_method = "failed_request"
        result.accuracy_score = 0.0 if (reference_answer or rubric) else None
        result.is_correct = False if (reference_answer or rubric) else None
        result.score_explanation = result.error_message or "Execution failed or produced empty response."
        return result

    # MODE A: Reference-Based Evaluation
    if reference_answer and reference_answer.strip():
        result.evaluation_mode = "reference"
        clean_ref = reference_answer.strip()
        method_to_use = eval_method.lower().replace(" ", "_")

        # Create temporary TestCase to reuse existing diagnostic evaluators
        fake_tc = TestCase(
            id="univ_eval",
            problem_statement=question,
            ground_truth=clean_ref,
            domain="General",
            evaluation_method="auto" if method_to_use == "auto" else method_to_use
        )

        # Method resolution if auto
        if method_to_use == "auto":
            gt_nums = extract_numbers(clean_ref)
            if gt_nums and len(clean_ref.split()) <= 4:
                method_to_use = "numeric_tolerance"
            elif clean_ref.startswith("{") or clean_ref.startswith("["):
                method_to_use = "json_schema"
            elif any(ch in clean_ref for ch in ["^", "$", ".*", "\\d"]):
                method_to_use = "regex"
            elif len(clean_ref.split()) > 15:
                method_to_use = "semantic"
            else:
                method_to_use = "exact_match"

        result.evaluation_method = method_to_use

        if method_to_use == "exact_match":
            is_corr, score, reason, _ = evaluate_exact_match(result.response_text, fake_tc)
            result.is_correct = is_corr
            result.accuracy_score = score
            result.score_explanation = reason
        elif method_to_use == "numeric_tolerance":
            is_corr, score, reason, _ = evaluate_numeric_tolerance(result.response_text, fake_tc)
            result.is_correct = is_corr
            result.accuracy_score = score
            result.score_explanation = reason
        elif method_to_use == "regex":
            is_corr, score, reason, _ = evaluate_regex(result.response_text, fake_tc)
            result.is_correct = is_corr
            result.accuracy_score = score
            result.score_explanation = reason
        elif method_to_use == "json_schema":
            is_corr, score, reason, _ = evaluate_structured_json(result.response_text, fake_tc)
            result.is_correct = is_corr
            result.accuracy_score = score
            result.score_explanation = reason
        elif method_to_use == "semantic":
            if enable_llm_judge and judge_adapter:
                try:
                    judge_out = await evaluate_with_judge(
                        fake_tc,
                        result.response_text,
                        provider_adapter=judge_adapter,
                        judge_model=judge_model
                    )
                    result.is_correct = judge_out.is_correct
                    result.accuracy_score = judge_out.score
                    result.score_explanation = f"[LLM Judge: {judge_model}] {judge_out.reasoning}"
                except Exception as e:
                    logger.warning("LLM judge failed, using heuristic judge: %s", str(e))
                    judge_out = evaluate_with_mock_judge(fake_tc, result.response_text)
                    result.is_correct = judge_out.is_correct
                    result.accuracy_score = judge_out.score
                    result.score_explanation = f"[Fallback Judge] {judge_out.reasoning}"
            else:
                judge_out = evaluate_with_mock_judge(fake_tc, result.response_text)
                result.is_correct = judge_out.is_correct
                result.accuracy_score = judge_out.score
                result.score_explanation = f"[Deterministic Semantic Heuristic] {judge_out.reasoning}"
        else:
            # Fallback to exact match
            is_corr, score, reason, _ = evaluate_exact_match(result.response_text, fake_tc)
            result.is_correct = is_corr
            result.accuracy_score = score
            result.score_explanation = reason

        return result

    # MODE B: Rubric-Based Evaluation
    elif rubric and rubric.strip():
        result.evaluation_mode = "rubric"
        result.evaluation_method = "rubric_scoring"
        criteria = parse_rubric_criteria(rubric)

        if not criteria:
            criteria = [rubric.strip()]

        score, crit_scores, missing, unsupported, expl = evaluate_response_rubric(
            result.response_text,
            criteria
        )
        result.accuracy_score = score
        result.is_correct = score >= 0.7
        result.rubric_scores = crit_scores
        result.missing_key_points = missing
        result.unsupported_claims = unsupported
        result.score_explanation = expl
        return result

    # MODE C: No Reference Answer (Unreferenced Quality & Agreement)
    else:
        result.evaluation_mode = "unreferenced"
        result.evaluation_method = "quality_heuristics"
        # Explicit honesty: Do NOT fabricate accuracy or correctness!
        result.accuracy_score = None
        result.is_correct = None
        result.score_explanation = "Accuracy not independently verified — no reference answer supplied."

        comp, rel, inst, uncert_det, uncert_list = evaluate_unreferenced_quality(
            question=question,
            response_text=result.response_text,
            system_prompt=system_prompt
        )
        result.completeness_score = comp
        result.relevance_score = rel
        result.instruction_following_score = inst
        result.uncertainty_detected = uncert_det
        result.uncertainty_indicators = uncert_list
        return result


def analyze_model_agreement(results: List[UniversalComparisonResult]) -> Dict[str, Any]:
    """Calculate pairwise agreement, consensus score, and identify response divergences across models."""
    valid_results = [r for r in results if r.status == "success" and r.response_text]
    if len(valid_results) < 2:
        return {
            "consensus_score": 0.0,
            "consensus_level": "Insufficient responses (need at least 2 successful models)",
            "pairwise_similarities": {},
            "divergence_notes": ["Fewer than 2 models succeeded; agreement analysis unavailable."],
            "disclaimer": "Model-to-model agreement is an indicator of consensus, NEVER proof of factual correctness."
        }

    # Tokenize responses
    stopwords = {"the", "a", "an", "is", "are", "and", "or", "in", "of", "to", "for", "with", "that", "this", "it"}
    tokens_by_model: Dict[str, set] = {}
    for r in valid_results:
        toks = set(re.findall(r"\w+", normalize_text(r.response_text)))
        tokens_by_model[r.model_id] = {t for t in toks if t not in stopwords and len(t) > 2}

    pairwise: Dict[str, float] = {}
    pair_scores: List[float] = []

    model_ids = [r.model_id for r in valid_results]
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            m1, m2 = model_ids[i], model_ids[j]
            s1, s2 = tokens_by_model[m1], tokens_by_model[m2]
            if s1 or s2:
                jaccard = len(s1.intersection(s2)) / max(1, len(s1.union(s2)))
            else:
                jaccard = 1.0
            score_val = round(jaccard, 3)
            pairwise[f"{m1} vs {m2}"] = score_val
            pair_scores.append(score_val)

    avg_agreement = round(sum(pair_scores) / max(1, len(pair_scores)), 3)
    if avg_agreement >= 0.65:
        level = "High Agreement (Models strongly converge on core content)"
    elif avg_agreement >= 0.35:
        level = "Moderate Agreement (Models share key topics with differing details)"
    else:
        level = "High Divergence (Models provide markedly different perspectives or answers)"

    divergence_notes = []
    for pair, sim in pairwise.items():
        if sim < 0.25:
            divergence_notes.append(f"Low semantic overlap ({sim:.1%}) between {pair}.")

    return {
        "consensus_score": avg_agreement,
        "consensus_level": level,
        "pairwise_similarities": pairwise,
        "divergence_notes": divergence_notes,
        "disclaimer": "Model-to-model agreement is an indicator of consensus, NEVER proof of factual correctness."
    }


async def run_universal_comparison(
    question: str,
    selected_models: List[str],
    system_prompt: Optional[str] = None,
    reference_answer: Optional[str] = None,
    rubric: Optional[str] = None,
    eval_method: str = "auto",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    enable_llm_judge: bool = False,
    judge_provider: str = "mock",
    judge_model: str = "mock-judge-v1",
    progress_callback: Optional[Callable[[str, str, Optional[UniversalComparisonResult]], None]] = None
) -> UniversalComparison:
    """Execute complete Universal Question comparison workflow across selected models."""
    clean_question = validate_question(question)
    comparison_id = f"cmp_{uuid.uuid4().hex[:10]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    if not selected_models:
        raise ValueError("At least one model must be selected for comparison.")

    # Prepare judge adapter if requested
    judge_adapter = None
    if enable_llm_judge and judge_provider != "mock":
        try:
            judge_adapter = get_adapter(judge_provider, judge_model)
        except Exception as e:
            logger.warning("Could not instantiate judge adapter: %s", str(e))

    # Determine execution mode
    if reference_answer and reference_answer.strip():
        eval_mode = "reference"
    elif rubric and rubric.strip():
        eval_mode = "rubric"
    else:
        eval_mode = "unreferenced"

    # Async task runner with progress callback
    async def run_model_task(mid: str) -> UniversalComparisonResult:
        if progress_callback:
            progress_callback(mid, "running", None)
        raw_res = await execute_single_model_question(
            model_id=mid,
            question=clean_question,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        eval_res = await evaluate_single_result(
            result=raw_res,
            question=clean_question,
            system_prompt=system_prompt,
            reference_answer=reference_answer,
            rubric=rubric,
            eval_method=eval_method,
            enable_llm_judge=enable_llm_judge,
            judge_adapter=judge_adapter,
            judge_model=judge_model
        )
        if progress_callback:
            progress_callback(mid, eval_res.status, eval_res)
        return eval_res

    # Dispatch all models concurrently
    tasks = [run_model_task(mid) for mid in selected_models]
    results: List[UniversalComparisonResult] = await asyncio.gather(*tasks, return_exceptions=False)

    # Determine run status
    success_count = sum(1 for r in results if r.status == "success")
    if success_count == len(results):
        overall_status = "completed"
    elif success_count > 0:
        overall_status = "partial_failure"
    else:
        overall_status = "failed"

    # Agreement analysis
    agreement = analyze_model_agreement(results)

    comparison = UniversalComparison(
        comparison_id=comparison_id,
        timestamp=timestamp,
        question=clean_question,
        system_prompt=system_prompt,
        reference_answer=reference_answer,
        rubric=rubric,
        evaluation_mode=eval_mode,
        selected_models=selected_models,
        generation_config={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "eval_method": eval_method,
            "enable_llm_judge": enable_llm_judge,
            "judge_model": judge_model if enable_llm_judge else "None"
        },
        results=results,
        status=overall_status,
        agreement_analysis=agreement
    )

    return comparison


# =====================================================================
# EXPORT HELPERS (CSV / JSON) WITHOUT SECRETS
# =====================================================================

def export_comparison_to_csv(comp: UniversalComparison) -> str:
    """Export comparison results to CSV string ensuring no secrets and explicitly unverified scores."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Comparison ID",
        "Timestamp",
        "Question",
        "Provider",
        "Model ID",
        "Display Name",
        "Is Simulated",
        "Status",
        "Evaluation Mode",
        "Evaluation Method",
        "Accuracy Score",
        "Is Correct",
        "Latency (ms)",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Tokens",
        "Estimated Cost (USD)",
        "Finish Reason",
        "Score Explanation",
        "Error Details",
        "Response Text"
    ]
    writer.writerow(headers)

    for r in comp.results:
        acc_str = f"{r.accuracy_score:.3f}" if r.accuracy_score is not None else "N/A"
        corr_str = str(r.is_correct) if r.is_correct is not None else "N/A"
        is_sim = getattr(r, "is_simulated", r.provider == "mock")
        writer.writerow([
            comp.comparison_id,
            r.timestamp or comp.timestamp,
            comp.question,
            r.provider,
            r.model_id,
            r.display_name,
            str(is_sim),
            r.status,
            r.evaluation_mode,
            r.evaluation_method,
            acc_str,
            corr_str,
            r.latency_ms,
            r.prompt_tokens,
            r.completion_tokens,
            r.total_tokens,
            f"{r.estimated_cost_usd:.6f}",
            r.finish_reason or "N/A",
            sanitize_text(r.score_explanation or "N/A"),
            sanitize_text(r.sanitized_error or r.error_message or "None"),
            sanitize_text(r.response_text)
        ])

    return output.getvalue()


def export_comparison_to_json(comp: UniversalComparison) -> str:
    """Export comparison results to clean JSON string without any secrets."""
    data = comp.model_dump()
    # Ensure all strings are sanitized
    def sanitize_obj(obj: Any) -> Any:
        if isinstance(obj, str):
            return sanitize_text(obj)
        elif isinstance(obj, dict):
            return {k: sanitize_obj(v) for k, v in obj.items() if "key" not in k.lower() and "secret" not in k.lower()}
        elif isinstance(obj, list):
            return [sanitize_obj(i) for i in obj]
        return obj

    cleaned = sanitize_obj(data)
    return json.dumps(cleaned, indent=2)
