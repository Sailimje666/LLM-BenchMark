"""Offline unit tests for the Universal Question Evaluator engine and workflow."""

import pytest
import json
import asyncio
from unittest.mock import patch, AsyncMock

from schema import UniversalComparisonResult, UniversalComparison
from storage.repository import BenchmarkRepository
from engine.universal_evaluator import (
    validate_question,
    sanitize_text,
    execute_single_model_question,
    evaluate_single_result,
    evaluate_response_rubric,
    evaluate_unreferenced_quality,
    analyze_model_agreement,
    run_universal_comparison,
    export_comparison_to_csv,
    export_comparison_to_json,
    parse_rubric_criteria
)


# --- 1. Question Validation Tests ---

def test_validate_question_valid():
    """Ensure non-empty questions are stripped and accepted."""
    assert validate_question("  What is 2 + 2?  ") == "What is 2 + 2?"


def test_validate_question_empty_raises_error():
    """Ensure empty or whitespace-only questions raise ValueError."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        validate_question("")
    with pytest.raises(ValueError, match="Question cannot be empty"):
        validate_question("   \n\t  ")


# --- 2. Secret Sanitization Tests ---

def test_sanitize_text_redacts_api_keys():
    """Ensure API keys and auth bearer tokens are redacted from messages and errors."""
    raw = "Failed with sk-abcdef1234567890abcdef123456 and Bearer eyJhbGciOiJIUzI1Ni.secret and gsk_12345678901234567890"
    sanitized = sanitize_text(raw)
    assert "sk-" not in sanitized or "[REDACTED_API_KEY]" in sanitized
    assert "gsk_" not in sanitized or "[REDACTED_API_KEY]" in sanitized
    assert "eyJ" not in sanitized
    assert "[REDACTED_TOKEN]" in sanitized


# --- 3. Mode A: Reference-Based Evaluation Tests ---

@pytest.mark.asyncio
async def test_reference_evaluation_exact_match():
    """Test Mode A exact match evaluation."""
    res = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        status="success",
        response_text="Paris"
    )
    evaluated = await evaluate_single_result(
        result=res,
        question="What is the capital of France?",
        reference_answer="Paris",
        eval_method="exact_match"
    )
    assert evaluated.is_correct is True
    assert evaluated.accuracy_score == 1.0
    assert evaluated.evaluation_mode == "reference"
    assert "Exact match confirmed" in (evaluated.score_explanation or "")


@pytest.mark.asyncio
async def test_reference_evaluation_numeric_tolerance():
    """Test Mode A numeric tolerance evaluation."""
    res = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        status="success",
        response_text="The calculated value is 3.1415"
    )
    evaluated = await evaluate_single_result(
        result=res,
        question="What is pi to 4 decimal places?",
        reference_answer="3.1416",
        eval_method="numeric_tolerance"
    )
    assert evaluated.is_correct is True
    assert evaluated.accuracy_score == 1.0


@pytest.mark.asyncio
async def test_reference_evaluation_regex():
    """Test Mode A regex matching."""
    res = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        status="success",
        response_text="Contact us at info@example.com for help"
    )
    evaluated = await evaluate_single_result(
        result=res,
        question="Provide an email",
        reference_answer=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        eval_method="regex"
    )
    assert evaluated.is_correct is True
    assert evaluated.accuracy_score == 1.0


@pytest.mark.asyncio
async def test_reference_evaluation_json_schema():
    """Test Mode A structured JSON evaluation."""
    res = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        status="success",
        response_text='```json\n{"name": "Alice", "age": 30}\n```'
    )
    evaluated = await evaluate_single_result(
        result=res,
        question="Return person json",
        reference_answer='{"name": "Alice", "age": 30}',
        eval_method="json_schema"
    )
    assert evaluated.is_correct is True
    assert evaluated.accuracy_score == 1.0


# --- 4. Mode B: Rubric-Based Evaluation Tests ---

def test_parse_rubric_criteria():
    """Test parsing multi-line and bulleted rubric points."""
    rubric = """
    - Explains Big O notation
    - Mentions hash tables
    1. Gives a practical code example
    """
    crits = parse_rubric_criteria(rubric)
    assert len(crits) == 3
    assert crits[0] == "Explains Big O notation"
    assert crits[1] == "Mentions hash tables"
    assert crits[2] == "Gives a practical code example"


def test_evaluate_response_rubric_scoring():
    """Test criteria-level scoring, missing points, and explanation in Mode B."""
    response = "Hash tables allow average O(1) lookups using Big O notation."
    criteria = [
        "Explains Big O notation",
        "Mentions hash tables",
        "Provides a complete Java code snippet"
    ]
    overall, crit_scores, missing, unsupported, expl = evaluate_response_rubric(response, criteria)
    assert overall > 0.5
    assert "Provides a complete Java code snippet" in missing
    assert len(crit_scores) == 3
    assert crit_scores["Mentions hash tables"] > 0.5
    assert "Satisfied 2/3 key points" in expl


@pytest.mark.asyncio
async def test_rubric_evaluation_integration():
    """Test Mode B evaluation through evaluate_single_result."""
    res = UniversalComparisonResult(
        model_id="mock-claude",
        provider="mock",
        status="success",
        response_text="The solution uses dynamic programming to optimize time complexity."
    )
    evaluated = await evaluate_single_result(
        result=res,
        question="How does the algorithm work?",
        rubric="- Uses dynamic programming\n- Discusses time complexity\n- Mentions recursion"
    )
    assert evaluated.evaluation_mode == "rubric"
    assert evaluated.accuracy_score is not None
    assert len(evaluated.rubric_scores) == 3
    assert "Mentions recursion" in evaluated.missing_key_points


# --- 5. Mode C: Unreferenced Evaluation & Honesty Tests ---

@pytest.mark.asyncio
async def test_unreferenced_evaluation_honesty_no_fabricated_accuracy():
    """CRITICAL: Ensure Mode C does NOT fabricate an accuracy score or treat agreement as truth."""
    res = UniversalComparisonResult(
        model_id="mock-llama",
        provider="mock",
        status="success",
        response_text="Photosynthesis is the process by which green plants produce glucose using sunlight and water."
    )
    evaluated = await evaluate_single_result(
        result=res,
        question="Explain photosynthesis",
        reference_answer=None,
        rubric=None
    )
    assert evaluated.evaluation_mode == "unreferenced"
    # Accuracy score and is_correct MUST be None (unverified)
    assert evaluated.accuracy_score is None
    assert evaluated.is_correct is None
    assert "Accuracy not independently verified" in (evaluated.score_explanation or "")
    assert evaluated.completeness_score is not None
    assert evaluated.relevance_score is not None
    assert evaluated.instruction_following_score is not None


def test_evaluate_unreferenced_uncertainty_detection():
    """Test detection of hedging and linguistic uncertainty."""
    hedging_resp = "I am not completely sure, but to the best of my knowledge, it might be 42."
    comp, rel, inst, uncert_det, uncert_list = evaluate_unreferenced_quality(
        question="What is the answer?",
        response_text=hedging_resp
    )
    assert uncert_det is True
    assert len(uncert_list) >= 1


# --- 6. Model Agreement Analysis Tests ---

def test_analyze_model_agreement_insufficient_models():
    """Test agreement analysis when fewer than 2 models succeed."""
    r1 = UniversalComparisonResult(model_id="mock-gpt", provider="mock", status="success", response_text="Answer")
    analysis = analyze_model_agreement([r1])
    assert analysis["consensus_score"] == 0.0
    assert "Insufficient" in analysis["consensus_level"]


def test_analyze_model_agreement_high_consensus():
    """Test agreement analysis when models converge."""
    r1 = UniversalComparisonResult(model_id="mock-gpt", provider="mock", status="success", response_text="Quantum computing utilizes qubits and superposition to process information.")
    r2 = UniversalComparisonResult(model_id="mock-claude", provider="mock", status="success", response_text="Quantum computing uses superposition of qubits to perform complex calculations.")
    analysis = analyze_model_agreement([r1, r2])
    assert analysis["consensus_score"] > 0.3
    assert "disclaimer" in analysis
    assert "NEVER proof of factual correctness" in analysis["disclaimer"]


# --- 7. Multi-Model Concurrent Execution & Fault Isolation ---

@pytest.mark.asyncio
async def test_concurrent_execution_all_mock_models():
    """Test concurrent execution across multiple mock models without API calls."""
    models = ["mock-gpt", "mock-claude", "mock-llama"]
    comp = await run_universal_comparison(
        question="What are three primary colors?",
        selected_models=models,
        reference_answer="Red, blue, and yellow"
    )
    assert comp.comparison_id.startswith("cmp_")
    assert comp.status == "completed"
    assert len(comp.results) == 3
    for r in comp.results:
        assert r.status == "success"
        assert r.latency_ms >= 0.0
        assert r.total_tokens > 0
        assert r.accuracy_score is not None


@pytest.mark.asyncio
async def test_fault_isolation_unconfigured_provider():
    """Ensure an unconfigured provider fails gracefully without blocking successful models."""
    with patch("engine.universal_evaluator.get_provider_status") as mock_status:
        # Mock status: NVIDIA not configured, Mock available
        mock_status.return_value = {
            "nvidia": {"configured": False, "provider_type": "Cloud API"},
            "mock": {"configured": True, "provider_type": "Demo"}
        }
        comp = await run_universal_comparison(
            question="What is gravity?",
            selected_models=["mock-gpt", "nvidia/llama-3.1-nemotron-70b-instruct"]
        )
        assert comp.status == "partial_failure"
        gpt_res = next(r for r in comp.results if r.model_id == "mock-gpt")
        real_res = next(r for r in comp.results if r.model_id == "nvidia/llama-3.1-nemotron-70b-instruct")

        # mock-gpt should succeed
        assert gpt_res.status == "success"
        # real model must NOT silently substitute a mock! It must record unconfigured status
        assert real_res.status == "unconfigured"
        assert "unconfigured" in (real_res.error_message or "").lower()


@pytest.mark.asyncio
async def test_fault_isolation_timeout_and_network_error():
    """Ensure network timeouts or adapter exceptions are isolated to the failing model."""
    async def mock_generate_err(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch("engine.universal_evaluator.get_adapter") as mock_adapter_getter:
        mock_adapter = AsyncMock()
        mock_adapter.generate.side_effect = mock_generate_err
        mock_adapter_getter.return_value = mock_adapter

        res = await execute_single_model_question(
            model_id="mock-gpt",
            question="Tell me a joke"
        )
        assert res.status == "timeout"
        assert "timed out" in (res.error_message or "").lower()


# --- 8. Database Persistence & History Retrieval Tests ---

def test_repository_universal_comparison_crud(tmp_path):
    """Test saving, retrieving, listing, and deleting Universal Comparisons in SQLite."""
    db_file = tmp_path / "test_univ.db"
    repo = BenchmarkRepository(db_path=str(db_file))

    r1 = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        status="success",
        response_text="42",
        latency_ms=120.5,
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        estimated_cost_usd=0.0001,
        accuracy_score=1.0,
        is_correct=True
    )
    comp = UniversalComparison(
        comparison_id="cmp_test123",
        timestamp="2026-10-13T10:00:00Z",
        question="What is 6 * 7?",
        reference_answer="42",
        evaluation_mode="reference",
        selected_models=["mock-gpt"],
        generation_config={"temperature": 0.0},
        results=[r1],
        status="completed",
        agreement_analysis={"consensus_score": 1.0}
    )

    # 1. Save
    repo.save_comparison(comp)

    # 2. Retrieve
    loaded = repo.get_comparison("cmp_test123")
    assert loaded is not None
    assert loaded.comparison_id == "cmp_test123"
    assert loaded.question == "What is 6 * 7?"
    assert len(loaded.results) == 1
    assert loaded.results[0].model_id == "mock-gpt"
    assert loaded.results[0].accuracy_score == 1.0

    # 3. List
    comp_list = repo.list_comparisons()
    assert len(comp_list) == 1
    assert comp_list[0]["comparison_id"] == "cmp_test123"

    # 4. Delete
    repo.delete_comparison("cmp_test123")
    assert repo.get_comparison("cmp_test123") is None
    assert len(repo.list_comparisons()) == 0


# --- 9. CSV and JSON Export Tests ---

def test_export_comparison_to_csv_and_json():
    """Test exporting comparison results ensuring no secrets and honest N/A values."""
    r1 = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        status="success",
        response_text="Gravity pulls objects with mass together.",
        latency_ms=150.0,
        prompt_tokens=8,
        completion_tokens=12,
        total_tokens=20,
        estimated_cost_usd=0.00005,
        evaluation_mode="unreferenced",
        accuracy_score=None,  # Mode C unverified
        is_correct=None
    )
    comp = UniversalComparison(
        comparison_id="cmp_export_test",
        timestamp="2026-10-13T10:00:00Z",
        question="What is gravity?",
        evaluation_mode="unreferenced",
        selected_models=["mock-gpt"],
        results=[r1],
        status="completed"
    )

    # CSV Export
    csv_str = export_comparison_to_csv(comp)
    assert "Comparison ID" in csv_str
    assert "cmp_export_test" in csv_str
    assert "N/A" in csv_str  # Accuracy score must be explicitly N/A!
    assert "sk-" not in csv_str

    # JSON Export
    json_str = export_comparison_to_json(comp)
    parsed = json.loads(json_str)
    assert parsed["comparison_id"] == "cmp_export_test"
    assert parsed["results"][0]["accuracy_score"] is None
    assert "sk-" not in json_str
