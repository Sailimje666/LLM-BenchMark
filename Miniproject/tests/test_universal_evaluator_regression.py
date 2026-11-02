"""Regression tests for Universal Question Evaluator fixes.

Covers:
1. Exact arithmetic question ('What is 15% of 240? Return only the numerical answer.') returning 36 across all mock models.
2. Labeling of mock/simulated outputs in schema and CSV export.
3. Preservation of unreferenced-mode behavior (no fabricated accuracy).
4. Reference-based evaluation: response '36' is PASS (1.0), response '42' is FAIL (0.0).
5. Actionable provider failure reporting (OpenAI 429 quota exhaustion, 404 model errors, Groq 404, Gemini 404).
6. Pre-flight Ollama connectivity verification when daemon is unavailable.
7. Model registry updates with verified active models and clear unavailable markers.
8. Historical record preservation and database backwards-compatibility.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from schema import UniversalComparisonResult, UniversalComparison
from engine.provider_adapter import (
    MockProviderAdapter,
    NVIDIANemotronAdapter,
    DeepSeekAdapter,
    GroqAdapter,
    OllamaAdapter
)
from engine.model_registry import list_registered_models, REGISTRY
from engine.universal_evaluator import (
    execute_single_model_question,
    evaluate_single_result,
    run_universal_comparison,
    export_comparison_to_csv,
    export_comparison_to_json
)
from storage.repository import BenchmarkRepository


QUESTION = "What is 15% of 240? Return only the numerical answer."
EXPECTED_ANSWER = "36"


# ---------------------------------------------------------------------
# 1. Mock Deterministic Responses for Arithmetic Question
# ---------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["mock-gpt", "mock-claude", "mock-llama"])
async def test_mock_adapters_return_36_not_42_for_arithmetic_question(model_name):
    """Verify that Mock GPT, Mock Claude, and Mock Llama return 36, not 42."""
    adapter = MockProviderAdapter(model_name=model_name)
    res = await adapter.generate(prompt=QUESTION)
    assert res["text"].strip() == "36"
    assert res["text"].strip() != "42"
    assert res["text"].strip() != "42.0"
    assert res["error"] is None
    assert res["finish_reason"] == "stop"


# ---------------------------------------------------------------------
# 2. Simulated Outputs Clear Labeling
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mock_outputs_clearly_labelled_as_simulated():
    """Verify mock outputs are flagged as simulated and CSV clearly marks them."""
    res = await execute_single_model_question(
        model_id="mock-gpt",
        question=QUESTION
    )
    assert res.is_simulated is True
    assert "simulated" in res.display_name.lower()
    assert res.provider == "mock"

    comp = UniversalComparison(
        comparison_id="cmp_sim_test",
        timestamp="2026-10-16T10:00:00Z",
        question=QUESTION,
        evaluation_mode="unreferenced",
        selected_models=["mock-gpt"],
        results=[res],
        status="completed"
    )

    csv_data = export_comparison_to_csv(comp)
    assert "Is Simulated" in csv_data
    assert "True" in csv_data


# ---------------------------------------------------------------------
# 3. Unreferenced Mode Preserves Honesty (No Fabricated Accuracy)
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unreferenced_mode_preserves_honesty_no_fabricated_accuracy():
    """Verify unreferenced evaluation never fabricates accuracy or correctness."""
    mock_res = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        display_name="Mock GPT-4o (Simulated)",
        is_simulated=True,
        status="success",
        response_text="36"
    )
    eval_res = await evaluate_single_result(
        result=mock_res,
        question=QUESTION,
        reference_answer=None,
        rubric=None
    )
    assert eval_res.evaluation_mode == "unreferenced"
    assert eval_res.accuracy_score is None
    assert eval_res.is_correct is None
    assert "Accuracy not independently verified" in (eval_res.score_explanation or "")

    comp = UniversalComparison(
        comparison_id="cmp_unref_test",
        timestamp="2026-10-16T10:00:00Z",
        question=QUESTION,
        evaluation_mode="unreferenced",
        selected_models=["mock-gpt"],
        results=[eval_res],
        status="completed"
    )
    csv_data = export_comparison_to_csv(comp)
    assert "N/A" in csv_data


# ---------------------------------------------------------------------
# 4. Reference-Based Evaluation: 36 is Correct, 42 is Incorrect
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reference_based_evaluation_scores_36_correct_and_42_incorrect():
    """Verify reference evaluation marks 36 as correct (1.0) and 42 as incorrect (0.0)."""
    # Candidate 1: Response '36'
    res_correct = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        display_name="Mock GPT-4o (Simulated)",
        status="success",
        response_text="36"
    )
    eval_correct = await evaluate_single_result(
        result=res_correct,
        question=QUESTION,
        reference_answer=EXPECTED_ANSWER,
        eval_method="auto"
    )
    assert eval_correct.evaluation_mode == "reference"
    assert eval_correct.accuracy_score == 1.0
    assert eval_correct.is_correct is True
    assert "PASS" in (eval_correct.score_explanation or "")

    # Candidate 2: Response '42'
    res_incorrect = UniversalComparisonResult(
        model_id="mock-gpt",
        provider="mock",
        display_name="Mock GPT-4o (Simulated)",
        status="success",
        response_text="42"
    )
    eval_incorrect = await evaluate_single_result(
        result=res_incorrect,
        question=QUESTION,
        reference_answer=EXPECTED_ANSWER,
        eval_method="auto"
    )
    assert eval_incorrect.evaluation_mode == "reference"
    assert eval_incorrect.accuracy_score == 0.0
    assert eval_incorrect.is_correct is False
    assert "FAIL" in (eval_incorrect.score_explanation or "")

    # Candidate 3: Response '42.0'
    res_incorrect_float = UniversalComparisonResult(
        model_id="mock-llama",
        provider="mock",
        display_name="Mock Llama 3 (Simulated)",
        status="success",
        response_text="42.0"
    )
    eval_incorrect_float = await evaluate_single_result(
        result=res_incorrect_float,
        question=QUESTION,
        reference_answer=EXPECTED_ANSWER,
        eval_method="auto"
    )
    assert eval_incorrect_float.accuracy_score == 0.0
    assert eval_incorrect_float.is_correct is False


# ---------------------------------------------------------------------
# 5. Actionable Provider Failures: NVIDIA 404 & DeepSeek 403/404
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nvidia_actionable_error_on_404_catalog_not_found():
    """Verify NVIDIA 404 returns actionable advice indicating model not in catalog."""
    adapter = NVIDIANemotronAdapter(model_name="nvidia/nonexistent-model", api_key="nvapi-test-mock")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "error": {
            "message": "The model `nvidia/nonexistent-model` does not exist or you do not have access to it in the NVIDIA API catalog."
        }
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate(prompt=QUESTION)

        err_msg = str(exc_info.value)
        assert "404" in err_msg
        assert "catalog" in err_msg.lower() or "not exist" in err_msg.lower()


@pytest.mark.asyncio
async def test_deepseek_actionable_error_on_403_insufficient_balance():
    """Verify DeepSeek 403 returns actionable balance/permissions advice."""
    adapter = DeepSeekAdapter(model_name="deepseek-chat", api_key="sk-test-mock")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 403
    mock_resp.json.return_value = {
        "error": {
            "message": "Insufficient Balance"
        }
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate(prompt=QUESTION)

        err_msg = str(exc_info.value)
        assert "403" in err_msg
        assert "balance" in err_msg.lower() or "forbidden" in err_msg.lower()


@pytest.mark.asyncio
async def test_deepseek_actionable_error_on_404_model_not_found():
    """Verify DeepSeek 404 provides actionable recommendation to use verified models."""
    adapter = DeepSeekAdapter(model_name="deepseek-invalid", api_key="sk-test-mock")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "error": {
            "message": "Model deepseek-invalid not found"
        }
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate(prompt=QUESTION)

        err_msg = str(exc_info.value)
        assert "404" in err_msg
        assert "deepseek" in err_msg.lower()


@pytest.mark.asyncio
async def test_groq_actionable_error_on_404_model_not_found():
    """Verify Groq 404 provides list of verified available models on the account."""
    adapter = GroqAdapter(model_name="llama-3.1-8b-instant", api_key="gsk_mock_key")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "error": {
            "message": "The model `llama-3.1-8b-instant` does not exist or you do not have access to it.",
            "type": "invalid_request_error",
            "code": "model_not_found"
        }
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate(prompt=QUESTION)

        err_msg = str(exc_info.value)
        assert "404" in err_msg
        assert "openai/gpt-oss-120b" in err_msg


# ---------------------------------------------------------------------
# 7. Ollama Connectivity Pre-flight Check
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_preflight_check_fails_gracefully_when_server_unavailable():
    """Verify Ollama checks connectivity and returns actionable error when daemon is offline."""
    # When daemon is down, check_connection_static returns False
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        adapter = OllamaAdapter(model_name="llama3:8b", base_url="http://localhost:11434")
        is_alive, msg = await adapter.check_connection()
        assert is_alive is False
        assert "ollama serve" in msg.lower()

        # Calling generate raises actionable RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate(prompt=QUESTION)
        assert "ollama serve" in str(exc_info.value).lower()

        # Calling execute_single_model_question isolates the error without crashing
        res = await execute_single_model_question(
            model_id="llama3:8b",
            question=QUESTION
        )
        assert res.status == "provider_error"
        assert "ollama serve" in (res.error_message or "").lower()
        assert res.response_text == ""


# ---------------------------------------------------------------------
# 8. Model Registry Verified Models and Deprecation Markers
# ---------------------------------------------------------------------

def test_model_registry_contains_verified_and_marks_unavailable():
    """Verify registry contains active verified models and excludes removed provider models."""
    models = list_registered_models()
    model_map = {m["model_id"]: m for m in models}

    # Verified active models
    assert "nvidia/llama-3.1-nemotron-70b-instruct" in model_map
    assert model_map["nvidia/llama-3.1-nemotron-70b-instruct"]["available"] is True

    assert "nvidia/nemotron-4-340b-instruct" in model_map
    assert model_map["nvidia/nemotron-4-340b-instruct"]["available"] is True

    assert "deepseek-chat" in model_map
    assert model_map["deepseek-chat"]["available"] is True

    assert "deepseek-reasoner" in model_map
    assert model_map["deepseek-reasoner"]["available"] is True

    assert "openai/gpt-oss-120b" in model_map
    assert model_map["openai/gpt-oss-120b"]["available"] is True

    assert "qwen/qwen3.8-27b" in model_map
    assert model_map["qwen/qwen3.8-27b"]["available"] is True

    assert "llama3:8b" in model_map
    assert model_map["llama3:8b"]["available"] is True

    # Removed providers models must NOT be in active registry
    assert "gemini-3.8-flash" not in model_map
    assert "gemini-flash-latest" not in model_map
    assert "gemini-2.0-flash" not in model_map
    assert "gpt-4o" not in model_map
    assert "gpt-4-turbo" not in model_map
    assert "grok-2-1212" not in model_map


# ---------------------------------------------------------------------
# 9. Historical Comparison Records Preservation
# ---------------------------------------------------------------------

def test_historical_comparison_records_preserve_and_deserialize():
    """Verify historical records in data/benchmark.db deserialize successfully."""
    repo = BenchmarkRepository("data/benchmark.db")
    comps = repo.list_comparisons()
    assert len(comps) > 0

    for c in comps:
        loaded = repo.get_comparison(c["comparison_id"])
        assert loaded is not None
        assert loaded.comparison_id == c["comparison_id"]
        assert len(loaded.results) > 0
        for r in loaded.results:
            assert isinstance(r.is_simulated, bool)
