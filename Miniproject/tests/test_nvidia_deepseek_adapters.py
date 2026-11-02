"""Offline unit and integration tests for NVIDIA Nemotron and DeepSeek adapters."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from engine.provider_adapter import NVIDIANemotronAdapter, DeepSeekAdapter
from engine.model_registry import get_adapter, list_registered_models, REGISTRY
from engine.universal_evaluator import (
    execute_single_model_question,
    run_universal_comparison,
    sanitize_text,
    export_comparison_to_csv,
    export_comparison_to_json
)
from schema import TestCase, ModelConfig, UniversalComparison, UniversalComparisonResult
from storage.repository import BenchmarkRepository
from engine.runner import BenchmarkRunner
import config


# ==============================================================================
# 1. NVIDIA Nemotron Adapter Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_nvidia_adapter_success():
    """Verify NVIDIA adapter request construction and response parsing."""
    adapter = NVIDIANemotronAdapter(
        model_name="nvidia/nemotron-3-super-120b-a12b",
        api_key="nvapi-test-key-12345"
    )

    mock_resp_data = {
        "id": "chatcmpl-nv-12345",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "36"},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 2,
            "total_tokens": 27
        },
        "model": "nvidia/nemotron-3-super-120b-a12b"
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        res = await adapter.generate(
            prompt="What is 15% of 240?",
            system_prompt="Return only the number.",
            temperature=0.0,
            max_tokens=64
        )

        assert res["text"] == "36"
        assert res["prompt_tokens"] == 25
        assert res["completion_tokens"] == 2
        assert res["total_tokens"] == 27
        assert res["finish_reason"] == "stop"
        assert res["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert res["error"] is None
        assert res["latency_ms"] >= 0

        # Validate headers and payload
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert call_kwargs["headers"]["Authorization"] == "Bearer nvapi-test-key-12345"
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

        payload = call_kwargs["json"]
        assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 64
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"


def test_nvidia_custom_base_url():
    """Verify custom NVIDIA_BASE_URL formatting without trailing slash or duplication."""
    with patch.dict("os.environ", {"NVIDIA_BASE_URL": "https://custom.api.nvidia.com/v1"}):
        adapter = NVIDIANemotronAdapter(model_name="nvidia/nemotron-3-super-120b-a12b", api_key="test-key")
        assert adapter.base_url == "https://custom.api.nvidia.com/v1/chat/completions"

    with patch.dict("os.environ", {"NVIDIA_BASE_URL": "https://custom.api.nvidia.com/v1/chat/completions"}):
        adapter2 = NVIDIANemotronAdapter(model_name="nvidia/nemotron-3-super-120b-a12b", api_key="test-key")
        assert adapter2.base_url == "https://custom.api.nvidia.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_nvidia_missing_key_raises_value_error():
    """Verify that missing NVIDIA API key raises a clear ValueError."""
    with patch("config.get_api_key", return_value=""):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": ""}, clear=True):
            adapter = NVIDIANemotronAdapter(model_name="nvidia/nemotron-3-super-120b-a12b", api_key="")
            with pytest.raises(ValueError, match="NVIDIA API key not configured"):
                await adapter.generate(prompt="Hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code,err_message,expected_match", [
    (401, "Invalid Bearer Token", "401"),
    (403, "Permission Denied", "403"),
    (404, "Model not in catalog", "404"),
    (429, "Rate limit exceeded or quota exhausted", "429"),
])
async def test_nvidia_api_error_handling(status_code, err_message, expected_match):
    """Verify NVIDIA error codes 401, 403, 404, 429 raise descriptive RuntimeErrors."""
    adapter = NVIDIANemotronAdapter(model_name="nvidia/nemotron-3-super-120b-a12b", api_key="nvapi-key")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = {"error": {"message": err_message}}
    mock_response.text = err_message

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(RuntimeError, match=expected_match):
            await adapter.generate(prompt="Ping")


@pytest.mark.asyncio
async def test_nvidia_timeout_and_malformed_response():
    """Verify timeout and empty choices handling in NVIDIA adapter."""
    adapter = NVIDIANemotronAdapter(model_name="nvidia/nemotron-3-super-120b-a12b", api_key="nvapi-key")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("NVIDIA API timed out")
        with pytest.raises(httpx.TimeoutException):
            await adapter.generate(prompt="Ping", timeout=5)

    # Malformed empty choices
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(RuntimeError, match="No candidate choices"):
            await adapter.generate(prompt="Ping")


# ==============================================================================
# 2. DeepSeek Adapter Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_deepseek_adapter_success():
    """Verify DeepSeek adapter request construction and response parsing."""
    adapter = DeepSeekAdapter(
        model_name="deepseek-flash",
        api_key="sk-deepseek-test-12345"
    )

    mock_resp_data = {
        "id": "chatcmpl-ds-12345",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The computed answer is 36."},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23
        },
        "model": "deepseek-flash"
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        res = await adapter.generate(
            prompt="What is 15% of 240?",
            system_prompt="Answer concisely.",
            temperature=0.0,
            max_tokens=100
        )

        assert res["text"] == "The computed answer is 36."
        assert res["prompt_tokens"] == 15
        assert res["completion_tokens"] == 8
        assert res["total_tokens"] == 23
        assert res["finish_reason"] == "stop"
        assert res["model"] == "deepseek-flash"

        # Validate headers and payload
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "https://api.deepseek.com/chat/completions"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-deepseek-test-12345"

        payload = call_kwargs["json"]
        assert payload["model"] == "deepseek-flash"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 100


@pytest.mark.asyncio
async def test_deepseek_reasoning_model_omits_temperature():
    """Verify that reasoning models (deepseek-reasoner) do NOT receive temperature."""
    adapter = DeepSeekAdapter(
        model_name="deepseek-reasoner",
        api_key="sk-deepseek-test"
    )

    mock_resp_data = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "15% of 240 is 0.15 * 240 = 36."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        res = await adapter.generate(
            prompt="Compute 15% of 240",
            temperature=0.7,
            max_tokens=256
        )

        # Verify temperature was omitted from payload
        payload = mock_post.call_args[1]["json"]
        assert "temperature" not in payload
        # Verify reasoning_content is utilized when content is empty
        assert res["text"] == "15% of 240 is 0.15 * 240 = 36."


@pytest.mark.asyncio
async def test_deepseek_missing_key_raises_value_error():
    """Verify that missing DeepSeek API key raises a clear ValueError."""
    with patch("config.get_api_key", return_value=""):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=True):
            adapter = DeepSeekAdapter(model_name="deepseek-flash", api_key="")
            with pytest.raises(ValueError, match="DeepSeek API key not configured"):
                await adapter.generate(prompt="Hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code,err_message,expected_match", [
    (401, "Invalid API key", "401"),
    (403, "Insufficient account balance", "403"),
    (404, "Model deepseek-foo does not exist", "404"),
    (429, "Rate limit reached", "429"),
])
async def test_deepseek_api_error_handling(status_code, err_message, expected_match):
    """Verify DeepSeek error codes 401, 403, 404, 429 raise descriptive RuntimeErrors."""
    adapter = DeepSeekAdapter(model_name="deepseek-flash", api_key="sk-test")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = {"error": {"message": err_message}}
    mock_response.text = err_message

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(RuntimeError, match=expected_match):
            await adapter.generate(prompt="Ping")


# ==============================================================================
# 3. Model Registry & Lineup Verification Tests
# ==============================================================================

def test_model_registry_adapter_instantiation():
    """Verify get_adapter properly instantiates NVIDIANemotronAdapter and DeepSeekAdapter."""
    nv_adapter = get_adapter("nvidia", "nvidia/nemotron-3-super-120b-a12b", api_key="test-key")
    assert isinstance(nv_adapter, NVIDIANemotronAdapter)
    assert nv_adapter.model_name == "nvidia/nemotron-3-super-120b-a12b"

    ds_adapter = get_adapter("deepseek", "deepseek-flash", api_key="test-key")
    assert isinstance(ds_adapter, DeepSeekAdapter)
    assert ds_adapter.model_name == "deepseek-flash"

    # Retired providers raise ValueError when requested
    with pytest.raises(ValueError, match="removed from active service"):
        get_adapter("openai", "gpt-4o-mini", api_key="test-key")
    with pytest.raises(ValueError, match="removed from active service"):
        get_adapter("google", "gemini-3.8-flash", api_key="test-key")
    with pytest.raises(ValueError, match="removed from active service"):
        get_adapter("xai", "grok-2-1212", api_key="test-key")


def test_model_registry_active_lineup_filtering():
    """Verify that active models lineup includes only active providers and excludes removed ones."""
    active_models = list_registered_models(active_only=True)
    active_providers = {m["provider"] for m in active_models}

    # Primary and supported providers MUST be active
    assert "nvidia" in active_providers
    assert "deepseek" in active_providers
    assert "groq" in active_providers
    assert "ollama" in active_providers
    assert "mock" in active_providers

    # Removed providers MUST NOT be in active lineup
    assert "openai" not in active_providers
    assert "google" not in active_providers
    assert "xai" not in active_providers
    assert "gemini" not in active_providers
    assert "grok" not in active_providers


# ==============================================================================
# 4. Universal Question Evaluator Integration & Scoring Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_universal_evaluator_with_mocked_nvidia_and_deepseek():
    """Verify Universal Question Evaluator executes with NVIDIA and DeepSeek and scores accurately."""
    nv_response = MagicMock(spec=httpx.Response)
    nv_response.status_code = 200
    nv_response.json.return_value = {
        "choices": [{"message": {"content": "36"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        "model": "nvidia/nemotron-3-super-120b-a12b"
    }

    ds_response = MagicMock(spec=httpx.Response)
    ds_response.status_code = 200
    ds_response.json.return_value = {
        "choices": [{"message": {"content": "36.0"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        "model": "deepseek-flash"
    }

    async def mock_post(url, *args, **kwargs):
        if "nvidia" in url:
            return nv_response
        return ds_response

    with patch("config.get_api_key", return_value="dummy-key"):
        with patch("config.get_provider_status", return_value={
            "nvidia": {"configured": True},
            "deepseek": {"configured": True},
            "mock": {"configured": True}
        }):
            with patch("httpx.AsyncClient.post", side_effect=mock_post):
                res = await run_universal_comparison(
                    question="What is 15% of 240? Return only the numerical answer.",
                    selected_models=["nvidia/nemotron-3-super-120b-a12b", "deepseek-flash"],
                    reference_answer="36",
                    eval_method="numeric_tolerance"
                )

                assert res.status == "completed"
                assert len(res.results) == 2
                for r in res.results:
                    assert r.status == "success"
                    assert r.accuracy_score == 1.0
                    assert r.is_correct is True
                    assert r.total_tokens > 0
                    assert r.latency_ms >= 0


# ==============================================================================
# 5. Fault Isolation & Secret Masking Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_provider_fault_isolation():
    """Verify that an error in NVIDIA does NOT crash DeepSeek or Mock models."""
    nv_error = MagicMock(spec=httpx.Response)
    nv_error.status_code = 401
    nv_error.json.return_value = {"error": {"message": "Invalid API key"}}
    nv_error.text = "Invalid API key"

    ds_success = MagicMock(spec=httpx.Response)
    ds_success.status_code = 200
    ds_success.json.return_value = {
        "choices": [{"message": {"content": "36"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        "model": "deepseek-flash"
    }

    async def mock_post(url, *args, **kwargs):
        if "nvidia" in url:
            return nv_error
        return ds_success

    with patch("config.get_api_key", return_value="dummy-key"):
        with patch("config.get_provider_status", return_value={
            "nvidia": {"configured": True},
            "deepseek": {"configured": True},
            "mock": {"configured": True}
        }):
            with patch("httpx.AsyncClient.post", side_effect=mock_post):
                res = await run_universal_comparison(
                    question="What is 15% of 240?",
                    selected_models=["nvidia/nemotron-3-super-120b-a12b", "deepseek-flash", "mock-gpt"],
                    reference_answer="36",
                    eval_method="numeric_tolerance"
                )

                # Check individual model results
                by_model = {r.model_id: r for r in res.results}
                assert by_model["nvidia/nemotron-3-super-120b-a12b"].status == "provider_error"
                assert "401" in by_model["nvidia/nemotron-3-super-120b-a12b"].sanitized_error

                # DeepSeek and Mock must still succeed
                assert by_model["deepseek-flash"].status == "success"
                assert by_model["deepseek-flash"].accuracy_score == 1.0

                assert by_model["mock-gpt"].status == "success"
                assert by_model["mock-gpt"].accuracy_score == 1.0


def test_secret_masking_nvapi_and_deepseek():
    """Verify that sanitize_text and mask_secret redact nvapi and deepseek keys."""
    nv_key = "nvapi-abcdef1234567890abcdef"
    ds_key = "sk-deepseek-1234567890abcdef"

    # Test config.mask_secret
    assert config.mask_secret(nv_key) == "nva...cdef"
    assert config.mask_secret(ds_key) == "sk-...cdef"

    # Test sanitize_text in logs / outputs
    leak_sample = f"Error calling endpoint with {nv_key} and {ds_key}!"
    sanitized = sanitize_text(leak_sample)
    assert nv_key not in sanitized
    assert ds_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
