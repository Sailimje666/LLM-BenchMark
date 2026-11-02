"""Tests for provider registry resolution and connectivity verification probes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import config
from engine.model_registry import list_registered_models, get_adapter, REGISTRY
from engine.provider_adapter import (
    NVIDIANemotronAdapter,
    DeepSeekAdapter,
    GroqAdapter,
    OllamaAdapter,
    MockProviderAdapter
)
from engine.connection_check import verify_provider_connectivity


def test_registry_contains_nvidia_and_deepseek_models():
    """Verify that NVIDIA Nemotron and DeepSeek models are actively registered."""
    models = list_registered_models()
    model_map = {m["model_id"]: m for m in models}

    # NVIDIA Nemotron models
    assert "nvidia/llama-3.1-nemotron-70b-instruct" in model_map
    assert model_map["nvidia/llama-3.1-nemotron-70b-instruct"]["provider"] == "nvidia"
    assert model_map["nvidia/llama-3.1-nemotron-70b-instruct"]["context_window"] == 131072

    assert "nvidia/nemotron-4-340b-instruct" in model_map
    assert model_map["nvidia/nemotron-4-340b-instruct"]["provider"] == "nvidia"

    # DeepSeek models
    assert "deepseek-chat" in model_map
    assert model_map["deepseek-chat"]["provider"] == "deepseek"
    assert model_map["deepseek-chat"]["context_window"] == 128000

    assert "deepseek-reasoner" in model_map
    assert model_map["deepseek-reasoner"]["provider"] == "deepseek"

    # Groq, Ollama, and Mock models preserved
    assert "openai/gpt-oss-120b" in model_map
    assert model_map["openai/gpt-oss-120b"]["provider"] == "groq"
    assert "llama3:8b" in model_map
    assert model_map["llama3:8b"]["provider"] == "ollama"
    assert "mock-gpt" in model_map
    assert model_map["mock-gpt"]["provider"] == "mock"


def test_registry_excludes_openai_gemini_and_grok():
    """Verify that OpenAI, Google Gemini, and xAI Grok are completely excluded from the active registry."""
    models = list_registered_models()
    active_providers = {m["provider"] for m in models}

    assert "openai" not in active_providers
    assert "google" not in active_providers
    assert "gemini" not in active_providers
    assert "xai" not in active_providers
    assert "grok" not in active_providers

    model_ids = {m["model_id"] for m in models}
    assert "gpt-4o" not in model_ids
    assert "gpt-4o-mini" not in model_ids
    assert "gemini-3.8-flash" not in model_ids
    assert "gemini-flash-latest" not in model_ids
    assert "grok-2-1212" not in model_ids
    assert "grok-beta" not in model_ids


def test_get_adapter_resolves_active_and_rejects_retired():
    """Verify adapter factory handles active providers and cleanly rejects retired providers."""
    # Active providers
    nv_adapter = get_adapter("nvidia", "nvidia/llama-3.1-nemotron-70b-instruct", api_key="nvapi-test")
    assert isinstance(nv_adapter, NVIDIANemotronAdapter)

    ds_adapter = get_adapter("deepseek", "deepseek-chat", api_key="sk-test")
    assert isinstance(ds_adapter, DeepSeekAdapter)

    groq_adapter = get_adapter("groq", "openai/gpt-oss-120b", api_key="gsk-test")
    assert isinstance(groq_adapter, GroqAdapter)

    ollama_adapter = get_adapter("ollama", "llama3:8b")
    assert isinstance(ollama_adapter, OllamaAdapter)

    mock_adapter = get_adapter("mock", "mock-gpt")
    assert isinstance(mock_adapter, MockProviderAdapter)

    # Retired providers must raise ValueError
    for retired in ["openai", "google", "gemini", "xai", "grok"]:
        with pytest.raises(ValueError, match="removed from active service"):
            get_adapter(retired, "some-model")


@pytest.mark.asyncio
async def test_connection_check_unconfigured_active_providers():
    """Verify connection probe returns 'unconfigured' when keys are unset for active cloud providers."""
    with patch("config.get_api_key", return_value=""):
        res_nv = await verify_provider_connectivity("nvidia")
        assert res_nv["status"] == "unconfigured"
        assert "not configured" in res_nv["message"].lower()

        res_ds = await verify_provider_connectivity("deepseek")
        assert res_ds["status"] == "unconfigured"
        assert "not configured" in res_ds["message"].lower()


@pytest.mark.asyncio
async def test_connection_check_retired_providers_rejected():
    """Verify connection probe cleanly rejects retired providers."""
    for retired in ["openai", "google", "gemini", "xai", "grok"]:
        res = await verify_provider_connectivity(retired)
        assert res["status"] == "failed"
        assert "removed from active service" in res["message"]


@pytest.mark.asyncio
async def test_connection_check_mock_success():
    """Verify connection probe succeeds on mock provider."""
    res = await verify_provider_connectivity("mock")
    assert res["status"] == "success"
    assert res["latency_ms"] > 0
    assert "Connected successfully" in res["message"]


@pytest.mark.asyncio
async def test_connection_check_failure_sanitizes_key():
    """Verify that connection probe catches errors and sanitizes any sensitive keys from error messages."""
    fake_secret = "nvapi-secret-key-abcdef12345678"
    with patch("config.get_api_key", return_value=fake_secret):
        with patch.object(config, "NVIDIA_API_KEY", fake_secret):
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.side_effect = RuntimeError(f"Connection failed for key {fake_secret}")

                res = await verify_provider_connectivity("nvidia")
                assert res["status"] == "failed"
                # Fake key must be masked
                assert fake_secret not in res["message"]
                assert config.mask_secret(fake_secret) in res["message"]
