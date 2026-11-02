"""Tests verifying the complete retirement and exclusion of Google Gemini provider."""

import pytest
from unittest.mock import patch

import config
from engine.model_registry import list_registered_models, get_adapter, resolve_model_display_name
from engine.provider_adapter import GoogleGeminiAdapter
from engine.connection_check import verify_provider_connectivity
from schema import TestCase, ModelConfig
from storage.repository import BenchmarkRepository
from engine.runner import BenchmarkRunner


def test_gemini_excluded_from_active_registry():
    """Verify that Google Gemini models are completely excluded from the active model registry."""
    active_models = list_registered_models()
    active_providers = {m["provider"] for m in active_models}
    assert "google" not in active_providers
    assert "gemini" not in active_providers

    model_ids = {m["model_id"] for m in active_models}
    assert "gemini-3.8-flash" not in model_ids
    assert "gemini-flash-latest" not in model_ids
    assert "gemini-2.5-flash" not in model_ids
    assert "gemini-2.5-pro" not in model_ids
    assert "gemini-2.0-flash" not in model_ids


def test_gemini_get_adapter_raises_value_error():
    """Verify get_adapter rejects Google and Gemini with clear retirement message."""
    with pytest.raises(ValueError, match="Provider 'google' has been removed from active service"):
        get_adapter("google", "gemini-3.8-flash")

    with pytest.raises(ValueError, match="Provider 'gemini' has been removed from active service"):
        get_adapter("gemini", "gemini-flash-latest")


def test_gemini_adapter_direct_instantiation_raises_runtime_error():
    """Verify direct instantiation of GoogleGeminiAdapter raises clear RuntimeError."""
    with pytest.raises(RuntimeError, match="Provider 'google' has been removed from active service"):
        GoogleGeminiAdapter(model_name="gemini-3.8-flash", api_key="AIzaSyTest")


@pytest.mark.asyncio
async def test_gemini_connection_check_returns_failed_retired():
    """Verify verify_provider_connectivity rejects Google and Gemini probes."""
    res_google = await verify_provider_connectivity("google")
    assert res_google["status"] == "failed"
    assert "removed from active service" in res_google["message"]

    res_gemini = await verify_provider_connectivity("gemini")
    assert res_gemini["status"] == "failed"
    assert "removed from active service" in res_gemini["message"]


def test_gemini_api_key_not_in_active_config():
    """Verify config does not require or return GEMINI_API_KEY for active configuration."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "AIzaSyTest", "GOOGLE_API_KEY": "AIzaSyTest"}):
        assert config.get_api_key("google") == ""
        assert config.get_api_key("gemini") == ""

    status = config.get_provider_status()
    assert "google" not in status
    assert "gemini" not in status


def test_historical_gemini_models_display_name_and_pricing_preserved():
    """Verify historical Gemini records can still resolve display names and pricing for historical runs."""
    assert resolve_model_display_name("gemini-3.8-flash") == "Google Gemini 3.8 Flash (Retired)"
    assert resolve_model_display_name("gemini-flash-latest") == "Google Gemini Flash Latest (Retired)"

    pricing_flash = config.get_token_pricing("gemini-3.8-flash")
    assert pricing_flash["input"] == 0.10
    assert pricing_flash["output"] == 0.40


@pytest.mark.asyncio
async def test_runner_rejects_gemini_in_new_benchmark(tmp_path):
    """Verify BenchmarkRunner rejects attempts to run new benchmarks with removed Gemini provider."""
    db_file = tmp_path / "test_gemini_runner.db"
    repo = BenchmarkRepository(db_path=str(db_file))
    runner = BenchmarkRunner(repository=repo, max_concurrent=2, max_retries=1)

    tc = TestCase(
        id="math-tc-01",
        domain="Math",
        category="Arithmetic",
        difficulty="Easy",
        problem_statement="What is 15% of 240?",
        ground_truth="36",
        evaluation_method="exact_match"
    )

    configs = [
        ModelConfig(provider="mock", model_name="mock-gpt", enabled=True),
        ModelConfig(provider="google", model_name="gemini-3.8-flash", enabled=True)
    ]

    benchmark_run = await runner.run_benchmark(
        test_cases=[tc],
        model_configs=configs,
        run_config={"enable_semantic_judge": False}
    )

    results = repo.get_results_for_run(benchmark_run.run_id)
    gemini_res = next(r for r in results if r.provider == "google")
    mock_res = next(r for r in results if r.provider == "mock")

    assert mock_res.status == "success"
    assert gemini_res.status == "provider_error"
    assert "removed from active service" in (gemini_res.provider_error or "")
