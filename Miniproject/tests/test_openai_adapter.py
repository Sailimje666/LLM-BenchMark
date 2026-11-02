"""Tests verifying the complete retirement and exclusion of the OpenAI provider."""

import pytest
from unittest.mock import patch

import config
from engine.model_registry import list_registered_models, get_adapter, resolve_model_display_name
from engine.provider_adapter import OpenAIAdapter
from engine.connection_check import verify_provider_connectivity
from schema import TestCase, ModelConfig
from storage.repository import BenchmarkRepository
from engine.runner import BenchmarkRunner


def test_openai_excluded_from_active_registry():
    """Verify that OpenAI models are completely excluded from active registry."""
    active_models = list_registered_models()
    active_providers = {m["provider"] for m in active_models}
    assert "openai" not in active_providers

    model_ids = {m["model_id"] for m in active_models}
    assert "gpt-4o" not in model_ids
    assert "gpt-4o-mini" not in model_ids
    assert "gpt-4-turbo" not in model_ids
    assert "gpt-3.5-turbo" not in model_ids
    assert "o1" not in model_ids
    assert "o1-mini" not in model_ids
    assert "o3-mini" not in model_ids


def test_openai_get_adapter_raises_value_error():
    """Verify get_adapter rejects OpenAI with clear retirement message."""
    with pytest.raises(ValueError, match="has been removed from active service"):
        get_adapter("openai", "gpt-4o-mini")

    with pytest.raises(ValueError, match="has been removed from active service"):
        get_adapter("OPENAI", "gpt-4o")


def test_openai_adapter_direct_instantiation_raises_runtime_error():
    """Verify direct instantiation of OpenAIAdapter raises clear RuntimeError."""
    with pytest.raises(RuntimeError, match="Provider 'openai' has been removed from active service"):
        OpenAIAdapter(model_name="gpt-4o-mini", api_key="sk-test")


@pytest.mark.asyncio
async def test_openai_connection_check_returns_failed_retired():
    """Verify verify_provider_connectivity rejects OpenAI probes."""
    res = await verify_provider_connectivity("openai")
    assert res["status"] == "failed"
    assert "removed from active service" in res["message"]


def test_openai_api_key_not_in_active_config():
    """Verify config does not require or return OPENAI_API_KEY for active configuration."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy-test-key"}):
        assert config.get_api_key("openai") == ""

    status = config.get_provider_status()
    assert "openai" not in status


def test_historical_openai_models_display_name_and_pricing_preserved():
    """Verify historical OpenAI records can still resolve display names and pricing for historical runs."""
    assert resolve_model_display_name("gpt-4o") == "OpenAI GPT-4o (Retired)"
    assert resolve_model_display_name("gpt-4o-mini") == "OpenAI GPT-4o-mini (Retired)"

    pricing_4o = config.get_token_pricing("gpt-4o")
    assert pricing_4o["input"] == 2.50
    assert pricing_4o["output"] == 10.00


@pytest.mark.asyncio
async def test_runner_rejects_openai_in_new_benchmark(tmp_path):
    """Verify BenchmarkRunner rejects attempts to run new benchmarks with removed OpenAI provider."""
    db_file = tmp_path / "test_openai_runner.db"
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
        ModelConfig(provider="openai", model_name="gpt-4o-mini", enabled=True)
    ]

    benchmark_run = await runner.run_benchmark(
        test_cases=[tc],
        model_configs=configs,
        run_config={"enable_semantic_judge": False}
    )

    results = repo.get_results_for_run(benchmark_run.run_id)
    openai_res = next(r for r in results if r.provider == "openai")
    mock_res = next(r for r in results if r.provider == "mock")

    assert mock_res.status == "success"
    assert openai_res.status == "provider_error"
    assert "removed from active service" in (openai_res.provider_error or "")
