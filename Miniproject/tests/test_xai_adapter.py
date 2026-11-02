"""Tests verifying the complete retirement and exclusion of xAI Grok provider."""

import pytest
from unittest.mock import patch

import config
from engine.model_registry import list_registered_models, get_adapter, resolve_model_display_name
from engine.provider_adapter import XAIGrokAdapter
from engine.connection_check import verify_provider_connectivity
from schema import TestCase, ModelConfig
from storage.repository import BenchmarkRepository
from engine.runner import BenchmarkRunner


def test_xai_excluded_from_active_registry():
    """Verify that xAI Grok models are completely excluded from the active model registry."""
    active_models = list_registered_models()
    active_providers = {m["provider"] for m in active_models}
    assert "xai" not in active_providers
    assert "grok" not in active_providers

    model_ids = {m["model_id"] for m in active_models}
    assert "grok-2-1212" not in model_ids
    assert "grok-2" not in model_ids
    assert "grok-beta" not in model_ids


def test_xai_get_adapter_raises_value_error():
    """Verify get_adapter rejects xai and grok with clear retirement message."""
    with pytest.raises(ValueError, match="Provider 'xai' has been removed from active service"):
        get_adapter("xai", "grok-2-1212")

    with pytest.raises(ValueError, match="Provider 'grok' has been removed from active service"):
        get_adapter("grok", "grok-beta")


def test_xai_adapter_direct_instantiation_raises_runtime_error():
    """Verify direct instantiation of XAIGrokAdapter raises clear RuntimeError."""
    with pytest.raises(RuntimeError, match="Provider 'xai' has been removed from active service"):
        XAIGrokAdapter(model_name="grok-2-1212", api_key="xai-test-key")


@pytest.mark.asyncio
async def test_xai_connection_check_returns_failed_retired():
    """Verify verify_provider_connectivity rejects xAI and Grok probes."""
    res_xai = await verify_provider_connectivity("xai")
    assert res_xai["status"] == "failed"
    assert "removed from active service" in res_xai["message"]

    res_grok = await verify_provider_connectivity("grok")
    assert res_grok["status"] == "failed"
    assert "removed from active service" in res_grok["message"]


def test_xai_api_key_not_in_active_config():
    """Verify config does not require or return XAI_API_KEY for active configuration."""
    with patch.dict("os.environ", {"XAI_API_KEY": "xai-dummy-key"}):
        assert config.get_api_key("xai") == ""
        assert config.get_api_key("grok") == ""

    status = config.get_provider_status()
    assert "xai" not in status
    assert "grok" not in status


def test_historical_xai_models_display_name_and_pricing_preserved():
    """Verify historical xAI records can still resolve display names and pricing for historical runs."""
    assert resolve_model_display_name("grok-2-1212") == "xAI Grok-2 1212 (Retired)"
    assert resolve_model_display_name("grok-beta") == "xAI Grok-Beta (Retired)"

    pricing_grok = config.get_token_pricing("grok-2-1212")
    assert pricing_grok["input"] == 2.00
    assert pricing_grok["output"] == 10.00


@pytest.mark.asyncio
async def test_runner_rejects_xai_in_new_benchmark(tmp_path):
    """Verify BenchmarkRunner rejects attempts to run new benchmarks with removed xAI provider."""
    db_file = tmp_path / "test_xai_runner.db"
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
        ModelConfig(provider="xai", model_name="grok-2-1212", enabled=True)
    ]

    benchmark_run = await runner.run_benchmark(
        test_cases=[tc],
        model_configs=configs,
        run_config={"enable_semantic_judge": False}
    )

    results = repo.get_results_for_run(benchmark_run.run_id)
    xai_res = next(r for r in results if r.provider == "xai")
    mock_res = next(r for r in results if r.provider == "mock")

    assert mock_res.status == "success"
    assert xai_res.status == "provider_error"
    assert "removed from active service" in (xai_res.provider_error or "")
