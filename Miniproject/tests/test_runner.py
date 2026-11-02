"""Unit tests for BenchmarkRunner concurrent execution, retry, and partial completion."""

import pytest
import asyncio
from schema import TestCase, ModelConfig, BenchmarkRun
from engine.runner import BenchmarkRunner
from storage.repository import BenchmarkRepository


@pytest.mark.asyncio
async def test_benchmark_runner_mock_models(tmp_path):
    test_db = str(tmp_path / "test_benchmark.db")
    repo = BenchmarkRepository(test_db)
    runner = BenchmarkRunner(repository=repo, max_concurrent=3)

    test_cases = [
        TestCase(
            id="t1",
            problem_statement="What is 47 multiplied by 18?",
            ground_truth="846",
            domain="Math",
            evaluation_method="numeric_tolerance"
        ),
        TestCase(
            id="t2",
            problem_statement="All roses are flowers. Some flowers fade quickly. Does some roses fade quickly? Yes or No.",
            ground_truth="No",
            domain="Logic",
            evaluation_method="exact_match"
        )
    ]

    model_configs = [
        ModelConfig(provider="mock", model_name="mock-gpt", enabled=True),
        ModelConfig(provider="mock", model_name="mock-llama", enabled=True)
    ]

    run = await runner.run_benchmark(
        test_cases=test_cases,
        model_configs=model_configs,
        run_config={"run_id": "test_run_01", "enable_semantic_judge": False}
    )

    assert isinstance(run, BenchmarkRun)
    assert run.run_id == "test_run_01"
    assert run.total_requests == 4
    assert run.completed_requests == 4
    assert run.status == "completed"
    assert "mock-gpt" in run.model_metrics
    assert "mock-llama" in run.model_metrics


@pytest.mark.asyncio
async def test_runner_fault_isolation(tmp_path):
    """Verify that a failing model does not crash the entire benchmark for other models."""
    test_db = str(tmp_path / "fault_test.db")
    repo = BenchmarkRepository(test_db)
    runner = BenchmarkRunner(repository=repo, max_concurrent=2)

    test_cases = [
        TestCase(id="t_iso", problem_statement="Capital of France?", ground_truth="Paris", domain="General")
    ]

    model_configs = [
        ModelConfig(provider="mock", model_name="mock-gpt", enabled=True),
        # An invalid provider/model that will raise or error
        ModelConfig(provider="invalid_provider", model_name="bad-model", enabled=True)
    ]

    run = await runner.run_benchmark(
        test_cases=test_cases,
        model_configs=model_configs,
        run_config={"run_id": "iso_run_02", "enable_semantic_judge": False}
    )

    assert run.completed_requests == 1
    assert run.failed_requests == 1
    assert run.status == "partial"
    results = repo.get_results_for_run("iso_run_02")
    gpt_res = next((r for r in results if r.model_name == "mock-gpt"), None)
    bad_res = next((r for r in results if r.model_name == "bad-model"), None)

    assert gpt_res is not None
    assert gpt_res.status == "success"
    assert bad_res is not None
    assert bad_res.status == "provider_error"


@pytest.mark.asyncio
async def test_runner_all_failed_run_summary(tmp_path):
    """Verify that a run where all requests fail records status='failed', completed_requests=0, and 100% failure rate."""
    test_db = str(tmp_path / "all_failed.db")
    repo = BenchmarkRepository(test_db)
    runner = BenchmarkRunner(repository=repo, max_concurrent=2)

    test_cases = [
        TestCase(id="tc_fail_1", problem_statement="Problem 1", ground_truth="Answer 1", domain="Math"),
        TestCase(id="tc_fail_2", problem_statement="Problem 2", ground_truth="Answer 2", domain="Logic")
    ]

    model_configs = [
        ModelConfig(provider="invalid_provider", model_name="unreachable-model", enabled=True)
    ]

    run = await runner.run_benchmark(
        test_cases=test_cases,
        model_configs=model_configs,
        run_config={"run_id": "failed_run_01", "enable_semantic_judge": False}
    )

    assert run.status == "failed"
    assert run.total_requests == 2
    assert run.completed_requests == 0
    assert run.failed_requests == 2
    assert run.skipped_requests == 0

    m_metrics = run.model_metrics["unreachable-model"]
    assert m_metrics["completed_requests"] == 0
    assert m_metrics["failed_requests"] == 2
    assert m_metrics["binary_accuracy"] == 0.0
    assert m_metrics["mean_score"] == 0.0
    assert m_metrics["median_latency_ms"] == 0.0
    assert m_metrics["total_tokens"] == 0
    assert m_metrics["provider_failure_rate"] == 100.0


@pytest.mark.asyncio
async def test_rate_limiter_does_not_retry_quota_exhaustion():
    """Verify rate limiter does not waste time retrying non-transient out-of-quota / credit balance errors."""
    from engine.rate_limiter import RateLimiter

    rate_limiter = RateLimiter(max_concurrent=1, max_retries=3, backoff_factor=0.01)
    call_count = 0

    async def fail_quota():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("OpenAI API Error (429): You have no credits remaining. Add credits to continue using the API.")

    with pytest.raises(RuntimeError, match="no credits remaining"):
        await rate_limiter.execute_with_retry(fail_quota, provider="openai")

    # Should only have been called once - no retries on permanent quota exhaustion
    assert call_count == 1
