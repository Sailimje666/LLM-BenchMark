"""Unit tests for statistical analysis, Wilson confidence intervals, and paired comparisons."""

import pytest
from schema import EvaluationResult
from engine.telemetry import wilson_score_interval, compute_paired_comparison, compute_model_statistics


def test_wilson_score_interval():
    # 0 successes out of 0
    low, high = wilson_score_interval(0, 0)
    assert low == 0.0 and high == 0.0

    # 10 out of 10 successes
    low, high = wilson_score_interval(10, 10)
    assert low > 60.0
    assert high == 100.0

    # 5 out of 10 (50%)
    low, high = wilson_score_interval(5, 10)
    assert low < 50.0 < high


def test_compute_paired_comparison_empty():
    stats = compute_paired_comparison([], [])
    assert stats["shared_tests"] == 0
    assert stats["statistically_significant"] is False


def test_compute_paired_comparison():
    res_a = [
        EvaluationResult(
            run_id="r1", test_case_id="tc1", provider="mock", model_name="m_a",
            response_text="ok", accuracy_score=1.0, is_correct=True, timestamp="2026-01-01"
        ),
        EvaluationResult(
            run_id="r1", test_case_id="tc2", provider="mock", model_name="m_a",
            response_text="ok", accuracy_score=1.0, is_correct=True, timestamp="2026-01-01"
        ),
        EvaluationResult(
            run_id="r1", test_case_id="tc3", provider="mock", model_name="m_a",
            response_text="fail", accuracy_score=0.0, is_correct=False, timestamp="2026-01-01"
        )
    ]
    res_b = [
        EvaluationResult(
            run_id="r2", test_case_id="tc1", provider="mock", model_name="m_b",
            response_text="fail", accuracy_score=0.0, is_correct=False, timestamp="2026-01-01"
        ),
        EvaluationResult(
            run_id="r2", test_case_id="tc2", provider="mock", model_name="m_b",
            response_text="ok", accuracy_score=1.0, is_correct=True, timestamp="2026-01-01"
        ),
        EvaluationResult(
            run_id="r2", test_case_id="tc3", provider="mock", model_name="m_b",
            response_text="fail", accuracy_score=0.0, is_correct=False, timestamp="2026-01-01"
        )
    ]

    comp = compute_paired_comparison(res_a, res_b)
    assert comp["shared_tests"] == 3
    assert comp["model_a_wins"] == 1
    assert comp["model_b_wins"] == 0
    assert comp["ties"] == 2


def test_compute_model_statistics_extended():
    results = [
        EvaluationResult(
            run_id="r1", test_case_id="t1", provider="mock", model_name="m",
            response_text="ok", accuracy_score=1.0, is_correct=True, status="success",
            execution_time_ms=100.0, timestamp="2026-01-01"
        ),
        EvaluationResult(
            run_id="r1", test_case_id="t2", provider="mock", model_name="m",
            response_text="", accuracy_score=0.0, is_correct=False, status="timeout",
            provider_error="Connection timed out", execution_time_ms=0.0, timestamp="2026-01-01"
        )
    ]
    stats = compute_model_statistics(results)
    assert stats["binary_accuracy"] == 50.0
    assert stats["timeout_rate"] == 50.0
    assert stats["provider_failure_rate"] == 50.0
    assert "ci_95_lower" in stats
    assert "ci_95_upper" in stats


def test_compute_model_statistics_all_provider_failures():
    """Verify that when all requests fail with provider errors, stats reflect 0 completed and 100% failure rate without crashing."""
    results = [
        EvaluationResult(
            run_id="run_fail_42",
            test_case_id=f"tc_{i}",
            provider="openai",
            model_name="gpt-4o",
            response_text="",
            accuracy_score=0.0,
            is_correct=False,
            status="provider_error",
            error_category="Provider / API Error",
            provider_error="OpenAI API Error (429): You have no credits remaining.",
            execution_time_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            timestamp="2026-10-12T10:41:00Z"
        )
        for i in range(42)
    ]
    stats = compute_model_statistics(results)
    assert stats["total_requests"] == 42
    assert stats["completed_requests"] == 0
    assert stats["failed_requests"] == 42
    assert stats["binary_accuracy"] == 0.0
    assert stats["mean_score"] == 0.0
    assert stats["success_rate"] == 0.0
    assert stats["provider_failure_rate"] == 100.0
    assert stats["mean_latency_ms"] == 0.0
    assert stats["median_latency_ms"] == 0.0
    assert stats["total_tokens"] == 0
    assert stats["avg_tokens_per_req"] == 0.0
    assert stats["total_cost_usd"] == 0.0
    assert stats["error_distribution"] == {"Provider / API Error": 42}


def test_compute_model_statistics_all_successful():
    """Verify comprehensive metrics calculation when all requests succeed."""
    results = [
        EvaluationResult(
            run_id="run_succ_10",
            test_case_id=f"tc_{i}",
            provider="mock",
            model_name="mock-model",
            response_text="Correct answer",
            accuracy_score=1.0,
            is_correct=True,
            status="success",
            error_category="Success",
            execution_time_ms=100.0 + (i * 10),  # 100, 110, ..., 190
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            estimated_cost_usd=0.00005,
            timestamp="2026-10-12T10:41:00Z"
        )
        for i in range(10)
    ]
    stats = compute_model_statistics(results)
    assert stats["total_requests"] == 10
    assert stats["completed_requests"] == 10
    assert stats["failed_requests"] == 0
    assert stats["binary_accuracy"] == 100.0
    assert stats["mean_score"] == 1.0
    assert stats["success_rate"] == 100.0
    assert stats["provider_failure_rate"] == 0.0
    assert stats["median_latency_ms"] == 145.0
    assert stats["total_tokens"] == 300
    assert stats["avg_tokens_per_req"] == 30.0
    assert round(stats["total_cost_usd"], 5) == 0.0005
    assert stats["ci_95_lower"] > 65.0


def test_compute_model_statistics_missing_metrics():
    """Verify metrics calculation when some results have zero latency or zero tokens (missing metrics)."""
    results = [
        EvaluationResult(
            run_id="run_mix_3",
            test_case_id="tc_1",
            provider="mock",
            model_name="mock-model",
            response_text="Correct",
            accuracy_score=1.0,
            is_correct=True,
            status="success",
            execution_time_ms=200.0,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            estimated_cost_usd=0.0001,
            timestamp="2026-10-12T10:41:00Z"
        ),
        EvaluationResult(
            run_id="run_mix_3",
            test_case_id="tc_2",
            provider="mock",
            model_name="mock-model",
            response_text="",
            accuracy_score=0.0,
            is_correct=False,
            status="provider_error",
            provider_error="HTTP 500 Server Error",
            execution_time_ms=0.0,  # Zero latency due to failure
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            timestamp="2026-10-12T10:41:00Z"
        )
    ]
    stats = compute_model_statistics(results)
    assert stats["total_requests"] == 2
    assert stats["completed_requests"] == 1
    assert stats["failed_requests"] == 1
    assert stats["binary_accuracy"] == 50.0
    # Latency should only consider executed calls (> 0 ms)
    assert stats["mean_latency_ms"] == 200.0
    assert stats["median_latency_ms"] == 200.0
    assert stats["total_tokens"] == 70

