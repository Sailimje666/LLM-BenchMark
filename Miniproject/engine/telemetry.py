"""Telemetry collection and statistical analysis for LLM benchmark runs."""

import math
from typing import List, Dict, Any, Optional
import numpy as np
from schema import EvaluationResult


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_input_per_1m: float,
    price_output_per_1m: float
) -> float:
    """Calculate monetary estimated cost in USD based on input and output tokens."""
    input_cost = (prompt_tokens / 1_000_000.0) * price_input_per_1m
    output_cost = (completion_tokens / 1_000_000.0) * price_output_per_1m
    return round(input_cost + output_cost, 6)


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate 95% Wilson score confidence interval for binary accuracy proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + (z ** 2) / total
    centre = (p + (z ** 2) / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * total)) / total)) / denom
    lower = max(0.0, (centre - spread) * 100.0)
    upper = min(100.0, (centre + spread) * 100.0)
    return round(lower, 2), round(upper, 2)


def compute_model_statistics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Calculate comprehensive performance, reliability, and accuracy metrics for an evaluated model."""
    if not results:
        return {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "binary_accuracy": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "mean_score": 0.0,
            "success_rate": 0.0,
            "timeout_rate": 0.0,
            "provider_failure_rate": 0.0,
            "median_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "total_tokens": 0,
            "avg_tokens_per_req": 0.0,
            "total_cost_usd": 0.0,
            "cost_per_1k_usd": 0.0,
            "error_distribution": {}
        }

    total_reqs = len(results)
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]
    timeouts = [r for r in results if r.status == "timeout" or "timeout" in (r.provider_error or "").lower()]
    provider_errors = [
        r for r in results
        if r.status in ["provider_error", "timeout"]
        or bool(r.provider_error)
        or r.error_category == "Provider / API Error"
    ]

    # Accuracy calculations
    correct_count = sum(1 for r in results if r.is_correct)
    binary_acc = (correct_count / total_reqs) * 100.0 if total_reqs > 0 else 0.0
    ci_low, ci_high = wilson_score_interval(correct_count, total_reqs)
    mean_score = float(np.mean([r.accuracy_score for r in results])) if results else 0.0

    # Latencies (from successful or executed calls)
    latencies = [r.execution_time_ms for r in results if r.execution_time_ms > 0]
    if latencies:
        mean_lat = float(np.mean(latencies))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        std_lat = float(np.std(latencies))
    else:
        mean_lat = med_lat = p95_lat = std_lat = 0.0

    # Token usage & Cost
    total_tokens = sum(r.total_tokens for r in results)
    avg_tokens = (total_tokens / total_reqs) if total_reqs > 0 else 0.0
    total_cost = sum(r.estimated_cost_usd for r in results)
    cost_per_1k = (total_cost / total_reqs * 1000.0) if total_reqs > 0 else 0.0

    # Error distribution
    error_counts: Dict[str, int] = {}
    for r in results:
        cat = r.error_category or "Unclassified"
        error_counts[cat] = error_counts.get(cat, 0) + 1

    return {
        "total_requests": total_reqs,
        "completed_requests": len(successful),
        "failed_requests": len(failed),
        "binary_accuracy": round(binary_acc, 2),
        "ci_95_lower": ci_low,
        "ci_95_upper": ci_high,
        "mean_score": round(mean_score, 4),
        "success_rate": round((len(successful) / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0,
        "timeout_rate": round((len(timeouts) / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0,
        "provider_failure_rate": round((len(provider_errors) / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "total_tokens": total_tokens,
        "avg_tokens_per_req": round(avg_tokens, 1),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_1k_usd": round(cost_per_1k, 4),
        "error_distribution": error_counts
    }


def compute_paired_comparison(results_a: List[EvaluationResult], results_b: List[EvaluationResult]) -> Dict[str, Any]:
    """Perform paired comparative statistical analysis across two models on identical test cases."""
    map_a = {r.test_case_id: r for r in results_a}
    map_b = {r.test_case_id: r for r in results_b}
    shared_ids = sorted(list(set(map_a.keys()).intersection(set(map_b.keys()))))

    if not shared_ids:
        return {"shared_tests": 0, "model_a_wins": 0, "model_b_wins": 0, "ties": 0, "statistically_significant": False}

    a_wins = 0
    b_wins = 0
    ties = 0
    contingency = {"both_correct": 0, "only_a_correct": 0, "only_b_correct": 0, "both_wrong": 0}

    for tid in shared_ids:
        ra = map_a[tid]
        rb = map_b[tid]
        if ra.is_correct and rb.is_correct:
            contingency["both_correct"] += 1
            ties += 1
        elif ra.is_correct and not rb.is_correct:
            contingency["only_a_correct"] += 1
            a_wins += 1
        elif not ra.is_correct and rb.is_correct:
            contingency["only_b_correct"] += 1
            b_wins += 1
        else:
            contingency["both_wrong"] += 1
            ties += 1

    b = contingency["only_a_correct"]
    c = contingency["only_b_correct"]
    if (b + c) > 0:
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if abs(b - c) > 0 else 0.0
        significant = bool(chi2 > 3.841)  # p < 0.05
    else:
        chi2 = 0.0
        significant = False

    return {
        "shared_tests": len(shared_ids),
        "model_a_wins": a_wins,
        "model_b_wins": b_wins,
        "ties": ties,
        "contingency": contingency,
        "mcnemar_chi2": round(chi2, 3),
        "statistically_significant": significant
    }
