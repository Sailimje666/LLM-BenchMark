"""Unit tests for dataset schema validation and report exports."""

import json
import pytest
from pydantic import ValidationError
from schema import TestCase, BenchmarkRun, EvaluationResult
from storage.exports import (
    export_results_csv,
    export_results_json,
    generate_consolidated_markdown_report
)


def test_testcase_validation_valid():
    tc = TestCase(
        id="valid_01",
        problem_statement="Question text",
        ground_truth="Answer text",
        domain="Math",
        difficulty="Hard",
        evaluation_method="exact_match"
    )
    assert tc.difficulty == "Hard"
    assert tc.evaluation_method == "exact_match"


def test_testcase_validation_invalid_difficulty():
    with pytest.raises(ValidationError):
        TestCase(
            id="bad_diff",
            problem_statement="Q",
            ground_truth="A",
            domain="Math",
            difficulty="SuperHard"  # Invalid
        )


def test_testcase_validation_invalid_method():
    with pytest.raises(ValidationError):
        TestCase(
            id="bad_method",
            problem_statement="Q",
            ground_truth="A",
            domain="Math",
            evaluation_method="unknown_method"  # Invalid
        )


def test_csv_and_json_exports():
    results = [
        EvaluationResult(
            run_id="run_exp",
            test_case_id="tc_1",
            provider="mock",
            model_name="mock-gpt",
            response_text="846",
            accuracy_score=1.0,
            is_correct=True,
            evaluation_method="exact_match",
            execution_time_ms=150.0,
            error_category="Success",
            timestamp="2026-01-01T00:00:00Z"
        )
    ]

    csv_out = export_results_csv(results)
    assert "Run ID" in csv_out
    assert "mock-gpt" in csv_out
    assert "846" in csv_out

    json_out = export_results_json(results)
    parsed = json.loads(json_out)
    assert isinstance(parsed, list)
    assert parsed[0]["model_name"] == "mock-gpt"


def test_markdown_report_generation():
    run = BenchmarkRun(
        run_id="run_md",
        start_time="2026-01-01T00:00:00Z",
        total_test_cases=1,
        total_requests=1,
        model_metrics={"mock-gpt": {"binary_accuracy": 100.0, "mean_score": 1.0}}
    )
    results = [
        EvaluationResult(
            run_id="run_md",
            test_case_id="tc_1",
            provider="mock",
            model_name="mock-gpt",
            response_text="846",
            accuracy_score=1.0,
            is_correct=True,
            error_category="Success",
            timestamp="2026-01-01T00:00:00Z"
        )
    ]

    report = generate_consolidated_markdown_report(run, results)
    assert "# LLM Benchmark & Diagnostics: Comparative Report" in report
    assert "mock-gpt" in report
    assert "Executive Summary" in report
