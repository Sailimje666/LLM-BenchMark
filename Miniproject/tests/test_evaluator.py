"""Unit tests for deterministic evaluation and semantic judge."""

import pytest
from schema import TestCase, JudgeOutput
from diagnostics.deterministic import (
    evaluate_exact_match,
    evaluate_numeric_tolerance,
    evaluate_regex,
    evaluate_structured_json
)
from diagnostics.semantic_judge import evaluate_with_mock_judge


def test_exact_match_success_and_failure():
    tc = TestCase(
        id="em_1",
        problem_statement="Capital of France?",
        ground_truth="Paris",
        domain="General",
        evaluation_method="exact_match"
    )
    # Success with whitespace and case differences
    is_corr, score, reason, viols = evaluate_exact_match("   paris  ", tc)
    assert is_corr is True
    assert score == 1.0

    # Failure
    is_corr, score, reason, viols = evaluate_exact_match("London", tc)
    assert is_corr is False
    assert score == 0.0


def test_whitespace_and_punctuation_normalization():
    tc = TestCase(
        id="em_2",
        problem_statement="Yes or No?",
        ground_truth="Yes",
        domain="Logic",
        evaluation_method="exact_match"
    )
    is_corr, score, _, _ = evaluate_exact_match("**Yes!**", tc)
    assert is_corr is True


def test_numeric_tolerance_decimals_and_negatives():
    tc = TestCase(
        id="num_1",
        problem_statement="Calculate change",
        ground_truth="-14.50",
        domain="Math",
        evaluation_method="numeric_tolerance",
        tolerance=0.01
    )
    # Within tolerance
    is_corr, score, _, _ = evaluate_numeric_tolerance("-14.505", tc)
    assert is_corr is True
    assert score == 1.0

    # Outside tolerance
    is_corr, score, _, _ = evaluate_numeric_tolerance("-14.75", tc)
    assert is_corr is False
    assert score == 0.0


def test_numeric_tolerance_ambiguous_multiple_numbers():
    tc = TestCase(
        id="num_2",
        problem_statement="Solve 2x=84",
        ground_truth="42",
        domain="Math",
        evaluation_method="numeric_tolerance",
        tolerance=0.0
    )
    # Step by step with final answer at the end
    response = "Step 1: 2x = 84. Step 2: x = 84 / 2. Therefore the answer is 42."
    is_corr, score, _, viols = evaluate_numeric_tolerance(response, tc)
    assert is_corr is True
    assert score == 1.0
    assert any("evaluated final number" in v for v in viols)


def test_regex_matching():
    tc = TestCase(
        id="reg_1",
        problem_statement="Supply email",
        ground_truth="test@example.com",
        domain="Text Extraction",
        evaluation_method="regex",
        regex_pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )
    is_corr, score, _, _ = evaluate_regex("alice.smith@domain.co.uk", tc)
    assert is_corr is True

    is_corr, score, _, viols = evaluate_regex("not an email address", tc)
    assert is_corr is False
    assert len(viols) > 0


def test_structured_json_validation():
    tc = TestCase(
        id="json_1",
        problem_statement="Produce person json",
        ground_truth='{"name": "Alice", "age": 30}',
        domain="Text Extraction",
        evaluation_method="json_schema"
    )
    # Valid markdown code-fenced JSON
    valid_resp = '```json\n{"name": "Alice", "age": 30}\n```'
    is_corr, score, _, _ = evaluate_structured_json(valid_resp, tc)
    assert is_corr is True
    assert score == 1.0

    # Malformed JSON
    bad_resp = '{"name": "Alice", "age": }'
    is_corr, score, _, viols = evaluate_structured_json(bad_resp, tc)
    assert is_corr is False
    assert any("Malformed" in v for v in viols)


def test_mock_semantic_judge():
    tc = TestCase(
        id="code_test",
        problem_statement="Write a function add(a, b)",
        ground_truth="def add(a: int, b: int) -> int:\n    return a + b",
        domain="Coding",
        evaluation_method="semantic"
    )
    resp = "def add(a, b):\n    return a + b"
    judge_out = evaluate_with_mock_judge(tc, resp)
    assert isinstance(judge_out, JudgeOutput)
    assert judge_out.is_correct is True
    assert judge_out.score >= 0.7
