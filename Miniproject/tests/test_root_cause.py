"""Unit tests for root-cause error diagnostics and taxonomy categorization."""

import pytest
from schema import TestCase, JudgeOutput
from diagnostics.root_cause import diagnose_failure


def test_diagnose_success():
    tc = TestCase(id="s1", problem_statement="2+2", ground_truth="4", domain="Math")
    diag = diagnose_failure(tc, "4", is_correct=True, accuracy_score=1.0)
    assert diag.primary_category == "Success"
    assert diag.confidence_score == 1.0


def test_diagnose_provider_error():
    tc = TestCase(id="pe1", problem_statement="Hello", ground_truth="Hi", domain="General")
    diag = diagnose_failure(tc, "", is_correct=False, accuracy_score=0.0, provider_error="HTTP 401 Unauthorized")
    assert diag.primary_category == "Provider / API Error"
    assert "401" in diag.evidence


def test_diagnose_truncation_length():
    tc = TestCase(id="tr1", problem_statement="Long essay", ground_truth="...", domain="General")
    diag = diagnose_failure(tc, "This response ends mid-sentence and", is_correct=False, accuracy_score=0.0, finish_reason="length")
    assert diag.primary_category == "Truncation / Length Error"


def test_diagnose_formatting_mismatch():
    tc = TestCase(
        id="fmt1",
        problem_statement="Return json",
        ground_truth='{"status": "ok"}',
        domain="Text Extraction",
        evaluation_method="json_schema",
        expected_format="JSON"
    )
    diag = diagnose_failure(tc, "The status is ok without json syntax.", is_correct=False, accuracy_score=0.0)
    assert diag.primary_category == "Formatting Mismatch"


def test_diagnose_instruction_ignored():
    tc = TestCase(id="ins1", problem_statement="Only single letter", ground_truth="A", domain="General")
    diag = diagnose_failure(
        tc, "The answer is definitely A, because it fits the criteria.",
        is_correct=False, accuracy_score=0.0,
        constraint_violations=["Constraint violated: Expected concise single letter."]
    )
    assert diag.primary_category == "Instruction Ignored"


def test_diagnose_arithmetic_failure():
    tc = TestCase(id="math1", problem_statement="What is 47 * 18?", ground_truth="846", domain="Math")
    diag = diagnose_failure(
        tc, "836", is_correct=False, accuracy_score=0.0,
        deterministic_reasoning="Extracted 836, expected 846."
    )
    assert diag.primary_category == "Arithmetic / Logic Failure"


def test_diagnose_knowledge_gap():
    tc = TestCase(id="kg1", problem_statement="What happened on date X?", ground_truth="Event Y", domain="Factual")
    diag = diagnose_failure(
        tc, "As an AI language model, I do not know or have access to that information.",
        is_correct=False, accuracy_score=0.0
    )
    assert diag.primary_category == "Knowledge Gap"
