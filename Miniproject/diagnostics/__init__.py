"""Diagnostics package for deterministic grading, semantic judge, and root-cause classification."""

from diagnostics.deterministic import (
    evaluate_exact_match,
    evaluate_numeric_tolerance,
    evaluate_regex,
    evaluate_structured_json
)
from diagnostics.semantic_judge import evaluate_with_judge, evaluate_with_mock_judge
from diagnostics.root_cause import diagnose_failure, PRIMARY_CATEGORIES
from diagnostics.evaluator import evaluate_response

__all__ = [
    "evaluate_exact_match",
    "evaluate_numeric_tolerance",
    "evaluate_regex",
    "evaluate_structured_json",
    "evaluate_with_judge",
    "evaluate_with_mock_judge",
    "diagnose_failure",
    "PRIMARY_CATEGORIES",
    "evaluate_response"
]
