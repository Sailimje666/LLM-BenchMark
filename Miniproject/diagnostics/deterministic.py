"""Deterministic evaluation engine for exact match, numeric tolerance, regex, and structured JSON."""

import re
import json
import math
from typing import Tuple, List, Optional, Any, Dict
from schema import TestCase


def normalize_text(text: str, remove_punct: bool = False, lower: bool = True) -> str:
    """Normalize text by collapsing whitespace, trimming, and optional case/punctuation adjustments."""
    if not text:
        return ""
    # Strip markdown bold/italics
    cleaned = re.sub(r"[*_`]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if lower:
        cleaned = cleaned.lower()
    if remove_punct:
        # Preserve math symbols (+, -, *, /, ^, =) and brackets
        cleaned = re.sub(r"[.,;:!?'\"]", "", cleaned).strip()
    return cleaned


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values safely from text, supporting negatives, decimals, and scientific notation."""
    pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def evaluate_exact_match(
    response: str,
    test_case: TestCase,
    case_sensitive: bool = False,
    strip_punctuation: bool = True
) -> Tuple[bool, float, str, List[str]]:
    """Perform exact match comparison between response and ground truth."""
    violations: List[str] = []
    norm_resp = normalize_text(response, remove_punct=strip_punctuation, lower=not case_sensitive)
    norm_gt = normalize_text(test_case.ground_truth, remove_punct=strip_punctuation, lower=not case_sensitive)

    # Check constraint violations
    for constraint in test_case.constraints:
        if "single" in constraint.lower() or "only" in constraint.lower():
            if len(response.strip().split("\n")) > 2:
                violations.append(f"Constraint violated: Expected concise answer, received multi-line response: {constraint}")

    if norm_resp == norm_gt:
        return True, 1.0, f"Exact match confirmed: '{norm_resp}' matches ground truth '{norm_gt}'.", violations

    # Check if ground truth is contained cleanly as the dominant answer
    if norm_gt in norm_resp and len(norm_resp) <= len(norm_gt) + 15:
        # Near match with minor conversational padding e.g. "The answer is X"
        return True, 1.0, f"Ground truth '{norm_gt}' found with minimal conversational padding in response.", violations

    return False, 0.0, f"Expected '{norm_gt}', got '{norm_resp}'.", violations


def evaluate_numeric_tolerance(
    response: str,
    test_case: TestCase
) -> Tuple[bool, float, str, List[str]]:
    """Evaluate numeric answer within specified absolute or relative tolerance."""
    violations: List[str] = []
    tolerance = test_case.tolerance if test_case.tolerance is not None else 0.001

    # Extract target number from ground truth
    gt_numbers = extract_numbers(test_case.ground_truth)
    if not gt_numbers:
        return False, 0.0, f"Ground truth '{test_case.ground_truth}' contains no parseable number.", ["Invalid ground truth"]
    target_val = gt_numbers[0]

    resp_numbers = extract_numbers(response)
    if not resp_numbers:
        return False, 0.0, "Response contains no numerical values.", ["No numeric value found in response"]

    # If multiple numbers exist, check if user provided steps or ambiguous output
    if len(resp_numbers) > 1:
        # Check if the last number matches (common for final answer in step-by-step reasoning)
        candidate = resp_numbers[-1]
        violations.append(f"Response contains {len(resp_numbers)} numbers; evaluated final number ({candidate}).")
    else:
        candidate = resp_numbers[0]

    # Absolute difference check
    abs_diff = abs(candidate - target_val)
    rel_diff = abs_diff / (abs(target_val) + 1e-9)

    is_correct = abs_diff <= tolerance or rel_diff <= tolerance
    score = 1.0 if is_correct else 0.0
    reasoning = (
        f"Extracted candidate: {candidate}, expected: {target_val}. "
        f"Diff: {abs_diff:.6f} (tolerance: {tolerance}). Verdict: {'PASS' if is_correct else 'FAIL'}."
    )

    return is_correct, score, reasoning, violations


def evaluate_regex(
    response: str,
    test_case: TestCase
) -> Tuple[bool, float, str, List[str]]:
    """Validate model response against a compiled regular expression."""
    pattern = test_case.regex_pattern or test_case.ground_truth
    if not pattern:
        return False, 0.0, "No regex pattern supplied for regex evaluation.", ["Missing regex pattern"]

    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        match = compiled.search(response.strip())
        if match:
            return True, 1.0, f"Pattern '{pattern}' matched response substring '{match.group(0)}'.", []
        else:
            return False, 0.0, f"Response failed to match pattern '{pattern}'.", [f"Regex pattern '{pattern}' not satisfied"]
    except re.error as e:
        return False, 0.0, f"Invalid regex pattern '{pattern}': {str(e)}", [f"Invalid regex: {str(e)}"]


def clean_json_string(raw: str) -> str:
    """Extract clean JSON substring from potential markdown code fences."""
    cleaned = raw.strip()
    # Handle ```json ... ``` blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if code_block_match:
        return code_block_match.group(1).strip()
    return cleaned


def evaluate_structured_json(
    response: str,
    test_case: TestCase
) -> Tuple[bool, float, str, List[str]]:
    """Validate JSON output structure, required keys, and content match against ground truth."""
    violations: List[str] = []
    json_str = clean_json_string(response)

    try:
        parsed_resp = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, 0.0, f"Invalid JSON format: {str(e)}", ["Malformed JSON syntax"]

    # Parse expected ground truth JSON
    try:
        parsed_gt = json.loads(clean_json_string(test_case.ground_truth))
    except Exception:
        # If ground truth is not strict JSON, just verify valid JSON was produced
        return True, 1.0, "Valid JSON successfully generated.", violations

    # Structural key validation
    if isinstance(parsed_gt, dict):
        if not isinstance(parsed_resp, dict):
            return False, 0.0, f"Expected JSON object (dict), received {type(parsed_resp).__name__}.", ["Type mismatch"]
        missing_keys = [k for k in parsed_gt.keys() if k not in parsed_resp]
        if missing_keys:
            violations.append(f"Missing mandatory keys: {missing_keys}")
            return False, 0.5, f"JSON valid but missing keys: {missing_keys}", violations

        # Check values
        matches = sum(1 for k in parsed_gt if str(parsed_resp.get(k, "")).strip().lower() == str(parsed_gt[k]).strip().lower())
        total = len(parsed_gt)
        score = matches / max(1, total)
        is_correct = score >= 0.99
        return is_correct, score, f"JSON key-value match: {matches}/{total} fields correct.", violations

    elif isinstance(parsed_gt, list):
        if not isinstance(parsed_resp, list):
            return False, 0.0, f"Expected JSON array (list), received {type(parsed_resp).__name__}.", ["Type mismatch"]
        if len(parsed_resp) != len(parsed_gt):
            violations.append(f"Expected {len(parsed_gt)} items, found {len(parsed_resp)}")
        is_correct = len(parsed_resp) == len(parsed_gt)
        score = 1.0 if is_correct else (0.5 if len(parsed_resp) > 0 else 0.0)
        return is_correct, score, f"JSON array items: {len(parsed_resp)}/{len(parsed_gt)}.", violations

    return True, 1.0, "Valid JSON matching specification.", violations
