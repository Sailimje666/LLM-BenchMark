"""Unit tests for SQLite storage, benchmark runs, and result queries."""

import pytest
from datetime import datetime, timezone
from schema import BenchmarkRun, EvaluationResult, TestCase
from storage.repository import BenchmarkRepository
from storage.database import init_db


def test_save_and_retrieve_benchmark_run(tmp_path):
    db_file = str(tmp_path / "test_store.db")
    init_db(db_file)
    repo = BenchmarkRepository(db_file)

    now = datetime.now(timezone.utc).isoformat()
    run = BenchmarkRun(
        run_id="run_test_100",
        name="Unit Test Run",
        start_time=now,
        selected_models=["mock-gpt", "mock-claude"],
        total_test_cases=10,
        total_requests=20,
        completed_requests=20,
        model_metrics={"mock-gpt": {"binary_accuracy": 95.0}},
        status="completed"
    )

    repo.create_run(run)
    retrieved = repo.get_run("run_test_100")

    assert retrieved is not None
    assert retrieved.run_id == "run_test_100"
    assert retrieved.name == "Unit Test Run"
    assert retrieved.selected_models == ["mock-gpt", "mock-claude"]
    assert retrieved.model_metrics["mock-gpt"]["binary_accuracy"] == 95.0


def test_save_and_query_evaluation_results(tmp_path):
    db_file = str(tmp_path / "test_eval_store.db")
    init_db(db_file)
    repo = BenchmarkRepository(db_file)

    now = datetime.now(timezone.utc).isoformat()
    run = BenchmarkRun(run_id="run_test_200", start_time=now, total_requests=1)
    repo.create_run(run)

    res = EvaluationResult(
        run_id="run_test_200",
        test_case_id="tc_01",
        provider="mock",
        model_name="mock-gpt",
        response_text="42",
        accuracy_score=1.0,
        is_correct=True,
        evaluation_method="exact_match",
        execution_time_ms=120.5,
        prompt_tokens=15,
        completion_tokens=2,
        total_tokens=17,
        estimated_cost_usd=0.00002,
        error_category="Success",
        timestamp=now
    )

    repo.save_evaluation_result(res)
    results = repo.get_results_for_run("run_test_200")

    assert len(results) == 1
    assert results[0].test_case_id == "tc_01"
    assert results[0].is_correct is True
    assert results[0].execution_time_ms == 120.5


def test_duplicate_result_prevention(tmp_path):
    db_file = str(tmp_path / "test_dup.db")
    init_db(db_file)
    repo = BenchmarkRepository(db_file)

    now = datetime.now(timezone.utc).isoformat()
    run = BenchmarkRun(run_id="run_dup", start_time=now)
    repo.create_run(run)

    res = EvaluationResult(
        run_id="run_dup",
        test_case_id="tc_dup",
        provider="mock",
        model_name="mock-gpt",
        response_text="initial",
        accuracy_score=0.5,
        is_correct=False,
        timestamp=now
    )
    repo.save_evaluation_result(res)

    # Save updated result for the same (run_id, test_case_id, model_name)
    res_updated = EvaluationResult(
        run_id="run_dup",
        test_case_id="tc_dup",
        provider="mock",
        model_name="mock-gpt",
        response_text="updated",
        accuracy_score=1.0,
        is_correct=True,
        timestamp=now
    )
    repo.save_evaluation_result(res_updated)

    all_results = repo.get_results_for_run("run_dup")
    assert len(all_results) == 1
    assert all_results[0].accuracy_score == 1.0
    assert all_results[0].response_text == "updated"
