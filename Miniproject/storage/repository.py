"""Repository layer for database transactions, incremental persistence, and historical queries."""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from schema import (
    BenchmarkRun,
    EvaluationResult,
    TestCase,
    UniversalComparison,
    UniversalComparisonResult
)
from storage.database import get_db_connection


class BenchmarkRepository:
    """Manages persistent SQLite operations for benchmark runs, test cases, and evaluation results."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        from storage.database import init_db
        init_db(self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return get_db_connection(self.db_path)

    # --- Benchmark Run Methods ---

    def create_run(self, run: BenchmarkRun) -> None:
        """Insert a new benchmark run record."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO benchmark_runs (
                    run_id, name, start_time, end_time, dataset_version, dataset_hash,
                    selected_models, benchmark_config, total_test_cases, total_requests,
                    completed_requests, failed_requests, skipped_requests, model_metrics, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.name,
                run.start_time,
                run.end_time,
                run.dataset_version,
                run.dataset_hash,
                json.dumps(run.selected_models),
                json.dumps(run.benchmark_config),
                run.total_test_cases,
                run.total_requests,
                run.completed_requests,
                run.failed_requests,
                run.skipped_requests,
                json.dumps(run.model_metrics),
                run.status
            ))
        conn.close()

    def update_run(self, run: BenchmarkRun) -> None:
        """Update an existing benchmark run without triggering ON DELETE CASCADE."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                UPDATE benchmark_runs SET
                    name = ?, end_time = ?, dataset_version = ?, dataset_hash = ?,
                    selected_models = ?, benchmark_config = ?, total_test_cases = ?,
                    total_requests = ?, completed_requests = ?, failed_requests = ?,
                    skipped_requests = ?, model_metrics = ?, status = ?
                WHERE run_id = ?
            """, (
                run.name,
                run.end_time,
                run.dataset_version,
                run.dataset_hash,
                json.dumps(run.selected_models),
                json.dumps(run.benchmark_config),
                run.total_test_cases,
                run.total_requests,
                run.completed_requests,
                run.failed_requests,
                run.skipped_requests,
                json.dumps(run.model_metrics),
                run.status,
                run.run_id
            ))
        conn.close()

    def get_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Fetch a specific benchmark run by its run_id."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return BenchmarkRun(
            run_id=row["run_id"],
            name=row["name"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            dataset_version=row["dataset_version"],
            dataset_hash=row["dataset_hash"],
            selected_models=json.loads(row["selected_models"] or "[]"),
            benchmark_config=json.loads(row["benchmark_config"] or "{}"),
            total_test_cases=row["total_test_cases"],
            total_requests=row["total_requests"],
            completed_requests=row["completed_requests"],
            failed_requests=row["failed_requests"],
            skipped_requests=row["skipped_requests"],
            model_metrics=json.loads(row["model_metrics"] or "{}"),
            status=row["status"]
        )

    def list_runs(self) -> List[BenchmarkRun]:
        """Retrieve all historical benchmark runs ordered by start_time descending."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM benchmark_runs ORDER BY start_time DESC")
        rows = cursor.fetchall()
        conn.close()
        runs = []
        for row in rows:
            runs.append(BenchmarkRun(
                run_id=row["run_id"],
                name=row["name"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                dataset_version=row["dataset_version"],
                dataset_hash=row["dataset_hash"],
                selected_models=json.loads(row["selected_models"] or "[]"),
                benchmark_config=json.loads(row["benchmark_config"] or "{}"),
                total_test_cases=row["total_test_cases"],
                total_requests=row["total_requests"],
                completed_requests=row["completed_requests"],
                failed_requests=row["failed_requests"],
                skipped_requests=row["skipped_requests"],
                model_metrics=json.loads(row["model_metrics"] or "{}"),
                status=row["status"]
            ))
        return runs

    # --- Evaluation Result Methods ---

    def save_evaluation_result(self, res: EvaluationResult) -> None:
        """Insert or replace a single evaluation result (incremental persistence)."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO evaluation_results (
                    run_id, test_case_id, provider, model_name, response_text,
                    accuracy_score, is_correct, evaluation_method, execution_time_ms,
                    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd,
                    error_category, error_subcategory, judge_reasoning,
                    deterministic_reasoning, constraint_violations, provider_error,
                    status, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                res.run_id,
                res.test_case_id,
                res.provider,
                res.model_name,
                res.response_text,
                res.accuracy_score,
                1 if res.is_correct else 0,
                res.evaluation_method,
                res.execution_time_ms,
                res.prompt_tokens,
                res.completion_tokens,
                res.total_tokens,
                res.estimated_cost_usd,
                res.error_category,
                res.error_subcategory,
                res.judge_reasoning,
                res.deterministic_reasoning,
                json.dumps(res.constraint_violations),
                res.provider_error,
                res.status,
                res.timestamp
            ))
        conn.close()

    def get_results_for_run(self, run_id: str) -> List[EvaluationResult]:
        """Fetch all evaluation results associated with a benchmark run."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM evaluation_results WHERE run_id = ? ORDER BY id ASC
        """, (run_id,))
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append(EvaluationResult(
                run_id=r["run_id"],
                test_case_id=r["test_case_id"],
                provider=r["provider"],
                model_name=r["model_name"],
                response_text=r["response_text"] or "",
                accuracy_score=r["accuracy_score"],
                is_correct=bool(r["is_correct"]),
                evaluation_method=r["evaluation_method"],
                execution_time_ms=r["execution_time_ms"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                total_tokens=r["total_tokens"],
                estimated_cost_usd=r["estimated_cost_usd"],
                error_category=r["error_category"] or "Unclassified",
                error_subcategory=r["error_subcategory"],
                judge_reasoning=r["judge_reasoning"],
                deterministic_reasoning=r["deterministic_reasoning"],
                constraint_violations=json.loads(r["constraint_violations"] or "[]"),
                provider_error=r["provider_error"],
                status=r["status"],
                timestamp=r["timestamp"]
            ))
        return results

    # --- Test Case Methods ---

    def save_test_cases(self, test_cases: List[TestCase]) -> None:
        """Persist or update test cases in the database."""
        conn = self._get_conn()
        with conn:
            for tc in test_cases:
                conn.execute("""
                    INSERT OR REPLACE INTO test_cases (
                        id, problem_statement, ground_truth, domain, category,
                        difficulty, evaluation_method, expected_format,
                        constraints, tolerance, regex_pattern, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tc.id,
                    tc.problem_statement,
                    tc.ground_truth,
                    tc.domain,
                    tc.category,
                    tc.difficulty,
                    tc.evaluation_method,
                    tc.expected_format,
                    json.dumps(tc.constraints),
                    tc.tolerance,
                    tc.regex_pattern,
                    json.dumps(tc.metadata)
                ))
        conn.close()

    def get_all_test_cases(self) -> List[TestCase]:
        """Fetch all stored test cases from the database."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM test_cases ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()
        tcs = []
        for r in rows:
            tcs.append(TestCase(
                id=r["id"],
                problem_statement=r["problem_statement"],
                ground_truth=r["ground_truth"],
                domain=r["domain"],
                category=r["category"] or "",
                difficulty=r["difficulty"],
                evaluation_method=r["evaluation_method"],
                expected_format=r["expected_format"] or "",
                constraints=json.loads(r["constraints"] or "[]"),
                tolerance=r["tolerance"],
                regex_pattern=r["regex_pattern"] or "",
                metadata=json.loads(r["metadata"] or "{}")
            ))
        return tcs

    # --- Universal Comparison Methods ---

    def save_comparison(self, comp: UniversalComparison) -> None:
        """Insert or replace a universal question comparison record."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO universal_comparisons (
                    comparison_id, timestamp, question, system_prompt, reference_answer,
                    rubric, evaluation_mode, selected_models, generation_config, results,
                    status, agreement_analysis, summary_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comp.comparison_id,
                comp.timestamp,
                comp.question,
                comp.system_prompt,
                comp.reference_answer,
                comp.rubric,
                comp.evaluation_mode,
                json.dumps(comp.selected_models),
                json.dumps(comp.generation_config),
                json.dumps([r.model_dump() for r in comp.results]),
                comp.status,
                json.dumps(comp.agreement_analysis),
                comp.summary_notes
            ))
        conn.close()

    def get_comparison(self, comparison_id: str) -> Optional[UniversalComparison]:
        """Fetch a specific universal comparison by comparison_id."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM universal_comparisons WHERE comparison_id = ?", (comparison_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        raw_results = json.loads(row["results"] or "[]")
        results = [UniversalComparisonResult(**r) for r in raw_results]
        return UniversalComparison(
            comparison_id=row["comparison_id"],
            timestamp=row["timestamp"],
            question=row["question"],
            system_prompt=row["system_prompt"],
            reference_answer=row["reference_answer"],
            rubric=row["rubric"],
            evaluation_mode=row["evaluation_mode"],
            selected_models=json.loads(row["selected_models"] or "[]"),
            generation_config=json.loads(row["generation_config"] or "{}"),
            results=results,
            status=row["status"] or "completed",
            agreement_analysis=json.loads(row["agreement_analysis"] or "{}"),
            summary_notes=row["summary_notes"]
        )

    def list_comparisons(self) -> List[Dict[str, Any]]:
        """List summary info for all stored comparisons ordered by timestamp descending."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT comparison_id, timestamp, question, evaluation_mode, selected_models, status
            FROM universal_comparisons
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        comps = []
        for r in rows:
            comps.append({
                "comparison_id": r["comparison_id"],
                "timestamp": r["timestamp"],
                "question": r["question"],
                "evaluation_mode": r["evaluation_mode"],
                "selected_models": json.loads(r["selected_models"] or "[]"),
                "status": r["status"]
            })
        return comps

    def delete_comparison(self, comparison_id: str) -> None:
        """Delete a universal comparison record."""
        conn = self._get_conn()
        with conn:
            conn.execute("DELETE FROM universal_comparisons WHERE comparison_id = ?", (comparison_id,))
        conn.close()

