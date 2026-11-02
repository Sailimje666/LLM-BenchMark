"""SQLite database initialization, schemas, and connection management."""

import sqlite3
import os
from pathlib import Path
from typing import Optional
from config import DATABASE_PATH


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a thread-safe connection to SQLite with WAL mode enabled."""
    target_path = Path(db_path or DATABASE_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables with indexes and unique constraints."""
    conn = get_db_connection(db_path)
    with conn:
        # Table 1: Benchmark Runs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                name TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                dataset_version TEXT,
                dataset_hash TEXT,
                selected_models TEXT,
                benchmark_config TEXT,
                total_test_cases INTEGER DEFAULT 0,
                total_requests INTEGER DEFAULT 0,
                completed_requests INTEGER DEFAULT 0,
                failed_requests INTEGER DEFAULT 0,
                skipped_requests INTEGER DEFAULT 0,
                model_metrics TEXT,
                status TEXT DEFAULT 'running'
            );
        """)

        # Table 2: Evaluation Results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                test_case_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                response_text TEXT,
                accuracy_score REAL DEFAULT 0.0,
                is_correct INTEGER DEFAULT 0,
                evaluation_method TEXT,
                execution_time_ms REAL DEFAULT 0.0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0.0,
                error_category TEXT,
                error_subcategory TEXT,
                judge_reasoning TEXT,
                deterministic_reasoning TEXT,
                constraint_violations TEXT,
                provider_error TEXT,
                status TEXT DEFAULT 'success',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
                UNIQUE(run_id, test_case_id, model_name)
            );
        """)

        # Table 3: Stored Test Cases
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                id TEXT PRIMARY KEY,
                problem_statement TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                domain TEXT NOT NULL,
                category TEXT,
                difficulty TEXT,
                evaluation_method TEXT,
                expected_format TEXT,
                constraints TEXT,
                tolerance REAL,
                regex_pattern TEXT,
                metadata TEXT
            );
        """)

        # Table 4: Universal Question Comparisons
        conn.execute("""
            CREATE TABLE IF NOT EXISTS universal_comparisons (
                comparison_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                system_prompt TEXT,
                reference_answer TEXT,
                rubric TEXT,
                evaluation_mode TEXT NOT NULL,
                selected_models TEXT NOT NULL,
                generation_config TEXT,
                results TEXT NOT NULL,
                status TEXT DEFAULT 'completed',
                agreement_analysis TEXT,
                summary_notes TEXT
            );
        """)

        # Performance Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_run ON evaluation_results(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_model ON evaluation_results(model_name);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_category ON evaluation_results(error_category);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_universal_timestamp ON universal_comparisons(timestamp);")
    conn.close()
