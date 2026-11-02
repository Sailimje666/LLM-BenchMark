"""End-to-end integration and verification script testing complete workflow without external API keys."""

import os
import json
import asyncio
import pytest
from pathlib import Path
from schema import TestCase, ModelConfig
from engine.runner import BenchmarkRunner
from storage.repository import BenchmarkRepository
from storage.exports import export_results_csv, export_results_json, generate_consolidated_markdown_report


@pytest.mark.asyncio
async def test_end_to_end_benchmark_and_diagnostics_pipeline(tmp_path):
    test_db = str(tmp_path / "e2e_benchmark.db")
    repo = BenchmarkRepository(test_db)
    runner = BenchmarkRunner(repository=repo, max_concurrent=3, max_retries=2)

    # 1. Load test cases from default dataset
    default_json = Path(__file__).resolve().parent.parent / "datasets" / "default_test_cases.json"
    assert default_json.exists()
    with open(default_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Take 4 diverse test cases (Math, Logic, Coding, Text Extraction)
    test_cases = [TestCase(**c) for c in data if c["id"] in ["math_01", "logic_01", "code_01", "extract_01"]]
    assert len(test_cases) == 4

    # 2. Configure mock models
    model_configs = [
        ModelConfig(provider="mock", model_name="mock-gpt", enabled=True),
        ModelConfig(provider="mock", model_name="mock-llama", enabled=True)
    ]

    # 3. Track progress callback
    progress_updates = []
    def on_prog(done, total, msg):
        progress_updates.append((done, total, msg))

    # 4. Execute benchmark run
    run = await runner.run_benchmark(
        test_cases=test_cases,
        model_configs=model_configs,
        run_config={
            "run_id": "e2e_run_01",
            "name": "E2E Verification Run",
            "enable_semantic_judge": True,
            "judge_provider": "mock",
            "judge_model": "mock-judge-v1"
        },
        progress_callback=on_prog
    )

    # 5. Assert run properties
    assert run.run_id == "e2e_run_01"
    assert run.status == "completed"
    assert run.total_requests == 8  # 4 test cases * 2 models
    assert run.completed_requests == 8
    assert len(progress_updates) == 8

    # 6. Verify incremental SQLite persistence
    stored_run = repo.get_run("e2e_run_01")
    assert stored_run is not None
    assert stored_run.name == "E2E Verification Run"

    stored_results = repo.get_results_for_run("e2e_run_01")
    assert len(stored_results) == 8

    # 7. Verify evaluations and root cause classifications
    for res in stored_results:
        assert res.run_id == "e2e_run_01"
        assert res.model_name in ["mock-gpt", "mock-llama"]
        assert 0.0 <= res.accuracy_score <= 1.0
        assert res.error_category in [
            "Success", "Instruction Ignored", "Hallucination / Fact Drift",
            "Arithmetic / Logic Failure", "Truncation / Length Error",
            "Formatting Mismatch", "Knowledge Gap", "Tool / Execution Failure",
            "Provider / API Error", "Unclassified"
        ]
        assert res.execution_time_ms >= 0.0
        assert res.total_tokens >= 0
        assert res.estimated_cost_usd >= 0.0

    # 8. Test exports from stored results
    csv_str = export_results_csv(stored_results)
    assert "Run ID" in csv_str
    assert "mock-gpt" in csv_str
    assert "e2e_run_01" in csv_str

    json_str = export_results_json(stored_results)
    parsed_json = json.loads(json_str)
    assert len(parsed_json) == 8

    md_report = generate_consolidated_markdown_report(stored_run, stored_results)
    assert "# LLM Benchmark & Diagnostics: Comparative Report" in md_report
    assert "mock-gpt" in md_report
    assert "mock-llama" in md_report
