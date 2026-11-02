"""Export utilities for benchmark results, summaries, CSV, JSON, Markdown, and HTML reports."""

import json
import io
from typing import List, Dict, Any
import pandas as pd
from schema import BenchmarkRun, EvaluationResult


def export_results_df(results: List[EvaluationResult]) -> pd.DataFrame:
    """Convert a list of EvaluationResult objects into a flattened pandas DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Run ID": r.run_id,
            "Test Case ID": r.test_case_id,
            "Provider": r.provider,
            "Model Name": r.model_name,
            "Accuracy Score": r.accuracy_score,
            "Is Correct": r.is_correct,
            "Evaluation Method": r.evaluation_method,
            "Execution Time (ms)": r.execution_time_ms,
            "Prompt Tokens": r.prompt_tokens,
            "Completion Tokens": r.completion_tokens,
            "Total Tokens": r.total_tokens,
            "Estimated Cost (USD)": r.estimated_cost_usd,
            "Primary Root Cause": r.error_category,
            "Secondary Root Cause": r.error_subcategory or "",
            "Deterministic Reasoning": r.deterministic_reasoning or "",
            "Judge Reasoning": r.judge_reasoning or "",
            "Constraint Violations": "; ".join(r.constraint_violations),
            "Provider Error": r.provider_error or "",
            "Status": r.status,
            "Response Text": r.response_text,
            "Timestamp": r.timestamp
        })
    return pd.DataFrame(rows)


def export_results_csv(results: List[EvaluationResult]) -> str:
    """Generate CSV string of benchmark evaluation results."""
    df = export_results_df(results)
    return df.to_csv(index=False)


def export_results_json(results: List[EvaluationResult]) -> str:
    """Generate structured JSON string of benchmark evaluation results."""
    raw_list = [r.model_dump() for r in results]
    return json.dumps(raw_list, indent=2)


def export_run_summary_json(run: BenchmarkRun, results: List[EvaluationResult]) -> str:
    """Generate a combined JSON export of the benchmark run configuration, metrics, and results."""
    combined = {
        "benchmark_run": run.model_dump(),
        "total_results_count": len(results),
        "results": [r.model_dump() for r in results]
    }
    return json.dumps(combined, indent=2)


def safe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Safely render DataFrame as Markdown table, with fallback to formatted string if tabulate is absent."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def generate_consolidated_markdown_report(run: BenchmarkRun, results: List[EvaluationResult]) -> str:
    """Generate an academic/portfolio-grade Markdown report of benchmark outcomes and error taxonomy."""
    df = export_results_df(results)

    md = []
    md.append(f"# LLM Benchmark & Diagnostics: Comparative Report")
    md.append(f"**Run Identifier:** `{run.run_id}`  ")
    md.append(f"**Execution Timestamp:** {run.start_time}  ")
    md.append(f"**Dataset Version / Hash:** {run.dataset_version} (`{run.dataset_hash or 'default'}`)  ")
    md.append(f"**Total Test Cases:** {run.total_test_cases} | **Total Requests:** {run.total_requests}  \n")

    md.append("## 1. Executive Summary & Model Leaderboard")
    if run.model_metrics:
        leaderboard_rows = []
        for model_name, metrics in run.model_metrics.items():
            leaderboard_rows.append({
                "Model": model_name,
                "Accuracy (%)": f"{metrics.get('binary_accuracy', 0.0):.1f}%",
                "Mean Score": f"{metrics.get('mean_score', 0.0):.3f}",
                "Success Rate": f"{metrics.get('success_rate', 0.0):.1f}%",
                "Median Latency (ms)": f"{metrics.get('median_latency_ms', 0.0):.0f}",
                "Avg Tokens": f"{metrics.get('avg_tokens_per_req', 0.0):.1f}",
                "Cost / 1K (USD)": f"${metrics.get('cost_per_1k_usd', 0.0):.4f}"
            })
        ldf = pd.DataFrame(leaderboard_rows)
        md.append(safe_to_markdown(ldf, index=False))
    else:
        md.append("No aggregate metrics recorded.")

    md.append("\n## 2. Root-Cause Error Analysis Taxonomy")
    if not df.empty and "Primary Root Cause" in df.columns:
        error_pivot = pd.crosstab(df["Model Name"], df["Primary Root Cause"])
        md.append(safe_to_markdown(error_pivot, index=True))
    else:
        md.append("No error records present.")

    md.append("\n## 3. Notable Failure Case Studies & Recommendations")
    failed_df = df[df["Is Correct"] == False].head(5) if not df.empty else pd.DataFrame()
    if not failed_df.empty:
        for _, row in failed_df.iterrows():
            md.append(f"### Test Case: `{row['Test Case ID']}` | Model: `{row['Model Name']}`")
            md.append(f"- **Primary Root Cause:** {row['Primary Root Cause']}")
            md.append(f"- **Deterministic Reasoning:** {row['Deterministic Reasoning']}")
            if row['Judge Reasoning']:
                md.append(f"- **Judge Reasoning:** {row['Judge Reasoning']}")
            md.append(f"- **Candidate Response Excerpt:** `{str(row['Response Text'])[:150]}...`\n")
    else:
        md.append("Zero failure cases recorded across evaluated test cases.")

    md.append("\n## 4. Methodology & Evaluation Protocol")
    md.append("- **Deterministic Grading:** Exact match, numeric tolerance check, regex matching, and JSON schema syntax validation.")
    md.append("- **Semantic Grading:** Automated LLM-as-a-Judge isolating untrusted candidate output.")
    md.append("- **Root-Cause Classification:** Rule-based hierarchical taxonomy categorizing failure modes.")

    return "\n".join(md)
