"""Regression tests for token usage analytics, DataFrame schema, missing columns, and empty datasets."""

import pytest
import pandas as pd
import plotly.graph_objects as go
from schema import EvaluationResult
from storage.exports import export_results_df


@pytest.fixture
def sample_evaluation_results():
    """Create sample evaluation results with realistic token counts."""
    return [
        EvaluationResult(
            run_id="run_test_001",
            test_case_id="tc_01",
            provider="openai",
            model_name="gpt-4o-mini",
            response_text="Answer 1",
            accuracy_score=1.0,
            is_correct=True,
            execution_time_ms=120.0,
            prompt_tokens=45,
            completion_tokens=15,
            total_tokens=60,
            estimated_cost_usd=0.0001
        ),
        EvaluationResult(
            run_id="run_test_001",
            test_case_id="tc_02",
            provider="openai",
            model_name="gpt-4o-mini",
            response_text="Answer 2",
            accuracy_score=1.0,
            is_correct=True,
            execution_time_ms=150.0,
            prompt_tokens=55,
            completion_tokens=25,
            total_tokens=80,
            estimated_cost_usd=0.00015
        ),
        EvaluationResult(
            run_id="run_test_001",
            test_case_id="tc_01",
            provider="mock",
            model_name="mock-gpt",
            response_text="Answer Mock 1",
            accuracy_score=1.0,
            is_correct=True,
            execution_time_ms=80.0,
            prompt_tokens=40,
            completion_tokens=10,
            total_tokens=50,
            estimated_cost_usd=0.0
        ),
    ]


def test_export_results_df_token_columns_exist(sample_evaluation_results):
    """Verify export_results_df contains the exact token usage columns expected by analytics."""
    df = export_results_df(sample_evaluation_results)
    assert "Prompt Tokens" in df.columns
    assert "Completion Tokens" in df.columns
    assert "Total Tokens" in df.columns
    assert "Model Name" in df.columns

    assert len(df) == 3
    assert df["Prompt Tokens"].tolist() == [45, 55, 40]
    assert df["Completion Tokens"].tolist() == [15, 25, 10]
    assert df["Total Tokens"].tolist() == [60, 80, 50]


def test_token_df_aggregation_and_column_access(sample_evaluation_results):
    """Verify token aggregation succeeds and both PromptTokens and 'Prompt Tokens' are accessible without KeyError."""
    df = export_results_df(sample_evaluation_results)

    prompt_col = next((c for c in ["Prompt Tokens", "PromptTokens", "prompt_tokens"] if c in df.columns), None)
    completion_col = next((c for c in ["Completion Tokens", "CompletionTokens", "completion_tokens"] if c in df.columns), None)

    assert prompt_col == "Prompt Tokens"
    assert completion_col == "Completion Tokens"

    token_df = df.groupby("Model Name").agg(
        PromptTokens=(prompt_col, "mean"),
        CompletionTokens=(completion_col, "mean")
    ).reset_index()

    token_df["Prompt Tokens"] = token_df["PromptTokens"]
    token_df["Completion Tokens"] = token_df["CompletionTokens"]

    # Verify both naming conventions can be accessed without KeyError
    assert "PromptTokens" in token_df.columns
    assert "Prompt Tokens" in token_df.columns
    assert "CompletionTokens" in token_df.columns
    assert "Completion Tokens" in token_df.columns

    # Verify gpt-4o-mini mean calculation
    gpt_row = token_df[token_df["Model Name"] == "gpt-4o-mini"].iloc[0]
    assert gpt_row["PromptTokens"] == 50.0
    assert gpt_row["Prompt Tokens"] == 50.0
    assert gpt_row["CompletionTokens"] == 20.0
    assert gpt_row["Completion Tokens"] == 20.0

    # Verify Plotly figure builds without KeyError
    fig = go.Figure(data=[
        go.Bar(name='Prompt Tokens', x=token_df['Model Name'], y=token_df['PromptTokens']),
        go.Bar(name='Completion Tokens', x=token_df['Model Name'], y=token_df['CompletionTokens'])
    ])
    assert len(fig.data) == 2
    assert fig.data[0].name == 'Prompt Tokens'
    assert fig.data[1].name == 'Completion Tokens'


def test_token_analytics_with_empty_dataframe():
    """Verify that empty DataFrame is handled safely without raising exceptions."""
    empty_df = pd.DataFrame(columns=["Model Name", "Prompt Tokens", "Completion Tokens"])

    prompt_col = next((c for c in ["Prompt Tokens", "PromptTokens", "prompt_tokens"] if c in empty_df.columns), None)
    completion_col = next((c for c in ["Completion Tokens", "CompletionTokens", "completion_tokens"] if c in empty_df.columns), None)

    assert prompt_col == "Prompt Tokens"
    assert completion_col == "Completion Tokens"
    assert empty_df.empty is True

    # Aggregating an empty dataframe should produce an empty dataframe
    token_df = empty_df.groupby("Model Name").agg(
        PromptTokens=(prompt_col, "mean"),
        CompletionTokens=(completion_col, "mean")
    ).reset_index()

    assert token_df.empty is True
    assert token_df["PromptTokens"].fillna(0).sum() == 0
    assert token_df["CompletionTokens"].fillna(0).sum() == 0


def test_token_analytics_missing_token_columns():
    """Verify graceful handling when token columns are completely absent from DataFrame."""
    incomplete_df = pd.DataFrame({
        "Model Name": ["gpt-4o", "mock-claude"],
        "Accuracy Score": [1.0, 0.8]
    })

    prompt_col = next((c for c in ["Prompt Tokens", "PromptTokens", "prompt_tokens"] if c in incomplete_df.columns), None)
    completion_col = next((c for c in ["Completion Tokens", "CompletionTokens", "completion_tokens"] if c in incomplete_df.columns), None)

    # Both should be None, indicating unavailable data
    assert prompt_col is None
    assert completion_col is None


def test_token_analytics_all_zero_tokens():
    """Verify detection when all models returned zero tokens."""
    zero_df = pd.DataFrame({
        "Model Name": ["model-a", "model-b"],
        "Prompt Tokens": [0, 0],
        "Completion Tokens": [0, 0]
    })

    token_df = zero_df.groupby("Model Name").agg(
        PromptTokens=("Prompt Tokens", "mean"),
        CompletionTokens=("Completion Tokens", "mean")
    ).reset_index()

    total_prompt = token_df["PromptTokens"].fillna(0).sum()
    total_completion = token_df["CompletionTokens"].fillna(0).sum()

    assert total_prompt == 0
    assert total_completion == 0
