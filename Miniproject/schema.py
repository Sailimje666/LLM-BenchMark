"""Strongly typed Pydantic models for LLM Benchmark & Diagnostics."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class TestCase(BaseModel):
    """Represents a standardized benchmark test case."""
    __test__ = False
    id: str = Field(..., description="Unique test case identifier")
    problem_statement: str = Field(..., description="Complete input question or prompt")
    ground_truth: str = Field(..., description="Expected answer or reference solution")
    domain: str = Field(..., description="Domain: Math, Logic, Coding, Text Extraction, etc.")
    category: Optional[str] = Field(default="", description="Finer-grained classification or subtopic")
    difficulty: str = Field(default="Medium", description="Easy, Medium, Hard, or Expert")
    evaluation_method: str = Field(
        default="auto",
        description="exact_match, numeric_tolerance, regex, semantic, json_schema, or auto"
    )
    expected_format: Optional[str] = Field(default="", description="Required format or structure")
    constraints: List[str] = Field(default_factory=list, description="Mandatory constraints or negative instructions")
    tolerance: Optional[float] = Field(default=0.001, description="Numeric tolerance if applicable")
    regex_pattern: Optional[str] = Field(default="", description="Regular expression pattern if regex method")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        valid_difficulties = {"Easy", "Medium", "Hard", "Expert"}
        v_title = v.strip().title()
        if v_title not in valid_difficulties:
            raise ValueError(f"Invalid difficulty '{v}'. Must be one of {valid_difficulties}")
        return v_title

    @field_validator("evaluation_method")
    @classmethod
    def validate_eval_method(cls, v: str) -> str:
        valid_methods = {"exact_match", "numeric_tolerance", "regex", "semantic", "json_schema", "auto"}
        v_clean = v.strip().lower().replace(" ", "_")
        if v_clean not in valid_methods:
            raise ValueError(f"Invalid evaluation method '{v}'. Must be one of {valid_methods}")
        return v_clean


class ModelConfig(BaseModel):
    """Configuration for an individual model candidate."""
    provider: str = Field(..., description="nvidia, deepseek, groq, ollama, or mock")
    model_name: str = Field(..., description="Model identifier e.g. nvidia/llama-3.1-nemotron-70b-instruct, deepseek-chat")
    enabled: bool = Field(default=True, description="Whether to include in the benchmark run")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_completion_tokens: int = Field(default=1024, ge=1, le=32768)
    timeout_seconds: int = Field(default=60, ge=5, le=300)
    retry_count: int = Field(default=3, ge=0, le=5)
    cost_input_per_1m_tokens: float = Field(default=0.0, ge=0.0)
    cost_output_per_1m_tokens: float = Field(default=0.0, ge=0.0)


class JudgeOutput(BaseModel):
    """Structured response from LLM-as-a-Judge."""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_correct: bool = Field(default=False)
    reasoning: str = Field(default="", description="Concise explanation for the verdict")
    factual_errors: List[str] = Field(default_factory=list)
    instruction_violations: List[str] = Field(default_factory=list)
    reasoning_errors: List[str] = Field(default_factory=list)
    truncation_detected: bool = Field(default=False)
    formatting_error: bool = Field(default=False)


class RootCauseDiagnosis(BaseModel):
    """Diagnostic categorization of model output error."""
    primary_category: str = Field(..., description="Primary error category")
    secondary_category: Optional[str] = Field(default=None)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="Supporting factual or behavioral evidence")
    recommended_action: str = Field(default="", description="Actionable recommendation for improvement")


class EvaluationResult(BaseModel):
    """Detailed record for a single model request on a single test case."""
    run_id: str
    test_case_id: str
    provider: str
    model_name: str
    response_text: str = ""
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_correct: bool = False
    evaluation_method: str = "auto"
    execution_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error_category: str = "Success"
    error_subcategory: Optional[str] = None
    judge_reasoning: Optional[str] = None
    deterministic_reasoning: Optional[str] = None
    constraint_violations: List[str] = Field(default_factory=list)
    provider_error: Optional[str] = None
    status: str = "success"  # success, provider_error, timeout, skipped
    timestamp: str = ""


class BenchmarkRun(BaseModel):
    """Metadata and aggregate overview of a benchmark experiment."""
    run_id: str
    name: Optional[str] = None
    start_time: str
    end_time: Optional[str] = None
    dataset_version: Optional[str] = "1.0.0"
    dataset_hash: Optional[str] = None
    selected_models: List[str] = Field(default_factory=list)
    benchmark_config: Dict[str, Any] = Field(default_factory=dict)
    total_test_cases: int = 0
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    skipped_requests: int = 0
    model_metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"  # running, completed, interrupted, failed


class UniversalComparisonResult(BaseModel):
    """Detailed result of a single model's execution on an ad-hoc universal question."""
    model_id: str
    provider: str
    display_name: str = ""
    is_simulated: bool = False
    status: str = "success"  # success, provider_error, timeout, unconfigured
    response_text: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    pricing_assumptions: str = ""
    finish_reason: Optional[str] = "stop"
    error_message: Optional[str] = None
    sanitized_error: Optional[str] = None
    evaluation_mode: str = "unreferenced"  # reference, rubric, unreferenced
    evaluation_method: str = "none"
    accuracy_score: Optional[float] = None  # None when unreferenced or invalid
    is_correct: Optional[bool] = None
    score_explanation: Optional[str] = None
    rubric_scores: Dict[str, float] = Field(default_factory=dict)
    missing_key_points: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    completeness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    instruction_following_score: Optional[float] = None
    uncertainty_detected: bool = False
    uncertainty_indicators: List[str] = Field(default_factory=list)
    timestamp: str = ""


class UniversalComparison(BaseModel):
    """Container for a multi-model comparative question run."""
    comparison_id: str
    timestamp: str
    question: str
    system_prompt: Optional[str] = None
    reference_answer: Optional[str] = None
    rubric: Optional[str] = None
    evaluation_mode: str = "unreferenced"  # reference, rubric, unreferenced
    selected_models: List[str] = Field(default_factory=list)
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    results: List[UniversalComparisonResult] = Field(default_factory=list)
    status: str = "completed"  # completed, partial_failure, failed
    agreement_analysis: Dict[str, Any] = Field(default_factory=dict)
    summary_notes: Optional[str] = None

