"""BenchmarkRunner coordinating concurrent multi-model execution, incremental persistence, and evaluation."""

import asyncio
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable

from schema import TestCase, ModelConfig, EvaluationResult, BenchmarkRun
from config import get_token_pricing
from engine.rate_limiter import RateLimiter
from engine.model_registry import get_adapter
from engine.telemetry import calculate_cost, compute_model_statistics
from diagnostics.evaluator import evaluate_response
from storage.repository import BenchmarkRepository

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrator for standardized multi-model LLM benchmarking."""

    def __init__(
        self,
        repository: Optional[BenchmarkRepository] = None,
        max_concurrent: int = 5,
        max_retries: int = 2
    ):
        self.repository = repository or BenchmarkRepository()
        self.rate_limiter = RateLimiter(max_concurrent=max_concurrent, max_retries=max_retries)
        self.is_cancelled = False

    def cancel(self) -> None:
        """Signal runner to halt further task dispatching gracefully."""
        self.is_cancelled = True

    async def _execute_single_request(
        self,
        test_case: TestCase,
        model_cfg: ModelConfig,
        run_id: str,
        system_prompt: Optional[str] = None,
        enable_semantic_judge: bool = True,
        judge_adapter: Optional[Any] = None,
        judge_model: str = "gpt-4o-mini"
    ) -> EvaluationResult:
        """Execute one model request against one test case with retry, telemetry, and evaluation."""
        pricing = get_token_pricing(model_cfg.model_name)
        price_in = model_cfg.cost_input_per_1m_tokens or pricing["input"]
        price_out = model_cfg.cost_output_per_1m_tokens or pricing["output"]

        try:
            adapter = get_adapter(model_cfg.provider, model_cfg.model_name)
        except Exception as e:
            eval_result = await evaluate_response(
                test_case=test_case,
                response_text="",
                run_id=run_id,
                provider=model_cfg.provider,
                model_name=model_cfg.model_name,
                provider_error=f"Adapter init failed: {str(e)}"
            )
            try:
                self.repository.save_evaluation_result(eval_result)
            except Exception as se:
                logger.error("Failed saving adapter error result: %s", str(se))
            return eval_result

        # Async generator call wrapped in rate limiter retry policy
        async def call_model():
            return await adapter.generate(
                prompt=test_case.problem_statement,
                system_prompt=system_prompt,
                temperature=model_cfg.temperature,
                max_tokens=model_cfg.max_completion_tokens,
                timeout=model_cfg.timeout_seconds
            )

        try:
            gen_res = await self.rate_limiter.execute_with_retry(
                call_model,
                provider=model_cfg.provider
            )
            raw_text = gen_res.get("text", "")
            prompt_toks = gen_res.get("prompt_tokens", 0)
            comp_toks = gen_res.get("completion_tokens", 0)
            lat_ms = gen_res.get("latency_ms", 0.0)
            fin_reason = gen_res.get("finish_reason", "stop")
            prov_err = gen_res.get("error")
        except Exception as e:
            logger.error("Request execution failed for %s on %s: %s", model_cfg.model_name, test_case.id, str(e))
            raw_text = ""
            prompt_toks = comp_toks = 0
            lat_ms = 0.0
            fin_reason = None
            prov_err = str(e)

        cost_usd = calculate_cost(prompt_toks, comp_toks, price_in, price_out)

        # Evaluate response and diagnose root causes
        eval_result = await evaluate_response(
            test_case=test_case,
            response_text=raw_text,
            run_id=run_id,
            provider=model_cfg.provider,
            model_name=model_cfg.model_name,
            execution_time_ms=lat_ms,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            estimated_cost_usd=cost_usd,
            provider_error=prov_err,
            finish_reason=fin_reason,
            enable_semantic_judge=enable_semantic_judge,
            judge_adapter=judge_adapter,
            judge_model=judge_model
        )

        # Incremental persistence: Save each result immediately to SQLite
        try:
            self.repository.save_evaluation_result(eval_result)
        except Exception as e:
            logger.error("Failed to persist evaluation result: %s", str(e))

        return eval_result

    async def run_benchmark(
        self,
        test_cases: List[TestCase],
        model_configs: List[ModelConfig],
        run_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> BenchmarkRun:
        """Execute benchmark suite across multiple models and return aggregated BenchmarkRun."""
        self.is_cancelled = False
        cfg = run_config or {}
        run_id = cfg.get("run_id", f"run_{uuid.uuid4().hex[:8]}")
        run_name = cfg.get("name", f"Benchmark Run - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
        system_prompt = cfg.get("system_prompt", None)
        enable_semantic_judge = cfg.get("enable_semantic_judge", True)
        judge_provider = cfg.get("judge_provider", "mock")
        judge_model = cfg.get("judge_model", "mock-judge-v1")

        # Prepare judge adapter if enabled
        judge_adapter = None
        if enable_semantic_judge and judge_provider != "none":
            try:
                judge_adapter = get_adapter(judge_provider, judge_model)
            except Exception as e:
                logger.warning("Could not instantiate judge adapter: %s", str(e))

        start_time = datetime.now(timezone.utc).isoformat()
        selected_model_names = [m.model_name for m in model_configs if m.enabled]
        total_requests = len(test_cases) * len(selected_model_names)

        # Initial benchmark run record
        run = BenchmarkRun(
            run_id=run_id,
            name=run_name,
            start_time=start_time,
            dataset_version=cfg.get("dataset_version", "1.0"),
            dataset_hash=cfg.get("dataset_hash", "default"),
            selected_models=selected_model_names,
            benchmark_config=cfg,
            total_test_cases=len(test_cases),
            total_requests=total_requests,
            completed_requests=0,
            failed_requests=0,
            skipped_requests=0,
            status="running"
        )
        self.repository.create_run(run)

        # Construct flat task list
        tasks_meta = []
        for tc in test_cases:
            for m in model_configs:
                if m.enabled:
                    tasks_meta.append((tc, m))

        all_results: List[EvaluationResult] = []
        completed_count = 0
        failed_count = 0

        # Run with worker queue to provide steady concurrency and progress updates
        async def worker(item):
            nonlocal completed_count, failed_count
            if self.is_cancelled:
                return None
            tc, m = item
            res = await self._execute_single_request(
                test_case=tc,
                model_cfg=m,
                run_id=run_id,
                system_prompt=system_prompt,
                enable_semantic_judge=enable_semantic_judge,
                judge_adapter=judge_adapter,
                judge_model=judge_model
            )
            completed_count += 1
            if res.status != "success":
                failed_count += 1

            if progress_callback:
                status_msg = f"Finished {m.model_name} on {tc.id} ({res.error_category})"
                progress_callback(completed_count, total_requests, status_msg)
            return res

        # Execute all tasks concurrently up to rate limiter max_concurrent
        coros = [worker(item) for item in tasks_meta]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        for r in raw_results:
            if isinstance(r, EvaluationResult):
                all_results.append(r)
            elif isinstance(r, Exception):
                print(f"UNCAUGHT WORKER EXCEPTION: {type(r).__name__}: {str(r)}")
                logger.error("Worker raised uncaught exception: %s", str(r))
                failed_count += 1

        # Compute per-model aggregated statistics
        model_metrics: Dict[str, Any] = {}
        for m_name in selected_model_names:
            m_res = [r for r in all_results if r.model_name == m_name]
            model_metrics[m_name] = compute_model_statistics(m_res)

        end_time = datetime.now(timezone.utc).isoformat()
        successful_count = sum(1 for r in all_results if r.status == "success")

        if self.is_cancelled:
            final_status = "interrupted"
        elif failed_count == 0:
            final_status = "completed"
        elif successful_count > 0:
            final_status = "partial"
        else:
            final_status = "failed"

        run.end_time = end_time
        run.completed_requests = successful_count
        run.failed_requests = failed_count
        run.skipped_requests = total_requests - len(all_results)
        run.model_metrics = model_metrics
        run.status = final_status

        self.repository.update_run(run)
        return run
