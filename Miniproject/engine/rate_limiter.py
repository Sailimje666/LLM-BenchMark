"""Concurrency limiting, rate limiting, and exponential backoff retry policies."""

import asyncio
import random
import time
import logging
from typing import Dict, Any, Callable, Coroutine, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Bounded concurrency and backoff retry controller with provider-specific limits."""

    def __init__(
        self,
        max_concurrent: int = 5,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        jitter: bool = True
    ):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.provider_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.provider_last_called: Dict[str, float] = {}
        # Minimum time gap between consecutive requests to prevent HTTP 429
        self.provider_min_spacing: Dict[str, float] = {
            "nvidia": 0.15,
            "nemotron": 0.15,
            "deepseek": 0.15,
            "groq": 0.25,
            "ollama": 0.15,
            "mock": 0.01,
            "default": 0.05
        }
        # Concurrency caps per provider
        self.provider_concurrency_limits: Dict[str, int] = {
            "nvidia": 3,
            "nemotron": 3,
            "deepseek": 3,
            "groq": 3,
            "ollama": 2,
            "mock": 10
        }

    def get_provider_semaphore(self, provider: str, default_max: int = 3) -> asyncio.Semaphore:
        """Get or initialize provider-specific concurrency semaphore based on provider tier limits."""
        pkey = provider.lower().strip()
        if pkey not in self.provider_semaphores:
            limit = self.provider_concurrency_limits.get(pkey, default_max)
            self.provider_semaphores[pkey] = asyncio.Semaphore(limit)
        return self.provider_semaphores[pkey]

    async def execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        provider: Optional[str] = None,
        **kwargs: Any
    ) -> T:
        """Execute async function bounded by concurrency limits with exponential backoff and jitter."""
        prov_key = (provider or "default").lower().strip()
        prov_sem = self.get_provider_semaphore(prov_key)

        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    async with prov_sem:
                        # Enforce minimum spacing between calls to respect provider RPM
                        min_space = self.provider_min_spacing.get(prov_key, 0.0)
                        last_time = self.provider_last_called.get(prov_key, 0.0)
                        now = time.perf_counter()
                        if min_space > 0 and (now - last_time) < min_space:
                            await asyncio.sleep(min_space - (now - last_time))
                        self.provider_last_called[prov_key] = time.perf_counter()

                        return await func(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                # Permanent non-retryable errors
                is_auth_error = any(w in err_msg for w in ["unauthorized", "invalid api key", "forbidden", "401", "403", "api_key_invalid", "api key not valid"])
                is_invalid_request = any(w in err_msg for w in ["bad request", "invalid model", "404", "model not found", "not supported"])
                is_quota_exhausted = any(w in err_msg for w in ["insufficient_quota", "no credits remaining", "exceeded your current quota", "billing details", "quota", "resource_exhausted", "quota_exceeded"])

                if is_auth_error or is_invalid_request or is_quota_exhausted or attempt == self.max_retries:
                    # Do not retry fatal errors
                    raise e

                # Calculate exponential backoff sleep with jitter
                delay = self.backoff_factor * (2 ** attempt)
                if self.jitter:
                    delay += random.uniform(0.1, 0.5)

                logger.warning(
                    "Request attempt %d/%d failed for provider %s: %s. Retrying in %.2fs...",
                    attempt + 1, self.max_retries, provider, str(e), delay
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Exhausted retry attempts.")
