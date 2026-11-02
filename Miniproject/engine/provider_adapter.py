"""Unified model provider adapters for NVIDIA Nemotron, DeepSeek, Groq, Ollama, and Mock."""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import httpx
import config
from config import GROQ_API_KEY, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class BaseProviderAdapter(ABC):
    """Abstract base class for all LLM provider integrations."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Generate a completion for the provided prompt."""
        pass


class OpenAIAdapter(BaseProviderAdapter):
    """Deprecated adapter for removed OpenAI provider."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        raise RuntimeError(
            "Provider 'openai' has been removed from active service. "
            "Active supported providers are: NVIDIA Nemotron ('nvidia'), DeepSeek ('deepseek'), "
            "Groq ('groq'), Ollama ('ollama'), and Mock ('mock')."
        )

    async def generate(self, *args, **kwargs) -> Dict[str, Any]:
        raise RuntimeError("Provider 'openai' has been removed from active service.")


class GroqAdapter(BaseProviderAdapter):
    """Adapter for Groq cloud API."""

    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: Optional[str] = None):
        super().__init__(model_name, api_key or GROQ_API_KEY)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("Groq API key not configured. Set GROQ_API_KEY in .env or environment.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.base_url, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_detail = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_detail = resp.text
                if resp.status_code == 404:
                    raise RuntimeError(f"Groq API Error (404): The model `{self.model_name}` does not exist or you do not have access to it. Verified active models on this account include `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. Raw error: {err_detail}")
                raise RuntimeError(f"Groq API Error ({resp.status_code}): {err_detail}")

            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return {
                "text": choice["message"]["content"],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": round(elapsed_ms, 2),
                "finish_reason": choice.get("finish_reason", "stop"),
                "error": None
            }


class XAIGrokAdapter(BaseProviderAdapter):
    """Deprecated adapter for removed xAI Grok provider."""

    def __init__(self, model_name: str = "grok-2-1212", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        raise RuntimeError(
            "Provider 'xai' has been removed from active service. "
            "Active supported providers are: NVIDIA Nemotron ('nvidia'), DeepSeek ('deepseek'), "
            "Groq ('groq'), Ollama ('ollama'), and Mock ('mock')."
        )

    async def generate(self, *args, **kwargs) -> Dict[str, Any]:
        raise RuntimeError("Provider 'xai' has been removed from active service.")


class NVIDIANemotronAdapter(BaseProviderAdapter):
    """Adapter for NVIDIA NIM cloud API (OpenAI-compatible chat completions endpoint)."""

    def __init__(self, model_name: str = "nvidia/nemotron-3-super-120b-a12b", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        custom_base = (os.getenv("NVIDIA_BASE_URL") or config.NVIDIA_BASE_URL).rstrip("/")
        if custom_base.endswith("/chat/completions"):
            self.base_url = custom_base
        else:
            self.base_url = f"{custom_base}/chat/completions"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60
    ) -> Dict[str, Any]:
        active_key = self.api_key if self.api_key is not None else config.get_api_key("nvidia")
        if not active_key:
            raise ValueError("NVIDIA API key not configured. Add NVIDIA_API_KEY to your .env file or environment.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.base_url, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_detail = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_detail = resp.text
                if resp.status_code == 401:
                    raise RuntimeError(f"NVIDIA API Error (401): Unauthorized - invalid NVIDIA API key. Details: {err_detail}")
                if resp.status_code == 403:
                    raise RuntimeError(f"NVIDIA API Error (403): Forbidden - permission denied or invalid credentials. Details: {err_detail}")
                if resp.status_code == 404:
                    raise RuntimeError(f"NVIDIA API Error (404): The model `{self.model_name}` does not exist or you do not have access to it in the NVIDIA API catalog. Details: {err_detail}")
                if resp.status_code == 429:
                    raise RuntimeError(f"NVIDIA API Error (429): Rate limit or quota exhausted. Details: {err_detail}")
                raise RuntimeError(f"NVIDIA API Error ({resp.status_code}): {err_detail}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("NVIDIA API Error: No candidate choices returned in response.")

            choice = choices[0]
            usage = data.get("usage", {})
            prompt_toks = usage.get("prompt_tokens", 0)
            comp_toks = usage.get("completion_tokens", 0)
            tot_toks = usage.get("total_tokens", prompt_toks + comp_toks)

            msg = choice.get("message", {})
            text = msg.get("content") or ""
            if not text and msg.get("reasoning_content"):
                text = msg["reasoning_content"]

            return {
                "text": text,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": tot_toks,
                "latency_ms": round(elapsed_ms, 2),
                "finish_reason": choice.get("finish_reason", "stop"),
                "model": data.get("model", self.model_name),
                "error": None
            }


class DeepSeekAdapter(BaseProviderAdapter):
    """Adapter for official DeepSeek API (OpenAI-compatible chat completions endpoint)."""

    def __init__(self, model_name: str = "deepseek-flash", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        custom_base = (os.getenv("DEEPSEEK_BASE_URL") or config.DEEPSEEK_BASE_URL).rstrip("/")
        if custom_base.endswith("/chat/completions"):
            self.base_url = custom_base
        else:
            self.base_url = f"{custom_base}/chat/completions"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60
    ) -> Dict[str, Any]:
        active_key = self.api_key if self.api_key is not None else config.get_api_key("deepseek")
        if not active_key:
            raise ValueError("DeepSeek API key not configured. Add DEEPSEEK_API_KEY to your .env file or environment.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # DeepSeek reasoning models do not accept temperature; standard models accept temperature
        is_reasoning_model = any(k in self.model_name.lower() for k in ["reasoner", "r1", "thinking"])

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens
        }
        if not is_reasoning_model:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.base_url, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_detail = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_detail = resp.text
                if resp.status_code == 401:
                    raise RuntimeError(f"DeepSeek API Error (401): Unauthorized - invalid DeepSeek API key. Details: {err_detail}")
                if resp.status_code in (402, 403):
                    raise RuntimeError(f"DeepSeek API Error ({resp.status_code}): Insufficient balance or permission denied. Please top up your account at platform.deepseek.com. Details: {err_detail}")
                if resp.status_code == 404:
                    raise RuntimeError(f"DeepSeek API Error (404): The model `{self.model_name}` does not exist. Verified models include `deepseek-chat` and `deepseek-reasoner`. Details: {err_detail}")
                if resp.status_code == 429:
                    raise RuntimeError(f"DeepSeek API Error (429): Rate limit or quota exhausted. Details: {err_detail}")
                raise RuntimeError(f"DeepSeek API Error ({resp.status_code}): {err_detail}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("DeepSeek API Error: No candidate choices returned in response.")

            choice = choices[0]
            usage = data.get("usage", {})
            prompt_toks = usage.get("prompt_tokens", 0)
            comp_toks = usage.get("completion_tokens", 0)
            tot_toks = usage.get("total_tokens", prompt_toks + comp_toks)

            msg = choice.get("message", {})
            text = msg.get("content", "") or ""
            # Support reasoning_content if content is empty in pure reasoning mode
            if not text and msg.get("reasoning_content"):
                text = msg["reasoning_content"]

            return {
                "text": text,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": tot_toks,
                "latency_ms": round(elapsed_ms, 2),
                "finish_reason": choice.get("finish_reason", "stop"),
                "model": data.get("model", self.model_name),
                "error": None
            }


class GoogleGeminiAdapter(BaseProviderAdapter):
    """Deprecated adapter for removed Google Gemini provider."""

    def __init__(self, model_name: str = "gemini-3.8-flash", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        raise RuntimeError(
            "Provider 'google' has been removed from active service. "
            "Active supported providers are: NVIDIA Nemotron ('nvidia'), DeepSeek ('deepseek'), "
            "Groq ('groq'), Ollama ('ollama'), and Mock ('mock')."
        )

    async def generate(self, *args, **kwargs) -> Dict[str, Any]:
        raise RuntimeError("Provider 'google' has been removed from active service.")


class OllamaAdapter(BaseProviderAdapter):
    """Adapter for locally hosted Ollama instances."""

    def __init__(self, model_name: str = "llama3:8b", base_url: Optional[str] = None):
        super().__init__(model_name)
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

    @staticmethod
    async def check_connection_static(base_url: Optional[str] = None, timeout: float = 2.0) -> Tuple[bool, str]:
        """Check if the local Ollama daemon is running and reachable."""
        endpoint = (base_url or OLLAMA_BASE_URL).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{endpoint}/api/version")
                if resp.status_code == 200:
                    ver = resp.json().get("version", "unknown")
                    return True, f"Ollama is online (version {ver}) at {endpoint}."
                return False, f"Ollama server returned HTTP {resp.status_code} at {endpoint}."
        except Exception:
            return False, f"Ollama local server is unavailable at {endpoint}. Ensure the Ollama daemon is running ('ollama serve') before requesting local models."

    async def check_connection(self, timeout: float = 2.0) -> Tuple[bool, str]:
        """Instance method checking connection to self.base_url."""
        return await self.check_connection_static(self.base_url, timeout=timeout)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60
    ) -> Dict[str, Any]:
        # Pre-flight check: verify local Ollama connectivity before attempting request
        is_alive, check_msg = await self.check_connection(timeout=2.0)
        if not is_alive:
            raise RuntimeError(check_msg)

        endpoint = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama Error ({resp.status_code}): {resp.text}")

                data = resp.json()
                return {
                    "text": data.get("message", {}).get("content", ""),
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "latency_ms": round(elapsed_ms, 2),
                    "finish_reason": data.get("done_reason", "stop"),
                    "error": None
                }
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama local server is unavailable at {self.base_url}. Ensure the Ollama daemon is running ('ollama serve') before requesting local models.")


class MockProviderAdapter(BaseProviderAdapter):
    """Simulated provider adapter for offline demo, rapid experimentation, and automated testing."""

    def __init__(self, model_name: str = "mock-gpt", accuracy_tier: str = "high"):
        super().__init__(model_name)
        self.accuracy_tier = accuracy_tier

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60
    ) -> Dict[str, Any]:
        import asyncio
        # Simulate realistic API round-trip network latency
        simulated_delay = 0.08 if "fast" in self.model_name else 0.15
        await asyncio.sleep(simulated_delay)

        prompt_tokens = max(10, len(prompt.split()) * 2)

        # Smart mock response generation based on prompt content
        clean_p = prompt.lower()
        ans = "42"

        # Evaluation Judge prompt simulation
        if "### evaluation task" in clean_p or "<candidate_response>" in clean_p:
            import re
            cand_match = re.search(r"<candidate_response>([\s\S]*?)</candidate_response>", prompt)
            cand_text = cand_match.group(1).strip() if cand_match else ""
            is_valid = bool(cand_text and len(cand_text) > 2 and "error" not in cand_text.lower())
            ans = json.dumps({
                "score": 1.0 if is_valid else 0.0,
                "is_correct": is_valid,
                "reasoning": f"Simulated judge verdict: candidate response verified ({len(cand_text)} chars).",
                "factual_errors": [] if is_valid else ["Empty or invalid candidate output"],
                "instruction_violations": [],
                "reasoning_errors": [],
                "truncation_detected": False,
                "formatting_error": False
            })

        # Math test cases
        elif "15% of 240" in clean_p or ("15%" in clean_p and "240" in clean_p) or ("15 percent" in clean_p and "240" in clean_p):
            ans = "36"
        elif "47 multiplied by 18" in clean_p:
            ans = "846"
        elif "discounted by 25%" in clean_p:
            ans = "90"
        elif "ratio of boys to girls is 3:2" in clean_p:
            ans = "21"
        elif "150 miles at 50 mph" in clean_p:
            ans = "54"
        elif "meters are in 3.5 kilometers" in clean_p:
            ans = "3500"
        elif "20 heads and 56 legs" in clean_p:
            ans = "8" if "mock-llama" not in self.model_name else "12"  # mock-llama makes arithmetic mistake!
        elif "volume of a sphere with radius 3" in clean_p:
            ans = "113.10"
        elif "3x - 7 = 2x + 15" in clean_p:
            ans = "22"
        elif "sum is 7" in clean_p:
            ans = "0.1667"
        elif "bat and a ball" in clean_p:
            # Classic cognitive reflection problem: inferior model says 10 (reasoning drift), superior says 5
            ans = "5" if "mock-gpt" in self.model_name else "10"
        elif "2 raised to the power of 8" in clean_p:
            ans = "256"
        
        # Logic test cases
        elif "roses are flowers" in clean_p:
            ans = "No"
        elif "alice is taller than bob" in clean_p:
            ans = "Alice, Bob, Charlie"
        elif "2, 6, 12, 20, 30" in clean_p:
            ans = "42"
        elif "contraposition" in clean_p or "marked fragile" in clean_p:
            ans = "No"
        elif "all cats are mammals" in clean_p:
            ans = "No"
        elif "knave" in clean_p and "paradox" in clean_p:
            ans = "Paradox"
        elif "yesterday was tomorrow" in clean_p:
            ans = "Sunday"
        elif "five houses in a row" in clean_p:
            ans = "White"
        elif "not (true and false)" in clean_p:
            ans = "True"
        elif "heavier than y" in clean_p:
            ans = "Z"
        elif "every square a quadrilateral" in clean_p:
            ans = "Yes"

        # Coding test cases
        elif "reverse_words" in clean_p:
            ans = "def reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])"
        elif "filter_even" in clean_p:
            ans = "def filter_even(numbers: list[int]) -> list[int]:\n    return [n for n in numbers if n % 2 == 0]"
        elif "invert_dict" in clean_p:
            ans = "def invert_dict(d: dict) -> dict:\n    return {v: k for k, v in d.items()}"
        elif "safe_divide" in clean_p:
            ans = "def safe_divide(a: float, b: float) -> float | None:\n    if b == 0:\n        return None\n    return a / b"
        elif "quicksort" in clean_p and "lomuto" in clean_p:
            ans = "O(n^2)"
        elif "10-digit us phone number" in clean_p:
            ans = "^\\d{3}-\\d{3}-\\d{4}$"
        elif "is_valid_brackets" in clean_p:
            ans = "def is_valid_brackets(s: str) -> bool:\n    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n    return len(stack) == 0"
        elif "fib(n: int)" in clean_p:
            ans = "def fib(n: int) -> int:\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
        elif "binary search" in clean_p:
            ans = "O(log n)"
        elif "employees where 'salary' is strictly greater" in clean_p:
            ans = "SELECT * FROM employees WHERE salary > 80000 ORDER BY last_name ASC;"

        # Extraction test cases
        elif "sarah at" in clean_p and "apex-corp" in clean_p:
            ans = "support.billing@apex-corp.org"
        elif "quarterly review will be held" in clean_p:
            ans = "2026-11-15"
        elif "raymond c. vance" in clean_p:
            ans = "Raymond C. Vance"
        elif "invoice #84920" in clean_p:
            ans = "486.00"
        elif "jane doe is a 34-year-old" in clean_p:
            if "mock-claude" in self.model_name and "padding" in self.model_name:
                ans = "Here is the JSON:\n{\"name\": \"Jane Doe\", \"age\": 34, \"occupation\": \"Data Scientist\", \"city\": \"Seattle\"}"
            else:
                ans = '{"name": "Jane Doe", "age": 34, "occupation": "Data Scientist", "city": "Seattle"}'
        elif "primary programming language" in clean_p:
            ans = "TypeScript"
        elif "carlos ruiz" in clean_p:
            ans = "N/A"
        elif "ord-99214-x" in clean_p:
            ans = "ORD-99214-X"
        elif "purchased 2 laptops" in clean_p:
            ans = '[{"item": "laptop", "quantity": 2, "price": 1200}, {"item": "mouse", "quantity": 3, "price": 25}]'
        elif "flight aa1422" in clean_p:
            ans = "AA1422"

        completion_tokens = max(5, len(ans.split()) * 2)

        return {
            "text": ans,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(simulated_delay * 1000.0, 2),
            "finish_reason": "stop",
            "error": None
        }
