"""Centralized application configuration for LLM Benchmark & Diagnostics."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "benchmark.db"))

# Concurrency & Network configuration
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "1.5"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Judge Model Configuration
DEFAULT_JUDGE_PROVIDER = os.getenv("DEFAULT_JUDGE_PROVIDER", "mock")
DEFAULT_JUDGE_MODEL = os.getenv("DEFAULT_JUDGE_MODEL", "mock-judge-v1")

# Ollama local settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Provider API Keys & Endpoints (Read exclusively from env vars)
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Retired providers list
RETIRED_PROVIDERS = {"openai", "google", "gemini", "xai", "grok"}

# Default Model Parameters
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_MAX_COMPLETION_TOKENS", "1024"))

# Per-model pricing per 1M tokens in USD (Input, Output)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # NVIDIA Nemotron Models
    "nvidia/llama-3.1-nemotron-70b-instruct": {"input": 0.35, "output": 0.80},
    "llama-3.1-nemotron-70b-instruct": {"input": 0.35, "output": 0.80},
    "nvidia/nemotron-4-340b-instruct": {"input": 0.80, "output": 2.00},
    "nemotron-4-340b-instruct": {"input": 0.80, "output": 2.00},
    "nvidia/nemotron-3-super-120b-a12b": {"input": 0.30, "output": 0.75},
    "nemotron-3-super-120b-a12b": {"input": 0.30, "output": 0.75},
    # DeepSeek Models
    "deepseek-flash": {"input": 0.15, "output": 0.60},
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98},
    "deepseek-chat": {"input": 0.15, "output": 0.60},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Groq (Hosted open models)
    "openai/gpt-oss-120b": {"input": 0.30, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.20},
    "qwen/qwen3.8-27b": {"input": 0.20, "output": 0.40},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    # Ollama / Local / Mock
    "llama3:8b": {"input": 0.0, "output": 0.0},
    "mistral:7b": {"input": 0.0, "output": 0.0},
    "phi3:mini": {"input": 0.0, "output": 0.0},
    "mock-gpt": {"input": 0.15, "output": 0.60},
    "mock-claude": {"input": 0.25, "output": 1.25},
    "mock-llama": {"input": 0.05, "output": 0.08},
    "mock-judge-v1": {"input": 0.0, "output": 0.0},
    # Historical Models (Preserved for cost calculation of historical runs)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "grok-2-1212": {"input": 2.00, "output": 10.00},
    "grok-2": {"input": 2.00, "output": 10.00},
    "grok-beta": {"input": 5.00, "output": 15.00},
    "gemini-3.8-flash": {"input": 0.10, "output": 0.40},
    "gemini-flash-latest": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def get_api_key(provider: str) -> str:
    """Retrieve API key dynamically from environment with dotenv fallback."""
    # Reload environment in case .env was recently created or modified
    load_dotenv(override=False)
    pkey = provider.lower().strip()
    if pkey in ("nvidia", "nemotron"):
        return os.getenv("NVIDIA_API_KEY", "")
    elif pkey == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY", "")
    elif pkey == "groq":
        return os.getenv("GROQ_API_KEY", "")
    return ""


def get_token_pricing(model_name: str) -> Dict[str, float]:
    """Retrieve input and output token pricing per 1M tokens for a given model."""
    clean_name = model_name.lower().strip()
    for key, pricing in MODEL_PRICING.items():
        if key in clean_name or clean_name in key:
            return pricing
    return {"input": 0.0, "output": 0.0}


def get_provider_status() -> Dict[str, Dict[str, Any]]:
    """Check availability of credentials for active supported providers without crashing."""
    nvidia_key = get_api_key("nvidia")
    deepseek_key = get_api_key("deepseek")
    groq_key = get_api_key("groq")

    def derive_status(key: str) -> str:
        return "Configured, connection not verified" if bool(key.strip()) else "Not configured"

    return {
        "nvidia": {
            "configured": bool(nvidia_key.strip()),
            "status_label": derive_status(nvidia_key),
            "masked_key": f"{nvidia_key[:4]}...{nvidia_key[-4:]}" if len(nvidia_key) > 8 else "Not configured",
            "provider_type": "Cloud API (NVIDIA NIM)",
            "endpoint": NVIDIA_BASE_URL,
            "setup_hint": "Add NVIDIA_API_KEY to your .env file (obtain from build.nvidia.com)."
        },
        "deepseek": {
            "configured": bool(deepseek_key.strip()),
            "status_label": derive_status(deepseek_key),
            "masked_key": f"{deepseek_key[:4]}...{deepseek_key[-4:]}" if len(deepseek_key) > 8 else "Not configured",
            "provider_type": "Cloud API (DeepSeek)",
            "endpoint": DEEPSEEK_BASE_URL,
            "setup_hint": "Add DEEPSEEK_API_KEY to your .env file (obtain from platform.deepseek.com)."
        },
        "groq": {
            "configured": bool(groq_key.strip()),
            "status_label": derive_status(groq_key),
            "masked_key": f"{groq_key[:3]}...{groq_key[-4:]}" if len(groq_key) > 7 else "Not configured",
            "provider_type": "Cloud API (Groq LPU)",
            "endpoint": "https://api.groq.com/openai/v1",
            "setup_hint": "Add GROQ_API_KEY to your .env file."
        },
        "ollama": {
            "configured": True,  # Checked via network endpoint
            "status_label": "Configured (Local host)",
            "endpoint": OLLAMA_BASE_URL,
            "provider_type": "Local Host",
            "setup_hint": "Ensure Ollama daemon is running locally ('ollama serve')."
        },
        "mock": {
            "configured": True,
            "status_label": "Ready (Offline Demo)",
            "provider_type": "Simulated (Demo Mode)",
            "setup_hint": "Ready for offline benchmarking."
        }
    }


def mask_secret(secret: Optional[str]) -> str:
    """Mask a secret key for safe display in UI or logs."""
    if not secret:
        return "[NOT SET]"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:3]}...{secret[-4:]}"
