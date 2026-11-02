"""Model registry managing active provider availability, supported models, and adapter instantiation."""

from typing import Dict, Any, List, Optional
from engine.provider_adapter import (
    BaseProviderAdapter,
    NVIDIANemotronAdapter,
    DeepSeekAdapter,
    GroqAdapter,
    OllamaAdapter,
    MockProviderAdapter
)
from config import get_provider_status, get_token_pricing

# Active Models Registry (NVIDIA Nemotron, DeepSeek, Groq, Ollama, Mock)
REGISTRY: Dict[str, Dict[str, Any]] = {
    # Mock / Demo Models (Always available offline)
    "mock-gpt": {
        "provider": "mock",
        "display_name": "Mock GPT-4o (Simulated)",
        "description": "High-accuracy simulated frontier model for fast testing.",
        "context_window": 128000,
        "default_enabled": True,
        "available": True
    },
    "mock-claude": {
        "provider": "mock",
        "display_name": "Mock Claude Sonnet (Simulated)",
        "description": "Balanced simulated model with formatting variations.",
        "context_window": 200000,
        "default_enabled": True,
        "available": True
    },
    "mock-llama": {
        "provider": "mock",
        "display_name": "Mock Llama 3 (Simulated)",
        "description": "Fast simulated open model exhibiting occasional arithmetic drift.",
        "context_window": 8192,
        "default_enabled": True,
        "available": True
    },

    # NVIDIA Nemotron Models (Primary Cloud Provider)
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "provider": "nvidia",
        "display_name": "NVIDIA Llama 3.1 Nemotron 70B Instruct",
        "description": "NVIDIA flagship Nemotron instruction-following model hosted on NVIDIA NIM.",
        "context_window": 131072,
        "default_enabled": True,
        "available": True,
        "active_lineup": True
    },
    "nvidia/nemotron-4-340b-instruct": {
        "provider": "nvidia",
        "display_name": "NVIDIA Nemotron 4 340B Instruct",
        "description": "NVIDIA frontier 340B synthetic data and instruction model hosted on NIM.",
        "context_window": 4096,
        "default_enabled": False,
        "available": True,
        "active_lineup": True
    },
    "nvidia/nemotron-3-super-120b-a12b": {
        "provider": "nvidia",
        "display_name": "NVIDIA Nemotron 3 Super 120B",
        "description": "NVIDIA hosted frontier reasoning and instruction-following Nemotron model via NIM catalog.",
        "context_window": 131072,
        "default_enabled": False,
        "available": True,
        "active_lineup": True
    },

    # DeepSeek Models (Primary Cloud Provider)
    "deepseek-flash": {
        "provider": "deepseek",
        "display_name": "DeepSeek Flash",
        "description": "High-throughput, ultra-fast general purpose DeepSeek chat model.",
        "context_window": 128000,
        "default_enabled": False,
        "available": True
    },
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "display_name": "DeepSeek V4 Pro",
        "description": "Advanced flagship reasoning and code intelligence model from DeepSeek.",
        "context_window": 128000,
        "default_enabled": False,
        "available": True
    },
    "deepseek-chat": {
        "provider": "deepseek",
        "display_name": "DeepSeek Chat",
        "description": "Standard general-purpose chat endpoint (V3).",
        "context_window": 128000,
        "default_enabled": False,
        "available": True
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "display_name": "DeepSeek Reasoner",
        "description": "Specialized reasoning model (R1) with chain-of-thought capability.",
        "context_window": 128000,
        "default_enabled": False,
        "available": True
    },

    # Groq Models (Hosted open weights)
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "display_name": "Groq OpenAI GPT-OSS 120B",
        "description": "Active open-weights flagship chat model hosted on Groq LPUs.",
        "context_window": 131072,
        "default_enabled": False,
        "available": True
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "display_name": "Groq OpenAI GPT-OSS 20B",
        "description": "Active ultra-low latency model hosted on Groq LPUs.",
        "context_window": 131072,
        "default_enabled": False,
        "available": True
    },
    "qwen/qwen3.8-27b": {
        "provider": "groq",
        "display_name": "Groq Qwen 3.8 27B",
        "description": "Active high-capability reasoning model hosted on Groq LPUs.",
        "context_window": 131072,
        "default_enabled": False,
        "available": True
    },

    # Ollama Models (Local hosting)
    "llama3:8b": {
        "provider": "ollama",
        "display_name": "Ollama Llama 3 8B (Local)",
        "description": "Locally hosted Llama 3 weights served via Ollama.",
        "context_window": 8192,
        "default_enabled": False,
        "available": True
    },
    "mistral:7b": {
        "provider": "ollama",
        "display_name": "Ollama Mistral 7B (Local)",
        "description": "Locally hosted Mistral model served via Ollama.",
        "context_window": 8192,
        "default_enabled": False,
        "available": True
    }
}

# Historical Models Metadata (Preserved to render historical records without crashing)
HISTORICAL_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {"provider": "openai", "display_name": "OpenAI GPT-4o (Retired)"},
    "gpt-4o-mini": {"provider": "openai", "display_name": "OpenAI GPT-4o-mini (Retired)"},
    "gpt-4-turbo": {"provider": "openai", "display_name": "OpenAI GPT-4-turbo (Retired)"},
    "gpt-3.5-turbo": {"provider": "openai", "display_name": "OpenAI GPT-3.5-turbo (Retired)"},
    "o1": {"provider": "openai", "display_name": "OpenAI o1 (Retired)"},
    "o1-mini": {"provider": "openai", "display_name": "OpenAI o1-mini (Retired)"},
    "o3-mini": {"provider": "openai", "display_name": "OpenAI o3-mini (Retired)"},
    "gemini-3.8-flash": {"provider": "google", "display_name": "Google Gemini 3.8 Flash (Retired)"},
    "gemini-flash-latest": {"provider": "google", "display_name": "Google Gemini Flash Latest (Retired)"},
    "gemini-2.5-flash": {"provider": "google", "display_name": "Google Gemini 2.5 Flash (Retired)"},
    "gemini-2.5-pro": {"provider": "google", "display_name": "Google Gemini 2.5 Pro (Retired)"},
    "gemini-2.0-flash": {"provider": "google", "display_name": "Google Gemini 2.0 Flash (Retired)"},
    "grok-2-1212": {"provider": "xai", "display_name": "xAI Grok-2 1212 (Retired)"},
    "grok-2": {"provider": "xai", "display_name": "xAI Grok-2 (Retired)"},
    "grok-beta": {"provider": "xai", "display_name": "xAI Grok-Beta (Retired)"},
    "llama-3.1-8b-instant": {"provider": "groq", "display_name": "Groq Llama 3.1 8B Instant (Retired)"},
    "llama-3.3-70b-versatile": {"provider": "groq", "display_name": "Groq Llama 3.3 70B Versatile (Retired)"}
}

RETIRED_PROVIDERS = {"openai", "google", "gemini", "xai", "grok"}


def get_adapter(provider: str, model_name: str, api_key: Optional[str] = None) -> BaseProviderAdapter:
    """Instantiate the appropriate provider adapter for the requested model.
    
    Cleanly prevents and rejects attempts to benchmark retired providers.
    """
    prov = provider.lower().strip()
    if prov in RETIRED_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' has been removed from active service. "
            f"Active supported providers are: NVIDIA Nemotron ('nvidia'), DeepSeek ('deepseek'), "
            f"Groq ('groq'), Ollama ('ollama'), and Mock ('mock')."
        )
    elif prov in ("nvidia", "nemotron"):
        return NVIDIANemotronAdapter(model_name=model_name, api_key=api_key)
    elif prov == "deepseek":
        return DeepSeekAdapter(model_name=model_name, api_key=api_key)
    elif prov == "groq":
        return GroqAdapter(model_name=model_name, api_key=api_key)
    elif prov == "ollama":
        return OllamaAdapter(model_name=model_name)
    elif prov == "mock":
        return MockProviderAdapter(model_name=model_name)
    else:
        raise ValueError(f"Unknown or unsupported provider: {provider}")


def list_registered_models(active_only: bool = True) -> List[Dict[str, Any]]:
    """Return all active models in the registry enriched with pricing and provider status."""
    statuses = get_provider_status()
    models = []
    for model_id, meta in REGISTRY.items():
        prov = meta["provider"]
        pricing = get_token_pricing(model_id)
        prov_info = statuses.get(prov, {})
        models.append({
            "model_id": model_id,
            "provider": prov,
            "display_name": meta["display_name"],
            "description": meta["description"],
            "context_window": meta["context_window"],
            "configured": prov_info.get("configured", False),
            "available": meta.get("available", True),
            "active_lineup": meta.get("active_lineup", True),
            "cost_input_per_1m": pricing["input"],
            "cost_output_per_1m": pricing["output"],
            "default_enabled": meta.get("default_enabled", False)
        })
    return models


def resolve_model_display_name(model_id: str) -> str:
    """Resolve human-readable display name for any active or historical model."""
    if model_id in REGISTRY:
        return REGISTRY[model_id]["display_name"]
    if model_id in HISTORICAL_MODELS:
        return HISTORICAL_MODELS[model_id]["display_name"]
    return model_id
