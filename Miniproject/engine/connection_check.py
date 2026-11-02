"""Secure provider connectivity verification utility.

Tests minimal round-trip connectivity to active cloud providers (NVIDIA Nemotron, DeepSeek, Groq, Ollama, Mock)
on-demand without running full benchmark test cases.
Never prints or logs secret credentials.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional

import config
from engine.model_registry import get_adapter

logger = logging.getLogger(__name__)

# Active probe models per supported provider
PROBE_MODELS: Dict[str, str] = {
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek": "deepseek-chat",
    "groq": "openai/gpt-oss-120b",
    "ollama": "llama3:8b",
    "mock": "mock-gpt"
}


async def list_deepseek_models(timeout: int = 15) -> Dict[str, Any]:
    """Securely query DeepSeek's /models endpoint to verify supported models.
    
    Returns available models without exposing API keys.
    """
    key = config.get_api_key("deepseek")
    if not key.strip():
        return {
            "status": "unconfigured",
            "message": "DeepSeek API key not configured. Add DEEPSEEK_API_KEY to your .env file."
        }

    import httpx
    url = f"{config.DEEPSEEK_BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                err_text = resp.text
                if key in err_text:
                    err_text = err_text.replace(key, config.mask_secret(key))
                return {
                    "status": "failed",
                    "code": resp.status_code,
                    "message": f"DeepSeek /models returned HTTP {resp.status_code}: {err_text[:200]}"
                }
            data = resp.json()
            raw_models = data.get("data", [])
            models = []
            for m in raw_models:
                models.append({
                    "model_id": m.get("id", ""),
                    "owned_by": m.get("owned_by", "deepseek")
                })
            return {
                "status": "success",
                "total_models": len(models),
                "models": models
            }
    except Exception as e:
        err_msg = str(e)
        if key in err_msg:
            err_msg = err_msg.replace(key, config.mask_secret(key))
        return {
            "status": "failed",
            "message": f"Failed querying DeepSeek /models: {err_msg}"
        }


def list_deepseek_models_sync(timeout: int = 15) -> Dict[str, Any]:
    """Synchronous wrapper for list_deepseek_models."""
    return asyncio.run(list_deepseek_models(timeout))


async def verify_provider_connectivity(
    provider: str,
    model_name: Optional[str] = None,
    timeout: int = 15
) -> Dict[str, Any]:
    """Test a single model provider connection with a minimal, non-billable or micro-payload probe.

    Returns a structured dictionary with:
    - status: 'success' | 'failed' | 'unconfigured'
    - provider: Provider identifier
    - model: Model ID probed
    - latency_ms: Round-trip response time in ms (if successful)
    - message: Sanitized status or error description
    """
    prov_key = provider.lower().strip()
    target_model = model_name or PROBE_MODELS.get(prov_key, "mock-gpt")

    # 1. Cleanly reject retired providers
    if prov_key in config.RETIRED_PROVIDERS:
        return {
            "status": "failed",
            "provider": prov_key,
            "model": target_model,
            "latency_ms": 0.0,
            "message": f"Provider '{prov_key}' has been removed from active service. Supported active providers: nvidia, deepseek, groq, ollama, mock."
        }

    # 2. Check credential configuration
    key = config.get_api_key(prov_key)
    if prov_key not in ("mock", "ollama") and not key.strip():
        setup_hint = config.get_provider_status().get(prov_key, {}).get("setup_hint", f"Configure {prov_key.upper()}_API_KEY.")
        return {
            "status": "unconfigured",
            "provider": prov_key,
            "model": target_model,
            "latency_ms": 0.0,
            "message": f"API key not configured. {setup_hint}"
        }

    # For Ollama, verify local server connectivity first
    if prov_key == "ollama":
        from engine.provider_adapter import OllamaAdapter
        is_alive, conn_msg = await OllamaAdapter.check_connection_static(config.OLLAMA_BASE_URL, timeout=min(5.0, timeout))
        if not is_alive:
            return {
                "status": "failed",
                "provider": prov_key,
                "model": target_model,
                "latency_ms": 0.0,
                "message": conn_msg
            }

    # 3. Instantiate provider adapter
    try:
        adapter = get_adapter(prov_key, target_model)
    except Exception as e:
        return {
            "status": "failed",
            "provider": prov_key,
            "model": target_model,
            "latency_ms": 0.0,
            "message": f"Adapter instantiation failed: {str(e)}"
        }

    # 4. Perform minimal probe
    start_time = time.perf_counter()
    try:
        res = await adapter.generate(
            prompt="Respond with one word: ping.",
            temperature=0.0,
            max_tokens=20,
            timeout=timeout
        )
        latency = res.get("latency_ms", round((time.perf_counter() - start_time) * 1000.0, 2))
        sample_out = (res.get("text") or "").strip()[:20]
        return {
            "status": "success",
            "provider": prov_key,
            "model": target_model,
            "latency_ms": latency,
            "message": f"Connected successfully. Latency: {latency:.1f}ms. Sample output: '{sample_out}'"
        }
    except Exception as e:
        latency = round((time.perf_counter() - start_time) * 1000.0, 2)
        err_msg = str(e)
        # Sanitize any potential API keys from error messages
        for k in [config.NVIDIA_API_KEY, config.DEEPSEEK_API_KEY, config.GROQ_API_KEY]:
            if k and len(k) > 6 and k in err_msg:
                err_msg = err_msg.replace(k, config.mask_secret(k))
        return {
            "status": "failed",
            "provider": prov_key,
            "model": target_model,
            "latency_ms": latency,
            "message": f"Connection check failed: {err_msg}"
        }


def check_provider_connection_sync(
    provider: str,
    model_name: Optional[str] = None,
    timeout: int = 15
) -> Dict[str, Any]:
    """Synchronous wrapper for verify_provider_connectivity."""
    return asyncio.run(verify_provider_connectivity(provider, model_name, timeout))
