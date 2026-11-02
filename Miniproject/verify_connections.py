"""CLI tool to securely verify provider API connectivity without running benchmark suites.

Usage:
    python verify_connections.py --provider nvidia
    python verify_connections.py --provider deepseek
    python verify_connections.py --provider groq
    python verify_connections.py --all
"""

import sys
import argparse
import asyncio
from tabulate import tabulate

import config
from engine.connection_check import (
    verify_provider_connectivity,
    list_deepseek_models,
    PROBE_MODELS
)


async def main():
    parser = argparse.ArgumentParser(description="Test LLM provider connectivity safely.")
    parser.add_argument(
        "--provider",
        choices=["nvidia", "deepseek", "groq", "ollama", "mock", "all"],
        default="all",
        help="Provider to probe (default: all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Probe all active configured providers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom model name to test with (optional)"
    )
    parser.add_argument(
        "--list-deepseek-models",
        action="store_true",
        help="Query DeepSeek official /models endpoint and display available models"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Timeout in seconds for probe request (default: 15)"
    )
    args = parser.parse_args()

    # Ensure stdout handles unicode/emoji on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.list_deepseek_models:
        print("\n==============================================================")
        print(" [*] DeepSeek /models Secure Diagnostic")
        print("==============================================================\n")
        res = await list_deepseek_models(timeout=args.timeout)
        if res["status"] == "success":
            print(f"Total available models: {res['total_models']}\n")
            table = [[m["model_id"], m["owned_by"]] for m in res["models"]]
            print(tabulate(table, headers=["Model ID", "Owned By"], tablefmt="github"))
        else:
            print(f"Status: {res['status']}\nMessage: {res['message']}")
        print("\n==============================================================\n")
        return

    # If --all is passed or default "all" is set
    if args.all or args.provider == "all":
        providers = list(PROBE_MODELS.keys())
    else:
        providers = [args.provider]

    print("\n==============================================================")
    print(" [*] LLM Provider Connectivity Verification Probe")
    print("==============================================================")
    print(" Note: Probes send a minimal 5-token ping only when credentials exist.\n")

    table_data = []
    for prov in providers:
        res = await verify_provider_connectivity(prov, model_name=args.model, timeout=args.timeout)
        status_symbol = {
            "success": "[OK] Success",
            "failed": "[FAIL] Error",
            "unconfigured": "[WARN] Not Configured"
        }.get(res["status"], res["status"])

        table_data.append([
            prov.upper(),
            res["model"],
            status_symbol,
            f"{res['latency_ms']:.1f} ms" if res["latency_ms"] > 0 else "-",
            res["message"]
        ])

    headers = ["Provider", "Model", "Status", "Latency", "Details"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("\n==============================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
