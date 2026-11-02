"""Application launcher and pre-flight validation script."""

import sys
import subprocess
from pathlib import Path
from storage.database import init_db
from config import get_provider_status


def preflight_check():
    """Verify database initialization and provider configurations."""
    print("=" * 65)
    print("LLM Benchmark & Diagnostics: Comparative Accuracy Engine")
    print("=" * 65)
    print("[1/3] Initializing SQLite schema...")
    init_db()
    print("      [OK] Database initialized successfully.")

    print("\n[2/3] Checking Model Provider Status:")
    statuses = get_provider_status()
    for prov, info in statuses.items():
        configured = info.get("configured", False)
        status_symbol = "+" if configured else "x"
        prov_type = info.get("provider_type", "")
        print(f"      [{status_symbol}] {prov.upper():<10} ({prov_type:<18}) -> {'Configured' if configured else 'No API Key (Demo mode available)'}")

    print("\n[3/3] Ready to launch Streamlit Dashboard.")
    print("=" * 65)


def launch():
    """Launch Streamlit web application."""
    preflight_check()
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"\nExecuting: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutdown requested by user. Exiting cleanly.")


if __name__ == "__main__":
    launch()
