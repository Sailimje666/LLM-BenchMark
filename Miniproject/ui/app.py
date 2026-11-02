"""LLM Benchmark & Diagnostics Streamlit Dashboard."""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from schema import (
    TestCase, ModelConfig, EvaluationResult, BenchmarkRun,
    UniversalComparison, UniversalComparisonResult
)
from engine.universal_evaluator import (
    run_universal_comparison,
    export_comparison_to_csv,
    export_comparison_to_json,
    validate_question
)
from config import (
    NVIDIA_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY, OLLAMA_BASE_URL,
    get_provider_status, get_token_pricing, mask_secret
)
from storage.database import init_db
from storage.repository import BenchmarkRepository
from storage.exports import (
    export_results_df, export_results_csv, export_results_json,
    export_run_summary_json, generate_consolidated_markdown_report
)
from engine.model_registry import list_registered_models, REGISTRY
from engine.runner import BenchmarkRunner
from engine.telemetry import compute_model_statistics, compute_paired_comparison
from engine.connection_check import check_provider_connection_sync
from diagnostics.root_cause import PRIMARY_CATEGORIES

# Initialize database tables on load
init_db()
repo = BenchmarkRepository()

# --- Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="LLM Benchmark & Diagnostics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Sleek Modern Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-error {
        background-color: #881337;
        color: #fda4af;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fcd34d;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Management ---
if "test_cases" not in st.session_state:
    default_dataset_file = BASE_DIR / "datasets" / "default_test_cases.json"
    if default_dataset_file.exists():
        with open(default_dataset_file, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
            st.session_state["test_cases"] = [TestCase(**c) for c in raw_cases]
    else:
        st.session_state["test_cases"] = []

if "selected_test_case_ids" not in st.session_state:
    st.session_state["selected_test_case_ids"] = {tc.id for tc in st.session_state["test_cases"]}

if "current_run" not in st.session_state:
    st.session_state["current_run"] = None

if "current_results" not in st.session_state:
    st.session_state["current_results"] = []


if "verified_connections" not in st.session_state:
    st.session_state["verified_connections"] = {}


# --- Sidebar: Provider Credentials & Environment Status ---
with st.sidebar:
    st.title("⚡ Benchmark Hub")
    st.caption("LLM Accuracy & Diagnostics Engine")
    st.divider()

    st.subheader("🔌 Active Providers")
    status_info = get_provider_status()

    primary_providers = [
        ("nvidia", "NVIDIA Nemotron", "NVIDIA NIM"),
        ("deepseek", "DeepSeek", "Official API"),
        ("groq", "Groq LPU", "Groq Fast LPUs"),
        ("ollama", "Ollama", "Local Host"),
        ("mock", "Mock Engine", "Offline Demo Mode")
    ]

    for prov_name, display_title, prov_desc in primary_providers:
        info = status_info.get(prov_name, {})
        is_configured = info.get("configured", False)
        v_info = st.session_state["verified_connections"].get(prov_name)

        if prov_name == "mock":
            status_text = "Connected (Offline Ready)"
            icon = "🟢"
        elif v_info and v_info.get("status") == "connected":
            status_text = "Connected (API Verified)"
            icon = "🟢"
        elif v_info and v_info.get("status") == "unavailable":
            status_text = "Unavailable (Connection Failed)"
            icon = "🔴"
        elif is_configured:
            status_text = "Configured (Unverified)"
            icon = "🟡"
        else:
            status_text = "Unavailable (Key Missing)"
            icon = "🔴"

        with st.expander(f"{icon} {display_title} ({prov_desc})"):
            st.markdown(f"**State:** `{status_text}`")
            if "masked_key" in info:
                st.text(f"Key: {info['masked_key']}")
            if "endpoint" in info:
                st.text(f"Endpoint: {info['endpoint']}")
            if not is_configured and prov_name not in ("mock", "ollama"):
                hint = info.get("setup_hint", f"Set {prov_name.upper()}_API_KEY in .env.")
                st.warning(hint)
            else:
                if st.button(f"🔍 Test {display_title} Connection", key=f"test_btn_{prov_name}"):
                    with st.spinner("Probing connection..."):
                        t_res = check_provider_connection_sync(prov_name)
                    if t_res["status"] == "success":
                        st.session_state["verified_connections"][prov_name] = {
                            "status": "connected",
                            "message": t_res["message"]
                        }
                        st.success(f"✅ {t_res['message']}")
                    elif t_res["status"] == "unconfigured":
                        st.session_state["verified_connections"][prov_name] = {
                            "status": "unavailable",
                            "message": t_res["message"]
                        }
                        st.warning(f"⚠️ {t_res['message']}")
                    else:
                        st.session_state["verified_connections"][prov_name] = {
                            "status": "unavailable",
                            "message": t_res["message"]
                        }
                        st.error(f"❌ {t_res['message']}")

    st.divider()
    st.info("💡 **Demo Mode:** Select `mock-gpt`, `mock-claude`, or `mock-llama` to run complete end-to-end benchmarks offline without external API costs.")


# --- Main Dashboard Tabs ---
tab_univ, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Universal Question Evaluator",
    "📂 1. Dataset & Test Suite",
    "🚀 2. Run Benchmarks",
    "📊 3. Comparative Analytics",
    "🔬 4. Root-Cause Diagnostics",
    "📑 5. Export Report"
])


# =====================================================================
# TAB: UNIVERSAL QUESTION EVALUATOR
# =====================================================================
with tab_univ:
    st.header("🔎 Universal Question Evaluator")
    st.markdown(
        "Directly test and compare multiple configured LLMs on **any arbitrary question or prompt** in real time. "
        "Evaluates responses honestly across three modes: **Mode A (Reference Ground Truth)**, "
        "**Mode B (Rubric / Key Points)**, or **Mode C (Unreferenced Quality & Model Consensus)**."
    )

    # Question Input
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        question_text = st.text_area(
            "💬 Enter Question or Prompt",
            value=st.session_state.get("univ_q_input", ""),
            height=120,
            placeholder="Type any arbitrary question, technical problem, coding task, or reasoning challenge here...",
            key="univ_question_area"
        )
    with q_col2:
        st.markdown("**💡 Evaluation Guidelines**")
        st.caption("• **No reference?** Model responses are compared for completeness, relevance, and agreement. Accuracy is marked unverified.")
        st.caption("• **Have an answer?** Provide Ground Truth for automated exact/tolerance/semantic grading.")
        st.caption("• **Open-ended?** Provide an expected Rubric or key bullet points.")

    # Advanced Inputs: System Prompt & Reference / Rubric
    with st.expander("📝 Optional System Prompt & Evaluation Guidance (Ground Truth / Rubric)", expanded=False):
        sys_col, eval_mode_col = st.columns(2)
        with sys_col:
            system_prompt_input = st.text_area(
                "System Prompt / Instruction (Optional)",
                value=st.session_state.get("univ_sys_input", ""),
                height=110,
                placeholder="e.g. 'You are an expert Python engineer. Provide clear code with comments.'",
                key="univ_system_prompt_area"
            )
        with eval_mode_col:
            eval_choice = st.radio(
                "Evaluation Mode",
                ["Mode C: No Reference (Quality & Agreement)", "Mode A: Reference Answer / Ground Truth", "Mode B: Evaluation Rubric / Key Points"],
                index=0,
                key="univ_eval_choice"
            )

        ref_answer_input = None
        rubric_input = None
        eval_method_input = "auto"
        enable_judge = False
        judge_model_sel = "mock-judge-v1"

        if eval_choice == "Mode A: Reference Answer / Ground Truth":
            r_col1, r_col2 = st.columns([2, 1])
            with r_col1:
                ref_answer_input = st.text_area(
                    "Reference Answer / Expected Ground Truth",
                    height=90,
                    placeholder="Enter the exact or expected answer, mathematical solution, or reference text...",
                    key="univ_ref_answer_area"
                )
            with r_col2:
                eval_method_input = st.selectbox(
                    "Evaluation Method",
                    ["auto", "exact_match", "numeric_tolerance", "regex", "json_schema", "semantic"],
                    index=0,
                    key="univ_eval_method_sel"
                )
                if eval_method_input == "semantic":
                    enable_judge = st.checkbox("Enable Live LLM Judge (Requires billable API call)", value=False, key="univ_judge_toggle_a")
                    st.caption("⚠️ Automated / LLM judges are estimates and can make mistakes.")

        elif eval_choice == "Mode B: Evaluation Rubric / Key Points":
            rubric_input = st.text_area(
                "Evaluation Rubric / Expected Key Points",
                height=100,
                placeholder="List expected key points or rubric criteria (one per line or bullet, e.g.):\n- Mentions time complexity O(N)\n- Handles edge case of empty list\n- Uses Python 3 type hints",
                key="univ_rubric_area"
            )
            enable_judge = st.checkbox("Enable LLM Judge for Rubric Grading (Requires API key)", value=False, key="univ_judge_toggle_b")
            st.caption("⚠️ LLM judge judgments are heuristic estimates, not definitive ground truth.")

        else:
            st.info("ℹ️ **Mode C Active:** Factual accuracy will remain **explicitly unverified**. Model responses will be measured for completeness, lexical relevance, instruction following, linguistic uncertainty, and pairwise model agreement.")

    # Model Selection & Configuration
    st.subheader("🤖 Model Selection & Generation Controls")
    registered_models = list_registered_models(active_only=True)
    model_id_list = [m["model_id"] for m in registered_models]

    cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 1, 1])
    with cfg_col1:
        default_selected = [
            m["model_id"] for m in registered_models
            if (m["default_enabled"] or m["configured"]) and m.get("available", True) and m.get("active_lineup", True)
        ]

        select_all_configured = st.checkbox("Select all configured & available models", value=False, key="univ_sel_all_configured")
        if select_all_configured:
            selected_models_choice = [m["model_id"] for m in registered_models if (m["configured"] or m["provider"] == "mock") and m.get("available", True)]
        else:
            selected_models_choice = default_selected

        def format_model_label(m_id: str) -> str:
            m = next((item for item in registered_models if item["model_id"] == m_id), None)
            if not m:
                return m_id
            prov = m.get("provider", "")
            v_info = st.session_state.get("verified_connections", {}).get(prov, {})
            if not m.get("available", True):
                status_tag = "🔴 Unavailable"
            elif prov == "mock":
                status_tag = "🟢 Ready"
            elif v_info.get("status") == "connected":
                status_tag = "🟢 Connected"
            elif v_info.get("status") == "unavailable":
                status_tag = "🔴 Unavailable"
            elif m.get("configured", False):
                status_tag = "🟡 Configured"
            else:
                status_tag = "🔴 Unavailable"
            cost_tag = f"${m['cost_input_per_1m']:.2f}/1M" if m['cost_input_per_1m'] > 0 else "Free / Local"
            return f"{m['display_name']} [{status_tag} | {cost_tag}]"

        selected_models = st.multiselect(
            "Select Models to Query",
            options=model_id_list,
            default=selected_models_choice,
            format_func=format_model_label,
            key="univ_models_multiselect"
        )

    with cfg_col2:
        temperature_val = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.0,
            step=0.1,
            help="Higher values produce more creative responses; 0.0 is deterministic.",
            key="univ_temp_slider"
        )
    with cfg_col3:
        max_tokens_val = st.number_input(
            "Max Output Tokens",
            min_value=64,
            max_value=4096,
            value=1024,
            step=128,
            help="Upper limit on generated tokens.",
            key="univ_max_tokens_input"
        )

    # Cost and Safety Control Banner
    paid_selected = [
        m for m in registered_models
        if m["model_id"] in selected_models and m["provider"] not in ("mock", "ollama")
    ]
    st.markdown(f"**Target Models:** `{len(selected_models)}` total selected ({len(paid_selected)} cloud API models, {len(selected_models) - len(paid_selected)} demo/local models).")
    if paid_selected:
        st.warning(
            f"⚠️ **Cost & Safety Advisory:** You have selected {len(paid_selected)} paid model(s) ({', '.join(m['model_id'] for m in paid_selected)}). "
            "Clicking **'Run Comparison'** will issue live HTTP requests to external APIs that may incur billing charges based on token usage. "
            "No requests will be made until you click the button below."
        )

    # Action Controls: Run Comparison and Clear/Reset
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
    with btn_col1:
        run_btn = st.button("🚀 Run Comparison", type="primary", use_container_width=True, key="univ_run_btn")
    with btn_col2:
        clear_btn = st.button("🔄 Clear / Reset", use_container_width=True, key="univ_clear_btn")

    if clear_btn:
        st.session_state["universal_current_comparison"] = None
        st.session_state["univ_question_area"] = ""
        st.rerun()

    # Progress and Execution
    if run_btn:
        if not question_text or not question_text.strip():
            st.error("❌ Validation Error: Please enter a valid question or prompt before running the comparison. Empty queries cannot be executed.")
        elif not selected_models:
            st.error("❌ Model Selection Error: Please select at least one model to evaluate.")
        else:
            progress_status = st.status("🚀 Executing Universal Question Comparison across models...", expanded=True)
            status_placeholders = {}
            for mid in selected_models:
                status_placeholders[mid] = progress_status.empty()
                status_placeholders[mid].markdown(f"⏳ **{mid}**: Pending dispatch...")

            def on_progress(mid: str, status: str, res: Optional[UniversalComparisonResult]):
                if status == "running":
                    status_placeholders[mid].markdown(f"🔄 **{mid}**: Querying provider...")
                elif status == "success":
                    lat = f"{res.latency_ms:.0f}ms" if res else ""
                    toks = f"{res.total_tokens} toks" if res else ""
                    status_placeholders[mid].markdown(f"✅ **{mid}**: Completed ({lat}, {toks})")
                elif status == "unconfigured":
                    status_placeholders[mid].markdown(f"⚠️ **{mid}**: Unconfigured provider")
                else:
                    err_hint = res.sanitized_error if res else "Failed"
                    status_placeholders[mid].markdown(f"❌ **{mid}**: Failed ({err_hint})")

            try:
                comparison_result = asyncio.run(run_universal_comparison(
                    question=question_text,
                    selected_models=selected_models,
                    system_prompt=system_prompt_input if system_prompt_input else None,
                    reference_answer=ref_answer_input if ref_answer_input else None,
                    rubric=rubric_input if rubric_input else None,
                    eval_method=eval_method_input,
                    temperature=temperature_val,
                    max_tokens=max_tokens_val,
                    enable_llm_judge=enable_judge,
                    progress_callback=on_progress
                ))
                # Persist to database
                repo.save_comparison(comparison_result)
                st.session_state["universal_current_comparison"] = comparison_result
                progress_status.update(label="✅ Universal Comparison Completed Successfully!", state="complete", expanded=False)
            except Exception as e:
                progress_status.update(label=f"❌ Comparison execution encountered an error: {str(e)}", state="error")
                st.error(f"Execution error: {str(e)}")

    st.divider()

    # DISPLAY COMPARISON RESULTS (if available)
    comp: Optional[UniversalComparison] = st.session_state.get("universal_current_comparison")

    # History Selector
    saved_comps = repo.list_comparisons()
    if saved_comps:
        with st.expander("📂 Comparison History & Previous Runs", expanded=False):
            hist_options = {
                f"{c['timestamp'][:19]} | {c['evaluation_mode'].upper()} | {c['question'][:45]}... ({c['status']})": c['comparison_id']
                for c in saved_comps
            }
            selected_hist_label = st.selectbox(
                "Select a previously saved comparison to view:",
                options=list(hist_options.keys()),
                key="univ_history_selector"
            )
            hist_col1, hist_col2 = st.columns([1, 1])
            with hist_col1:
                if st.button("📥 Load Selected Comparison", key="univ_load_hist_btn"):
                    cid = hist_options[selected_hist_label]
                    loaded = repo.get_comparison(cid)
                    if loaded:
                        st.session_state["universal_current_comparison"] = loaded
                        st.success(f"Loaded comparison {cid}")
                        st.rerun()
            with hist_col2:
                if st.button("🗑️ Delete Selected Comparison", key="univ_del_hist_btn"):
                    cid = hist_options[selected_hist_label]
                    repo.delete_comparison(cid)
                    st.info("Comparison deleted.")
                    st.rerun()

    if comp:
        # Comparison Overview Banner
        st.subheader("📊 Comparison Results")
        status_color = "badge-success" if comp.status == "completed" else ("badge-warning" if comp.status == "partial_failure" else "badge-error")
        st.markdown(f"""
        **Comparison ID:** `{comp.comparison_id}` &nbsp;|&nbsp; 
        **Timestamp:** `{comp.timestamp}` &nbsp;|&nbsp; 
        **Execution Status:** <span class="{status_color}">{comp.status.upper()}</span> &nbsp;|&nbsp; 
        **Evaluation Mode:** `{comp.evaluation_mode.upper()}`
        """, unsafe_allow_html=True)

        # Mode C Honest Disclosure Notice
        if comp.evaluation_mode == "unreferenced":
            st.warning(
                "⚠️ **Accuracy not independently verified — no reference answer supplied.**\n\n"
                "When only a question is submitted, models cannot be graded for factual truth without ground truth. "
                "The scores below reflect **response completeness**, **question relevance**, **instruction following**, "
                "and **cross-model agreement**. Model-to-model consensus must **never** be treated as proof of factual correctness."
            )
        elif comp.evaluation_mode == "rubric":
            st.info("📋 **Rubric-Based Evaluation:** Models were graded on specific key points. Automated and LLM judgments are estimates.")
        else:
            st.success(f"🎯 **Reference-Based Evaluation:** Models evaluated against ground truth using method: `{comp.generation_config.get('eval_method', 'auto')}`.")

        # Best Model Highlights (Strictly Qualified Metrics!)
        succ_results = [r for r in comp.results if r.status == "success"]
        if succ_results:
            hl_cols = st.columns(4)
            # Verified accuracy only when reference or rubric exists
            with hl_cols[0]:
                if comp.evaluation_mode in ("reference", "rubric"):
                    best_acc = max(succ_results, key=lambda x: x.accuracy_score or 0.0)
                    acc_val = f"{(best_acc.accuracy_score or 0.0)*100:.1f}%"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">🏆 Best Verified Accuracy</div>
                        <div class="metric-value">{acc_val}</div>
                        <div class="metric-sub">{best_acc.display_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">🔍 Verified Accuracy</div>
                        <div class="metric-value" style="color: #94a3b8; font-size: 1.3rem;">UNVERIFIED</div>
                        <div class="metric-sub">No reference answer provided</div>
                    </div>
                    """, unsafe_allow_html=True)

            with hl_cols[1]:
                fastest = min(succ_results, key=lambda x: x.latency_ms if x.latency_ms > 0 else 999999)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">⚡ Lowest Latency</div>
                    <div class="metric-value">{fastest.latency_ms:.0f} ms</div>
                    <div class="metric-sub">{fastest.display_name}</div>
                </div>
                """, unsafe_allow_html=True)

            with hl_cols[2]:
                cheapest = min(succ_results, key=lambda x: x.estimated_cost_usd)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">💰 Lowest Estimated Cost</div>
                    <div class="metric-value">${cheapest.estimated_cost_usd:.5f}</div>
                    <div class="metric-sub">{cheapest.display_name}</div>
                </div>
                """, unsafe_allow_html=True)

            with hl_cols[3]:
                agreement_info = comp.agreement_analysis
                c_score = agreement_info.get("consensus_score", 0.0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🤝 Model Agreement Index</div>
                    <div class="metric-value">{c_score*100:.1f}%</div>
                    <div class="metric-sub">{agreement_info.get('consensus_level', 'Consensus')}</div>
                </div>
                """, unsafe_allow_html=True)

        # Cross-Model Agreement & Disagreement Section
        if comp.agreement_analysis and len(succ_results) >= 2:
            with st.expander("🤝 Model Agreement & Disagreement Analysis", expanded=False):
                ag = comp.agreement_analysis
                st.markdown(f"**Agreement Level:** `{ag.get('consensus_level', 'N/A')}` (Average Token Similarity: **{ag.get('consensus_score', 0.0)*100:.1f}%**)")
                st.caption(f"ℹ️ {ag.get('disclaimer', '')}")
                if ag.get("pairwise_similarities"):
                    st.write("**Pairwise Similarity Matrix:**")
                    sim_df = pd.DataFrame([
                        {"Pair": k, "Lexical Overlap": f"{v*100:.1f}%", "Raw Score": v}
                        for k, v in ag["pairwise_similarities"].items()
                    ])
                    st.dataframe(sim_df, hide_index=True, use_container_width=True)
                if ag.get("divergence_notes"):
                    st.write("**Noted Divergences:**")
                    for note in ag["divergence_notes"]:
                        st.markdown(f"- ⚠️ {note}")

        # Comparative Table
        st.subheader("📋 Model Comparison Table")
        table_rows = []
        for r in comp.results:
            acc_display = f"{r.accuracy_score*100:.1f}%" if r.accuracy_score is not None else "N/A (Unverified)"
            quality_desc = ""
            if comp.evaluation_mode == "unreferenced":
                c = f"Comp: {int((r.completeness_score or 0)*100)}%"
                rel = f"Rel: {int((r.relevance_score or 0)*100)}%"
                quality_desc = f"{c}, {rel}"
                if r.uncertainty_detected:
                    quality_desc += " [⚠️ Hedging]"
            elif comp.evaluation_mode == "rubric":
                quality_desc = f"Rubric: {len(r.rubric_scores) - len(r.missing_key_points)}/{len(r.rubric_scores)} points"
            else:
                quality_desc = "PASS" if r.is_correct else "FAIL"

            table_rows.append({
                "Provider": r.provider.upper(),
                "Model ID": r.model_id,
                "Status": r.status,
                "Score": acc_display,
                "Quality / Verdict": quality_desc,
                "Latency (ms)": r.latency_ms,
                "Prompt Tokens": r.prompt_tokens,
                "Completion Tokens": r.completion_tokens,
                "Total Tokens": r.total_tokens,
                "Est. Cost ($)": f"${r.estimated_cost_usd:.5f}",
                "Error Details": r.sanitized_error or "None"
            })

        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Visualizations Section
        st.subheader("📈 Visual Performance Diagnostics")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Latency Bar Chart
            succ_table = df_table[df_table["Status"] == "success"]
            if not succ_table.empty:
                lat_fig = px.bar(
                    succ_table,
                    x="Model ID",
                    y="Latency (ms)",
                    color="Provider",
                    title="⚡ Latency by Model (Milliseconds)",
                    text_auto=".0f"
                )
                lat_fig.update_layout(xaxis_tickangle=-30, height=320, margin=dict(t=40, b=40, l=40, r=20))
                st.plotly_chart(lat_fig, use_container_width=True)
            else:
                st.info("No successful latency records to plot.")

        with viz_col2:
            # Token Usage Breakdown
            tok_df = []
            for r in comp.results:
                if r.status == "success":
                    tok_df.append({"Model ID": r.model_id, "Type": "Prompt Tokens", "Count": r.prompt_tokens})
                    tok_df.append({"Model ID": r.model_id, "Type": "Completion Tokens", "Count": r.completion_tokens})
            if tok_df:
                tok_fig = px.bar(
                    pd.DataFrame(tok_df),
                    x="Model ID",
                    y="Count",
                    color="Type",
                    title="🔢 Token Consumption (Prompt vs Completion)",
                    barmode="stack"
                )
                tok_fig.update_layout(xaxis_tickangle=-30, height=320, margin=dict(t=40, b=40, l=40, r=20))
                st.plotly_chart(tok_fig, use_container_width=True)
            else:
                st.info("No successful token records to plot.")

        viz_col3, viz_col4 = st.columns(2)
        with viz_col3:
            # Cost Chart
            cost_fig = px.bar(
                df_table,
                x="Model ID",
                y=[float(c.replace("$", "")) for c in df_table["Est. Cost ($)"]],
                labels={"y": "Estimated Cost (USD)"},
                title="💰 Estimated Cost per Query (USD)",
                color="Provider"
            )
            cost_fig.update_layout(xaxis_tickangle=-30, height=300, margin=dict(t=40, b=40, l=40, r=20))
            st.plotly_chart(cost_fig, use_container_width=True)

        with viz_col4:
            # Accuracy chart IF valid; otherwise explain unavailable
            if comp.evaluation_mode in ("reference", "rubric"):
                score_records = [
                    {"Model ID": r.model_id, "Score (%)": (r.accuracy_score or 0.0) * 100, "Provider": r.provider}
                    for r in comp.results if r.status == "success"
                ]
                if score_records:
                    score_fig = px.bar(
                        pd.DataFrame(score_records),
                        x="Model ID",
                        y="Score (%)",
                        color="Provider",
                        title="🎯 Verified Accuracy / Rubric Score (%)",
                        range_y=[0, 105],
                        text_auto=".1f"
                    )
                    score_fig.update_layout(xaxis_tickangle=-30, height=300, margin=dict(t=40, b=40, l=40, r=20))
                    st.plotly_chart(score_fig, use_container_width=True)
                else:
                    st.info("No successful score records available to chart.")
            else:
                st.markdown("""
                <div style="border: 1px dashed #64748b; border-radius: 8px; padding: 30px; text-align: center; height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">📉 Accuracy Chart Unavailable</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        No reference ground truth or rubric was supplied.<br/>
                        To prevent misleading zero-accuracy representations, accuracy scoring is omitted for Mode C.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Individual Model Result Cards
        st.subheader("💬 Individual Model Responses & In-Depth Details")
        for r in comp.results:
            with st.container():
                is_mock = getattr(r, "is_simulated", r.provider == "mock" or "mock" in r.model_id.lower())
                sim_badge = " [SIMULATED / MOCK]" if is_mock else ""
                st.markdown(f"#### 🤖 {r.display_name}{sim_badge} (`{r.provider.upper()}`)")
                if is_mock:
                    st.info("ℹ️ **Simulated Offline Output:** This response was deterministically generated by a local test adapter for demonstration and testing. It is **not** an external model API response.")
                card_badge = "🟢 SUCCESS" if r.status == "success" else f"🔴 {r.status.upper()}"
                meta_cols = st.columns(5)
                meta_cols[0].markdown(f"**Status:** `{card_badge}`")
                meta_cols[1].markdown(f"**Latency:** `{r.latency_ms:.0f} ms`")
                meta_cols[2].markdown(f"**Tokens:** `{r.total_tokens}` ({r.prompt_tokens} in / {r.completion_tokens} out)")
                meta_cols[3].markdown(f"**Est. Cost:** `${r.estimated_cost_usd:.5f}`")
                meta_cols[4].markdown(f"**Finish:** `{r.finish_reason or 'stop'}`")

                if r.status == "success":
                    st.markdown("**Complete Response:**")
                    st.code(r.response_text, language="text", wrap_lines=True)

                    if r.evaluation_mode == "reference":
                        st.markdown(f"**Verdict:** `{'PASS' if r.is_correct else 'FAIL'}` &nbsp;|&nbsp; **Score:** `{r.accuracy_score}` &nbsp;|&nbsp; **Method:** `{r.evaluation_method}`")
                        if r.score_explanation:
                            st.caption(f"Reasoning: {r.score_explanation}")
                    elif r.evaluation_mode == "rubric":
                        st.markdown(f"**Rubric Score:** `{r.accuracy_score*100:.1f}%`")
                        if r.rubric_scores:
                            crit_cols = st.columns(min(3, len(r.rubric_scores)))
                            for idx, (crit_name, c_score) in enumerate(r.rubric_scores.items()):
                                with crit_cols[idx % len(crit_cols)]:
                                    crit_tag = "✅ Met" if c_score >= 0.5 else "❌ Missing"
                                    st.markdown(f"• **{crit_name}**: `{crit_tag}` ({c_score*100:.0f}%)")
                        if r.missing_key_points:
                            st.markdown(f"⚠️ **Missing Key Points:** {', '.join(r.missing_key_points)}")
                        if r.unsupported_claims:
                            st.markdown(f"⚠️ **Unsupported Claims:** {', '.join(r.unsupported_claims)}")
                        st.caption(f"Explanation: {r.score_explanation}")
                    else:
                        st.markdown(f"**Quality Heuristics:** Completeness: `{int((r.completeness_score or 0)*100)}%` &nbsp;|&nbsp; Relevance: `{int((r.relevance_score or 0)*100)}%` &nbsp;|&nbsp; Instruction Following: `{int((r.instruction_following_score or 0)*100)}%`")
                        if r.uncertainty_detected:
                            st.warning(f"⚠️ Uncertainty Detected: {'; '.join(r.uncertainty_indicators)}")
                        st.caption(f"Status: {r.score_explanation}")

                else:
                    st.error(f"❌ Execution Failure: {r.sanitized_error or r.error_message or 'Unknown provider error'}")
                    with st.expander("Diagnostics & Troubleshooting Info"):
                        st.write(f"Provider: {r.provider}")
                        st.write(f"Model ID: {r.model_id}")
                        st.write(f"Error: {r.sanitized_error or r.error_message}")
                        err_text = (r.sanitized_error or r.error_message or "").lower()
                        if r.status == "unconfigured":
                            st.info(f"💡 **Configuration Needed:** To configure this provider, add {r.provider.upper()}_API_KEY to your .env file or environment.")
                        elif "credits" in err_text or "429" in err_text:
                            st.warning("💡 **Action Required:** Your provider account has exhausted its quota or credits. Visit your provider's account dashboard to verify balance and add credits.")
                        elif "404" in err_text or "not exist" in err_text or "no longer available" in err_text:
                            st.warning("💡 **Action Required:** This model ID returned HTTP 404 (model retired or unavailable on your account tier). Please update your model selection to an active model.")
                        elif r.provider == "ollama" or "ollama" in err_text:
                            st.warning("💡 **Action Required:** Local Ollama daemon is unreachable. Start the service with `ollama serve` and ensure local weights are downloaded (`ollama pull llama3:8b`).")

                st.caption(f"Pricing Model: {r.pricing_assumptions}")
                st.divider()

        # Export Comparison Controls
        st.subheader("📥 Export Comparison Results")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = export_comparison_to_csv(comp)
            st.download_button(
                "📥 Download Results as CSV",
                data=csv_data,
                file_name=f"comparison_{comp.comparison_id}.csv",
                mime="text/csv",
                use_container_width=True,
                key="univ_download_csv_btn"
            )
        with exp_col2:
            json_data = export_comparison_to_json(comp)
            st.download_button(
                "📥 Download Results as JSON",
                data=json_data,
                file_name=f"comparison_{comp.comparison_id}.json",
                mime="application/json",
                use_container_width=True,
                key="univ_download_json_btn"
            )


# =====================================================================
# TAB 1: DATASET & TEST SUITE MANAGER
# =====================================================================
with tab1:
    st.header("📂 Dataset & Test Suite Manager")
    st.markdown("Inspect, validate, and customize standardized evaluation test cases across domains and difficulty tiers.")

    # Summary KPI cards
    all_tcs: List[TestCase] = st.session_state["test_cases"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Test Cases</div>
            <div class="metric-value">{len(all_tcs)}</div>
            <div class="metric-sub">{len(st.session_state['selected_test_case_ids'])} selected for testing</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        domains_count = len(set(tc.domain for tc in all_tcs))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Domains Covered</div>
            <div class="metric-value">{domains_count}</div>
            <div class="metric-sub">Math, Logic, Coding, Extraction</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        hard_count = sum(1 for tc in all_tcs if tc.difficulty in ["Hard", "Expert"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Hard / Expert Cases</div>
            <div class="metric-value">{hard_count}</div>
            <div class="metric-sub">{round((hard_count/max(1, len(all_tcs)))*100)}% of suite</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        eval_methods_count = len(set(tc.evaluation_method for tc in all_tcs))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Evaluation Methods</div>
            <div class="metric-value">{eval_methods_count}</div>
            <div class="metric-sub">Exact, Numeric, Regex, JSON, Semantic</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Upload and Download Section
    with st.expander("📤 Upload Custom Dataset (JSON or CSV) / Download Templates"):
        up_col1, up_col2 = st.columns([2, 1])
        with up_col1:
            uploaded_file = st.file_uploader("Upload test dataset file", type=["json", "csv"])
            if uploaded_file is not None:
                try:
                    loaded_cases = []
                    validation_errors = []
                    if uploaded_file.name.endswith(".json"):
                        data = json.load(uploaded_file)
                        for idx, item in enumerate(data):
                            try:
                                loaded_cases.append(TestCase(**item))
                            except Exception as e:
                                validation_errors.append(f"Row {idx+1}: {str(e)}")
                    elif uploaded_file.name.endswith(".csv"):
                        df_up = pd.read_csv(uploaded_file)
                        for idx, row in df_up.iterrows():
                            try:
                                row_dict = row.dropna().to_dict()
                                # Parse list strings if present
                                if "constraints" in row_dict and isinstance(row_dict["constraints"], str):
                                    row_dict["constraints"] = [c.strip() for c in row_dict["constraints"].split(";")]
                                loaded_cases.append(TestCase(**row_dict))
                            except Exception as e:
                                validation_errors.append(f"Row {idx+1}: {str(e)}")

                    if validation_errors:
                        st.error(f"Found {len(validation_errors)} validation errors:")
                        for err in validation_errors[:10]:
                            st.write(f"- {err}")
                    else:
                        st.success(f"Successfully loaded and validated {len(loaded_cases)} test cases!")
                        if st.button("Apply Uploaded Dataset"):
                            st.session_state["test_cases"] = loaded_cases
                            st.session_state["selected_test_case_ids"] = {tc.id for tc in loaded_cases}
                            st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse file: {str(e)}")

        with up_col2:
            st.markdown("**Download Example Templates:**")
            # Example JSON template
            sample_json = json.dumps([
                {
                    "id": "custom_01",
                    "domain": "Math",
                    "category": "Algebra",
                    "difficulty": "Easy",
                    "problem_statement": "What is 2x + 5 = 15? Solve for x.",
                    "ground_truth": "5",
                    "evaluation_method": "numeric_tolerance",
                    "tolerance": 0.0,
                    "constraints": ["Single integer only"]
                }
            ], indent=2)
            st.download_button("Download JSON Template", data=sample_json, file_name="sample_template.json", mime="application/json")

            # Example CSV template
            sample_csv = "id,domain,category,difficulty,problem_statement,ground_truth,evaluation_method,tolerance,constraints\ncustom_01,Math,Algebra,Easy,What is 2x + 5 = 15? Solve for x.,5,numeric_tolerance,0.0,Single integer only\n"
            st.download_button("Download CSV Template", data=sample_csv, file_name="sample_template.csv", mime="text/csv")

    # Search & Filter Controls
    fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])
    with fcol1:
        search_query = st.text_input("🔍 Search problem statement or ground truth", "")
    with fcol2:
        available_domains = ["All"] + sorted(list(set(tc.domain for tc in all_tcs)))
        sel_domain = st.selectbox("Filter Domain", available_domains)
    with fcol3:
        available_diffs = ["All", "Easy", "Medium", "Hard", "Expert"]
        sel_diff = st.selectbox("Filter Difficulty", available_diffs)
    with fcol4:
        available_methods = ["All", "exact_match", "numeric_tolerance", "regex", "json_schema", "semantic", "auto"]
        sel_method = st.selectbox("Filter Eval Method", available_methods)

    # Filtered test cases
    filtered_tcs = []
    for tc in all_tcs:
        if sel_domain != "All" and tc.domain != sel_domain:
            continue
        if sel_diff != "All" and tc.difficulty != sel_diff:
            continue
        if sel_method != "All" and tc.evaluation_method != sel_method:
            continue
        if search_query:
            q = search_query.lower()
            if q not in tc.problem_statement.lower() and q not in tc.ground_truth.lower() and q not in tc.id.lower():
                continue
        filtered_tcs.append(tc)

    # Selection helpers
    scol1, scol2, _ = st.columns([1, 1, 4])
    with scol1:
        if st.button("Select All Filtered"):
            st.session_state["selected_test_case_ids"].update({tc.id for tc in filtered_tcs})
            st.rerun()
    with scol2:
        if st.button("Deselect All Filtered"):
            st.session_state["selected_test_case_ids"].difference_update({tc.id for tc in filtered_tcs})
            st.rerun()

    # Render interactive table of test cases
    tc_table_data = []
    for tc in filtered_tcs:
        is_sel = tc.id in st.session_state["selected_test_case_ids"]
        tc_table_data.append({
            "Selected": is_sel,
            "ID": tc.id,
            "Domain": tc.domain,
            "Category": tc.category,
            "Difficulty": tc.difficulty,
            "Eval Method": tc.evaluation_method,
            "Problem Statement": tc.problem_statement[:80] + ("..." if len(tc.problem_statement) > 80 else ""),
            "Ground Truth": tc.ground_truth[:40] + ("..." if len(tc.ground_truth) > 40 else "")
        })

    tc_df = pd.DataFrame(tc_table_data)
    edited_df = st.data_editor(
        tc_df,
        column_config={
            "Selected": st.column_config.CheckboxColumn("Include in Run", default=True),
            "ID": st.column_config.TextColumn("Test ID", width="small"),
            "Domain": st.column_config.TextColumn("Domain", width="small"),
            "Difficulty": st.column_config.TextColumn("Difficulty", width="small"),
            "Eval Method": st.column_config.TextColumn("Method", width="small")
        },
        disabled=["ID", "Domain", "Category", "Difficulty", "Eval Method", "Problem Statement", "Ground Truth"],
        hide_index=True,
        use_container_width=True
    )

    # Sync selection changes
    for _, row in edited_df.iterrows():
        t_id = row["ID"]
        if row["Selected"]:
            st.session_state["selected_test_case_ids"].add(t_id)
        else:
            st.session_state["selected_test_case_ids"].discard(t_id)

    # Individual detail inspector
    if filtered_tcs:
        with st.expander("🔍 Deep-Dive Test Case Inspector"):
            selected_inspect_id = st.selectbox("Select Test Case ID to view full details", [tc.id for tc in filtered_tcs])
            selected_tc = next(tc for tc in filtered_tcs if tc.id == selected_inspect_id)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown(f"**Problem Statement:**\n```\n{selected_tc.problem_statement}\n```")
                st.markdown(f"**Ground Truth:**\n```\n{selected_tc.ground_truth}\n```")
            with dcol2:
                st.markdown(f"- **Domain:** `{selected_tc.domain}`")
                st.markdown(f"- **Category:** `{selected_tc.category}`")
                st.markdown(f"- **Difficulty:** `{selected_tc.difficulty}`")
                st.markdown(f"- **Evaluation Method:** `{selected_tc.evaluation_method}`")
                if selected_tc.constraints:
                    st.markdown(f"- **Constraints:** {'; '.join(selected_tc.constraints)}")
                if selected_tc.tolerance is not None:
                    st.markdown(f"- **Numeric Tolerance:** `{selected_tc.tolerance}`")
                if selected_tc.regex_pattern:
                    st.markdown(f"- **Regex Pattern:** `{selected_tc.regex_pattern}`")


# =====================================================================
# TAB 2: RUN BENCHMARKS
# =====================================================================
with tab2:
    st.header("🚀 Benchmark Execution Engine")
    st.markdown("Configure target LLMs, concurrency bounding, evaluation judges, and launch standardized comparative runs.")

    registered_models = list_registered_models(active_only=True)
    model_choices = {m["model_id"]: m for m in registered_models}

    # Configuration Form
    with st.container():
        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            st.subheader("1. Candidate Model Selection")
            default_selection = ["mock-gpt", "mock-claude", "mock-llama"]

            def format_bench_model_label(mid: str) -> str:
                m = model_choices.get(mid, {})
                prov = m.get("provider", "")
                v_info = st.session_state.get("verified_connections", {}).get(prov, {})
                if prov == "mock":
                    tag = "Ready"
                elif v_info.get("status") == "connected":
                    tag = "Connected"
                elif v_info.get("status") == "unavailable":
                    tag = "Unavailable"
                elif m.get("configured", False):
                    tag = "Configured"
                else:
                    tag = "Unavailable"
                return f"{m.get('display_name', mid)} ({prov.upper()}) [{tag}]"

            selected_model_ids = st.multiselect(
                "Select Models to Benchmark",
                options=list(model_choices.keys()),
                default=default_selection,
                format_func=format_bench_model_label
            )

            # Check provider credential warnings
            has_unconfigured = False
            for mid in selected_model_ids:
                if not model_choices[mid]["configured"] and model_choices[mid]["provider"] != "mock":
                    st.warning(f"⚠️ Provider `{model_choices[mid]['provider']}` for `{mid}` is not configured with an API key. Requests may fail unless credentials are provided.")
                    has_unconfigured = True

            st.divider()
            st.subheader("2. Hyperparameters & Concurrency")
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                run_temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05,
                                           help="Non-zero temperature reduces benchmark reproducibility.")
                if run_temperature > 0.0:
                    st.caption("⚠️ Non-zero temperature selected; results may exhibit variance between runs.")
                max_tokens = st.number_input("Max Completion Tokens", min_value=64, max_value=8192, value=1024, step=128)
            with param_col2:
                concurrency_limit = st.slider("Max Concurrent Requests", 1, 10, 5)
                timeout_sec = st.number_input("Request Timeout (seconds)", min_value=10, max_value=180, value=60, step=10)

        with cfg_col2:
            st.subheader("3. Evaluation Judge Configuration")
            enable_judge = st.checkbox("Enable LLM-as-a-Judge Grading", value=True,
                                      help="Evaluates open-ended reasoning, coding, and instruction adherence.")
            judge_provider = st.selectbox("Judge Provider", ["mock", "nvidia", "deepseek", "groq"], index=0)
            judge_defaults = {
                "mock": "mock-judge-v1",
                "nvidia": "nvidia/llama-3.1-nemotron-70b-instruct",
                "deepseek": "deepseek-chat",
                "groq": "openai/gpt-oss-120b"
            }
            judge_model = st.text_input("Judge Model Name", judge_defaults.get(judge_provider, "mock-judge-v1"))

            st.divider()
            st.subheader("4. System Prompt & Test Subset")
            system_prompt = st.text_area(
                "Optional Uniform System Prompt",
                value="You are an expert, precise AI assistant. Follow all instructions and format requirements exactly.",
                help="Applied uniformly across all evaluated models."
            )

            active_test_count = len(st.session_state["selected_test_case_ids"])
            test_subset_choice = st.radio(
                "Test Suite Execution Scope",
                [f"Run on selected test cases ({active_test_count} cases)", f"Run on entire suite ({len(all_tcs)} cases)"],
                index=0
            )

    st.divider()

    # Pre-execution confirmation and trigger
    run_cases = [tc for tc in all_tcs if (tc.id in st.session_state["selected_test_case_ids"] or "entire suite" in test_subset_choice)]
    total_calls = len(run_cases) * len(selected_model_ids)

    exec_col1, exec_col2 = st.columns([3, 1])
    with exec_col1:
        st.write(f"📊 **Planned Benchmark Matrix:** `{len(selected_model_ids)}` models × `{len(run_cases)}` test cases = **`{total_calls}` total model requests**")

    start_button = st.button("🚀 Start Benchmark Run", type="primary", disabled=len(selected_model_ids) == 0 or len(run_cases) == 0)

    if start_button:
        # Build model configs
        active_configs = []
        for mid in selected_model_ids:
            meta = model_choices[mid]
            pricing = get_token_pricing(mid)
            active_configs.append(ModelConfig(
                provider=meta["provider"],
                model_name=mid,
                enabled=True,
                temperature=run_temperature,
                max_completion_tokens=max_tokens,
                timeout_seconds=timeout_sec,
                retry_count=2,
                cost_input_per_1m_tokens=pricing["input"],
                cost_output_per_1m_tokens=pricing["output"]
            ))

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        status_text.info("Initiating benchmark runner and connection pools...")

        runner = BenchmarkRunner(repository=repo, max_concurrent=concurrency_limit, max_retries=2)

        def on_progress(completed: int, total: int, msg: str):
            frac = completed / max(1, total)
            progress_bar.progress(frac)
            status_text.text(f"Progress: {completed}/{total} ({frac:.0%}) | {msg}")

        # Run async benchmark loop
        run_config = {
            "name": f"Benchmark - {datetime.now(timezone.utc).strftime('%b %d, %H:%M')}",
            "system_prompt": system_prompt,
            "enable_semantic_judge": enable_judge,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "concurrency": concurrency_limit
        }

        with st.spinner("Executing concurrent requests..."):
            completed_run = asyncio.run(runner.run_benchmark(
                test_cases=run_cases,
                model_configs=active_configs,
                run_config=run_config,
                progress_callback=on_progress
            ))

        # Retrieve saved results
        results = repo.get_results_for_run(completed_run.run_id)
        st.session_state["current_run"] = completed_run
        st.session_state["current_results"] = results

        progress_bar.progress(1.0)
        if completed_run.status == "failed" or completed_run.completed_requests == 0:
            status_text.error(f"❌ Benchmark Run `{completed_run.run_id}` failed: All requests failed ({completed_run.failed_requests}/{completed_run.total_requests} requests failed with provider errors).")
        elif completed_run.failed_requests > 0:
            status_text.warning(f"⚠️ Benchmark Run `{completed_run.run_id}` completed with errors: {completed_run.completed_requests} succeeded, {completed_run.failed_requests} failed.")
        else:
            status_text.success(f"✅ Benchmark Run `{completed_run.run_id}` completed successfully!")

        st.subheader("🏁 Run Summary Overview")
        sum_cols = st.columns(max(1, len(completed_run.model_metrics)))
        for idx, (m_name, metrics) in enumerate(completed_run.model_metrics.items()):
            with sum_cols[idx]:
                fail_rate = metrics.get('provider_failure_rate', 0.0)
                if fail_rate == 100.0:
                    st.markdown(f"""
                    <div class="metric-card" style="border: 1px solid #ef4444;">
                        <div class="metric-title">{m_name} <span class="badge-error">PROVIDER FAILED</span></div>
                        <div class="metric-value" style="color: #ef4444;">0.0%</div>
                        <div class="metric-sub" style="color: #fca5a5;">100% Provider / API Errors ({metrics.get('failed_requests', 0)} failed)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    fail_info = f" | Failures: {metrics.get('failed_requests', 0)}" if metrics.get('failed_requests', 0) > 0 else ""
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{m_name}</div>
                        <div class="metric-value">{metrics.get('binary_accuracy', 0.0)}%</div>
                        <div class="metric-sub">Mean Score: {metrics.get('mean_score', 0.0):.3f} | Latency: {metrics.get('median_latency_ms', 0):.0f}ms{fail_info}</div>
                    </div>
                    """, unsafe_allow_html=True)

        for m_name, metrics in completed_run.model_metrics.items():
            if metrics.get('provider_failure_rate', 0.0) == 100.0:
                sample_err = next((r.provider_error for r in results if r.model_name == m_name and r.provider_error), "Provider error")
                st.error(f"🚨 **Provider Failure for `{m_name}`:** All requests failed with provider error:\n```\n{sample_err}\n```\n*Check your API key quota, balance, or billing at your provider console.*")

        st.info("👉 Switch to **📊 3. Comparative Analytics** or **🔬 4. Root-Cause Diagnostics** to explore interactive charts and failure analysis.")


# =====================================================================
# TAB 3: COMPARATIVE ANALYTICS
# =====================================================================
with tab3:
    st.header("📊 Comparative Performance Analytics")
    st.markdown("Interactive multi-dimensional evaluation charts comparing accuracy, latency, costs, and reliability.")

    # Select Run from history
    historical_runs = repo.list_runs()
    if not historical_runs:
        st.info("No benchmark runs recorded yet. Execute a benchmark in Tab 2 to generate analytics.")
    else:
        run_options = {r.run_id: f"{r.name or r.run_id} ({r.start_time[:16]})" for r in historical_runs}
        selected_run_id = st.selectbox(
            "Select Benchmark Run to Analyze",
            options=list(run_options.keys()),
            format_func=lambda x: run_options[x]
        )

        curr_results = repo.get_results_for_run(selected_run_id)
        curr_run = repo.get_run(selected_run_id)
        if not curr_results:
            st.warning("No evaluation results associated with this run.")
        else:
            df_results = export_results_df(curr_results)

            if curr_run and (curr_run.status == "failed" or curr_run.completed_requests == 0):
                sample_err = next((r.provider_error for r in curr_results if r.provider_error), "Provider / API Error")
                st.error(f"🚨 **Benchmark Run `{selected_run_id}` encountered provider failure ({curr_run.failed_requests}/{curr_run.total_requests} requests failed).**\n\nProvider Error: `{sample_err}`\n\n*Note: Zero accuracy (0.0%), zero mean score (0.000), and zero latency (0ms) indicate that requests were rejected by the provider API before execution.*")

            # Join domain and difficulty from test cases for segmentation
            tc_meta_map = {tc.id: (tc.domain, tc.difficulty) for tc in st.session_state["test_cases"]}
            df_results["Domain"] = df_results["Test Case ID"].map(lambda tid: tc_meta_map.get(tid, ("General", "Medium"))[0])
            df_results["Difficulty"] = df_results["Test Case ID"].map(lambda tid: tc_meta_map.get(tid, ("General", "Medium"))[1])

            # Filters row
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                models_filter = st.multiselect("Filter Models", options=df_results["Model Name"].unique(), default=df_results["Model Name"].unique())
            with f_col2:
                domains_filter = st.multiselect("Filter Domains", options=df_results["Domain"].unique(), default=df_results["Domain"].unique())
            with f_col3:
                diff_filter = st.multiselect("Filter Difficulties", options=df_results["Difficulty"].unique(), default=df_results["Difficulty"].unique())

            filtered_df = df_results[
                (df_results["Model Name"].isin(models_filter)) &
                (df_results["Domain"].isin(domains_filter)) &
                (df_results["Difficulty"].isin(diff_filter))
            ]

            # Statistical Performance Leaderboard Table
            st.subheader("📋 Statistical Performance Leaderboard")
            leaderboard_data = []
            for m in models_filter:
                m_records = [r for r in curr_results if r.model_name == m and r.test_case_id in set(filtered_df["Test Case ID"])]
                if m_records:
                    stats = compute_model_statistics(m_records)
                    leaderboard_data.append({
                        "Model": m,
                        "Accuracy (%)": f"{stats['binary_accuracy']:.1f}%",
                        "95% CI (Wilson)": f"[{stats.get('ci_95_lower', 0.0):.1f}%, {stats.get('ci_95_upper', 0.0):.1f}%]",
                        "Mean Score": f"{stats['mean_score']:.3f}",
                        "Success Rate": f"{stats['success_rate']:.1f}%",
                        "Timeout Rate": f"{stats.get('timeout_rate', 0.0):.1f}%",
                        "Provider Error Rate": f"{stats.get('provider_failure_rate', 0.0):.1f}%",
                        "Median Latency (ms)": f"{stats['median_latency_ms']:.0f}",
                        "Avg Tokens": f"{stats['avg_tokens_per_req']:.1f}",
                        "Cost / 1K ($)": f"${stats['cost_per_1k_usd']:.4f}"
                    })
            if leaderboard_data:
                st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)

            st.divider()

            # Row 1 Charts: Overall Accuracy & Accuracy by Domain
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                st.subheader("1. Overall Accuracy (%)")
                acc_summary = filtered_df.groupby("Model Name")["Is Correct"].agg(
                    Total="count",
                    Correct=lambda x: (x == True).sum()
                ).reset_index()
                acc_summary["Accuracy (%)"] = (acc_summary["Correct"] / acc_summary["Total"]) * 100.0

                fig_acc = px.bar(
                    acc_summary,
                    x="Model Name",
                    y="Accuracy (%)",
                    color="Model Name",
                    text_auto=".1f",
                    title="Binary Accuracy by Model",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_acc.update_layout(yaxis_range=[0, 105], showlegend=False)
                st.plotly_chart(fig_acc, use_container_width=True)

            with r1_c2:
                st.subheader("2. Accuracy by Domain")
                domain_acc = filtered_df.groupby(["Domain", "Model Name"])["Is Correct"].agg(
                    Total="count",
                    Correct=lambda x: (x == True).sum()
                ).reset_index()
                domain_acc["Accuracy (%)"] = (domain_acc["Correct"] / domain_acc["Total"]) * 100.0

                fig_domain = px.bar(
                    domain_acc,
                    x="Domain",
                    y="Accuracy (%)",
                    color="Model Name",
                    barmode="group",
                    text_auto=".0f",
                    title="Accuracy Across Domains",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_domain.update_layout(yaxis_range=[0, 105])
                st.plotly_chart(fig_domain, use_container_width=True)

            st.divider()

            # Row 2 Charts: Latency vs Accuracy & Token Usage
            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                st.subheader("3. Latency vs. Accuracy")
                tot_token_col = next((c for c in ["Total Tokens", "TotalTokens", "total_tokens"] if c in filtered_df.columns), None)
                lat_col = next((c for c in ["Execution Time (ms)", "latency_ms", "execution_time_ms"] if c in filtered_df.columns), None)

                if filtered_df.empty or not lat_col:
                    st.info("ℹ️ Latency and accuracy data not available for the current selection.")
                else:
                    agg_dict = {
                        "MeanLatency": (lat_col, "mean"),
                        "Accuracy": ("Is Correct", lambda x: (x == True).sum() / len(x) * 100.0 if len(x) > 0 else 0.0)
                    }
                    if tot_token_col:
                        agg_dict["AvgTokens"] = (tot_token_col, "mean")
                    scatter_df = filtered_df.groupby("Model Name").agg(**agg_dict).reset_index()
                    if "AvgTokens" not in scatter_df.columns or scatter_df["AvgTokens"].fillna(0).sum() == 0:
                        scatter_df["AvgTokens"] = 10  # fallback non-zero bubble size

                    fig_scatter = px.scatter(
                        scatter_df,
                        x="MeanLatency",
                        y="Accuracy",
                        size="AvgTokens",
                        color="Model Name",
                        text="Model Name",
                        title="Mean Latency (ms) vs. Binary Accuracy (%) [Bubble = Avg Tokens]",
                        labels={"MeanLatency": "Mean Latency (ms)", "Accuracy": "Accuracy (%)"}
                    )
                    fig_scatter.update_traces(textposition="top center")
                    fig_scatter.update_layout(yaxis_range=[0, 105])
                    st.plotly_chart(fig_scatter, use_container_width=True)

            with r2_c2:
                st.subheader("4. Token Usage Breakdown")
                prompt_col = next((c for c in ["Prompt Tokens", "PromptTokens", "prompt_tokens"] if c in filtered_df.columns), None)
                completion_col = next((c for c in ["Completion Tokens", "CompletionTokens", "completion_tokens"] if c in filtered_df.columns), None)

                if filtered_df.empty:
                    st.info("ℹ️ No records match the current filter selection for token usage analysis.")
                elif not prompt_col or not completion_col:
                    st.info("ℹ️ Token usage metrics are not available in the selected benchmark results.")
                else:
                    token_df = filtered_df.groupby("Model Name").agg(
                        PromptTokens=(prompt_col, "mean"),
                        CompletionTokens=(completion_col, "mean")
                    ).reset_index()

                    # Guarantee both naming conventions exist on the resulting DataFrame
                    token_df["Prompt Tokens"] = token_df["PromptTokens"]
                    token_df["Completion Tokens"] = token_df["CompletionTokens"]

                    total_prompt = token_df["PromptTokens"].fillna(0).sum()
                    total_completion = token_df["CompletionTokens"].fillna(0).sum()

                    if token_df.empty or (total_prompt == 0 and total_completion == 0):
                        st.info("ℹ️ Token usage was not recorded or returned zero tokens for the selected models.")
                    else:
                        fig_tokens = go.Figure(data=[
                            go.Bar(name='Prompt Tokens', x=token_df['Model Name'], y=token_df['PromptTokens'], marker_color='#38bdf8'),
                            go.Bar(name='Completion Tokens', x=token_df['Model Name'], y=token_df['CompletionTokens'], marker_color='#f43f5e')
                        ])
                        fig_tokens.update_layout(
                            barmode='stack',
                            title="Average Token Distribution per Request",
                            yaxis_title="Average Tokens",
                            xaxis_title="Model Name"
                        )
                        st.plotly_chart(fig_tokens, use_container_width=True)

            st.divider()

            # Row 3 Charts: Root-Cause Error Distribution & Difficulty Analysis
            r3_c1, r3_c2 = st.columns(2)
            with r3_c1:
                st.subheader("5. Root-Cause Error Distribution")
                err_df = filtered_df[filtered_df["Is Correct"] == False]
                if not err_df.empty:
                    err_counts = err_df.groupby(["Model Name", "Primary Root Cause"]).size().reset_index(name="Count")
                    fig_err = px.bar(
                        err_counts,
                        x="Model Name",
                        y="Count",
                        color="Primary Root Cause",
                        title="Failure Distribution by Root Cause",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_err, use_container_width=True)
                else:
                    st.success("🎉 No failure cases found in current filter selection!")

            with r3_c2:
                st.subheader("6. Performance by Difficulty")
                diff_summary = filtered_df.groupby(["Difficulty", "Model Name"])["Is Correct"].agg(
                    Total="count",
                    Correct=lambda x: (x == True).sum()
                ).reset_index()
                diff_summary["Accuracy (%)"] = (diff_summary["Correct"] / diff_summary["Total"]) * 100.0

                fig_diff = px.bar(
                    diff_summary,
                    x="Difficulty",
                    y="Accuracy (%)",
                    color="Model Name",
                    barmode="group",
                    category_orders={"Difficulty": ["Easy", "Medium", "Hard", "Expert"]},
                    title="Accuracy by Problem Difficulty",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_diff.update_layout(yaxis_range=[0, 105])
                st.plotly_chart(fig_diff, use_container_width=True)


# =====================================================================
# TAB 4: ROOT-CAUSE DIAGNOSTICS
# =====================================================================
with tab4:
    st.header("🔬 Root-Cause Error Diagnostics")
    st.markdown("Investigate specific failure modes, compare models side-by-side, and explore corrective engineering actions.")

    historical_runs = repo.list_runs()
    if not historical_runs:
        st.info("No benchmark data available. Run a benchmark in Tab 2 to view diagnostics.")
    else:
        run_options = {r.run_id: f"{r.name or r.run_id} ({r.start_time[:16]})" for r in historical_runs}
        sel_diag_run_id = st.selectbox(
            "Select Benchmark Run for Diagnostics",
            options=list(run_options.keys()),
            format_func=lambda x: run_options[x],
            key="diag_run_selector"
        )

        diag_results = repo.get_results_for_run(sel_diag_run_id)
        df_diag = export_results_df(diag_results)

        # Filters
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            cat_filter = st.selectbox(
                "Filter by Error Category",
                ["All Failure Categories", "All (Including Success)"] + PRIMARY_CATEGORIES
            )
        with dcol2:
            model_diag_filter = st.selectbox(
                "Filter by Model",
                ["All Models"] + list(df_diag["Model Name"].unique())
            )
        with dcol3:
            search_diag = st.text_input("🔍 Search Test Case ID or text", "")

        # Apply filters
        active_diag_df = df_diag.copy()
        if cat_filter == "All Failure Categories":
            active_diag_df = active_diag_df[active_diag_df["Primary Root Cause"] != "Success"]
        elif cat_filter != "All (Including Success)":
            active_diag_df = active_diag_df[active_diag_df["Primary Root Cause"] == cat_filter]

        if model_diag_filter != "All Models":
            active_diag_df = active_diag_df[active_diag_df["Model Name"] == model_diag_filter]

        if search_diag:
            sd = search_diag.lower()
            active_diag_df = active_diag_df[
                active_diag_df["Test Case ID"].str.lower().str.contains(sd) |
                active_diag_df["Response Text"].str.lower().str.contains(sd)
            ]

        st.subheader(f"Diagnostic Cases ({len(active_diag_df)} records)")
        st.dataframe(
            active_diag_df[[
                "Test Case ID", "Model Name", "Primary Root Cause", "Secondary Root Cause",
                "Accuracy Score", "Execution Time (ms)", "Status"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Detailed Case Inspector
        if not active_diag_df.empty:
            st.subheader("🔍 Deep-Dive Failure Inspector")
            target_tc_id = st.selectbox("Select Test Case ID to inspect", options=active_diag_df["Test Case ID"].unique())
            
            # Find matching test case from memory
            matched_tc = next((tc for tc in st.session_state["test_cases"] if tc.id == target_tc_id), None)
            if matched_tc:
                st.markdown(f"### Problem: `{matched_tc.id}` ({matched_tc.domain} - {matched_tc.difficulty})")
                st.markdown(f"**Problem Statement:**\n```\n{matched_tc.problem_statement}\n```")
                st.markdown(f"**Ground Truth:**\n```\n{matched_tc.ground_truth}\n```")

            # Side-by-side comparison across all models for this test case
            tc_models_df = df_diag[df_diag["Test Case ID"] == target_tc_id]
            st.markdown("#### Multi-Model Comparison for this Test Case")
            cols = st.columns(len(tc_models_df))
            for idx, (_, row) in enumerate(tc_models_df.iterrows()):
                with cols[idx]:
                    is_corr = row["Is Correct"]
                    badge = "badge-success" if is_corr else "badge-error"
                    verdict = "PASS" if is_corr else "FAIL"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{row['Model Name']}</div>
                        <div class="{badge}">{verdict} - {row['Primary Root Cause']}</div>
                        <div class="metric-sub" style="margin-top: 8px;">Score: {row['Accuracy Score']:.2f} | Latency: {row['Execution Time (ms)']:.0f}ms</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Candidate Output:**\n```\n{row['Response Text']}\n```")
                    if row["Deterministic Reasoning"]:
                        st.caption(f"**Deterministic:** {row['Deterministic Reasoning']}")
                    if row["Judge Reasoning"]:
                        st.caption(f"**Judge Reasoning:** {row['Judge Reasoning']}")
                    if row["Constraint Violations"]:
                        st.caption(f"**Violations:** {row['Constraint Violations']}")
                    if row.get("Provider Error"):
                        st.error(f"**Provider Error:** {row['Provider Error']}")


# =====================================================================
# TAB 5: EXPORT REPORT
# =====================================================================
with tab5:
    st.header("📑 Export Benchmark Reports")
    st.markdown("Export comprehensive evaluation logs, aggregate summaries, and portfolio-ready documentation.")

    historical_runs = repo.list_runs()
    if not historical_runs:
        st.info("No benchmark runs recorded yet.")
    else:
        run_options = {r.run_id: f"{r.name or r.run_id} ({r.start_time[:16]})" for r in historical_runs}
        export_run_id = st.selectbox(
            "Select Run to Export",
            options=list(run_options.keys()),
            format_func=lambda x: run_options[x],
            key="export_run_selector"
        )

        exp_run = repo.get_run(export_run_id)
        exp_results = repo.get_results_for_run(export_run_id)

        st.subheader("📥 Downloadable Formats")
        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            csv_data = export_results_csv(exp_results)
            st.download_button(
                "📄 Download Results CSV",
                data=csv_data,
                file_name=f"benchmark_{export_run_id}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dcol2:
            json_data = export_run_summary_json(exp_run, exp_results)
            st.download_button(
                "📦 Download Complete JSON Report",
                data=json_data,
                file_name=f"benchmark_{export_run_id}.json",
                mime="application/json",
                use_container_width=True
            )

        with dcol3:
            md_report = generate_consolidated_markdown_report(exp_run, exp_results)
            st.download_button(
                "📝 Download Markdown Report",
                data=md_report,
                file_name=f"benchmark_report_{export_run_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.divider()

        # In-Dashboard Markdown Preview
        st.subheader("👀 Markdown Report Preview")
        with st.expander("Expand to preview Markdown document", expanded=True):
            st.markdown(md_report)

        st.divider()

        # Run Comparator Section
        st.subheader("⚖️ Compare Two Historical Runs")
        if len(historical_runs) >= 2:
            cmp_c1, cmp_c2 = st.columns(2)
            with cmp_c1:
                run_a_id = st.selectbox("Base Run (A)", options=list(run_options.keys()), index=0)
            with cmp_c2:
                run_b_id = st.selectbox("Comparison Run (B)", options=list(run_options.keys()), index=min(1, len(historical_runs)-1))

            run_a = repo.get_run(run_a_id)
            run_b = repo.get_run(run_b_id)

            if run_a and run_b:
                cmp_data = []
                all_models = set(run_a.selected_models).union(set(run_b.selected_models))
                for m in all_models:
                    m_a = run_a.model_metrics.get(m, {})
                    m_b = run_b.model_metrics.get(m, {})
                    acc_a = m_a.get("binary_accuracy", 0.0)
                    acc_b = m_b.get("binary_accuracy", 0.0)
                    delta_acc = acc_b - acc_a
                    lat_a = m_a.get("median_latency_ms", 0.0)
                    lat_b = m_b.get("median_latency_ms", 0.0)
                    cmp_data.append({
                        "Model": m,
                        "Run A Accuracy (%)": f"{acc_a:.1f}%",
                        "Run B Accuracy (%)": f"{acc_b:.1f}%",
                        "Accuracy Δ": f"{delta_acc:+.1f}%",
                        "Run A Median Latency": f"{lat_a:.0f}ms",
                        "Run B Median Latency": f"{lat_b:.0f}ms",
                        "Latency Δ": f"{(lat_b - lat_a):+.0f}ms"
                    })
                st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

                # Paired statistical analysis between models across runs
                st.markdown("##### 🔬 Paired Statistical Significance Analysis (McNemar Test)")
                results_a = repo.get_results_for_run(run_a_id)
                results_b = repo.get_results_for_run(run_b_id)
                paired_rows = []
                for m in all_models:
                    m_res_a = [r for r in results_a if r.model_name == m]
                    m_res_b = [r for r in results_b if r.model_name == m]
                    if m_res_a and m_res_b:
                        pair_stats = compute_paired_comparison(m_res_a, m_res_b)
                        paired_rows.append({
                            "Model": m,
                            "Shared Tests": pair_stats["shared_tests"],
                            "Run A Wins": pair_stats["model_a_wins"],
                            "Run B Wins": pair_stats["model_b_wins"],
                            "Ties": pair_stats["ties"],
                            "McNemar Chi²": pair_stats["mcnemar_chi2"],
                            "Significant (p < 0.05)": "Yes" if pair_stats["statistically_significant"] else "No"
                        })
                if paired_rows:
                    st.dataframe(pd.DataFrame(paired_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Execute at least two runs to unlock comparative delta analysis.")
