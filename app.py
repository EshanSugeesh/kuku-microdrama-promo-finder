import os
import json
import zipfile
import tempfile
import importlib
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

import pipeline
importlib.reload(pipeline)

from pipeline import (
    load_screenplay_directory,
    parse_single_episode_text,
    read_any_file,
    extract_episode_number,
    run_full_pipeline,
    validate_and_rank_promos,
    parse_time_to_seconds,
    seconds_to_mmss,
    resolve_ui_provider_and_model
)

st.set_page_config(
    page_title="Kuku AI Microdrama Promo Finder",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Kuku AI Microdrama Promo Finder & Growth Ranking Engine")
st.caption("Automated 20-60s Promo Extraction, Hook & Cliffhanger Scoring, and Quality Guardrails for Vertical Microdramas")


def coalesce_contiguous_segments(promo: dict) -> dict:
    """
    Merges adjacent or overlapping segments within the same episode (e.g., 00:00->00:12 + 00:12->00:21)
    so single continuous scenes are never mislabeled as Multi-Segment Stitches.
    """
    segs = promo.get("segments", [])
    if len(segs) <= 1:
        return promo

    merged = []
    for s in segs:
        ep = int(s.get("episode", 1))
        st_sec = parse_time_to_seconds(str(s.get("start_time", "00:00")))
        en_sec = parse_time_to_seconds(str(s.get("end_time", "00:00")))
        if merged and merged[-1]["episode"] == ep and abs(st_sec - merged[-1]["_end_sec"]) <= 3:
            merged[-1]["_end_sec"] = max(merged[-1]["_end_sec"], en_sec)
            merged[-1]["end_time"] = seconds_to_mmss(merged[-1]["_end_sec"])
            merged[-1]["segment_duration_sec"] = merged[-1]["_end_sec"] - merged[-1]["_start_sec"]
        else:
            merged.append({
                "episode": ep,
                "start_time": seconds_to_mmss(st_sec),
                "end_time": seconds_to_mmss(en_sec),
                "segment_duration_sec": max(0, en_sec - st_sec),
                "_start_sec": st_sec,
                "_end_sec": en_sec
            })

    clean_merged = [
        {
            "episode": m["episode"],
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "segment_duration_sec": m["segment_duration_sec"]
        }
        for m in merged
    ]
    updated = dict(promo)
    updated["segments"] = clean_merged
    updated["total_duration_sec"] = sum(m["segment_duration_sec"] for m in clean_merged)
    updated["promo_type"] = "Multi-Segment Stitch" if len(clean_merged) > 1 else "Continuous"
    return updated


# =====================================================================
# SIDEBAR: MODEL SWAP & AUTO-NORMALIZED RUBRIC WEIGHTS (SUM = 1.00)
# =====================================================================
with st.sidebar:
    st.header("⚙️ Model Setup")

    MODEL_CATALOG = {
        "Auto (Best Available API)": ("Auto-Detect", ""),
        "Google Gemini 3 Flash": ("Google Gemini", "gemini-3-flash-preview"),
        "NVIDIA Nemotron 3.5 Super 120B": ("NVIDIA NIM", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
        "NVIDIA Llama 3.3 70B Instruct": ("NVIDIA NIM", "meta/llama-3.3-70b-instruct"),
        "OpenAI GPT-4o Mini (Interview Swap)": ("OpenAI", "gpt-4o-mini"),
        "Anthropic Claude 3.5 Sonnet (Interview Swap)": ("Anthropic", "claude-3-5-sonnet-latest"),
        "Custom Model / Endpoint...": ("Custom", "")
    }

    selected_label = st.selectbox("Select AI Model", list(MODEL_CATALOG.keys()))
    mapped_provider, mapped_model = MODEL_CATALOG[selected_label]

    custom_base_url = None
    if selected_label == "Custom Model / Endpoint...":
        mapped_model = st.text_input("Model ID", value="gpt-4o-mini")
        custom_base_url = st.text_input("Base URL (Optional)", value="https://api.openai.com/v1")

    user_api_key = st.text_input(
        "API Key Override (Optional)",
        type="password",
        help="Leave blank to use your .env keys (Gemini / NVIDIA), or paste an interviewer's API key to swap models live."
    )

    resolved_provider, resolved_model, active_api_key = resolve_ui_provider_and_model(
        mapped_provider, mapped_model, user_api_key
    )

    st.divider()
    with st.expander("🎚️ Customize Ranking Weights", expanded=True):
        raw_w_hook = st.slider("3s Hook Shock", 0.0, 1.0, 0.35, 0.05)
        raw_w_cliff = st.slider("Cliffhanger Tension", 0.0, 1.0, 0.30, 0.05)
        raw_w_clarity = st.slider("Standalone Clarity", 0.0, 1.0, 0.20, 0.05)
        raw_w_trope = st.slider("Trope Virality", 0.0, 1.0, 0.15, 0.05)

        weight_sum = raw_w_hook + raw_w_cliff + raw_w_clarity + raw_w_trope
        if weight_sum > 0:
            w_hook = raw_w_hook / weight_sum
            w_cliff = raw_w_cliff / weight_sum
            w_clarity = raw_w_clarity / weight_sum
            w_trope = raw_w_trope / weight_sum
        else:
            w_hook, w_cliff, w_clarity, w_trope = 0.35, 0.30, 0.20, 0.15

        st.caption(
            f"**Normalized (Sum = 1.00):**\n"
            f"Hook `{w_hook:.2f}` | Cliff `{w_cliff:.2f}` | Clarity `{w_clarity:.2f}` | Trope `{w_trope:.2f}`"
        )

# =====================================================================
# TOP BAR: INPUT SOURCE SELECTION
# =====================================================================
col_src1, col_src2 = st.columns([1, 2])
with col_src1:
    data_mode = st.radio(
        "Select Screenplay Input Source:",
        [
            "I Married a Broken Soldier. He was a Duke (50 Episodes)",
            "Upload Custom Microdrama Screenplay(s)"
        ]
    )

uploaded_files = None
if data_mode == "Upload Custom Microdrama Screenplay(s)":
    with col_src2:
        uploaded_files = st.file_uploader(
            "Upload a ZIP archive of episode files, or select TXT / DOCX / PDF screenplay files",
            type=["zip", "txt", "pdf", "docx", "md"],
            accept_multiple_files=True
        )


def load_from_uploads(files):
    episodes_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, uf in enumerate(files, start=1):
            if uf.name.lower().endswith(".zip"):
                zpath = Path(tmpdir) / uf.name
                zpath.write_bytes(uf.read())
                extract_dir = Path(tmpdir) / "extracted"
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(extract_dir)
                for sub_idx, fpath in enumerate(sorted(extract_dir.rglob("*")), start=1):
                    if fpath.is_file() and not fpath.name.startswith("."):
                        ep_num = extract_episode_number(fpath.name, sub_idx)
                        txt = read_any_file(fpath)
                        if txt.strip():
                            episodes_list.append(parse_single_episode_text(ep_num, txt, fpath.name))
            else:
                fpath = Path(tmpdir) / uf.name
                fpath.write_bytes(uf.read())
                ep_num = extract_episode_number(uf.name, idx)
                txt = read_any_file(fpath)
                if txt.strip():
                    episodes_list.append(parse_single_episode_text(ep_num, txt, uf.name))
    episodes_list.sort(key=lambda x: x["episode_num"])
    return episodes_list


if st.button("🚀 Run Promo Extraction & Ranking Pipeline", type="primary"):
    with st.spinner("Analyzing screenplay beats, extracting 20-60s promo cuts, and running Stage-3 quality checks..."):
        if data_mode.startswith("I Married a Broken Soldier"):
            if os.path.exists("data/input_scripts") and os.listdir("data/input_scripts"):
                eps = load_screenplay_directory("data/input_scripts")
            elif os.path.exists("raw_outputs/01_parsed_screenplay_50eps.json"):
                with open("raw_outputs/01_parsed_screenplay_50eps.json", "r", encoding="utf-8") as f:
                    eps = json.load(f)
            else:
                st.error("Screenplay files not found locally. Please use the Custom Screenplay Uploader.")
                st.stop()
        else:
            if not uploaded_files:
                st.error("Please upload at least one screenplay file or ZIP archive.")
                st.stop()
            eps = load_from_uploads(uploaded_files)

        run_out = run_full_pipeline(
            eps,
            provider=resolved_provider,
            model_name=resolved_model,
            api_key=user_api_key if user_api_key.strip() else active_api_key,
            batch_size=5,
            output_dir="raw_outputs",
            custom_base_url=custom_base_url,
            include_heuristic_supplement=False
        )
        st.session_state["pipeline_results"] = run_out
        st.toast("Promo extraction and ranking complete!", icon="✅")

# Auto-load precomputed results on startup
if "pipeline_results" not in st.session_state:
    stats_file = (
        "raw_outputs/episode_summary_stats.json"
        if os.path.exists("raw_outputs/episode_summary_stats.json")
        else "raw_outputs/01_parsed_screenplay_50eps.json"
    )
    if os.path.exists("raw_outputs/02_all_extracted_promos_raw.json") and os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            eps_cached = json.load(f)
        with open("raw_outputs/02_all_extracted_promos_raw.json", "r", encoding="utf-8") as f:
            raw_cand_cached = json.load(f)
        ranked_cached = validate_and_rank_promos(raw_cand_cached)
        st.session_state["pipeline_results"] = {
            "episodes": eps_cached,
            "accepted": ranked_cached["accepted"],
            "rejected": ranked_cached["rejected"]
        }
    elif os.path.exists("data/input_scripts") and os.listdir("data/input_scripts"):
        eps_init = load_screenplay_directory("data/input_scripts")
        st.session_state["pipeline_results"] = run_full_pipeline(
            eps_init,
            provider=resolved_provider,
            model_name=resolved_model,
            api_key=active_api_key,
            output_dir="raw_outputs"
        )

results = st.session_state.get("pipeline_results")
if not results:
    st.info("Select an input source above and click 'Run Promo Extraction & Ranking Pipeline' to begin.")
    st.stop()

episodes = results["episodes"]
accepted = [coalesce_contiguous_segments(p) for p in results["accepted"]]
rejected = [coalesce_contiguous_segments(r) for r in results["rejected"]]

# Dynamically re-score and re-sort accepted promos using normalized weights (sum = 1.00)
rescored_accepted = []
for p in accepted:
    raw_calc = round(
        w_hook * p["hook_shock_score"]
        + w_cliff * p["cliffhanger_tension_score"]
        + w_clarity * p["standalone_clarity_score"]
        + w_trope * p["trope_virality_score"]
        - p.get("spoiler_penalty_applied", 0.0),
        2
    )
    c_score = round(max(0.0, min(100.0, raw_calc)), 2)
    p_copy = dict(p)
    p_copy["custom_score"] = c_score
    rescored_accepted.append(p_copy)

rescored_accepted.sort(key=lambda x: x["custom_score"], reverse=True)
for idx_r, item in enumerate(rescored_accepted, start=1):
    item["live_rank"] = idx_r

# =====================================================================
# SUMMARY KPI METRICS (3 Clean Columns)
# =====================================================================
m1, m2, m3 = st.columns(3)
m1.metric("Episodes Parsed", len(episodes))
m2.metric("Validated Top Promos (20-60s)", len(rescored_accepted))
m3.metric("Rejected by Quality Gate", len(rejected))

# =====================================================================
# THREE CORE EVALUATION TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "🏆 Top Ranked Promos & Editor Trim",
    "🔥 Episode Tension Heatmap (Eps 1-50)",
    "🛡️ Quality Audit & Rejected Picks"
])

with tab1:
    st.subheader("Ranked Promo Leaderboard")
    approved_exports = []

    for promo in rescored_accepted:
        custom_score = promo["custom_score"]
        live_rank = promo["live_rank"]
        with st.expander(
            f"Rank #{live_rank} | {promo['promo_id']} ({promo['promo_type']}) - Duration: {promo['total_duration_sec']}s | Score: {custom_score}/100",
            expanded=(live_rank <= 3)
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Ordered Segments:**")
                for seg_i, seg in enumerate(promo["segments"], start=1):
                    st.code(
                        f"Segment {seg_i}: Episode {seg['episode']} | {seg['start_time']} -> {seg['end_time']} ({seg['segment_duration_sec']}s)"
                    )
                st.markdown(f"**⚡ First 3-Second Hook:** {promo['opening_hook_3s']}")
                st.markdown(f"**🛑 Unresolved Cliffhanger Cut:** {promo['cliffhanger_cut']}")
                st.markdown(f"**🌍 Standalone Clarity Check:** {promo['standalone_clarity_check']}")
                st.success(f"**💡 Why It Works:** {promo['why_it_works']}")

            with c2:
                st.markdown("**Rubric Breakdown:**")
                st.progress(int(promo["hook_shock_score"]), text=f"Hook Shock (0-3s): {int(promo['hook_shock_score'])}/100")
                st.progress(int(promo["cliffhanger_tension_score"]), text=f"Cliffhanger Cut: {int(promo['cliffhanger_tension_score'])}/100")
                st.progress(int(promo["standalone_clarity_score"]), text=f"Standalone Clarity: {int(promo['standalone_clarity_score'])}/100")
                st.progress(int(promo["trope_virality_score"]), text=f"Trope Virality: {int(promo['trope_virality_score'])}/100")

                st.divider()
                st.markdown("**👨‍💻 Editor Fine-Tuning (HITL):**")
                nudge_start = st.number_input("Nudge Start (sec)", -5, 5, 0, key=f"ns_{promo['promo_id']}")
                nudge_end = st.number_input("Nudge End (sec)", -5, 5, 0, key=f"ne_{promo['promo_id']}")
                is_approved = st.checkbox("✅ Approve for Video Editor Export", value=(live_rank <= 5), key=f"ap_{promo['promo_id']}")

                adjusted_segments = list(promo["segments"])
                if nudge_start != 0 or nudge_end != 0:
                    s0 = dict(adjusted_segments[0])
                    s_last = dict(adjusted_segments[-1])
                    new_s = seconds_to_mmss(parse_time_to_seconds(s0["start_time"]) + nudge_start)
                    new_e = seconds_to_mmss(parse_time_to_seconds(s_last["end_time"]) + nudge_end)
                    s0["start_time"] = new_s
                    s_last["end_time"] = new_e
                    adjusted_segments[0] = s0
                    adjusted_segments[-1] = s_last
                    st.caption(f"Adjusted Cut Window: `{new_s} -> {new_e}`")

                if is_approved:
                    export_item = dict(promo)
                    export_item["segments"] = adjusted_segments
                    approved_exports.append(export_item)

    if approved_exports:
        st.download_button(
            label=f"📥 Download {len(approved_exports)} Approved Promo Cuts (JSON)",
            data=json.dumps(approved_exports, indent=2, ensure_ascii=False),
            file_name="approved_kuku_promo_cuts.json",
            mime="application/json"
        )

with tab2:
    st.subheader("Narrative Tension & Promo Density Across All Episodes")
    df_ep = pd.DataFrame([
        {
            "Episode": ep["episode_num"],
            "Heuristic Tension Score": ep["total_tension_score"],
            "Candidate 20-60s Windows": len(ep.get("candidate_windows", [])),
            "Duration (sec)": ep["total_duration_sec"]
        }
        for ep in episodes
    ])
    fig = px.bar(
        df_ep,
        x="Episode",
        y="Heuristic Tension Score",
        color="Candidate 20-60s Windows",
        title="Episode-Wise Dramatic Tension Score (Stage 1 Kinetic Verbs + Status Tropes + Dialogue Pacing)",
        color_continuous_scale="Magma"
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df_ep, width="stretch")

with tab3:
    st.subheader("Quality Audit: Candidate Clips Rejected by Deterministic Guardrails")
    st.markdown(
        "Below are candidate picks flagged and filtered out by Stage 3 checks (duration violations outside 20-60s, lore-heavy dialogue lacking standalone context, duplicate scene overlaps, or late-series spoilers)."
    )
    for rej in rejected:
        with st.expander(f"❌ {rej['promo_id']} | Duration: {rej['total_duration_sec']}s | Clarity: {int(rej['standalone_clarity_score'])}/100", expanded=True):
            for reason in rej.get("rejection_reasons", []):
                st.error(reason)
            st.write("**Segments:**", rej["segments"])
            st.write("**Hook:**", rej["opening_hook_3s"])
            st.write("**Cliffhanger:**", rej["cliffhanger_cut"])
            st.write("**Initial Rationale:**", rej["why_it_works"])\
            