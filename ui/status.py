"""System Status page: models, GPU, checkpoints, environment."""

from __future__ import annotations

import subprocess

import streamlit as st

from core.config import LOG_DIR
from core.llm import llm_endpoint_ready
from ui.common import procedure_log, registry, settings

cfg = settings()
reg = registry()
log = procedure_log()

st.title("🛠️ System Status")

st.subheader("Models")
st.dataframe(reg.status(), hide_index=True, width="stretch")
st.caption(
    "Models load lazily on first use. Missing checkpoints can be fetched with "
    "`python downloader.py`."
)

load_col1, load_col2 = st.columns(2)
with load_col1:
    if st.button("Preload MedSAM2 now", use_container_width=True):
        with st.spinner("Loading MedSAM2 (image + video predictors)…"):
            try:
                reg.medsam2.image_predictor()
                reg.medsam2.video_predictor()
                st.toast("MedSAM2 loaded.")
            except Exception as exc:
                st.error(f"MedSAM2 load failed: {exc}")
        st.rerun()
with load_col2:
    if st.button("Preload Medical-SAM3 now (10 GB)", use_container_width=True):
        with st.spinner("Loading Medical-SAM3 — this takes a while…"):
            try:
                reg.medical_sam3.processor()
                st.toast("Medical-SAM3 loaded.")
            except Exception as exc:
                st.error(f"Medical-SAM3 load failed: {exc}")
        st.rerun()

st.subheader("GPUs")
try:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    rows = [
        dict(zip(["gpu", "name", "util", "mem_used", "mem_total"],
                 [part.strip() for part in line.split(",")]))
        for line in output.strip().splitlines()
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
except (OSError, subprocess.SubprocessError):
    st.caption("nvidia-smi is not available on this host.")

st.subheader("Session & environment")
st.json(
    {
        "procedure_session": log.session_id,
        "phase": log.phase,
        "events": len(log.events),
        "device": cfg.device,
        "llm_endpoint": cfg.llm_base_url or "OpenAI cloud",
        "llm_model": cfg.llm_model,
        "llm_status": llm_endpoint_ready()[1],
        "playback_fps": cfg.playback_fps,
        "max_video_frames": cfg.max_video_frames,
        "pixel_spacing_mm": cfg.pixel_spacing_mm or "not set (px-only measurements)",
        "log_dir": str(LOG_DIR),
    }
)
