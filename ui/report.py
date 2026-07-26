"""Procedure Report page: event log, area trend, snapshots, operative note."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from core.agent.report import build_ablation_report, build_report
from core.imaging import render_overlay
from ui.common import procedure_log

log = procedure_log()

st.title("📄 Procedure Report")
st.caption(f"Session `{log.session_id}` — everything below is reconstructed from the event log.")

summary = log.area_summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", len(log.events))
c2.metric("Snapshots", len(log.snapshots()))
c3.metric("Final phase", log.phase)
c4.metric(
    "Area vs session max",
    f"{summary['current_area_ratio_to_reference']:.0%}"
    if summary["current_area_px"] is not None
    else "—",
)

if log.area_series:
    st.markdown("##### Tracked PTMC area over time")
    st.line_chart(
        {"area_px": [a for _, a in log.area_series]},
        x_label="tracking step",
        y_label="area (px²)",
    )

st.markdown("##### Event log")
st.dataframe(log.to_rows(), hide_index=True, width="stretch")

snapshots = log.snapshots()
if snapshots:
    st.markdown("##### Snapshots")
    grid = st.columns(4)
    for idx, event in enumerate(snapshots):
        with grid[idx % 4]:
            st.image(Image.open(event.image_path), caption=f"{event.time_hms} · {event.phase}")

st.divider()
st.markdown("### 🔥 Ablation coverage")
tracker = (st.session_state.get("video_state") or {}).get("ablation")
if tracker is None or not tracker.active:
    st.caption(
        "Ablation monitoring starts automatically when PTMC tracking starts on the "
        "Live Procedure page: the first segmented frame becomes the baseline and the "
        "tracked mask is sampled every ~5 s."
    )
else:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Coverage (proxy)", f"{tracker.coverage_percent:.0f}%")
    a2.metric("Residual PTMC", f"{tracker.residual_percent:.0f}%")
    a3.metric("Baseline area", f"{tracker.baseline_area_px} px²")
    a4.metric("Samples (~5 s)", len(tracker.samples))

    img1, img2 = st.columns(2)
    with img1:
        st.image(
            render_overlay(tracker.baseline_image, [tracker.baseline_mask]),
            caption=f"Baseline PTMC (frame {tracker.baseline_frame_index})",
            width="stretch",
        )
    with img2:
        coverage_map = tracker.coverage_map()
        if coverage_map is not None:
            st.image(
                coverage_map,
                caption="Ablation pattern — green treated (proxy), red residual",
                width="stretch",
            )
    if tracker.samples:
        st.line_chart(
            {
                "coverage %": [s.coverage_percent for s in tracker.samples],
                "captured %": [100 * s.captured_ratio for s in tracker.samples],
            },
            x_label="sample (~5 s apart)",
            y_label="% of baseline PTMC",
        )

    if st.button("🔥 Generate ablation report", type="primary"):
        with st.spinner("Analyzing ablation coverage (single LLM call)…"):
            st.session_state["ablation_report_md"] = build_ablation_report(tracker)
    if st.session_state.get("ablation_report_md"):
        st.markdown(st.session_state["ablation_report_md"])
        st.download_button(
            "⬇ Download ablation report",
            st.session_state["ablation_report_md"],
            file_name=f"rfa_ablation_{log.session_id}.md",
            mime="text/markdown",
        )

st.divider()
if st.button("🧾 Generate session summary note"):
    with st.spinner("Drafting the summary note…"):
        st.session_state["report_md"] = build_report(log)

if st.session_state.get("report_md"):
    st.markdown(st.session_state["report_md"])
    st.download_button(
        "⬇ Download as Markdown",
        st.session_state["report_md"],
        file_name=f"rfa_session_{log.session_id}.md",
        mime="text/markdown",
    )
