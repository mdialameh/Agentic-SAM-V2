"""Image Analysis page: chat about any ultrasound still with the agent."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from core.agent.graph import ask_agent
from core.imaging import box_text, draw_prompt, points_to_box
from ui.common import (
    append_chat,
    clickable_image,
    init_chat_state,
    pill_question,
    procedure_log,
    render_chat,
    settings,
    thread_id,
)

cfg = settings()
log = procedure_log()
init_chat_state()

st.title("🖼️ Image Analysis")
st.caption("Draw a PTMC box with two clicks, then ask the assistant anything about the image.")

IMAGE_QUICK_QUESTIONS = (
    "Where is the PTMC in this ultrasound image?",
    "Describe this ultrasound image.",
    "How large is the segmented target?",
)

st.session_state.setdefault("ia_points", [])
st.session_state.setdefault("ia_source_label", None)
st.session_state.setdefault("ia_latest_overlay", None)

image_col, chat_col = st.columns([5, 6], gap="large")

with image_col:
    source = st.radio(
        "Image source", ["Sample PTMC image", "Upload"],
        horizontal=True, label_visibility="collapsed",
    )
    image: Image.Image | None = None
    label = None
    if source == "Upload":
        uploaded = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is not None:
            label = f"upload:{uploaded.name}:{uploaded.size}"
            image = Image.open(uploaded).convert("RGB")
    elif cfg.sample_image.is_file():
        label = f"sample:{cfg.sample_image}"
        image = Image.open(cfg.sample_image).convert("RGB")
    else:
        st.warning(f"Sample image not found: {cfg.sample_image}")

    if image is not None:
        if st.session_state.ia_source_label != label:
            st.session_state.ia_source_label = label
            st.session_state.ia_points = []
            st.session_state.ia_latest_overlay = None

        points = st.session_state.ia_points
        box = points_to_box(points)
        preview = draw_prompt(image, points=points, box=box)
        click = clickable_image(
            preview, key="ia_click", display_width=cfg.click_display_width
        )
        if click is not None:
            if len(points) >= 2:
                points = []
            points.append([click[0], click[1]])
            st.session_state.ia_points = points
            st.rerun()

        cap_col, clear_col = st.columns(2)
        with cap_col:
            if box is not None:
                st.caption(f"PTMC box: `{box_text(box)}`")
            elif len(points) == 1:
                st.caption("Click the opposite corner to finish the box.")
            else:
                st.caption("Click two corners to define the PTMC box (optional).")
        with clear_col:
            if st.button("Clear box", use_container_width=True):
                st.session_state.ia_points = []
                st.rerun()

        if st.session_state.ia_latest_overlay is not None:
            with st.expander("Latest segmentation overlay", expanded=True):
                st.image(st.session_state.ia_latest_overlay, width="stretch")

with chat_col:
    st.subheader("Assistant chat")
    chat_box = st.container(height=460)
    render_chat(chat_box)

    question = pill_question(
        "ia_quick", IMAGE_QUICK_QUESTIONS, "Ask about this ultrasound image…"
    )
    if question:
        if image is None:
            st.warning("Load or upload an image before asking a question.")
        else:
            box = points_to_box(st.session_state.ia_points)
            append_chat("user", question)
            with st.spinner("Assistant is analyzing…"):
                result = ask_agent(
                    question,
                    image=image,
                    box=[float(v) for v in box] if box else None,
                    procedure_context=log.context_for_agent(max_events=4),
                    thread_id=thread_id(),
                )
            if result.error:
                append_chat("assistant", f"⚠️ {result.error}")
            else:
                if result.overlay is not None:
                    st.session_state.ia_latest_overlay = result.overlay
                append_chat(
                    "assistant",
                    result.answer,
                    overlay=result.overlay,
                    tool_trace=result.tool_trace,
                    tool_logs=result.tool_logs,
                )
            st.rerun()
