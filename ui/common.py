"""Shared UI plumbing: cached resources, session state, click widget."""

from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from core.config import Settings, get_settings
from core.models.registry import ModelRegistry, get_registry
from core.procedure import ProcedureLog


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    return get_settings()


@st.cache_resource(show_spinner=False)
def registry() -> ModelRegistry:
    return get_registry()


def procedure_log() -> ProcedureLog:
    """One ProcedureLog per browser session; also provides the agent thread id."""
    if "procedure_log" not in st.session_state:
        st.session_state.procedure_log = ProcedureLog()
    return st.session_state.procedure_log


def thread_id() -> str:
    return f"session-{procedure_log().session_id}"


def init_chat_state() -> None:
    st.session_state.setdefault("chat_history", [])


def append_chat(
    role: str,
    content: str,
    *,
    overlay: Image.Image | None = None,
    tool_trace: list[str] | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> None:
    st.session_state.chat_history.append(
        {
            "role": role,
            "content": content,
            "overlay": overlay,
            "tool_trace": tool_trace or [],
            "tool_logs": tool_logs or [],
        }
    )


def render_chat(container: Any, *, show_tool_trace: bool = True) -> None:
    for message in st.session_state.chat_history:
        with container.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("overlay") is not None:
                st.image(message["overlay"], caption="Segmentation overlay", width="stretch")
            if show_tool_trace and message.get("tool_trace"):
                with st.expander(f"Tools used: {', '.join(message['tool_trace'])}"):
                    st.json(message.get("tool_logs") or message["tool_trace"])


def clickable_image(image: Image.Image, *, key: str, display_width: int) -> tuple[int, int] | None:
    """Show an image, return NEW clicks scaled to original pixel coordinates.

    The component re-reports the last click every rerun; a per-key token
    filters to fresh clicks only.
    """
    width = min(display_width, image.width)
    result = streamlit_image_coordinates(image, key=key, width=width)
    if not result or result.get("x") is None:
        return None
    token = (result["x"], result["y"])
    token_key = f"_last_click_{key}"
    if st.session_state.get(token_key) == token:
        return None
    st.session_state[token_key] = token
    scale = image.width / width
    x = int(round(result["x"] * scale))
    y = int(round(result["y"] * scale))
    return min(x, image.width - 1), min(y, image.height - 1)


def pill_question(pill_key: str, options: tuple[str, ...], input_placeholder: str) -> str | None:
    """Quick-question pills + chat input; returns the submitted question.

    Used pills are cleared on the NEXT run, before the widget is instantiated
    (Streamlit forbids writing a widget key after its widget exists).
    """
    reset_key = f"_reset_{pill_key}"
    if st.session_state.pop(reset_key, False):
        st.session_state[pill_key] = None
    quick = st.pills(
        "Quick questions",
        options,
        selection_mode="single",
        key=pill_key,
        label_visibility="collapsed",
    )
    typed = st.chat_input(input_placeholder, key=f"{pill_key}_input")
    question = typed or quick
    if question and quick and not typed:
        st.session_state[reset_key] = True
    return question
