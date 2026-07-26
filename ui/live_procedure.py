"""Live Procedure page — the intraoperative workflow.

Load the RFA ultrasound video, pause on a frame, click the PTMC, and MedSAM2
tracks it frame-by-frame — each segmented frame is displayed as soon as it is
produced. A live AI monitor posts short situational notes, alerts land in the
procedure event log, and the surgeon can snapshot frames or ask questions at
any time.

Concurrency rules learned in v1: exactly ONE `run_every` fragment on the page
(two timers interfere and silently stop each other), and every LLM call made
from timed code runs in a background thread.
"""

from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from core.ablation import AblationTracker
from core.agent.graph import ask_agent
from core.agent.monitor import generate_note
from core.agent.tools import set_ablation_tracker
from core.imaging import box_around_point, box_text, draw_prompt, mask_area_px, render_overlay
from core.procedure import RFA_PHASES
from core.video import is_lfs_pointer_stub, sample_video
from ui.common import (
    append_chat,
    clickable_image,
    init_chat_state,
    pill_question,
    procedure_log,
    registry,
    render_chat,
    settings,
    thread_id,
)

cfg = settings()
log = procedure_log()
init_chat_state()

st.title("🫀 Live Procedure")
st.caption(
    f"Session `{log.session_id}` · decision support only — verify all overlays visually."
)

VIDEO_QUICK_QUESTIONS = (
    "Is the PTMC tracking stable in this frame?",
    "Has the visible PTMC area changed recently?",
    "What should be visually re-checked right now?",
)


def _fresh_video_state(label: str = "") -> dict:
    return {
        "source_label": label,
        "sampled": None,
        "current_index": 0,
        "playing": False,
        "point": None,
        "box": None,
        "session": None,
        "overlays": {},  # sampled index -> overlay jpeg path
        "overlay_dir": Path(tempfile.gettempdir()) / "agentic_sam_v2_overlays" / str(uuid.uuid4()),
        "monitor_text": "",
        "last_monitor_index": None,
        "_monitor_inflight": False,
        "_stop_event": None,  # threading.Event of the running tracking worker
        "_worker_alive": False,
        "size_drop_active": False,
        "ablation": AblationTracker(pixel_spacing_mm=cfg.pixel_spacing_mm),
        "_ablation_t0": None,  # monotonic time of the baseline frame
        "_next_sample_at": 0.0,  # monotonic deadline of the next 5 s sample
        "status": "Load a video to begin.",
    }


st.session_state.setdefault("video_state", _fresh_video_state())
# Expose this session's ablation tracker to the agent's `ablation_status` tool.
set_ablation_tracker(st.session_state.video_state["ablation"])


def _resolve_video_path() -> tuple[str | None, str]:
    source = st.radio(
        "Video source",
        ["Sample RFA video", "Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Upload":
        uploaded = st.file_uploader("RFA video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is None:
            return None, ""
        label = f"upload:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("_upload_label") != label:
            target = (
                Path(tempfile.gettempdir())
                / "agentic_sam_v2_uploads"
                / f"{uuid.uuid4()}_{uploaded.name}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getvalue())
            st.session_state["_upload_path"] = str(target)
            st.session_state["_upload_label"] = label
        return st.session_state["_upload_path"], label

    path = cfg.sample_video
    if not path.is_file() or is_lfs_pointer_stub(path):
        st.warning(f"Sample video unavailable at {path} — upload a video instead.")
        return None, ""
    return str(path), f"sample:{path}"


def _load_video(video_path: str, label: str) -> None:
    with st.status("Extracting video frames…", expanded=False):
        sampled = sample_video(video_path, cfg.playback_fps, cfg.max_video_frames)
    st.session_state.pop("video_seek", None)
    state = _fresh_video_state(label)
    state["sampled"] = sampled
    state["status"] = (
        f"Loaded {sampled.frame_count} frames at {sampled.playback_fps} FPS "
        f"(source {sampled.source_fps:.1f} FPS, stride {sampled.stride})."
    )
    st.session_state.video_state = state
    log.add("note", f"Video loaded: {Path(video_path).name} ({sampled.frame_count} frames)")


def _current_frame(state: dict) -> Image.Image | None:
    sampled = state["sampled"]
    if sampled is None or sampled.frame_count == 0:
        return None
    index = max(0, min(state["current_index"], sampled.frame_count - 1))
    overlay_path = state["overlays"].get(index)
    # Base frame: tracking overlay when one exists, else the raw frame.
    if overlay_path:
        frame = Image.open(overlay_path).convert("RGB")
    else:
        frame = sampled.frame_image(index)
    # While paused, draw the surgeon's fresh PTMC mark on top — including over
    # tracking overlays, so re-marking mid-procedure is visible.
    if state["point"] is not None and state["box"] is not None and not state["playing"]:
        return draw_prompt(frame, points=[state["point"]], box=state["box"])
    return frame


def _tracking_worker(state: dict, stop_event: threading.Event) -> None:
    """Step MedSAM2 continuously in the background, paced to playback speed.

    Tracking must NOT ride the UI timer: a GPU step can exceed the fragment's
    `run_every`, and an overrunning timed fragment stops being rescheduled by
    the browser (observed live). The worker owns stepping; the UI fragment
    only displays whatever the worker has produced so far.
    """
    import time

    session = state["session"]
    sampled = state["sampled"]
    interval = 1.0 / cfg.playback_fps
    try:
        while not stop_event.is_set():
            started = time.monotonic()
            step = session.step()
            if step is None:
                state["status"] = "Tracking reached the end of the video."
                log.add("tracking", "Tracking completed at end of video.")
                break

            index, result = step
            frame = sampled.frame_image(index)
            overlay = render_overlay(frame, result.masks, result.boxes)
            state["overlay_dir"].mkdir(parents=True, exist_ok=True)
            overlay_path = state["overlay_dir"] / f"{index:06d}.jpg"
            overlay.save(overlay_path, format="JPEG", quality=92)
            state["overlays"][index] = str(overlay_path)
            state["current_index"] = index

            # Ablation coverage: baseline on the first tracked frame, then
            # low-cadence samples (~1 per ablation_sample_seconds) — cheap
            # pixel math only, the LLM never runs here.
            tracker: AblationTracker = state["ablation"]
            now = time.monotonic()
            if result.masks:
                if not tracker.active:
                    tracker.set_baseline(index, result.masks[0], frame)
                    state["_ablation_t0"] = now
                    state["_next_sample_at"] = now + cfg.ablation_sample_seconds
                    log.add(
                        "snapshot",
                        f"Ablation baseline saved: first segmented PTMC at frame "
                        f"{index} ({tracker.baseline_area_px} px²).",
                        data={"baseline_frame": index},
                        image=overlay,
                    )
                elif now >= state["_next_sample_at"]:
                    tracker.add_sample(
                        index, now - state["_ablation_t0"], result.masks[0], frame
                    )
                    state["_next_sample_at"] = now + cfg.ablation_sample_seconds

            area = mask_area_px(result.masks[0]) if result.masks else 0
            summary = log.record_area(index, area)
            if summary["size_drop_flag"] and not state["size_drop_active"]:
                state["size_drop_active"] = True
                log.add(
                    "alert",
                    "Apparent PTMC area drop: "
                    f"~{summary['estimated_area_not_captured_percent']}% of the "
                    "session-max tracked area not captured in this frame.",
                    data=summary,
                )
            elif not summary["size_drop_flag"]:
                state["size_drop_active"] = False

            # Pace to playback speed so the surgeon can follow the video.
            stop_event.wait(max(0.0, interval - (time.monotonic() - started)))
    except Exception as exc:
        state["status"] = f"Tracking failed: {exc.__class__.__name__}: {exc}"
        log.add("alert", f"Tracking error: {exc}")
    finally:
        state["_worker_alive"] = False


def _spawn_tracking_worker(state: dict) -> None:
    stop_event = threading.Event()
    state["_stop_event"] = stop_event
    state["_worker_alive"] = True
    threading.Thread(target=_tracking_worker, args=(state, stop_event), daemon=True).start()


def _stop_tracking_worker(state: dict) -> None:
    if state.get("_stop_event") is not None:
        state["_stop_event"].set()


def _launch_monitor_note(state: dict) -> None:
    """Kick the live note LLM in a background thread (never blocks ticks)."""
    payload = {
        "current_sampled_frame_index": state["current_index"],
        "prompt_box": state["box"],
        "area_summary": log.area_summary(),
        "area_series_tail": log.area_series[-8:],
        "phase": log.phase,
    }
    frame = _current_frame(state)
    state["_monitor_inflight"] = True
    state["last_monitor_index"] = state["current_index"]

    def _worker() -> None:
        try:
            note, _ok = generate_note(payload, frame)
            state["monitor_text"] = note
        finally:
            state["_monitor_inflight"] = False

    threading.Thread(target=_worker, daemon=True).start()


def _start_tracking(state: dict) -> None:
    sampled = state["sampled"]
    try:
        with st.spinner("Starting MedSAM2 tracking (first run loads the model)…"):
            state["session"] = registry().medsam2.start_tracking(
                frames_dir=sampled.frames_dir,
                prompt_box=[float(v) for v in state["box"]],
                start_index=state["current_index"],
            )
    except Exception as exc:
        state["status"] = f"Could not start tracking: {exc.__class__.__name__}: {exc}"
        log.add("alert", f"Tracking start failed: {exc}")
        return
    log.add(
        "tracking",
        f"MedSAM2 tracking started from frame {state['current_index']} "
        f"with box {box_text(state['box'])}.",
    )
    state["monitor_text"] = "Tracking started — live notes will appear shortly."
    state["last_monitor_index"] = None
    # The tracked overlay replaces the manual mark from here on; requiring a
    # fresh click before the next Track press prevents stale-box restarts.
    state["point"] = None
    state["box"] = None
    state["playing"] = True
    state["status"] = (
        f"Tracking from frame {state['current_index']} — the video keeps playing "
        "with the PTMC overlay as each frame is segmented. Pause any time."
    )
    _spawn_tracking_worker(state)


# ------------------------------------------------------------- source row
source_col, phase_col = st.columns([3, 2], gap="large")
with source_col:
    video_path, video_label = _resolve_video_path()
    if video_path and st.session_state.video_state["source_label"] != video_label:
        _load_video(video_path, video_label)
with phase_col:
    phase = st.selectbox("RFA phase", RFA_PHASES, index=RFA_PHASES.index(log.phase))
    if phase != log.phase:
        log.set_phase(phase)
    box_size = st.number_input(
        "Tracking box size (px)", min_value=16, max_value=512,
        value=cfg.tracking_box_size, step=8,
    )

video_state = st.session_state.video_state
_tick_interval = 1.0 / cfg.playback_fps if video_state["playing"] else None


@st.fragment(run_every=_tick_interval)
def procedure_stage() -> None:
    """The single timed fragment: playback, tracking, live notes, events."""
    state = st.session_state.video_state
    sampled = state["sampled"]
    if sampled is None:
        st.info("Load the sample video or upload one to start.")
        return

    if state["playing"]:
        tracking_live = state["session"] is not None and not state["session"].done
        if tracking_live and state.get("_worker_alive"):
            # The background worker advances frames; this tick only displays
            # the latest overlay and refreshes the live note.
            if not state.get("_monitor_inflight") and state[
                "last_monitor_index"
            ] != state["current_index"]:
                _launch_monitor_note(state)
        elif tracking_live and not state.get("_worker_alive"):
            # Worker exited without finishing (error) — stop playback.
            state["playing"] = False
            st.rerun(scope="app")
        elif state["session"] is not None and state["session"].done and state.get("_worker_alive"):
            pass  # worker finishing its last iteration
        elif state["current_index"] < sampled.frame_count - 1:
            state["current_index"] += 1
        else:
            state["playing"] = False
            state["status"] = "Reached the end of the video."
            st.rerun(scope="app")

    frame_col, side_col = st.columns([5, 3], gap="large")

    with frame_col:
        frame = _current_frame(state)
        if frame is None:
            st.warning("No frames could be extracted from this video.")
            return
        if state["playing"]:
            st.image(frame, width="stretch")
        else:
            click = clickable_image(
                frame, key="live_frame_click", display_width=cfg.click_display_width
            )
            if click is not None:
                raw = sampled.frame_image(state["current_index"])
                state["point"] = [click[0], click[1]]
                state["box"] = box_around_point(click[0], click[1], raw, int(box_size))
                state["status"] = f"PTMC box set: {box_text(state['box'])}"
                st.rerun()

        slider_max = max(1, sampled.frame_count - 1)
        if state["playing"]:
            st.session_state.video_seek = min(state["current_index"], slider_max)
        else:
            st.session_state.setdefault("video_seek", min(state["current_index"], slider_max))

        def _on_seek() -> None:
            vstate = st.session_state.video_state
            _stop_tracking_worker(vstate)
            vstate["playing"] = False
            vstate["current_index"] = int(st.session_state.video_seek)

        st.slider(
            "Sampled frame", min_value=0, max_value=slider_max,
            disabled=state["playing"], key="video_seek", on_change=_on_seek,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("▶ Play", use_container_width=True, disabled=state["playing"]):
                if state["current_index"] >= sampled.frame_count - 1:
                    state["current_index"] = 0
                state["playing"] = True
                # With an active tracking session, Play means "keep tracking".
                if (
                    state["session"] is not None
                    and not state["session"].done
                    and not state.get("_worker_alive")
                ):
                    _spawn_tracking_worker(state)
                st.rerun(scope="app")
        with c2:
            if st.button("⏸ Pause", use_container_width=True, disabled=not state["playing"]):
                _stop_tracking_worker(state)
                state["playing"] = False
                log.add("tracking", f"Paused at frame {state['current_index']}.")
                st.rerun(scope="app")
        with c3:
            if st.button(
                "🎯 Track PTMC",
                type="primary",
                use_container_width=True,
                disabled=state["playing"] or state["box"] is None,
                help="Segment the clicked PTMC on this frame and keep tracking it.",
            ):
                _start_tracking(state)
                st.rerun(scope="app")
        with c4:
            if st.button(
                "⏭ Continue",
                use_container_width=True,
                disabled=state["session"] is None or state["session"].done,
            ):
                state["playing"] = True
                if not state.get("_worker_alive"):
                    _spawn_tracking_worker(state)
                st.rerun(scope="app")

        st.caption(state["status"])

    with side_col:
        st.markdown("##### 🩺 Live assistant")
        st.markdown(
            state["monitor_text"]
            or "_Live tracking notes will appear after PTMC tracking starts._"
        )
        summary = log.area_summary()
        if summary["current_area_px"] is not None:
            st.metric(
                "Tracked PTMC area (px²)",
                f"{summary['current_area_px']:.0f}",
                delta=f"{(summary['current_area_ratio_to_reference'] - 1) * 100:.0f}% vs session max",
                delta_color="off",
            )
        tracker = state["ablation"]
        if tracker.active:
            abl1, abl2 = st.columns(2)
            abl1.metric("Ablation coverage*", f"{tracker.coverage_percent:.0f}%")
            abl2.metric("Residual PTMC*", f"{tracker.residual_percent:.0f}%")
            st.caption(
                f"*tracking-based proxy · {len(tracker.samples)} samples at "
                f"~{cfg.ablation_sample_seconds:.0f}s — final report on the Report page"
            )
        if st.button("📸 Snapshot to event log", use_container_width=True):
            snap = _current_frame(state)
            if snap is not None:
                log.add(
                    "snapshot",
                    f"Snapshot at frame {state['current_index']} ({log.phase}).",
                    data=log.area_summary(),
                    image=snap,
                )
                st.toast("Snapshot saved to the procedure log.")

        st.markdown("##### 📋 Recent events")
        for event in reversed(log.events[-6:]):
            st.caption(f"[{event.time_hms}] **{event.kind}** — {event.message}")


procedure_stage()

st.divider()
ask_col, chat_col = st.columns([2, 3], gap="large")
with ask_col:
    st.markdown("##### Ask during the procedure")
    question = pill_question(
        "live_quick", VIDEO_QUICK_QUESTIONS, "Ask about the current frame…"
    )
    if question:
        state = st.session_state.video_state
        _stop_tracking_worker(state)
        state["playing"] = False
        frame = _current_frame(state)
        if frame is None:
            st.warning("Load a video frame before asking a question.")
        else:
            append_chat("user", f"[frame {state['current_index']}] {question}")
            with st.spinner("Assistant is analyzing the current frame…"):
                result = ask_agent(
                    question,
                    image=frame,
                    box=[float(v) for v in state["box"]] if state["box"] else None,
                    procedure_context=log.context_for_agent(),
                    thread_id=thread_id(),
                )
            if result.error:
                append_chat("assistant", f"⚠️ {result.error}")
            else:
                append_chat(
                    "assistant",
                    result.answer,
                    overlay=result.overlay,
                    tool_trace=result.tool_trace,
                    tool_logs=result.tool_logs,
                )
                log.add(
                    "question",
                    f"Q: {question} — A: {result.answer[:160]}",
                    data={"tools": result.tool_trace},
                )
            state["status"] = "Question answered. Click Continue to resume tracking."
            st.rerun(scope="app")

with chat_col:
    chat_box = st.container(height=300)
    render_chat(chat_box, show_tool_trace=False)
