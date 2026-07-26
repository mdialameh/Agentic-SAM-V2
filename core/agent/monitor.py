"""Live tracking monitor: short situational notes for the surgeon.

`generate_note` is blocking (an LLM call) — the UI always runs it in a
background thread so it can never stall the frame-by-frame tracking loop.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image


MONITOR_PROMPT = """You are an intraoperative ultrasound RFA tracking assistant.

You receive periodic MedSAM2 PTMC tracking metadata and the latest overlay
frame. Help the surgeon maintain situational awareness; do not diagnose or
make treatment decisions. Phrase suggestions as observations to verify.

Every response must be short and useful during a live procedure:
- First line: one high-level tracking status in 8 words or fewer.
- Then 1-3 bullets maximum.
- Focus on high-level changes, apparent PTMC segmentation size percentage,
  and whether the surgeon may want to visually re-check the target.
- No raw coordinates, no JSON, no filler, no repetition of unchanged details.
- When `area_summary.size_drop_flag` is true, include exactly one bullet with
  `area_summary.estimated_area_not_captured_percent` rounded to a whole
  percent, phrased as "approximately N% of the previously tracked PTMC area
  is not captured/visible in this frame" — an imaging/tracking/probe-pressure
  concern, not a biological conclusion.
- If tracking looks unstable, say which visual check would help (for example
  re-clicking the PTMC on the current frame).
"""


def _image_to_data_url(image: Image.Image, max_size: int = 768) -> str:
    frame = image.convert("RGB")
    frame.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def generate_note(payload: dict[str, Any], frame: Image.Image | None) -> tuple[str, bool]:
    """One live note from tracking metadata (+ optional frame). Never raises."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from core.llm import build_chat_llm

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Provide a live RFA tracking note from this MedSAM2 tracking data. "
            f"Tracking JSON: {json.dumps(payload, default=str)}",
        }
    ]
    if frame is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(frame)}})

    try:
        llm = build_chat_llm()
        response = llm.invoke(
            [SystemMessage(content=MONITOR_PROMPT), HumanMessage(content=content)]
        )
        text = response.text.strip()
        return (text or "No note produced.", bool(text))
    except Exception as exc:
        return f"Tracking monitor unavailable: {exc.__class__.__name__}: {exc}", False
