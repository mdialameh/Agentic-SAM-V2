"""Agent tools. All run in-process against the model registry.

Tools return compact JSON strings for the LLM; full results (with masks and
overlay images) go to a per-turn holder the UI pops after the agent answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from langchain_core.tools import tool
from PIL import Image

from core.config import get_settings
from core.imaging import SegmentationResult, measure_mask, parse_box
from core.models.registry import get_registry

_HOLDER_LOCK = Lock()
_TURN_RESULT: SegmentationResult | None = None  # cleared every agent turn
_KEPT_RESULT: SegmentationResult | None = None  # survives turns, for measuring
_ABLATION_TRACKER = None  # AblationTracker registered by the live-procedure UI


def set_ablation_tracker(tracker) -> None:
    """UI-side: expose the session's AblationTracker to the agent tools."""
    global _ABLATION_TRACKER
    with _HOLDER_LOCK:
        _ABLATION_TRACKER = tracker


def _store_result(result: SegmentationResult) -> None:
    global _TURN_RESULT, _KEPT_RESULT
    with _HOLDER_LOCK:
        _TURN_RESULT = result
        _KEPT_RESULT = result


def pop_last_result() -> SegmentationResult | None:
    """UI-side: take this turn's segmentation result (with overlay)."""
    global _TURN_RESULT
    with _HOLDER_LOCK:
        result, _TURN_RESULT = _TURN_RESULT, None
    return result


def peek_last_result() -> SegmentationResult | None:
    """Most recent segmentation from ANY turn (for `measure_target`)."""
    with _HOLDER_LOCK:
        return _KEPT_RESULT


def _load_image(image_path: str) -> Image.Image:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    return Image.open(path).convert("RGB")


@tool
def segment_with_box(image_path: str, box: str) -> str:
    """Segment the target inside a user-drawn box with MedSAM2. Use this first
    whenever a user-drawn PTMC box is available. `box` is x1,y1,x2,y2."""
    image = _load_image(image_path)
    result = get_registry().medsam2.segment_image(image, parse_box(box))
    _store_result(result)
    return json.dumps(result.summary(get_settings().pixel_spacing_mm))


@tool
def segment_by_text(image_path: str, query: str) -> str:
    """Segment a structure by a short text prompt (e.g. 'PTMC', 'thyroid
    nodule', 'ablation zone') using Medical-SAM3, falling back to SAM 3.1 when
    Medical-SAM3 finds nothing. Use when no user-drawn box is available."""
    image = _load_image(image_path)
    registry = get_registry()
    attempts = []
    try:
        result = registry.medical_sam3.segment(image, query)
        attempts.append({"model": "medical_sam3", "num_masks": result.num_masks})
    except Exception as exc:
        result = None
        attempts.append({"model": "medical_sam3", "error": f"{exc.__class__.__name__}: {exc}"})

    if result is None or result.num_masks == 0:
        try:
            result = registry.sam3.segment(image, query)
            attempts.append({"model": "sam3", "num_masks": result.num_masks})
        except Exception as exc:
            attempts.append({"model": "sam3", "error": f"{exc.__class__.__name__}: {exc}"})
            if result is None:
                return json.dumps({"status": "error", "attempts": attempts})

    _store_result(result)
    payload = result.summary(get_settings().pixel_spacing_mm)
    payload["attempts"] = attempts
    return json.dumps(payload)


@tool
def measure_target() -> str:
    """Measure the most recently segmented target: mask area and bounding-box
    dimensions in pixels (and mm when pixel spacing is configured)."""
    result = peek_last_result()
    if result is None or not result.masks:
        return json.dumps(
            {"status": "error", "message": "No segmentation available to measure yet."}
        )
    measurement = measure_mask(result.masks[0], get_settings().pixel_spacing_mm)
    return json.dumps(
        {
            "source": result.source,
            "query": result.query,
            "measurement": measurement.describe(),
            "area_px": measurement.area_px,
        }
    )


@tool
def ablation_status() -> str:
    """Current RFA ablation-coverage status from the live tracker: percent of
    the baseline PTMC treated (proxy), residual un-covered percent, sample
    timeline. Use for questions like 'how much is ablated?' or 'did I cover
    the whole PTMC?'."""
    with _HOLDER_LOCK:
        tracker = _ABLATION_TRACKER
    if tracker is None or not tracker.active:
        return json.dumps(
            {
                "active": False,
                "note": "No ablation baseline yet — start PTMC tracking first.",
            }
        )
    payload = tracker.summary()
    payload["recent_samples"] = tracker.timeline()[-6:]
    return json.dumps(payload)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for background medical knowledge (terminology, procedure
    context). Never use this to guess image-specific findings."""
    from ddgs import DDGS

    settings = get_settings()
    count = max(1, min(max_results, settings.search_max_results))
    results = DDGS(timeout=30).text(query, region="us-en", max_results=count)
    normalized = [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("href", item.get("url", ""))),
            "snippet": str(item.get("body", item.get("snippet", ""))),
        }
        for item in results or []
    ]
    return json.dumps({"query": query, "results": normalized})


AGENT_TOOLS = [segment_with_box, segment_by_text, measure_target, ablation_status, web_search]
