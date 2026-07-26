"""Operative-note and ablation-report generation.

Both reports run exactly ONE LLM call, and only when the user presses the
corresponding button — all monitoring data is collected as cheap pixel math
during the procedure, so token cost is deferred to a single on-demand call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from core.ablation import AblationTracker
from core.llm import build_chat_llm, llm_endpoint_ready
from core.procedure import ProcedureLog

REPORT_PROMPT = """You are drafting a concise post-procedure summary note for an
ultrasound-guided RFA of a PTMC, from a structured event log produced by an
AI tracking assistant.

Rules:
- This is an observational technical summary of what the assistant recorded,
  NOT a clinical operative note; open with one line saying exactly that.
- Structure: Session overview → Timeline by phase → Target tracking &
  size-trend summary → Alerts raised → Questions asked and answers given →
  Limitations.
- Report only what the log supports. No diagnoses, no outcome claims, no
  treatment recommendations. Sizes are pixel-based estimates unless the log
  carries mm values.
- Markdown, terse professional tone, no filler.
"""


def _fallback_report(log: ProcedureLog) -> str:
    """Deterministic report when no LLM is available."""
    lines = [
        "# RFA Assistant Session Summary (auto-generated, no LLM)",
        "",
        f"- Session: `{log.session_id}`",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Final phase: {log.phase}",
        f"- Events: {len(log.events)}, snapshots: {len(log.snapshots())}",
        "",
        "| Time | Phase | Kind | Message |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['time']} | {r['phase']} | {r['kind']} | {r['message']} |"
        for r in log.to_rows()
    ]
    summary = log.area_summary()
    if summary["current_area_px"] is not None:
        lines += [
            "",
            f"Area trend: current {summary['current_area_px']} px² "
            f"({summary['current_area_ratio_to_reference']:.0%} of session max).",
        ]
    return "\n".join(lines)


def build_report(log: ProcedureLog) -> str:
    """Generate the session report (LLM when available, deterministic otherwise)."""
    ready, _detail = llm_endpoint_ready()
    if not ready:
        return _fallback_report(log)

    from langchain_core.messages import HumanMessage, SystemMessage

    payload = {
        "session_id": log.session_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_phase": log.phase,
        "events": log.to_rows(),
        "area_summary": log.area_summary(),
        "area_series_tail": log.area_series[-40:],
        "num_snapshots": len(log.snapshots()),
    }
    try:
        llm = build_chat_llm()
        response = llm.invoke(
            [
                SystemMessage(content=REPORT_PROMPT),
                HumanMessage(content="Event log JSON:\n" + json.dumps(payload, default=str)),
            ]
        )
        text = response.text.strip()
        return text or _fallback_report(log)
    except Exception:
        return _fallback_report(log)


ABLATION_REPORT_PROMPT = """You are drafting the final RFA ablation-coverage report
for an ultrasound-guided PTMC ablation, from an AI tracking assistant's data.

You receive:
1. The baseline image — the first tracked frame with the segmented PTMC.
2. The final coverage map — green = baseline PTMC regions the tracker lost at
   least once (treated/obscured proxy), red = baseline regions never lost
   (possibly NOT covered by ablation).
3. A compact numeric timeline sampled ~every 5 seconds: captured ratio,
   cumulative coverage %, mask area, mean echo brightness in the baseline
   region.

Write, in Markdown, tersely:
- One line stating this is a tracking-based proxy analysis, not a thermal
  measurement, and must be verified visually.
- **Ablation extent**: final coverage % of the baseline PTMC and how it
  evolved over time (fast/slow phases, when it plateaued).
- **Ablation pattern**: describe the spatial pattern from the coverage map —
  where treatment concentrated, whether it progressed evenly.
- **Residual assessment**: whether the whole PTMC appears covered; if red
  residual regions remain, describe their location (e.g. inferior margin) and
  approximate share, and note they may warrant visual re-checking.
- **Supporting signals**: echo-brightness trend (hyperechoic change supports
  ablation) and captured-ratio trend.
- **Limitations**: probe/plane motion, tracking loss vs true ablation
  ambiguity, no thermal data.
No diagnoses, no treatment instructions, no filler.
"""


def _fallback_ablation_report(tracker: AblationTracker) -> str:
    summary = tracker.summary()
    lines = [
        "# RFA Ablation Coverage Report (auto-generated, no LLM)",
        "",
        "Tracking-based proxy analysis — verify visually. Not a thermal measurement.",
        "",
        f"- Baseline PTMC: frame {summary.get('baseline_frame')}, "
        f"{summary.get('baseline_area_px')} px²",
        f"- Coverage (proxy): **{summary.get('coverage_percent')}%** treated at least once",
        f"- Residual (possibly uncovered): **{summary.get('residual_percent')}%**",
        f"- Samples: {summary.get('samples_recorded')} at ~5 s cadence",
    ]
    if summary.get("brightness_change") is not None:
        lines.append(
            f"- Echo brightness change in baseline region: {summary['brightness_change']:+.1f}"
        )
    lines += ["", "| t (s) | captured | coverage % | area px | brightness |", "|---|---|---|---|---|"]
    lines += [
        f"| {s['t_s']} | {s['captured_ratio']} | {s['coverage_percent']} | "
        f"{s['area_px']} | {s['brightness']} |"
        for s in tracker.timeline()
    ]
    return "\n".join(lines)


def build_ablation_report(tracker: AblationTracker) -> str:
    """One on-demand vision-LLM call over baseline + coverage map + timeline."""
    if not tracker.active:
        return "No ablation data: start PTMC tracking first so a baseline exists."
    ready, _detail = llm_endpoint_ready()
    if not ready:
        return _fallback_ablation_report(tracker)

    from langchain_core.messages import HumanMessage, SystemMessage

    from core.agent.monitor import _image_to_data_url
    from core.imaging import render_overlay

    baseline_overlay = render_overlay(tracker.baseline_image, [tracker.baseline_mask])
    coverage_map = tracker.coverage_map()
    content = [
        {
            "type": "text",
            "text": (
                "Ablation summary JSON:\n"
                + json.dumps(tracker.summary(), default=str)
                + "\n\nSample timeline JSON:\n"
                + json.dumps(tracker.timeline(), default=str)
                + "\n\nImage 1 is the baseline PTMC segmentation; image 2 is the "
                "final coverage map (green treated, red residual)."
            ),
        },
        {"type": "image_url", "image_url": {"url": _image_to_data_url(baseline_overlay)}},
        {"type": "image_url", "image_url": {"url": _image_to_data_url(coverage_map)}},
    ]
    try:
        llm = build_chat_llm()
        response = llm.invoke(
            [SystemMessage(content=ABLATION_REPORT_PROMPT), HumanMessage(content=content)]
        )
        text = response.text.strip()
        return text or _fallback_ablation_report(tracker)
    except Exception:
        return _fallback_ablation_report(tracker)
