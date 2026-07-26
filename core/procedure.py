"""RFA procedure state: phases, timestamped event log, snapshots, area trend.

The event log is the spine of the assistant — tracking events, phase changes,
alerts, Q&A, and snapshots all land here, and the operative report is
generated from it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from core.config import SESSION_DIR

RFA_PHASES = (
    "Pre-op review",
    "Targeting",
    "Ablation",
    "Post-ablation assessment",
)

SIZE_DROP_THRESHOLD_PERCENT = 15.0


@dataclass
class ProcedureEvent:
    """One timestamped entry in the procedure log."""

    timestamp: str
    phase: str
    kind: str  # phase_change | tracking | alert | snapshot | question | note
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None

    @property
    def time_hms(self) -> str:
        return self.timestamp.split("T")[1][:8]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProcedureLog:
    """Event-sourced state of one RFA procedure session."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.phase: str = RFA_PHASES[0]
        self.events: list[ProcedureEvent] = []
        self.area_series: list[tuple[int, float]] = []  # (frame_index, area_px)
        self.snapshot_dir = SESSION_DIR / self.session_id
        self.add("note", f"Procedure session {self.session_id} started.")

    # ------------------------------------------------------------- events
    def add(
        self,
        kind: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        image: Image.Image | None = None,
    ) -> ProcedureEvent:
        image_path: str | None = None
        if image is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(
                self.snapshot_dir / f"{len(self.events):04d}_{kind}.png"
            )
            image.save(image_path)
        event = ProcedureEvent(
            timestamp=_now_iso(),
            phase=self.phase,
            kind=kind,
            message=message,
            data=data or {},
            image_path=image_path,
        )
        self.events.append(event)
        return event

    def set_phase(self, phase: str) -> None:
        if phase not in RFA_PHASES or phase == self.phase:
            return
        self.phase = phase
        self.add("phase_change", f"Phase changed to: {phase}")

    # -------------------------------------------------------- area trend
    def record_area(self, frame_index: int, area_px: float) -> dict[str, Any]:
        """Record a tracked-mask area and return the running trend summary."""
        if area_px > 0:
            self.area_series.append((frame_index, float(area_px)))
        return self.area_summary()

    def area_summary(self) -> dict[str, Any]:
        areas = [area for _, area in self.area_series]
        if not areas:
            return {
                "reference_area_px": None,
                "current_area_px": None,
                "current_area_ratio_to_reference": None,
                "estimated_area_not_captured_percent": None,
                "size_drop_flag": False,
            }
        reference = max(areas)
        current = areas[-1]
        ratio = current / reference if reference > 0 else 1.0
        missing = max(0.0, min(100.0, (1.0 - ratio) * 100.0))
        return {
            "reference_area_px": round(reference, 1),
            "current_area_px": round(current, 1),
            "current_area_ratio_to_reference": round(ratio, 3),
            "estimated_area_not_captured_percent": round(missing),
            "size_drop_flag": missing >= SIZE_DROP_THRESHOLD_PERCENT,
        }

    # ----------------------------------------------------------- exports
    def to_rows(self) -> list[dict[str, str]]:
        """Flat rows for UI tables and the report generator."""
        return [
            {
                "time": event.time_hms,
                "phase": event.phase,
                "kind": event.kind,
                "message": event.message,
            }
            for event in self.events
        ]

    def snapshots(self) -> list[ProcedureEvent]:
        return [event for event in self.events if event.image_path]

    def context_for_agent(self, max_events: int = 12) -> str:
        """Compact procedure context injected into agent prompts."""
        lines = [f"Current RFA phase: {self.phase}."]
        summary = self.area_summary()
        if summary["current_area_px"] is not None:
            lines.append(
                "Tracked PTMC area: "
                f"{summary['current_area_px']} px² "
                f"({summary['current_area_ratio_to_reference']:.0%} of session max)."
            )
        if self.events:
            lines.append("Recent events:")
            for event in self.events[-max_events:]:
                lines.append(f"- [{event.time_hms}] ({event.kind}) {event.message}")
        return "\n".join(lines)
