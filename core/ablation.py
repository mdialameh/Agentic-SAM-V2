"""Ablation coverage monitoring — pure pixel math, no LLM.

When PTMC tracking starts, the first segmented mask becomes the **baseline**.
During ablation the tracked mask is sampled at a low cadence (default 1 frame
per 5 s). Baseline regions that stop being captured by the tracker are — as a
decision-support proxy — treated as ablated/obscured (RFA turns tissue
hyperechoic and degrades the tracker's target): their cumulative union is the
**treated map**, and baseline regions never lost are the **residual** the
physician may not have covered yet.

All of this is cheap numpy; the LLM sees it only once, when the final
ablation report is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from core.imaging import bbox_from_mask, to_numpy_mask

TREATED_COLOR = (80, 210, 120, 150)  # green — baseline regions lost at least once
RESIDUAL_COLOR = (255, 64, 64, 170)  # red — baseline regions never lost
CURRENT_COLOR = (64, 170, 255, 110)  # blue — currently captured mask


@dataclass
class AblationSample:
    """One low-cadence observation of the tracked PTMC during ablation."""

    frame_index: int
    elapsed_s: float
    captured_ratio: float  # |mask ∩ baseline| / |baseline|
    coverage_percent: float  # cumulative treated map / baseline
    mask_area_px: int
    mean_intensity: float  # echo brightness inside the baseline region

    def compact(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "t_s": round(self.elapsed_s, 1),
            "captured_ratio": round(self.captured_ratio, 3),
            "coverage_percent": round(self.coverage_percent, 1),
            "area_px": self.mask_area_px,
            "brightness": round(self.mean_intensity, 1),
        }


@dataclass
class AblationTracker:
    """Baseline PTMC + cumulative treated/residual coverage over time."""

    pixel_spacing_mm: float = 0.0
    baseline_mask: np.ndarray | None = None
    baseline_frame_index: int | None = None
    baseline_area_px: int = 0
    baseline_image: Image.Image | None = None
    treated_map: np.ndarray | None = None
    samples: list[AblationSample] = field(default_factory=list)

    # ------------------------------------------------------------ baseline
    def set_baseline(self, frame_index: int, mask: Any, frame_image: Image.Image) -> None:
        """The first segmented PTMC frame becomes the reference."""
        baseline = to_numpy_mask(mask)
        self.baseline_mask = baseline
        self.baseline_frame_index = frame_index
        self.baseline_area_px = int(np.count_nonzero(baseline))
        self.baseline_image = frame_image.convert("RGB").copy()
        self.treated_map = np.zeros_like(baseline, dtype=bool)
        self.samples = []

    @property
    def active(self) -> bool:
        return self.baseline_mask is not None and self.baseline_area_px > 0

    # ------------------------------------------------------------- samples
    def add_sample(
        self, frame_index: int, elapsed_s: float, mask: Any, frame_image: Image.Image
    ) -> AblationSample:
        """Record one observation and update the cumulative treated map."""
        if not self.active:
            raise RuntimeError("Set a baseline before adding ablation samples.")
        current = to_numpy_mask(mask)
        if current.shape != self.baseline_mask.shape:
            resized = Image.fromarray(current.astype(np.uint8) * 255).resize(
                (self.baseline_mask.shape[1], self.baseline_mask.shape[0])
            )
            current = np.array(resized) > 127

        overlap = int(np.count_nonzero(current & self.baseline_mask))
        self.treated_map |= self.baseline_mask & ~current

        gray = np.asarray(frame_image.convert("L"), dtype=np.float32)
        if gray.shape != self.baseline_mask.shape:
            gray = np.asarray(
                frame_image.convert("L").resize(
                    (self.baseline_mask.shape[1], self.baseline_mask.shape[0])
                ),
                dtype=np.float32,
            )
        mean_intensity = float(gray[self.baseline_mask].mean())

        sample = AblationSample(
            frame_index=frame_index,
            elapsed_s=elapsed_s,
            captured_ratio=overlap / self.baseline_area_px,
            coverage_percent=self.coverage_percent,
            mask_area_px=int(np.count_nonzero(current)),
            mean_intensity=mean_intensity,
        )
        self.samples.append(sample)
        return sample

    # ------------------------------------------------------------- metrics
    @property
    def coverage_percent(self) -> float:
        """Percent of the baseline PTMC treated at least once (proxy)."""
        if not self.active:
            return 0.0
        return 100.0 * np.count_nonzero(self.treated_map) / self.baseline_area_px

    @property
    def residual_percent(self) -> float:
        """Percent of the baseline PTMC never lost — possibly not covered."""
        if not self.active:
            return 0.0
        return 100.0 - self.coverage_percent

    def residual_mask(self) -> np.ndarray | None:
        if not self.active:
            return None
        return self.baseline_mask & ~self.treated_map

    # -------------------------------------------------------------- images
    def coverage_map(self, background: Image.Image | None = None) -> Image.Image | None:
        """Final ablation pattern: treated (green) vs residual (red) baseline."""
        if not self.active:
            return None
        base_image = (background or self.baseline_image).convert("RGBA")
        canvas = np.array(base_image, copy=True)

        def blend(region: np.ndarray, color: tuple[int, int, int, int]) -> None:
            rgba = np.array(color, dtype=np.float32)
            alpha = rgba[3] / 255.0
            canvas[region, :3] = (
                (1 - alpha) * canvas[region, :3].astype(np.float32) + alpha * rgba[:3]
            ).astype(np.uint8)

        blend(self.treated_map, TREATED_COLOR)
        blend(self.baseline_mask & ~self.treated_map, RESIDUAL_COLOR)
        return Image.fromarray(canvas, mode="RGBA").convert("RGB")

    # ------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        """Compact JSON-safe summary (what the agent tool and the report see)."""
        if not self.active:
            return {"active": False, "note": "No ablation baseline recorded yet."}
        residual = self.residual_mask()
        payload: dict[str, Any] = {
            "active": True,
            "baseline_frame": self.baseline_frame_index,
            "baseline_area_px": self.baseline_area_px,
            "samples_recorded": len(self.samples),
            "coverage_percent": round(self.coverage_percent, 1),
            "residual_percent": round(self.residual_percent, 1),
            "residual_bbox": bbox_from_mask(residual) if residual is not None else None,
            "interpretation_note": (
                "Coverage is a tracking-based proxy: baseline PTMC regions the "
                "tracker stopped capturing are counted as treated/obscured "
                "(ablation turns tissue hyperechoic). It is NOT a direct "
                "measurement of thermal ablation. Verify visually."
            ),
        }
        if self.pixel_spacing_mm > 0:
            payload["baseline_area_mm2"] = round(
                self.baseline_area_px * self.pixel_spacing_mm**2, 1
            )
        if self.samples:
            first, last = self.samples[0], self.samples[-1]
            payload["duration_s"] = round(last.elapsed_s - first.elapsed_s, 1)
            payload["brightness_change"] = round(
                last.mean_intensity - first.mean_intensity, 1
            )
        return payload

    def timeline(self) -> list[dict[str, Any]]:
        return [sample.compact() for sample in self.samples]
