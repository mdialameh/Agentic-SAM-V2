"""Mask, overlay, box, and measurement math. Pure functions on PIL/numpy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

MASK_PALETTE = np.array(
    [
        [255, 64, 64, 140],
        [64, 170, 255, 140],
        [80, 210, 120, 140],
        [255, 190, 64, 140],
        [200, 90, 255, 140],
    ],
    dtype=np.uint8,
)


def to_numpy_mask(value: Any) -> np.ndarray:
    """Coerce a tensor/array mask into a 2-D boolean array."""
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().float().numpy()
    except ImportError:
        pass
    mask = np.squeeze(np.asarray(value))
    while mask.ndim > 2:
        mask = mask[0]
    return mask > 0


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    """Tight x1,y1,x2,y2 bounding box of a boolean mask (None when empty)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_area_px(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def render_overlay(
    image: Image.Image,
    masks: list[np.ndarray],
    boxes: list[list[float]] | None = None,
) -> Image.Image:
    """Blend colored masks (and optional boxes) over an image."""
    base = np.array(image.convert("RGBA"), copy=True)
    for idx, mask in enumerate(masks):
        color = MASK_PALETTE[idx % len(MASK_PALETTE)]
        active = to_numpy_mask(mask)
        base[active] = (
            0.45 * base[active].astype(np.float32) + 0.55 * color.astype(np.float32)
        ).astype(np.uint8)
    output = Image.fromarray(base, mode="RGBA")
    if boxes:
        draw = ImageDraw.Draw(output)
        for idx, box in enumerate(boxes):
            if box and len(box) == 4:
                color = tuple(int(v) for v in MASK_PALETTE[idx % len(MASK_PALETTE)][:3])
                draw.rectangle(list(box), outline=(*color, 255), width=3)
    return output.convert("RGB")


def points_to_box(points: list[list[int]]) -> list[int] | None:
    """Last two clicked points -> x1,y1,x2,y2 box."""
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[-2], points[-1]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def box_text(box: list[int] | list[float] | None) -> str:
    if box is None:
        return ""
    return ",".join(str(int(round(float(v)))) for v in box)


def parse_box(box: str | list[float]) -> list[float]:
    """Parse and validate an `x1,y1,x2,y2` box string or list."""
    values = (
        [float(part.strip()) for part in box.split(",")]
        if isinstance(box, str)
        else [float(v) for v in box]
    )
    if len(values) != 4:
        raise ValueError("box must contain four values: x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have x2 > x1 and y2 > y1")
    return values


def box_around_point(x: int, y: int, image: Image.Image, box_size: int) -> list[int]:
    """Fixed-size box centered on a clicked point, clamped to the image."""
    half = max(4, int(box_size) // 2)
    return [
        max(0, x - half),
        max(0, y - half),
        min(image.width - 1, x + half),
        min(image.height - 1, y + half),
    ]


def draw_prompt(
    image: Image.Image,
    points: list[list[int]] | None = None,
    box: list[int] | None = None,
) -> Image.Image:
    """Render click markers and the prompt box on a copy of the image."""
    output = image.convert("RGBA")
    draw = ImageDraw.Draw(output)
    for x, y in (points or [])[-2:]:
        draw.ellipse(
            [x - 5, y - 5, x + 5, y + 5],
            fill=(255, 64, 64, 230),
            outline=(255, 255, 255, 240),
            width=2,
        )
    if box is not None:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64, 240), width=3)
        label = box_text(box)
        position = (x1, max(0, y1 - 14))
        draw.rectangle(draw.textbbox(position, label), fill=(0, 0, 0, 140))
        draw.text(position, label, fill=(255, 96, 96, 255))
    return output.convert("RGB")


@dataclass
class Measurement:
    """Size measurement of a segmented target."""

    area_px: int
    width_px: float
    height_px: float
    area_mm2: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None

    def describe(self) -> str:
        parts = [
            f"area {self.area_px} px²",
            f"box {self.width_px:.0f}×{self.height_px:.0f} px",
        ]
        if self.area_mm2 is not None:
            parts.append(f"≈ {self.area_mm2:.1f} mm², {self.width_mm:.1f}×{self.height_mm:.1f} mm")
        return ", ".join(parts)


def measure_mask(mask: np.ndarray, pixel_spacing_mm: float = 0.0) -> Measurement:
    """Area + bounding-box dimensions of a mask, in px and optionally mm."""
    mask = to_numpy_mask(mask)
    area = mask_area_px(mask)
    box = bbox_from_mask(mask)
    width = (box[2] - box[0]) if box else 0.0
    height = (box[3] - box[1]) if box else 0.0
    result = Measurement(area_px=area, width_px=width, height_px=height)
    if pixel_spacing_mm > 0:
        result.area_mm2 = area * pixel_spacing_mm**2
        result.width_mm = width * pixel_spacing_mm
        result.height_mm = height * pixel_spacing_mm
    return result


@dataclass
class SegmentationResult:
    """In-process segmentation outcome shared by models, tools, and UI."""

    source: str
    query: str
    masks: list[np.ndarray] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    overlay: Image.Image | None = None
    prompt_box: list[float] | None = None
    note: str = ""

    @property
    def num_masks(self) -> int:
        return len(self.masks)

    def summary(self, pixel_spacing_mm: float = 0.0) -> dict[str, Any]:
        """JSON-safe summary for the LLM (no image payloads)."""
        payload: dict[str, Any] = {
            "source": self.source,
            "query": self.query,
            "num_masks": self.num_masks,
            "boxes": [[round(v, 1) for v in b] for b in self.boxes],
            "scores": [round(s, 4) for s in self.scores],
        }
        if self.prompt_box:
            payload["prompt_box"] = [round(v, 1) for v in self.prompt_box]
        if self.masks:
            payload["measurement"] = measure_mask(self.masks[0], pixel_spacing_mm).describe()
        if self.note:
            payload["note"] = self.note
        return payload


def keep_smallest_box_candidate(result: SegmentationResult) -> SegmentationResult:
    """For PTMC queries keep only the smallest bounding-box candidate.

    Mirrors the v1 heuristic: text-prompted models often return the whole
    thyroid plus the nodule; the PTMC is the smallest plausible candidate.
    """
    if result.num_masks <= 1 or not result.boxes:
        return result

    def area(box: list[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    total = len(result.boxes)
    idx = min(range(total), key=lambda i: area(result.boxes[i]))
    result.masks = [result.masks[idx]]
    result.boxes = [result.boxes[idx]]
    result.scores = [result.scores[idx]] if idx < len(result.scores) else []
    result.note = f"kept smallest of {total} candidates (PTMC rule)"
    return result
