"""Process-wide lazy model registry.

One instance per process (the UI wraps `get_registry()` in a cached
resource). Models construct cheaply here; weights load on first inference.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.config import Settings, get_settings
from core.models.medsam2 import MedSam2
from core.models.text_seg import TextSegmenter


class ModelRegistry:
    """Holds the segmentation models and reports their state."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.medsam2 = MedSam2(self.settings)
        self.medical_sam3 = TextSegmenter(self.settings, "medical_sam3")
        self.sam3 = TextSegmenter(self.settings, "sam3")

    def status(self) -> list[dict[str, Any]]:
        """Checkpoint presence and load state for every model."""
        entries = [
            ("MedSAM2 (box + tracking)", self.settings.medsam2_checkpoint,
             self.medsam2._image_predictor is not None or self.medsam2._video_predictor is not None),
            ("Medical-SAM3 (text)", self.settings.medical_sam3_checkpoint,
             self.medical_sam3.loaded),
            ("SAM 3.1 (text fallback)", self.settings.sam3_checkpoint, self.sam3.loaded),
        ]
        rows = []
        for name, path, loaded in entries:
            present = path.is_file() and path.stat().st_size > 1024
            rows.append(
                {
                    "model": name,
                    "checkpoint": path.name,
                    "present": present,
                    "size_gb": round(path.stat().st_size / 1e9, 2) if present else 0.0,
                    "loaded": loaded,
                }
            )
        return rows


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry()
