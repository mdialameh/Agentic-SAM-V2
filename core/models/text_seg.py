"""Text-prompted segmentation: Medical-SAM3 (primary) and SAM 3.1 (fallback).

Both are SAM3-architecture models loaded through the `sam3` package; the
Medical-SAM3 checkpoint is a fine-tune whose state dict needs key cleanup.
They are heavy (10 GB / 3.5 GB) and load lazily on first use.
"""

from __future__ import annotations

from contextlib import nullcontext
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image

from core.config import Settings
from core.imaging import (
    SegmentationResult,
    keep_smallest_box_candidate,
    render_overlay,
    to_numpy_mask,
)


def _inference_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError("Unsupported checkpoint format for Medical-SAM3.")


def _clean_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    if any(key.startswith("detector.") for key in state_dict):
        state_dict = {
            key.removeprefix("detector."): value
            for key, value in state_dict.items()
            if key.startswith("detector.")
        }
    return state_dict


def _normalize(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()
    return np.asarray(value)


class TextSegmenter:
    """One text-prompted SAM3-family model (`variant` is 'medical_sam3' or 'sam3')."""

    def __init__(self, settings: Settings, variant: str) -> None:
        if variant not in {"medical_sam3", "sam3"}:
            raise ValueError(f"Unknown text segmenter variant: {variant}")
        self.settings = settings
        self.variant = variant
        self.device = settings.device if torch.cuda.is_available() else "cpu"
        self.lock = Lock()
        self._processor = None

    @property
    def checkpoint(self):
        if self.variant == "medical_sam3":
            return self.settings.medical_sam3_checkpoint
        return self.settings.sam3_checkpoint

    def _load(self) -> None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"{self.variant} checkpoint missing: {self.checkpoint} "
                "(run: python downloader.py)"
            )
        if self.variant == "sam3":
            model = build_sam3_image_model(
                checkpoint_path=str(self.checkpoint), load_from_HF=False
            )
        else:
            model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False)
            raw = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            state_dict = _clean_state_dict_keys(_extract_state_dict(raw))
            _missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected and len(unexpected) == len(state_dict):
                raise RuntimeError(
                    "Medical-SAM3 checkpoint could not be matched to the SAM3 model."
                )
        model = model.to(self.device)
        model.eval()
        self._processor = Sam3Processor(model)

    def processor(self):
        if self._processor is None:
            with self.lock:
                if self._processor is None:
                    self._load()
        return self._processor

    @property
    def loaded(self) -> bool:
        return self._processor is not None

    def segment(
        self,
        image: Image.Image,
        query: str,
        *,
        score_threshold: float = 0.0,
        max_masks: int = 4,
        smallest_for_ptmc: bool = True,
    ) -> SegmentationResult:
        """Segment by natural-language prompt (e.g. 'thyroid nodule')."""
        processor = self.processor()
        with self.lock, _inference_context(self.device):
            state = processor.set_image(image.convert("RGB"))
            output = processor.set_text_prompt(state=state, prompt=query)

        masks_np = _normalize(output.get("masks"))
        boxes_np = _normalize(output.get("boxes"))
        scores_np = _normalize(output.get("scores"))

        masks = np.squeeze(np.asarray(masks_np)) if masks_np is not None else np.zeros((0, 1, 1))
        if masks.ndim == 2:
            masks = masks[None, ...]
        boxes = np.asarray(boxes_np).reshape(-1, 4) if boxes_np is not None and np.asarray(boxes_np).size else np.zeros((0, 4))
        scores = np.asarray(scores_np).reshape(-1) if scores_np is not None else np.zeros(0)

        kept: list[int] = []
        for idx in range(len(masks)):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            if score >= score_threshold:
                kept.append(idx)
            if len(kept) >= max(1, max_masks):
                break

        result = SegmentationResult(
            source=self.variant,
            query=query,
            masks=[to_numpy_mask(masks[i]) for i in kept],
            boxes=[[float(v) for v in boxes[i]] for i in kept if i < len(boxes)],
            scores=[float(scores[i]) for i in kept if i < len(scores)],
        )
        if smallest_for_ptmc and "ptmc" in query.lower():
            result = keep_smallest_box_candidate(result)
        result.overlay = render_overlay(image, result.masks, result.boxes) if result.masks else None
        return result
