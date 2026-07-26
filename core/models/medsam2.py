"""MedSAM2: box-prompted image segmentation and streaming video tracking.

Wraps the upstream MedSAM2/SAM2 checkout in `third_party/MedSAM2` and the
`checkpoints/MedSAM2_latest.pt` weights. Both predictors load lazily and are
shared process-wide; a lock serializes GPU calls.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image

from core.config import Settings
from core.imaging import SegmentationResult, bbox_from_mask, render_overlay, to_numpy_mask


def _inference_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _activate_device(device: str) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available.")
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_device(index)


class TrackingSession:
    """A live MedSAM2 video-tracking session that yields one frame per step."""

    def __init__(
        self,
        *,
        model: "MedSam2",
        frames_dir: Path,
        prompt_box: list[float],
        start_index: int,
        score_threshold: float = 0.0,
    ) -> None:
        self._model = model
        self.prompt_box = prompt_box
        self.start_index = start_index
        self.score_threshold = score_threshold
        self.last_index: int | None = None
        self.done = False

        predictor = model.video_predictor()
        with model.lock, torch.inference_mode(), _inference_context(model.device):
            _activate_device(model.device)
            self._state = predictor.init_state(
                video_path=str(frames_dir), async_loading_frames=False
            )
            predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=start_index,
                obj_id=1,
                box=np.array(prompt_box, dtype=np.float32),
            )
            self._generator: Iterator[Any] = predictor.propagate_in_video(
                self._state, start_frame_idx=start_index
            )

    def step(self, frame_image: Image.Image | None = None) -> tuple[int, SegmentationResult] | None:
        """Track the next frame. Returns (sampled_index, result), or None at end."""
        if self.done:
            return None
        with self._model.lock, torch.inference_mode(), _inference_context(self._model.device):
            _activate_device(self._model.device)
            try:
                frame_idx, _obj_ids, mask_logits = next(self._generator)
            except StopIteration:
                self.done = True
                return None
            mask = to_numpy_mask(mask_logits[0] > self.score_threshold)

        index = int(frame_idx)
        self.last_index = index
        box = bbox_from_mask(mask)
        result = SegmentationResult(
            source="medsam2_tracking",
            query="PTMC",
            masks=[mask],
            boxes=[box] if box else [],
            scores=[1.0],
            prompt_box=self.prompt_box,
        )
        if frame_image is not None:
            result.overlay = render_overlay(frame_image, result.masks, result.boxes)
        return index, result


class MedSam2:
    """Lazy holder for the MedSAM2 image and video predictors."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.device if torch.cuda.is_available() else "cpu"
        self.lock = Lock()
        self._image_predictor = None
        self._video_predictor = None

    # ------------------------------------------------------------- loading
    def _ensure_source_on_path(self) -> None:
        source = str(self.settings.medsam2_source_root)
        if source not in sys.path:
            sys.path.insert(0, source)

    def _hydra_context(self):
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        config_dir = self.settings.medsam2_source_root / "sam2" / "configs"
        return initialize_config_dir(config_dir=str(config_dir), version_base=None)

    def _validate(self) -> None:
        if not self.settings.medsam2_source_root.is_dir():
            raise FileNotFoundError(
                f"MedSAM2 source checkout missing: {self.settings.medsam2_source_root}"
            )
        if not self.settings.medsam2_checkpoint.is_file():
            raise FileNotFoundError(
                f"MedSAM2 checkpoint missing: {self.settings.medsam2_checkpoint} "
                "(run: python downloader.py)"
            )

    def image_predictor(self):
        if self._image_predictor is None:
            with self.lock:
                if self._image_predictor is None:
                    self._validate()
                    self._ensure_source_on_path()
                    _activate_device(self.device)
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor

                    with self._hydra_context():
                        model = build_sam2(
                            self.settings.medsam2_config_file,
                            ckpt_path=str(self.settings.medsam2_checkpoint),
                            device=self.device,
                            apply_postprocessing=False,
                        )
                    self._image_predictor = SAM2ImagePredictor(model)
        return self._image_predictor

    def video_predictor(self):
        if self._video_predictor is None:
            with self.lock:
                if self._video_predictor is None:
                    self._validate()
                    self._ensure_source_on_path()
                    _activate_device(self.device)
                    from sam2.build_sam import build_sam2_video_predictor

                    with self._hydra_context():
                        self._video_predictor = build_sam2_video_predictor(
                            self.settings.medsam2_config_file,
                            ckpt_path=str(self.settings.medsam2_checkpoint),
                            device=self.device,
                            apply_postprocessing=False,
                            hydra_overrides_extra=["++model.non_overlap_masks=true"],
                        )
        return self._video_predictor

    # ------------------------------------------------------------ inference
    def segment_image(
        self,
        image: Image.Image,
        box: list[float],
        *,
        score_threshold: float = 0.0,
        max_masks: int = 1,
    ) -> SegmentationResult:
        """Box-prompted segmentation of a single image."""
        predictor = self.image_predictor()
        with self.lock, torch.inference_mode(), _inference_context(self.device):
            _activate_device(self.device)
            predictor.set_image(np.array(image.convert("RGB")))
            masks, scores_np, _ = predictor.predict(
                box=np.array(box, dtype=np.float32),
                multimask_output=max_masks > 1,
            )

        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        scores = [float(s) for s in np.asarray(scores_np).reshape(-1).tolist()]
        order = sorted(range(len(masks)), key=lambda i: -(scores[i] if i < len(scores) else 0))
        kept = [
            i for i in order if (scores[i] if i < len(scores) else 0.0) >= score_threshold
        ][: max(1, max_masks)] or order[:1]

        kept_masks = [to_numpy_mask(masks[i]) for i in kept]
        kept_boxes = [b for b in (bbox_from_mask(m) for m in kept_masks) if b]
        result = SegmentationResult(
            source="medsam2",
            query="box prompt",
            masks=kept_masks,
            boxes=kept_boxes,
            scores=[scores[i] if i < len(scores) else 0.0 for i in kept],
            prompt_box=list(box),
        )
        result.overlay = render_overlay(image, result.masks, result.boxes)
        return result

    def start_tracking(
        self,
        *,
        frames_dir: Path,
        prompt_box: list[float],
        start_index: int,
        score_threshold: float = 0.0,
    ) -> TrackingSession:
        """Open a streaming tracking session over a sampled-frames directory."""
        return TrackingSession(
            model=self,
            frames_dir=frames_dir,
            prompt_box=prompt_box,
            start_index=start_index,
            score_threshold=score_threshold,
        )
