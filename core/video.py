"""Video frame sampling for playback and tracking."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
from PIL import Image


def is_lfs_pointer_stub(path: str | Path) -> bool:
    """True when the file is a Git LFS pointer stub instead of real media."""
    try:
        file_path = Path(path)
        if file_path.stat().st_size > 1024:
            return False
        return file_path.read_bytes().startswith(b"version https://git-lfs")
    except OSError:
        return False


@dataclass
class SampledVideo:
    """Frames extracted from a source video at a reduced playback rate.

    `frames_dir` holds sequentially numbered JPEGs, which is exactly the
    layout SAM2's video predictor consumes — the same directory backs both
    UI playback and MedSAM2 tracking.
    """

    source_path: str
    frames_dir: Path
    frame_paths: list[str]
    original_indices: list[int]
    source_fps: float
    playback_fps: int
    stride: int

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    def frame_image(self, index: int) -> Image.Image:
        index = max(0, min(index, self.frame_count - 1))
        return Image.open(self.frame_paths[index]).convert("RGB")


def sample_video(video_path: str | Path, playback_fps: int, max_frames: int) -> SampledVideo:
    """Extract frames at ~playback_fps into a temp directory."""
    metadata = iio.immeta(str(video_path))
    source_fps = float(metadata.get("fps") or 30.0)
    stride = max(1, int(round(source_fps / max(1, playback_fps))))

    frames_dir = Path(tempfile.gettempdir()) / "agentic_sam_v2_frames" / str(uuid.uuid4())
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    original_indices: list[int] = []
    for original_idx, frame in enumerate(iio.imiter(str(video_path))):
        if original_idx % stride != 0:
            continue
        if len(frame_paths) >= max_frames:
            break
        image = Image.fromarray(frame).convert("RGB")
        frame_file = frames_dir / f"{len(frame_paths):06d}.jpg"
        image.save(frame_file, format="JPEG", quality=92)
        frame_paths.append(str(frame_file))
        original_indices.append(original_idx)

    return SampledVideo(
        source_path=str(video_path),
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        original_indices=original_indices,
        source_fps=source_fps,
        playback_fps=playback_fps,
        stride=stride,
    )
