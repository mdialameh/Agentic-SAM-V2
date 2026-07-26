"""Central configuration. Every tunable lives here, env-overridable.

Nothing outside this module reads `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
LOG_DIR = PROJECT_ROOT / ".logs"
SESSION_DIR = PROJECT_ROOT / ".sessions"

CHECKPOINTS = {
    "medsam2": PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt",
    "medical_sam3": PROJECT_ROOT / "checkpoints" / "medical_sam3_2D.pt",
    "sam3": PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt",
}
MEDSAM2_SOURCE_ROOT = PROJECT_ROOT / "third_party" / "MedSAM2"
MEDSAM2_CONFIG_FILE = "sam2.1_hiera_t512.yaml"


def load_env_file() -> None:
    """Populate os.environ from the project `.env` without overriding real env."""
    if not ENV_PATH.is_file():
        return
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for models, agent, and UI behavior."""

    # Models
    device: str = "cuda:0"
    medsam2_checkpoint: Path = CHECKPOINTS["medsam2"]
    medical_sam3_checkpoint: Path = CHECKPOINTS["medical_sam3"]
    sam3_checkpoint: Path = CHECKPOINTS["sam3"]
    medsam2_source_root: Path = MEDSAM2_SOURCE_ROOT
    medsam2_config_file: str = MEDSAM2_CONFIG_FILE

    # LLM (agent + monitor + report) — a local vLLM OpenAI-compatible endpoint
    # by default; set ASAM_LLM_BASE_URL="" plus OPENAI_API_KEY to use OpenAI.
    llm_base_url: str = "http://127.0.0.1:8601/v1"
    llm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm_api_key: str = "EMPTY"  # vLLM ignores it; real key needed for OpenAI

    # Video workflow
    playback_fps: int = 2
    max_video_frames: int = 240
    tracking_box_size: int = 96

    # Imaging / measurement
    pixel_spacing_mm: float = 0.0  # 0 disables mm conversion (px-only)
    ablation_sample_seconds: float = 5.0  # cadence of ablation-coverage samples

    # Agent behavior
    tool_budget: int = 2
    search_max_results: int = 5

    # Sample assets
    sample_image: Path = PROJECT_ROOT / "data" / "sample_ptmc_2.jpg"
    sample_video: Path = PROJECT_ROOT / "data" / "video" / "RFA-video-240716.mp4"

    # UI
    click_display_width: int = 560


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the settings snapshot from environment + `.env`."""
    load_env_file()
    return Settings(
        device=os.getenv("ASAM_DEVICE", Settings.device),
        llm_base_url=os.getenv("ASAM_LLM_BASE_URL", Settings.llm_base_url),
        llm_model=os.getenv(
            "ASAM_LLM_MODEL", os.getenv("OPENAI_MODEL", Settings.llm_model)
        ),
        llm_api_key=os.getenv("OPENAI_API_KEY", "") or Settings.llm_api_key,
        playback_fps=_env_int("ASAM_PLAYBACK_FPS", Settings.playback_fps),
        max_video_frames=_env_int("ASAM_MAX_VIDEO_FRAMES", Settings.max_video_frames),
        tracking_box_size=_env_int("ASAM_TRACKING_BOX_SIZE", Settings.tracking_box_size),
        pixel_spacing_mm=_env_float("ASAM_PIXEL_SPACING_MM", Settings.pixel_spacing_mm),
        ablation_sample_seconds=_env_float(
            "ASAM_ABLATION_SAMPLE_SECONDS", Settings.ablation_sample_seconds
        ),
        tool_budget=_env_int("ASAM_TOOL_BUDGET", Settings.tool_budget),
        search_max_results=_env_int("ASAM_SEARCH_MAX_RESULTS", Settings.search_max_results),
        click_display_width=_env_int("ASAM_CLICK_DISPLAY_WIDTH", Settings.click_display_width),
    )
