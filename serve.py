"""Warm start: load every model BEFORE the app accepts connections.

Run by `run.sh` (or directly). Order:
1. Wait for the LLM endpoint (the local vLLM server) to be ready.
2. Preload segmentation models in THIS process — Streamlit script runs are
   threads of the same process, so they share the warmed registry and the
   surgeon never pays a lazy-load stall mid-procedure.
3. Start the Streamlit server.

Env knobs: ASAM_PRELOAD (comma list of medsam2,medical_sam3,sam3; default all),
ASAM_LLM_WAIT_S (default 600), ASAM_PORT / ASAM_ADDRESS for the UI.
"""

from __future__ import annotations

import os
import sys
import time


def wait_for_llm(timeout_s: float) -> None:
    from core.config import get_settings
    from core.llm import llm_endpoint_ready

    settings = get_settings()
    if not settings.llm_base_url:
        print("[serve] LLM: using OpenAI cloud (no local endpoint to wait for)")
        return
    print(f"[serve] waiting for LLM endpoint {settings.llm_base_url} …", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, detail = llm_endpoint_ready()
        if ready:
            print(f"[serve] LLM ready: {detail}")
            return
        time.sleep(3)
    raise SystemExit(
        f"[serve] LLM endpoint {settings.llm_base_url} not ready after {timeout_s:.0f}s "
        "— check .logs/vllm.log"
    )


def preload_models() -> None:
    from core.models.registry import get_registry

    selected = {
        name.strip()
        for name in os.getenv("ASAM_PRELOAD", "medsam2,medical_sam3,sam3").split(",")
        if name.strip()
    }
    registry = get_registry()
    if "medsam2" in selected:
        started = time.monotonic()
        registry.medsam2.image_predictor()
        registry.medsam2.video_predictor()
        print(f"[serve] MedSAM2 loaded in {time.monotonic() - started:.1f}s", flush=True)
    if "medical_sam3" in selected:
        started = time.monotonic()
        registry.medical_sam3.processor()
        print(f"[serve] Medical-SAM3 loaded in {time.monotonic() - started:.1f}s", flush=True)
    if "sam3" in selected:
        started = time.monotonic()
        registry.sam3.processor()
        print(f"[serve] SAM 3.1 loaded in {time.monotonic() - started:.1f}s", flush=True)


def main() -> None:
    wait_for_llm(float(os.getenv("ASAM_LLM_WAIT_S", "600")))
    preload_models()

    port = int(os.getenv("ASAM_PORT", "7870"))
    address = os.getenv("ASAM_ADDRESS", "0.0.0.0")
    print(f"[serve] all models warm — starting UI on {address}:{port}", flush=True)

    from streamlit.web import bootstrap

    flag_options = {
        "server.port": port,
        "server.address": address,
        "server.headless": True,
        "browser.gatherUsageStats": False,
    }
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run("app.py", False, [], flag_options)


if __name__ == "__main__":
    sys.exit(main())
