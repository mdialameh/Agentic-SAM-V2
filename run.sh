#!/usr/bin/env bash
# Launch Agentic-SAM v2 fully warm on one GPU.
#
#   bash run.sh            # uses the settings below
#   bash run.sh 2          # ...but on GPU 2 (arg 1 overrides GPU)
#
# Order: start the vLLM OpenAI-compatible server -> wait until it answers ->
# preload every segmentation model in the app process -> open the UI port.

set -euo pipefail

# ==========================================================================
# CONFIGURATION — edit these
# ==========================================================================
# Which GPU the whole stack runs on (arg 1 or ASAM_GPU override this).
GPU="${1:-${ASAM_GPU:-0}}"

# Share of that GPU's memory vLLM may reserve (0.0-1.0).
# The segmentation models (MedSAM2 + Medical-SAM3 + SAM 3.1, ~8 GB) use the
# rest, so leave headroom: 0.35 of an 80 GB card is ~28 GB for the LLM.
VLLM_MEM_FRACTION="${ASAM_VLLM_MEM_FRACTION:-0.35}"

# Python environment that runs the app (Streamlit + segmentation models).
APP_ENV="${ASAM_APP_ENV:-/mnt/data/ubuntu/research/env/agentic}"

# Python environment that runs the vLLM server. Point it at APP_ENV once vLLM
# is installed there; until then it must be an env that actually has vLLM.
VLLM_ENV="${ASAM_VLLM_ENV:-/home/ubuntu/miniconda3/envs/vllm_m3}"

# LLM served by vLLM (vision + tool calling required).
LLM_MODEL="${ASAM_LLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
LLM_PORT="${ASAM_LLM_PORT:-8601}"
VLLM_MAX_LEN="${ASAM_VLLM_MAX_LEN:-16384}"

# Streamlit UI.
UI_PORT="${ASAM_PORT:-7870}"
UI_ADDRESS="${ASAM_ADDRESS:-0.0.0.0}"
# ==========================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p .logs

# The stack sees exactly one GPU, so inside every process it is index 0.
# (Do NOT set this to cuda:$GPU — after CUDA_VISIBLE_DEVICES the selected
# card is renumbered to 0 and cuda:2 would not exist.)
export CUDA_VISIBLE_DEVICES="$GPU"
export ASAM_DEVICE="cuda:0"
export LD_LIBRARY_PATH=""   # keep torch's bundled cuBLAS, not the host's

export ASAM_LLM_BASE_URL="${ASAM_LLM_BASE_URL:-http://127.0.0.1:${LLM_PORT}/v1}"
export ASAM_LLM_MODEL="$LLM_MODEL"
export ASAM_PORT="$UI_PORT"
export ASAM_ADDRESS="$UI_ADDRESS"

APP_PYTHON="$APP_ENV/bin/python"
VLLM_BIN="${ASAM_VLLM_BIN:-$VLLM_ENV/bin/vllm}"

[[ -x "$APP_PYTHON" ]] || { echo "App python not found: $APP_PYTHON (set ASAM_APP_ENV)" >&2; exit 1; }

VLLM_PID=""
APP_PID=""
cleanup() {
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM (pid $VLLM_PID)…"
    # Kill the whole process group: vLLM forks an engine subprocess that
    # otherwise survives and keeps holding GPU memory.
    kill -- "-$VLLM_PID" 2>/dev/null || kill "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$VLLM_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 -- "-$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "GPU $GPU | app env: $APP_ENV | vLLM env: $VLLM_ENV | vLLM memory: $VLLM_MEM_FRACTION"

if curl -sf -m 3 "${ASAM_LLM_BASE_URL%/}/models" > /dev/null 2>&1; then
  echo "vLLM already serving at ${ASAM_LLM_BASE_URL} — reusing it."
else
  [[ -x "$VLLM_BIN" ]] || {
    echo "vLLM binary not found: $VLLM_BIN" >&2
    echo "Install vLLM in an env and set ASAM_VLLM_ENV (or ASAM_VLLM_BIN)." >&2
    exit 1
  }
  echo "Starting vLLM: $LLM_MODEL on port $LLM_PORT"
  # vLLM's JIT kernel build shells out to `ninja`, so its env bin MUST be on
  # PATH — calling the binary by absolute path alone is not enough.
  # setsid: own process group so cleanup takes the engine subprocess down too.
  PATH="$VLLM_ENV/bin:$PATH" setsid "$VLLM_BIN" serve "$LLM_MODEL" \
    --port "$LLM_PORT" \
    --gpu-memory-utilization "$VLLM_MEM_FRACTION" \
    --max-model-len "$VLLM_MAX_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > .logs/vllm.log 2>&1 &
  VLLM_PID=$!
  echo "vLLM pid $VLLM_PID — logs: .logs/vllm.log (first run downloads the model)"
fi

echo "Preloading models, then opening the UI on ${UI_ADDRESS}:${UI_PORT}…"
# Run the app in the background and wait on it, so this shell stays responsive
# to signals (inside a `| tee` pipeline it would not react until the pipe
# closed, leaving vLLM alive and holding GPU memory).
PYTHONUNBUFFERED=1 "$APP_PYTHON" serve.py > >(tee .logs/app.log) 2>&1 &
APP_PID=$!
wait "$APP_PID" || true
