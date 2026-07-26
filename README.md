# Agentic-SAM v2 — RFA Surgical Assistant

An agentic medical-image-segmentation assistant that supports a surgeon
before, during, and after ultrasound-guided RFA (radiofrequency ablation) of
PTMC. Built on **LangGraph** (in-process tool agent) and **Streamlit**
(single-server UI), with **MedSAM2** for box-prompted segmentation and video
tracking, and **Medical-SAM3 / SAM 3.1** for text-prompted segmentation.

> Research and workflow prototyping only — decision support, not autonomous
> diagnosis or treatment. All overlays must be verified visually.

## Architecture (one process, no micro-services)

```
streamlit run app.py
 ├─ ui/            Live Procedure · Image Analysis · Procedure Report · System Status
 ├─ core/agent/    LangGraph agent (MemorySaver memory, 4 tools, tool budget),
 │                 live tracking monitor, operative-note generator
 ├─ core/models/   lazy in-process registry: MedSAM2 (box + tracking),
 │                 Medical-SAM3 (text), SAM 3.1 (text fallback)
 └─ core/          config, imaging math, video sampling, procedure event log
```

Everything runs inside the Streamlit process: the agent is invoked directly
(no HTTP hop), models load lazily behind locks, and the sampled-frames
directory used for playback is the same one MedSAM2's video predictor tracks
over.

## Features

- **Live Procedure** — play/pause the intraoperative video, click the PTMC,
  MedSAM2 segments it on that frame and keeps tracking it, displaying each
  frame as soon as it is segmented; a live AI monitor posts short situational
  notes; apparent size-drops raise alerts; snapshots and questions land in
  the procedure event log; RFA phases (Pre-op → Targeting → Ablation →
  Post-ablation) stamp everything.
- **Image Analysis** — chat about any ultrasound still with multi-turn
  memory; the agent picks tools: `segment_with_box` (MedSAM2),
  `segment_by_text` (Medical-SAM3 → SAM 3.1 fallback), `measure_target`,
  `web_search`.
- **Procedure Report** — area-trend chart, event table, snapshot gallery, and
  a generated Markdown session summary note (downloadable).
- **System Status** — model/checkpoint/GPU state and lazy preload buttons.

## Setup

```bash
conda create -n agentic-sam-v2 python=3.11 -y
conda activate agentic-sam-v2
pip install -r requirements.txt

# restore model weights + MedSAM2 source (skips whatever already exists)
python downloader.py

# optional secrets (.env): HF_TOKEN for the gated SAM 3.1 repo;
# OPENAI_API_KEY only if you want the OpenAI cloud instead of local vLLM
cp .env.example .env
```

The LLM is served locally by **vLLM** through the OpenAI-compatible API
(default model `Qwen/Qwen2.5-VL-7B-Instruct` — vision + tool calling; the
first launch downloads it). A separate vLLM installation is used to serve
(`ASAM_VLLM_BIN` points at its `vllm` binary).

`data/video/` holds the sample RFA video; it is not downloadable — copy it
from wherever your recordings live.

## Run

```bash
bash run.sh                    # uses the CONFIGURATION block in run.sh
bash run.sh 2                  # ...but on GPU 2 -> http://<host>:7870
ASAM_PORT=8080 bash run.sh 2   # custom UI port
```

The top of `run.sh` is a small **CONFIGURATION block** — edit it instead of
passing env vars:

| Setting | Purpose |
|---|---|
| `GPU` | which GPU the whole stack uses (arg 1 overrides) |
| `VLLM_MEM_FRACTION` | share of that GPU vLLM may reserve (rest is for the segmentation models) |
| `APP_ENV` | Python env running Streamlit + segmentation models |
| `VLLM_ENV` | Python env running the vLLM server (may be the same as `APP_ENV` once vLLM is installed there) |
| `LLM_MODEL` / `LLM_PORT` | model vLLM serves and its port |
| `UI_PORT` / `UI_ADDRESS` | Streamlit bind address |

`CUDA_VISIBLE_DEVICES` pins the chosen card, so inside every process it is
`cuda:0` — that is why `ASAM_DEVICE` stays `cuda:0` regardless of which GPU
number you pick. Ctrl+C stops the UI and the vLLM server together and frees
the GPU.

`run.sh` brings the whole stack up **warm before the UI opens**: it starts
the vLLM server on the chosen GPU, waits until it answers, preloads MedSAM2 +
Medical-SAM3 + SAM 3.1 in the app process (`serve.py`), and only then binds
the Streamlit port — the surgeon never hits a model cold-start mid-procedure.
Everything (LLM + all segmentation models, ~45 GB total) shares the one GPU.

## Configuration (env vars, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `ASAM_GPU` / arg 1 | `0` | GPU number the whole stack runs on |
| `ASAM_LLM_BASE_URL` | `http://127.0.0.1:8601/v1` | OpenAI-compatible endpoint; `""` = OpenAI cloud |
| `ASAM_LLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | model vLLM serves / the app requests |
| `ASAM_APP_ENV` | `/mnt/data/ubuntu/research/env/agentic` | env running the app |
| `ASAM_VLLM_ENV` | `…/envs/vllm_m3` | env running vLLM (its `bin/` goes on `PATH` so vLLM's JIT can find `ninja`) |
| `ASAM_VLLM_MEM_FRACTION` | `0.35` | GPU memory share vLLM may take |
| `ASAM_PRELOAD` | `medsam2,medical_sam3,sam3` | models warmed before the UI opens |
| `OPENAI_API_KEY` | — | only for the OpenAI-cloud fallback |
| `ASAM_DEVICE` | `cuda:0` | torch device (within `CUDA_VISIBLE_DEVICES`) |
| `ASAM_PLAYBACK_FPS` | `2` | sampled playback/tracking rate |
| `ASAM_MAX_VIDEO_FRAMES` | `240` | frame cap per loaded video |
| `ASAM_TRACKING_BOX_SIZE` | `96` | default click-to-box size (px) |
| `ASAM_PIXEL_SPACING_MM` | `0` | mm-per-pixel; enables mm measurements |
| `ASAM_TOOL_BUDGET` | `2` | agent tool calls per request |

## Testing

```bash
python -m pytest tests/ -q      # pure-logic unit tests (no GPU needed)
```
