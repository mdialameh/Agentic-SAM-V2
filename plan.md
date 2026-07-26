# Agentic-SAM v2 — RFA Surgical Assistant (LangGraph + Streamlit)

A ground-up rebuild of Agentic-SAM as a single-process agentic medical image
segmentation assistant for surgeons during RFA (radiofrequency ablation) of
PTMC. LangGraph runs the agent **in-process**, segmentation models load
**in-process** behind a lazy registry, and Streamlit is the only server —
no FastAPI micro-services, no port orchestration, no cross-service state.

## 1. Lessons from v1 (why rebuild)

- Four FastAPI services + UI meant port juggling, restart choreography, and
  state split across processes. v2: one `streamlit run`, models and agent
  in-process.
- The agent lived behind HTTP; conversation memory was a custom store. v2:
  LangGraph with a checkpointer keyed by session, invoked directly.
- Two `run_every` fragments interfered (tracking froze). v2: exactly one timed
  fragment per page, background threads for LLM work.
- Model checkpoints already exist locally (copied/hardlinked from v1):
  MedSAM2 (box-prompt image + video tracking), Medical-SAM3 (text-prompt,
  10 GB), SAM 3.1 (text-prompt fallback, 3.5 GB). All load lazily.

## 2. Product: what the surgeon gets

- **Live Procedure page (core)** — load/pause the intraoperative ultrasound
  video, click the PTMC, MedSAM2 tracks it frame-by-frame with each segmented
  frame displayed as it is produced; a live AI monitor posts short situational
  notes (area trend, size-drop alerts); every notable event lands in the
  procedure event log; the surgeon can ask questions at any time (tracking
  pauses, agent answers on the current frame).
- **RFA phase tracking** — Pre-op review → Targeting → Ablation →
  Post-ablation assessment; the current phase stamps every event and shapes
  the agent's answers.
- **Snapshots & measurements** — one click captures the current overlay +
  mask area into the event log; area trend is measured continuously during
  tracking.
- **Image Analysis page** — chat with the agent about any ultrasound still:
  box-prompted MedSAM2, text-prompted Medical-SAM3 (SAM 3.1 fallback), web
  lookup for background knowledge; multi-turn memory via LangGraph
  checkpointer.
- **Procedure Report page** — the agent turns the event log, phases,
  measurements, and snapshots into a structured Markdown operative note,
  downloadable.
- **System Status page** — model/checkpoint/GPU state, lazy-load controls,
  log viewer.

## 3. Architecture

```
agentic-sam-v2/
  app.py                    # Streamlit entry: st.navigation over ui/ pages
  run.sh                    # launcher (streamlit run app.py)
  downloader.py             # restores checkpoints from Hugging Face if absent
  core/                     # NO streamlit imports — unit-testable engine
    config.py               # every tunable, env-overridable
    imaging.py              # masks, overlays, boxes, area/measurement math
    video.py                # frame sampling, sampled-video handle
    models/
      registry.py           # lazy singletons + per-model locks
      medsam2.py            # box-prompt image seg + streaming video tracker
      text_seg.py           # Medical-SAM3 / SAM 3.1 text-prompted seg
    procedure.py            # RFA phases, event log, snapshots
    agent/
      graph.py              # LangGraph agent, MemorySaver, tool budget
      tools.py              # segment_with_box / segment_by_text / measure / web_search
      monitor.py            # tracking-note generator (area trend + vision LLM)
      report.py             # operative-note generator
  ui/                       # Streamlit pages + shared UI plumbing
    common.py               # cached resources, session state, clickable image
    live_procedure.py       # tracking + monitor + phases + events (ONE fragment)
    image_analysis.py       # chat + box/text segmentation
    report.py               # generate/download operative note
    status.py               # models, GPU, checkpoints, logs
  tests/test_core.py        # pure-logic unit tests
  checkpoints/              # hardlinked from v1 (no extra disk)
  third_party/MedSAM2/      # upstream MedSAM2 source (sam2 package + configs)
  data/                     # sample image + RFA video (copied from v1)
```

Key decisions:
- **In-process models.** `core/models/registry.py` lazy-loads each model once
  per server process behind a lock (`st.cache_resource` on the UI side).
  MedSAM2 (~150 MB) loads at first use in seconds; the 10 GB Medical-SAM3 and
  3.5 GB SAM 3.1 only load when a text-prompt tool actually runs.
- **In-process LangGraph.** `create_agent_graph()` compiled once, invoked with
  `thread_id` = Streamlit session id; `MemorySaver` provides multi-turn memory.
  Tool budget of 2 per request, mirrored from v1's proven prompt policy.
- **One timed fragment per page.** The live page's single fragment owns
  playback, tracking steps, monitor kicks, and note display. All LLM calls
  from timed code run in background threads.
- **Procedure log is the spine.** Everything (phase changes, tracking events,
  size-drop alerts, snapshots, Q&A) appends `ProcedureEvent`s; the report
  generator and the agent both read from it.

## 4. Work items

- [DONE] **P1 — Scaffold + assets**: project tree, `.gitignore`, copy `data/`,
  hardlink `checkpoints/` (MedSAM2 + Medical-SAM3 + SAM 3.1), copy
  `third_party/MedSAM2` source, copy `.env`.
- [DONE] **P2 — Core engine**: `config.py`, `imaging.py` (mask/overlay/box/
  measurement math), `video.py` (frame sampling), `models/registry.py`,
  `models/medsam2.py` (image seg + streaming tracker), `models/text_seg.py`
  (Medical-SAM3, SAM 3.1 fallback).
- [DONE] **P3 — Procedure spine**: `procedure.py` — RFA phases, timestamped
  event log, snapshot records, area-trend series.
- [DONE] **P4 — Agent**: LangGraph graph with MemorySaver + tool budget;
  tools (`segment_with_box`, `segment_by_text`, `measure_target`,
  `web_search`); surgical system prompt with phase context; `monitor.py`
  (live note, background-thread safe); `report.py` (operative note from the
  event log).
- [DONE] **P5 — UI**: `app.py` shell + `common.py`; Live Procedure page
  (single fragment: playback/tracking/monitor/phases/events/snapshot/Q&A);
  Image Analysis page (chat + click-box + quick questions); Report page;
  Status page.
- [DONE] **P6 — Ops**: `run.sh`, `requirements.txt`, `downloader.py` (v2
  paths), README.
- [DONE] **P7 — Verification**: unit tests for core math; live MedSAM2 image
  segmentation + video tracking through the registry; agent chat end-to-end
  (OpenAI); AppTest on every page; headless `streamlit run` smoke test.

- [DONE] **P8 — Ablation coverage monitor**: when tracking starts, save the
  first segmented PTMC as the baseline; sample tracked masks at 1 frame / 5 s
  (`core/ablation.py` AblationTracker — pure pixel math, no LLM): captured
  ratio vs baseline, cumulative treated-coverage map, residual un-covered
  regions, echo-brightness trend. New `ablation_status` agent tool for
  mid-procedure questions. Live coverage metrics on the procedure page. On
  the Report page, a **Generate ablation report** button runs ONE vision-LLM
  call (baseline image + final coverage map + compact numeric timeline) —
  token consumption is deferred entirely to that single press.

- [DONE] **P9 — Local vLLM inference + warm start**: all LLM roles (agent,
  live monitor, both reports) go through one `core/llm.py` builder pointing
  at a local vLLM OpenAI-compatible server (default
  `Qwen/Qwen2.5-VL-7B-Instruct` — vision + hermes tool calling); OpenAI cloud
  stays available via env. `run.sh <gpu>` pins the whole stack to one GPU,
  starts vLLM, waits for it, then `serve.py` preloads MedSAM2 + Medical-SAM3
  + SAM 3.1 **in the app process before the UI port opens** (Streamlit script
  threads share the warmed registry), so nothing cold-starts mid-procedure.

## 5. Quality bar

- [DONE] `core/` imports no Streamlit and is fully unit-testable.
- [DONE] Every model/LLM call has a timeout-or-lock story and surfaces errors
  as user-readable text, never tracebacks.
- [DONE] One timed fragment per page; background threads for LLM work.
- [DONE] All tunables in `core/config.py`, env-overridable.
- [DONE] Event-sourced procedure state; report reproducible from the log.
- [DONE] Graceful degradation: pages render with no GPU, no checkpoints, or
  no OpenAI key, with actionable messages.
