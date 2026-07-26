"""The LangGraph surgical-assistant agent.

Compiled once per process with a `MemorySaver` checkpointer; multi-turn memory
comes from invoking with a stable `thread_id` per UI session. A per-request
tool budget keeps latency bounded during live procedures.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from PIL import Image

from core.config import get_settings
from core.agent.tools import AGENT_TOOLS, pop_last_result
from core.imaging import SegmentationResult
from core.llm import build_chat_llm

SYSTEM_PROMPT = """You are an intraoperative ultrasound assistant for PTMC RFA workflows.

You support a surgeon before, during, and after radiofrequency ablation of a
papillary thyroid microcarcinoma. You are decision support, not a decision
maker: never diagnose, never instruct, phrase suggestions as observations the
surgeon can verify.

Evidence policy:
- The current ultrasound image/frame is the primary evidence for image
  questions. Never invent findings not visible or not supported by a tool.
- Be concise and explicit about uncertainty; an operating room has no time
  for filler.

Tool policy:
- If a user-drawn PTMC box is available and the question asks for PTMC
  segmentation, localization, boundaries, or an overlay, call
  `segment_with_box` FIRST with exactly that box.
- With no box available, use `segment_by_text` with a short prompt such as
  'PTMC', 'thyroid nodule', or 'ablation zone'.
- Use `measure_target` when asked about size, area, or change in size after a
  segmentation exists.
- Use `ablation_status` for questions about ablation progress, coverage, or
  residual PTMC ("how much is ablated?", "did I cover everything?"); report
  its numbers as tracking-based proxies to verify visually, never as measured
  thermal ablation.
- Use `web_search` only for general background knowledge, never for
  image-specific findings.
- Reuse recent tool results from the conversation instead of re-running tools
  unless the user asks to re-segment or the image changed.
- You have a hard budget of {tool_budget} tool calls per request; make them
  count, then answer with the best available evidence.

Answering policy:
- When a segmentation ran, say an overlay is available and give the key
  numbers (score, size) rather than raw coordinates.
- During the procedure, weave in the provided procedure context (phase, area
  trend) when it is relevant to the question.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int


def _build_llm() -> ChatOpenAI:
    return build_chat_llm()


def _agent_node(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    system = SystemMessage(content=SYSTEM_PROMPT.format(tool_budget=settings.tool_budget))
    llm = _build_llm()
    if state.get("tool_call_count", 0) >= settings.tool_budget:
        system = SystemMessage(
            content=system.content
            + "\nTool budget is exhausted for this request: do NOT call tools; "
            "answer from the conversation evidence."
        )
    else:
        llm = llm.bind_tools(AGENT_TOOLS)
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


_TOOL_NODE = ToolNode(AGENT_TOOLS)


def _tools_node(state: AgentState) -> dict[str, Any]:
    result = _TOOL_NODE.invoke(state)
    last = state["messages"][-1]
    calls = len(last.tool_calls) if isinstance(last, AIMessage) else 0
    return {
        "messages": result["messages"],
        "tool_call_count": state.get("tool_call_count", 0) + calls,
    }


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    if (
        isinstance(last, AIMessage)
        and last.tool_calls
        and state.get("tool_call_count", 0) < get_settings().tool_budget
    ):
        return "tools"
    return END


@lru_cache(maxsize=1)
def get_agent_graph():
    """Compile the agent graph once per process, with in-memory checkpoints."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


@dataclass
class AgentAnswer:
    """What the UI needs from one agent turn."""

    answer: str
    overlay: Image.Image | None = None
    tool_trace: list[str] = field(default_factory=list)
    tool_logs: list[dict[str, Any]] = field(default_factory=list)
    segmentation: SegmentationResult | None = None
    error: str | None = None


def _save_temp_image(image: Image.Image) -> str:
    path = Path(tempfile.gettempdir()) / "agentic_sam_v2_turns" / f"{uuid.uuid4()}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return str(path)


_TARGET_TERMS = ("ptmc", "papillary thyroid microcarcinoma", "tumor", "nodule", "target")
_SEGMENT_TERMS = (
    "segment", "segmentation", "where", "locat", "boundar", "overlay",
    "mask", "highlight", "show", "outline",
)


_ABLATION_TERMS = (
    "ablat",     # ablate / ablated / ablation
    "cover",     # cover / covered / coverage / uncovered
    "residual",
    "remaining",
    "burn",
    "treated",
    "untreated",
    "missed",
    "how much",
)


def _should_force_ablation_status(question: str) -> bool:
    """Ablation questions must be answered from measured data, never guessed.

    Small local models will happily invent a coverage percentage; grounding
    this route deterministically is a safety requirement, not an optimization.
    """
    normalized = question.lower()
    return any(term in normalized for term in _ABLATION_TERMS)


def _run_forced_ablation_status(
    *, question: str, procedure_context: str, thread_id: str
) -> AgentAnswer:
    """Read the live ablation tracker, then summarize with one LLM pass."""
    from core.agent.tools import ablation_status

    tool_output = ablation_status.invoke({})
    prompt = (
        "Ablation status from the live tracker (authoritative — never invent or "
        f"round away these numbers):\n{tool_output}\n"
        + (f"Procedure context:\n{procedure_context}\n" if procedure_context else "")
        + f"User question: {question}\n"
        "Answer concisely using ONLY these numbers. State that coverage is a "
        "tracking-based proxy to verify visually. If the tracker is not active, "
        "say tracking has not started yet and give no percentages."
    )
    try:
        response = build_chat_llm().invoke(
            [SystemMessage(content=SYSTEM_PROMPT.format(tool_budget=0)), HumanMessage(content=prompt)]
        )
        answer = response.text.strip()
    except Exception as exc:
        return AgentAnswer(answer="", error=f"{exc.__class__.__name__}: {exc}")

    try:
        get_agent_graph().update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [HumanMessage(content=f"User question: {question}"),
                          AIMessage(content=answer)]},
        )
    except Exception:
        pass
    return AgentAnswer(
        answer=answer,
        tool_trace=["ablation_status"],
        tool_logs=[{"tool": "ablation_status", "content": str(tool_output)[:2000]}],
    )


def _should_force_box_segmentation(question: str) -> bool:
    """Deterministic route: user box + a PTMC localization/segmentation ask.

    Small local models don't reliably obey 'call the tool first' instructions,
    so this guarantee cannot be left to the LLM. It also saves one
    tool-selection round-trip.
    """
    normalized = question.lower()
    return any(t in normalized for t in _TARGET_TERMS) and any(
        t in normalized for t in _SEGMENT_TERMS
    )


def _run_forced_box_segmentation(
    *, question: str, image_path: str, box: list[float],
    procedure_context: str, thread_id: str,
) -> AgentAnswer:
    """Run MedSAM2 on the user box directly, then one LLM summarization pass."""
    from core.agent.tools import segment_with_box

    box_text = ",".join(str(round(v, 1)) for v in box)
    try:
        tool_output = segment_with_box.invoke({"image_path": image_path, "box": box_text})
    except Exception as exc:
        return AgentAnswer(answer="", error=f"Segmentation failed: {exc}")

    prompt = (
        f"The user drew a PTMC box, so MedSAM2 segmentation ran first as required.\n"
        f"MedSAM2 result: {tool_output}\n"
        + (f"Procedure context:\n{procedure_context}\n" if procedure_context else "")
        + f"User question: {question}\n"
        "Answer concisely from this segmentation result; mention that an overlay is available."
    )
    try:
        response = build_chat_llm().invoke(
            [SystemMessage(content=SYSTEM_PROMPT.format(tool_budget=0)), HumanMessage(content=prompt)]
        )
        answer = response.text.strip()
    except Exception as exc:
        return AgentAnswer(answer="", error=f"{exc.__class__.__name__}: {exc}")

    # Record the turn in the thread so follow-up questions have this context.
    try:
        get_agent_graph().update_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [HumanMessage(content=f"User question: {question}"),
                          AIMessage(content=answer)]},
        )
    except Exception:
        pass  # memory is best-effort; the answer itself already succeeded

    segmentation = pop_last_result()
    return AgentAnswer(
        answer=answer,
        overlay=segmentation.overlay if segmentation else None,
        tool_trace=["segment_with_box"],
        tool_logs=[{"tool": "segment_with_box", "content": str(tool_output)[:2000]}],
        segmentation=segmentation,
    )


def ask_agent(
    question: str,
    *,
    image: Image.Image | None,
    box: list[float] | None = None,
    procedure_context: str = "",
    thread_id: str = "default",
) -> AgentAnswer:
    """Run one agent turn. Never raises: errors come back in `AgentAnswer.error`."""
    question = question.strip()
    if not question:
        return AgentAnswer(answer="", error="Question must not be empty.")

    if _should_force_ablation_status(question):
        return _run_forced_ablation_status(
            question=question,
            procedure_context=procedure_context,
            thread_id=thread_id,
        )

    if image is not None and box is not None and _should_force_box_segmentation(question):
        pop_last_result()
        return _run_forced_box_segmentation(
            question=question,
            image_path=_save_temp_image(image),
            box=box,
            procedure_context=procedure_context,
            thread_id=thread_id,
        )

    blocks = []
    if image is not None:
        blocks.append(f"Image path available for tools: {_save_temp_image(image)}")
    if box is not None:
        blocks.append(
            "Available user-drawn PTMC box for `segment_with_box`: "
            + ",".join(str(round(v, 1)) for v in box)
        )
    if procedure_context:
        blocks.append("Procedure context:\n" + procedure_context)
    blocks.append("User question:\n" + question)
    message = HumanMessage(content="\n\n".join(blocks))

    pop_last_result()  # clear any stale segmentation from a previous turn
    try:
        graph = get_agent_graph()
        before = graph.get_state({"configurable": {"thread_id": thread_id}})
        n_before = len(before.values.get("messages", [])) if before and before.values else 0
        result = graph.invoke(
            {"messages": [message], "tool_call_count": 0},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        return AgentAnswer(answer="", error=f"{exc.__class__.__name__}: {exc}")

    new_messages = result["messages"][n_before:]
    final = next(
        (m for m in reversed(new_messages) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    answer = final.text.strip() if final else "No final answer was generated."

    tool_trace = [m.name or "tool" for m in new_messages if isinstance(m, ToolMessage)]
    tool_logs = [
        {"tool": m.name, "content": str(m.content)[:2000]}
        for m in new_messages
        if isinstance(m, ToolMessage)
    ]
    segmentation = pop_last_result()
    return AgentAnswer(
        answer=answer,
        overlay=segmentation.overlay if segmentation else None,
        tool_trace=tool_trace,
        tool_logs=tool_logs,
        segmentation=segmentation,
    )
