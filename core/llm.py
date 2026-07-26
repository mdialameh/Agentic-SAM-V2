"""Shared LLM construction: one place decides where inference runs.

Default target is a local vLLM server speaking the OpenAI API
(`settings.llm_base_url`). Setting `ASAM_LLM_BASE_URL=""` with a real
`OPENAI_API_KEY` switches everything (agent, live monitor, reports) to the
OpenAI cloud with the Responses API.
"""

from __future__ import annotations

import requests

from core.config import get_settings


def build_chat_llm(temperature: float = 0.0):
    """ChatOpenAI wired to the configured endpoint (vLLM by default)."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if settings.llm_base_url:
        return ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "EMPTY",
            model=settings.llm_model,
            temperature=temperature,
        )
    if not settings.llm_api_key or settings.llm_api_key == "EMPTY":
        raise RuntimeError(
            "No LLM configured: set ASAM_LLM_BASE_URL (local vLLM) or OPENAI_API_KEY."
        )
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=temperature,
        use_responses_api=True,
    )


def llm_endpoint_ready(timeout: float = 3.0) -> tuple[bool, str]:
    """Probe the configured endpoint. Returns (ready, detail)."""
    settings = get_settings()
    if not settings.llm_base_url:
        ready = bool(settings.llm_api_key and settings.llm_api_key != "EMPTY")
        return ready, "OpenAI cloud" if ready else "no OPENAI_API_KEY set"
    url = settings.llm_base_url.rstrip("/") + "/models"
    try:
        response = requests.get(url, timeout=timeout)
        if not response.ok:
            return False, f"HTTP {response.status_code} from {url}"
        models = [m.get("id") for m in response.json().get("data", [])]
        if settings.llm_model in models:
            return True, f"vLLM serving {settings.llm_model}"
        return True, f"endpoint up, models: {models}"
    except requests.RequestException as exc:
        return False, f"{url} unreachable ({exc.__class__.__name__})"
