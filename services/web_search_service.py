# services/web_search_service.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List

from openai import OpenAI

from core.logging import logger


@dataclass(frozen=True)
class WebSearchSource:
    title: str | None
    url: str | None
    snippet: str | None = None


@dataclass(frozen=True)
class WebSearchResult:
    query: str
    answer: str
    sources: List[WebSearchSource]
    latency_ms: float | None = None
    model: str | None = None


_CLIENT: OpenAI | None = None


def web_search_enabled() -> bool:
    return (os.getenv("WEB_SEARCH_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")

def _env_flag(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


WEB_SEARCH_TOOL_TYPE = (os.getenv("WEB_SEARCH_TOOL_TYPE") or "web_search_preview").strip() or "web_search_preview"
WEB_SEARCH_REQUIRE_SOURCES = _env_flag("WEB_SEARCH_REQUIRE_SOURCES", "1")
WEB_SEARCH_MIN_SOURCES = int((os.getenv("WEB_SEARCH_MIN_SOURCES") or "1").strip() or "1")
WEB_SEARCH_EMPTY_SOURCES_MESSAGE = (
    (os.getenv("WEB_SEARCH_EMPTY_SOURCES_MESSAGE") or "").strip()
    or "I couldn't find reliable sources for that right now."
)

def _get_client() -> OpenAI | None:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()
    if not api_key:
        return None
    try:
        _CLIENT = OpenAI(api_key=api_key)
        return _CLIENT
    except Exception:
        return None


def _safe_get_output_text(resp: Any) -> str:
    try:
        out = getattr(resp, "output_text", None)
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception:
        pass

    try:
        parts: list[str] = []
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if itype == "message":
                    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    if isinstance(content, list):
                        for c in content:
                            ctype = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
                            if ctype in ("output_text", "text"):
                                txt = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                                if isinstance(txt, str) and txt.strip():
                                    parts.append(txt.strip())
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass

    return ""


def _extract_sources(resp: Any) -> List[WebSearchSource]:
    sources: list[WebSearchSource] = []
    try:
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if itype != "web_search_call":
                    continue
                action = item.get("action") if isinstance(item, dict) else getattr(item, "action", None)
                if not action:
                    continue
                srcs = action.get("sources") if isinstance(action, dict) else getattr(action, "sources", None)
                if isinstance(srcs, list):
                    for s in srcs:
                        if isinstance(s, dict):
                            sources.append(
                                WebSearchSource(
                                    title=s.get("title"),
                                    url=s.get("url"),
                                    snippet=s.get("snippet"),
                                )
                            )
    except Exception:
        return sources

    dedup: list[WebSearchSource] = []
    seen: set[str] = set()
    for s in sources:
        u = (s.url or "").strip()
        if u and u in seen:
            continue
        if u:
            seen.add(u)
        dedup.append(s)
    return dedup


def run_web_search(
    query: str,
    *,
    model: str | None = None,
    max_sources: int = 6,
    timeout_s: float = 15.0,
) -> WebSearchResult:
    q = (query or "").strip()
    if not q:
        return WebSearchResult(query="", answer="", sources=[], latency_ms=0.0, model=model)

    if not web_search_enabled():
        return WebSearchResult(
            query=q,
            answer="Web search is currently disabled.",
            sources=[],
            latency_ms=0.0,
            model=model,
        )

    client = _get_client()
    if client is None:
        return WebSearchResult(
            query=q,
            answer="Web search is not available because the OpenAI API key is not configured.",
            sources=[],
            latency_ms=0.0,
            model=model,
        )

    chosen_model = (model or os.getenv("WEB_SEARCH_MODEL") or os.getenv("OPENAI_ROUTER_MODEL") or "gpt-4o-mini").strip()

    t0 = time.perf_counter()
    try:
        resp = None
        last_err: Exception | None = None
        # Try configured tool type first, then fall back between known variants.
        tool_types: list[str] = []
        for tt in [WEB_SEARCH_TOOL_TYPE, 'web_search_preview', 'web_search']:
            if tt and tt not in tool_types:
                tool_types.append(tt)

        for tt in tool_types:
            try:
                resp = client.responses.create(
                    model=chosen_model,
                    input=[{'role': 'user', 'content': q}],
                    tools=[{'type': tt}],
                    tool_choice={'type': tt},
                    include=['web_search_call.action.sources'],
                    timeout=timeout_s,
                )
                break
            except TypeError:
                # Some SDK versions do not accept tool_choice/timeout.
                try:
                    resp = client.responses.create(
                        model=chosen_model,
                        input=[{'role': 'user', 'content': q}],
                        tools=[{'type': tt}],
                        tool_choice={'type': tt},
                        include=['web_search_call.action.sources'],
                    )
                    break
                except TypeError as e:
                    last_err = e
                    continue
            except Exception as e:
                last_err = e
                continue

        if resp is None:
            raise last_err or RuntimeError('web_search_failed')
    except Exception as e:
        logger.exception("WEB_SEARCH_FAIL query=%r err=%s", q, e)
        return WebSearchResult(query=q, answer="I couldn't complete the web search due to an error.", sources=[], model=chosen_model)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    answer = _safe_get_output_text(resp)
    sources = _extract_sources(resp)

    if max_sources and len(sources) > max_sources:
        sources = sources[:max_sources]


    # Grounding guard: if we cannot produce any sources, do NOT return a freeform answer.
    if WEB_SEARCH_REQUIRE_SOURCES and len(sources) < max(0, WEB_SEARCH_MIN_SOURCES):
        logger.warning("WEB_SEARCH_NO_SOURCES q_len=%s model=%s", len(q), chosen_model)
        return WebSearchResult(query=q, answer=WEB_SEARCH_EMPTY_SOURCES_MESSAGE, sources=[], latency_ms=dt_ms, model=chosen_model)
    logger.info("WEB_SEARCH_OK dt_ms=%s q_len=%s sources=%s", round(dt_ms, 1), len(q), len(sources))
    return WebSearchResult(query=q, answer=(answer or "").strip(), sources=sources, latency_ms=dt_ms, model=chosen_model)