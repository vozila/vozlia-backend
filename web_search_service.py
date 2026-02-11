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
    """Extract sources from a Responses API web_search tool call.

    Primary: web_search_call.action.sources (when include=['web_search_call.action.sources'] works).
    Fallback: message annotations (url_citation), which some SDK versions expose even when
    action.sources isn't present.

    Returns a de-duplicated list by URL.
    """

    sources: list[WebSearchSource] = []

    def _add(title: Any, url: Any, snippet: Any = None) -> None:
        try:
            t = title if isinstance(title, str) else None
            u = url if isinstance(url, str) else None
            s = snippet if isinstance(snippet, str) else None
            if u or t or s:
                sources.append(WebSearchSource(title=t, url=u, snippet=s))
        except Exception:
            return

    try:
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            # 1) web_search_call.action.sources
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
                            _add(s.get("title"), s.get("url"), s.get("snippet"))
                        else:
                            _add(getattr(s, "title", None), getattr(s, "url", None), getattr(s, "snippet", None))

            # 2) message annotations (url_citation)
            for item in output:
                itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if itype != "message":
                    continue
                content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    annotations = c.get("annotations") if isinstance(c, dict) else getattr(c, "annotations", None)
                    if not isinstance(annotations, list):
                        continue
                    for a in annotations:
                        atype = a.get("type") if isinstance(a, dict) else getattr(a, "type", None)
                        if atype != "url_citation":
                            continue
                        if isinstance(a, dict):
                            _add(a.get("title"), a.get("url"), a.get("snippet"))
                        else:
                            _add(getattr(a, "title", None), getattr(a, "url", None), getattr(a, "snippet", None))
    except Exception:
        # Best-effort only.
        pass

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
        resp = client.responses.create(
            model=chosen_model,
            input=[{"role": "user", "content": q}],
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
            include=["web_search_call.action.sources"],
            timeout=timeout_s,
        )
    except TypeError:
        resp = client.responses.create(
            model=chosen_model,
            input=[{"role": "user", "content": q}],
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
        )
    except Exception as e:
        logger.exception("WEB_SEARCH_FAIL query=%r err=%s", q, e)
        return WebSearchResult(query=q, answer="I couldn't complete the web search due to an error.", sources=[], model=chosen_model)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    answer = _safe_get_output_text(resp)
    sources = _extract_sources(resp)

    if max_sources and len(sources) > max_sources:
        sources = sources[:max_sources]

    logger.info("WEB_SEARCH_OK dt_ms=%s q_len=%s sources=%s", round(dt_ms, 1), len(q), len(sources))
    return WebSearchResult(query=q, answer=(answer or "").strip(), sources=sources, latency_ms=dt_ms, model=chosen_model)
