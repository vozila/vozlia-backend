# services/investment_service.py
from __future__ import annotations

import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.logging import logger


# In-process quote cache to reduce Yahoo throttling (429).
# Key: sorted comma-joined tickers; Value: (ts_epoch, quotes_by_symbol)
_INVREP_QUOTES_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
_INVREP_QUOTES_CACHE_LOCK = threading.Lock()
from openai import OpenAI
from core import config as cfg

#from typing import List, Dict, Any


_client: OpenAI | None = OpenAI(api_key=cfg.OPENAI_API_KEY) if getattr(cfg, 'OPENAI_API_KEY', None) else None

def _get_openai_client() -> OpenAI:
    if _client is None:
        raise RuntimeError('OPENAI_API_KEY is not set')
    return _client

YF_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YF_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YF_QUOTESUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"

DEFAULT_NEWS_COUNT = 5


def _split_tickers(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        out = []
        for t in raw:
            s = str(t).strip().upper()
            if s:
                out.append(s)
        return out
    # string
    parts = []
    for chunk in str(raw).replace("\n", ",").split(","):
        s = chunk.strip().upper()
        if s:
            parts.append(s)
    # de-dupe, preserve order
    seen = set()
    out = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out



_YAHOO_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

def _yahoo_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": _YAHOO_UA,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://finance.yahoo.com",
            "Referer": "https://finance.yahoo.com/",
            "Connection": "keep-alive",
        },
        timeout=httpx.Timeout(10.0, connect=5.0),
        follow_redirects=True,
    )

def _prime_yahoo_cookies(client: httpx.Client) -> None:
    """Optionally prime Yahoo cookies.

    Historically we used `https://fc.yahoo.com` to set a `B` cookie, but on some edges this returns
    `404 Not Found on Accelerator` and it adds extra requests that can trigger 429 throttling.
    If you see 401/403 again, enable priming via `YF_COOKIE_PRIME=1` and we will hit the main site.
    """
    if os.getenv("YF_COOKIE_PRIME", "0") not in ("1", "true", "True", "yes"):
        return
    try:
        # A lightweight HTML request to establish cookies. Avoid calling quote endpoints here.
        client.get("https://finance.yahoo.com/quote/AAPL")
    except Exception:
        pass



def fetch_quotes(tickers: List[str]) -> Dict[str, Dict]:
    """Fetch quote snapshots for the provided tickers.

    Returns a mapping: {SYMBOL: quote_dict_from_yahoo}.

    Reliability notes:
    - Yahoo often rate-limits server IPs (429). We therefore:
      1) Use an in-process TTL cache (default 60s).
      2) Implement 429-aware backoff with jitter and *avoid hammering* query1/query2.
      3) Prefer `query2` over `query1`.
    - Cookie priming is OFF by default to reduce request volume; enable with `YF_COOKIE_PRIME=1` if 401/403 returns.
    """
    # Normalize symbols
    syms = [t.strip().upper() for t in tickers if t and t.strip()]
    if not syms:
        return {}

    # Cache key is sorted symbols so AAPL,MSFT equals MSFT,AAPL
    cache_key = ",".join(sorted(syms))
    now = time.time()
    ttl_s = float(os.getenv("YF_QUOTE_CACHE_TTL_S", "60"))
    stale_ttl_s = float(os.getenv("YF_QUOTE_CACHE_STALE_TTL_S", "600"))  # allow stale on 429

    with _INVREP_QUOTES_CACHE_LOCK:
        hit = _INVREP_QUOTES_CACHE.get(cache_key)
        if hit and (now - hit[0]) < ttl_s:
            return hit[1]

    params = {
        "symbols": ",".join(syms),
        "formatted": "false",
        "region": "US",
        "lang": "en-US",
    }

    bases = [
        "https://query2.finance.yahoo.com",
        "https://query1.finance.yahoo.com",
    ]

    def _parse_quotes(payload: dict) -> Dict[str, Dict]:
        results = (((payload.get("quoteResponse") or {}).get("result")) or [])
        out: Dict[str, Dict] = {}
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol") or "").strip().upper()
                if sym:
                    out[sym] = item
        return out

    max_attempts = int(os.getenv("YF_QUOTE_MAX_ATTEMPTS", "3"))
    last_err: Exception | None = None

    with _yahoo_client() as client:
        _prime_yahoo_cookies(client)

        for attempt in range(max_attempts):
            url = f"{bases[0]}/v7/finance/quote"  # query2 preferred
            try:
                r = client.get(url, params=params)

                if r.status_code == 429:
                    # Throttled: backoff + retry query2 only (query1 is usually also throttled).
                    ra = r.headers.get("Retry-After")
                    try:
                        base_sleep = float(ra) if ra else (1.0 * (2 ** attempt))
                    except Exception:
                        base_sleep = 1.0 * (2 ** attempt)
                    sleep_s = min(8.0, max(0.5, base_sleep)) + random.random() * 0.35
                    logger.info("INVREP_YF_THROTTLED status=429 sleep_s=%.2f attempt=%d", sleep_s, attempt + 1)

                    # If we have any cached value (even stale), return it rather than failing the call.
                    with _INVREP_QUOTES_CACHE_LOCK:
                        hit2 = _INVREP_QUOTES_CACHE.get(cache_key)
                        if hit2 and (now - hit2[0]) < stale_ttl_s:
                            logger.info("INVREP_QUOTES_STALE_CACHE_HIT key=%s age_s=%.1f", cache_key, now - hit2[0])
                            return hit2[1]

                    time.sleep(sleep_s)
                    continue

                if r.status_code in (401, 403):
                    # Try query1 as a fallback *once per attempt*.
                    last_err = httpx.HTTPStatusError(f"{r.status_code} from {url}", request=r.request, response=r)
                    url2 = f"{bases[1]}/v7/finance/quote"
                    r2 = client.get(url2, params=params)
                    if r2.status_code == 429:
                        ra = r2.headers.get("Retry-After")
                        base_sleep = float(ra) if ra and ra.replace('.', '', 1).isdigit() else (1.0 * (2 ** attempt))
                        sleep_s = min(8.0, max(0.5, base_sleep)) + random.random() * 0.35
                        logger.info("INVREP_YF_THROTTLED status=429 base=query1 sleep_s=%.2f attempt=%d", sleep_s, attempt + 1)
                        with _INVREP_QUOTES_CACHE_LOCK:
                            hit3 = _INVREP_QUOTES_CACHE.get(cache_key)
                            if hit3 and (now - hit3[0]) < stale_ttl_s:
                                logger.info("INVREP_QUOTES_STALE_CACHE_HIT key=%s age_s=%.1f", cache_key, now - hit3[0])
                                return hit3[1]
                        time.sleep(sleep_s)
                        continue
                    r2.raise_for_status()
                    payload = r2.json() or {}
                    out = _parse_quotes(payload)
                    with _INVREP_QUOTES_CACHE_LOCK:
                        _INVREP_QUOTES_CACHE[cache_key] = (time.time(), out)
                    return out

                r.raise_for_status()
                payload = r.json() or {}
                out = _parse_quotes(payload)
                with _INVREP_QUOTES_CACHE_LOCK:
                    _INVREP_QUOTES_CACHE[cache_key] = (time.time(), out)
                return out

            except Exception as e:
                last_err = e
                # backoff on transient errors too
                sleep_s = min(4.0, 0.5 * (attempt + 1)) + random.random() * 0.2
                time.sleep(sleep_s)
                continue

    # Final: return stale cache if we have it, otherwise raise
    with _INVREP_QUOTES_CACHE_LOCK:
        hit4 = _INVREP_QUOTES_CACHE.get(cache_key)
        if hit4 and (now - hit4[0]) < stale_ttl_s:
            logger.info("INVREP_QUOTES_STALE_CACHE_HIT key=%s age_s=%.1f", cache_key, now - hit4[0])
            return hit4[1]

    if last_err:
        raise last_err
    raise RuntimeError("Yahoo Finance quote fetch failed unexpectedly")


def fetch_news(symbol: str, news_count: int = DEFAULT_NEWS_COUNT, timeout_s: float = 8.0) -> list[dict]:
    """Fetch recent news items for a ticker using Yahoo Finance search endpoint."""
    headers = {"User-Agent": "Mozilla/5.0 (Vozlia)"}
    params = {"q": symbol, "newsCount": str(int(news_count)), "quotesCount": "0"}
    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        r = client.get(YF_SEARCH_URL, params=params)
        r.raise_for_status()
        data = r.json() or {}
    news = data.get("news") or []
    out: list[dict] = []
    if isinstance(news, list):
        for n in news:
            if isinstance(n, dict):
                out.append(n)
    return out


def fetch_analyst_guidance(symbol: str, timeout_s: float = 8.0) -> dict:
    """Best-effort analyst signals from quoteSummary. If Yahoo blocks, return {}."""
    headers = {"User-Agent": "Mozilla/5.0 (Vozlia)"}
    url = YF_QUOTESUMMARY_URL.format(symbol=symbol)
    params = {"modules": "upgradeDowngradeHistory,recommendationTrend"}
    try:
        with httpx.Client(timeout=timeout_s, headers=headers) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json() or {}
        result = (((data.get("quoteSummary") or {}).get("result")) or [])
        if isinstance(result, list) and result:
            return result[0] if isinstance(result[0], dict) else {}
    except Exception as e:
        logger.info("INVREP analyst fetch skipped symbol=%s err=%s", symbol, type(e).__name__)
    return {}


def _epoch_to_dt(ts: Any) -> Optional[datetime]:
    try:
        if ts is None:
            return None
        v = int(ts)
        return datetime.fromtimestamp(v, tz=timezone.utc)
    except Exception:
        return None


def _filter_news_last_24h(items: list[dict]) -> list[dict]:
    now = datetime.now(tz=timezone.utc)
    out = []
    for n in items:
        ts = _epoch_to_dt(n.get("providerPublishTime"))
        if not ts:
            continue
        if (now - ts).total_seconds() <= 24 * 3600:
            out.append(n)
    # newest first
    out.sort(key=lambda x: x.get("providerPublishTime") or 0, reverse=True)
    return out


def _extract_latest_upgrade_downgrade(analyst: dict) -> Optional[dict]:
    hist = (((analyst.get("upgradeDowngradeHistory") or {}).get("history")) or [])
    if not isinstance(hist, list) or not hist:
        return None
    # already reverse-chronological usually, but ensure
    hist = [h for h in hist if isinstance(h, dict)]
    hist.sort(key=lambda h: h.get("epochGradeDate") or 0, reverse=True)
    return hist[0] if hist else None


def build_stock_report_item(symbol: str, quote: dict, news: list[dict], analyst: dict) -> dict:
    price = quote.get("regularMarketPrice")
    prev = quote.get("regularMarketPreviousClose")
    currency = quote.get("currency") or ""
    long_name = quote.get("longName") or quote.get("shortName") or ""
    change = None
    try:
        if price is not None and prev is not None and float(prev) != 0:
            change = (float(price) - float(prev)) / float(prev) * 100.0
    except Exception:
        change = None

    news_24h = _filter_news_last_24h(news)
    titles = [n.get("title") for n in news_24h if isinstance(n.get("title"), str)]
    titles = [t.strip() for t in titles if t and t.strip()]

    latest_ud = _extract_latest_upgrade_downgrade(analyst)
    analyst_line = None
    if latest_ud:
        firm = latest_ud.get("firm")
        to_grade = latest_ud.get("toGrade")
        from_grade = latest_ud.get("fromGrade")
        if firm and to_grade:
            if from_grade:
                analyst_line = f"{firm} changed rating from {from_grade} to {to_grade}."
            else:
                analyst_line = f"{firm} rated it {to_grade}."

    rec_trend = (analyst.get("recommendationTrend") or {}).get("trend")
    consensus = None
    if isinstance(rec_trend, list) and rec_trend:
        # pick most recent period
        rec_trend = [t for t in rec_trend if isinstance(t, dict)]
        rec_trend.sort(key=lambda t: t.get("period") or "", reverse=True)
        if rec_trend:
            consensus = rec_trend[0].get("strongBuy")  # not great; keep for future

    return {
        "symbol": symbol,
        "name": long_name,
        "currency": currency,
        "price": price,
        "previous_close": prev,
        "pct_change": change,
        "news_titles": titles[:5],
        "analyst_update": analyst_line,
    }


def llm_summarize_reports(items: list[dict], llm_prompt: str) -> list[str]:
    """Return one short spoken paragraph per ticker."""
    client = _get_openai_client()
    model = os.getenv("INVREP_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    out: list[str] = []
    for it in items:
        sym = it.get("symbol")
        # Build compact context
        lines = []
        lines.append(f"Ticker: {sym}")
        if it.get("name"):
            lines.append(f"Name: {it['name']}")
        lines.append(f"Current price: {it.get('price')} {it.get('currency')}")
        lines.append(f"Previous close: {it.get('previous_close')} {it.get('currency')}")
        if it.get("pct_change") is not None:
            lines.append(f"Percent change: {it['pct_change']:.2f}%")
        news_titles = it.get("news_titles") or []
        if news_titles:
            lines.append("Recent news headlines (last 24h):")
            for t in news_titles:
                lines.append(f"- {t}")
        else:
            lines.append("Recent news headlines (last 24h): none")
        if it.get("analyst_update"):
            lines.append(f"Analyst update: {it['analyst_update']}")
        else:
            lines.append("Analyst update: none")
        context = "\n".join(lines)

        system = (
            "You are Vozlia. Create a short spoken stock update for ONE ticker. "
            "Include current price and previous close, then briefly summarize today's news, "
            "and mention any analyst update if provided. Keep it under 2-3 sentences."
        )
        if llm_prompt and llm_prompt.strip():
            system = system + "\n\n" + llm_prompt.strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            text = f"{sym}: I couldn't generate a report right now."
        out.append(text)
    return out


def get_investment_reports(
    tickers: list[str],
    llm_prompt: str,
    news_count: int = DEFAULT_NEWS_COUNT,
) -> dict:
    """Returns {'items': [...], 'spoken_reports': [...]}"""
    tickers = _split_tickers(tickers)
    if not tickers:
        return {"items": [], "spoken_reports": []}

    quotes = fetch_quotes(tickers)
    items: list[dict] = []
    for sym in tickers:
        q = quotes.get(sym) or {}
        try:
            news = fetch_news(sym, news_count=news_count)
        except Exception as e:
            logger.info("INVREP news fetch failed symbol=%s err=%s", sym, type(e).__name__)
            news = []
        analyst = fetch_analyst_guidance(sym)
        items.append(build_stock_report_item(sym, q, news, analyst))

    spoken = llm_summarize_reports(items, llm_prompt=llm_prompt)
    return {"items": items, "spoken_reports": spoken}
