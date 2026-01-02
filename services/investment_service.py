# services/investment_service.py
from __future__ import annotations

import os
import time
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.logging import logger
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

# Investment Reporting market-data provider:
# - yfinance: use yfinance library (cookie/crumb/session handling). Requires dependency install.
# - httpx: direct Yahoo endpoints (more likely to be rate-limited / blocked).
# Default is yfinance.
INVREP_DATA_PROVIDER = os.getenv("INVREP_DATA_PROVIDER", "yfinance").lower().strip()

# In-process caches to reduce throttling (Yahoo often returns 429 from server IP ranges).
_QUOTE_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
_NEWS_CACHE: dict[str, tuple[float, list[dict]]] = {}



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

def _cache_key(symbols: list[str]) -> str:
    return ",".join(sorted([s.strip().upper() for s in symbols if s and s.strip()]))

def _cache_ttl_seconds(env_name: str, default: int) -> int:
    try:
        v = int(os.getenv(env_name, str(default)))
        return max(1, v)
    except Exception:
        return default

def _cache_get_quotes(key: str) -> dict[str, dict] | None:
    ttl = _cache_ttl_seconds("YF_QUOTE_CACHE_TTL_S", 60)
    hit = _QUOTE_CACHE.get(key)
    if not hit:
        return None
    ts, data = hit
    if (time.time() - ts) <= ttl:
        return data
    return None

def _cache_get_quotes_stale(key: str) -> dict[str, dict] | None:
    ttl = _cache_ttl_seconds("YF_QUOTE_CACHE_STALE_TTL_S", 600)
    hit = _QUOTE_CACHE.get(key)
    if not hit:
        return None
    ts, data = hit
    if (time.time() - ts) <= ttl:
        return data
    return None

def _cache_set_quotes(key: str, data: dict[str, dict]) -> None:
    _QUOTE_CACHE[key] = (time.time(), data)

def _cache_get_news(symbol: str) -> list[dict] | None:
    ttl = _cache_ttl_seconds("YF_NEWS_CACHE_TTL_S", 300)
    hit = _NEWS_CACHE.get(symbol)
    if not hit:
        return None
    ts, data = hit
    if (time.time() - ts) <= ttl:
        return data
    return None

def _cache_set_news(symbol: str, data: list[dict]) -> None:
    _NEWS_CACHE[symbol] = (time.time(), data)


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
    # This typically sets the `B` cookie and helps avoid 401/403.
    try:
        client.get("https://fc.yahoo.com")
    except Exception:
        pass

def fetch_quotes(tickers: List[str]) -> dict[str, dict]:
    """Fetch quote snapshot keyed by symbol.

    Prefer yfinance (handles cookie/crumb/session) to reduce bot-blocking issues.
    Fallback to direct Yahoo endpoints if yfinance is unavailable or fails.

    Returns mapping:
      { 'AAPL': {'regularMarketPrice': ..., 'regularMarketPreviousClose': ..., 'currency': ..., 'longName': ...}, ... }
    """
    tickers = _split_tickers(tickers)
    if not tickers:
        return {}

    key = _cache_key(tickers)
    cached = _cache_get_quotes(key)
    if cached is not None:
        return cached

    provider = os.getenv("INVREP_DATA_PROVIDER", "yfinance").lower().strip()
    last_err: Exception | None = None

    if provider in ("yfinance", "yf"):
        try:
            data = _fetch_quotes_yfinance(tickers)
            if data:
                _cache_set_quotes(key, data)
                return data
        except Exception as e:
            last_err = e
            logger.warning("INVREP_YFINANCE_QUOTES_FAIL tickers=%s err=%s", tickers, type(e).__name__)

    try:
        data = _fetch_quotes_httpx_with_backoff(tickers)
        if data:
            _cache_set_quotes(key, data)
            return data
    except Exception as e:
        last_err = e

    stale = _cache_get_quotes_stale(key)
    if stale is not None:
        logger.warning("INVREP_QUOTES_STALE_CACHE_HIT tickers=%s", tickers)
        return stale

    if last_err:
        raise last_err
    raise RuntimeError("Quote fetch failed")


def _fetch_quotes_yfinance(tickers: list[str]) -> dict[str, dict]:
    """Fetch quotes via yfinance.

    Requires: yfinance + pandas + numpy installed in the runtime.
    """
    import yfinance as yf

    syms = [t.strip().upper() for t in tickers if t and t.strip()]
    if not syms:
        return {}

    tks = yf.Tickers(" ".join(syms))

    out: dict[str, dict] = {}
    for sym in syms:
        t = tks.tickers.get(sym) or yf.Ticker(sym)

        info: dict = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        price = info.get("regularMarketPrice")
        prev = info.get("regularMarketPreviousClose")
        currency = info.get("currency")
        long_name = info.get("longName") or info.get("shortName")

        if price is None or prev is None:
            fi = getattr(t, "fast_info", None) or {}
            price = price if price is not None else fi.get("last_price") or fi.get("lastPrice")
            prev = prev if prev is not None else fi.get("previous_close") or fi.get("previousClose")

        if price is None or prev is None:
            hist = t.history(period="2d", interval="1d")
            if len(hist.index) >= 2:
                prev = float(hist["Close"].iloc[-2])
                price = float(hist["Close"].iloc[-1])
            elif len(hist.index) == 1:
                price = float(hist["Close"].iloc[-1])

        out[sym] = {
            "symbol": sym,
            "regularMarketPrice": price,
            "regularMarketPreviousClose": prev,
            "currency": currency,
            "longName": long_name,
        }

        time.sleep(0.05)

    logger.info("INVREP_YFINANCE_QUOTES_OK tickers=%s", syms)
    return out


def _fetch_quotes_httpx_with_backoff(tickers: list[str]) -> dict[str, dict]:
    """Fallback quote fetch using direct Yahoo endpoints with 429-aware backoff."""
    symbols = ",".join([t.strip().upper() for t in tickers if t.strip()])
    if not symbols:
        return {}

    params = {"symbols": symbols, "formatted": "false", "region": "US", "lang": "en-US"}
    bases = ["https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"]

    max_attempts = _cache_ttl_seconds("YF_QUOTE_MAX_ATTEMPTS", 3)

    with _yahoo_client() as client:
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            for base in bases:
                url = f"{base}/v7/finance/quote"
                try:
                    r = client.get(url, params=params)
                    if r.status_code == 429:
                        ra = r.headers.get("Retry-After")
                        sleep_s = float(ra) if ra and ra.isdigit() else min(8.0, 0.75 * (2 ** (attempt - 1)))
                        sleep_s = sleep_s + random.random() * 0.35
                        logger.info("INVREP_YF_THROTTLED status=429 sleep_s=%.2f attempt=%d", sleep_s, attempt)
                        time.sleep(sleep_s)
                        last_err = httpx.HTTPStatusError("429 Too Many Requests", request=r.request, response=r)
                        continue
                    if r.status_code in (401, 403):
                        last_err = httpx.HTTPStatusError(f"{r.status_code} Unauthorized/Forbidden", request=r.request, response=r)
                        continue
                    r.raise_for_status()
                    data = r.json() or {}
                    results = (((data.get("quoteResponse") or {}).get("result")) or [])
                    out: dict[str, dict] = {}
                    if isinstance(results, list):
                        for item in results:
                            if not isinstance(item, dict):
                                continue
                            sym = (item.get("symbol") or "").strip().upper()
                            if sym:
                                out[sym] = item
                    if out:
                        logger.info("INVREP_HTTPX_QUOTES_OK tickers=%s", tickers)
                        return out
                except Exception as e:
                    last_err = e
                    continue

        if last_err:
            raise last_err
        raise RuntimeError("Yahoo Finance quote fetch failed unexpectedly")



def fetch_news(symbol: str, news_count: int = DEFAULT_NEWS_COUNT, timeout_s: float = 8.0) -> list[dict]:
    """Fetch recent news for a ticker.

    Prefer yfinance (Ticker.news) to reduce bot-blocking and rate-limit issues.
    Falls back to Yahoo search endpoint if yfinance fails.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return []

    cached = _cache_get_news(sym)
    if cached is not None:
        return cached

    provider = os.getenv("INVREP_DATA_PROVIDER", "yfinance").lower().strip()

    if provider in ("yfinance", "yf"):
        try:
            import yfinance as yf
            tkr = yf.Ticker(sym)
            news = tkr.news or []
            out: list[dict] = []
            for n in news[: int(news_count)]:
                if not isinstance(n, dict):
                    continue
                out.append({
                    "title": n.get("title"),
                    "publisher": n.get("publisher"),
                    "link": n.get("link"),
                    "providerPublishTime": n.get("providerPublishTime"),
                })
            _cache_set_news(sym, out)
            logger.info("INVREP_YFINANCE_NEWS_OK symbol=%s n=%d", sym, len(out))
            return out
        except Exception as e:
            logger.info("INVREP_YFINANCE_NEWS_FAIL symbol=%s err=%s", sym, type(e).__name__)

    headers = {"User-Agent": "Mozilla/5.0 (Vozlia)"}
    params = {"q": sym, "newsCount": str(int(news_count)), "quotesCount": "0"}
    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        r = client.get(YF_SEARCH_URL, params=params)
        r.raise_for_status()
        data = r.json() or {}
    news = data.get("news", [])
    out = _normalize_news_items(news)
    _cache_set_news(sym, out)
    return out


def fetch_analyst_guidance(symbol: str, timeout_s: float = 8.0) -> dict:
    """Best-effort analyst signals.

    Prefer yfinance upgrades/downgrades if available; fallback to Yahoo quoteSummary.
    Return {} if unavailable.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {}

    provider = os.getenv("INVREP_DATA_PROVIDER", "yfinance").lower().strip()

    if provider in ("yfinance", "yf"):
        try:
            import yfinance as yf
            tkr = yf.Ticker(sym)
            ud = getattr(tkr, "upgrades_downgrades", None)
            if ud is not None and hasattr(ud, "empty") and not ud.empty:
                last = ud.tail(1)
                row = last.iloc[0].to_dict()
                ts = None
                try:
                    idx = last.index[-1]
                    ts = int(idx.timestamp())
                except Exception:
                    ts = None
                return {
                    "upgradeDowngradeHistory": {
                        "history": [
                            {
                                "epochGradeDate": ts,
                                "firm": row.get("Firm") or row.get("firm"),
                                "fromGrade": row.get("From Grade") or row.get("fromGrade"),
                                "toGrade": row.get("To Grade") or row.get("toGrade"),
                                "action": row.get("Action") or row.get("action"),
                            }
                        ]
                    }
                }
        except Exception as e:
            logger.info("INVREP_YFINANCE_ANALYST_FAIL symbol=%s err=%s", sym, type(e).__name__)

    headers = {"User-Agent": "Mozilla/5.0 (Vozlia)"}
    url = YF_QUOTESUMMARY_URL.format(symbol=sym)
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
        logger.info("INVREP analyst fetch skipped symbol=%s err=%s", sym, type(e).__name__)
    return {}


def _epoch_to_dt(ts: Any) -> Optional[datetime]:
    try:
        if ts is None:
            return None
        v = int(ts)
        return datetime.fromtimestamp(v, tz=timezone.utc)
    except Exception:
        return None


def _normalize_news_items(items: Any) -> list[dict]:
    """Normalize Yahoo-style news items into the stable shape used by _filter_news_last_24h."""
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for n in items:
        if not isinstance(n, dict):
            continue
        out.append(
            {
                "title": n.get("title"),
                "publisher": n.get("publisher") or (n.get("provider") or ""),
                "link": n.get("link") or n.get("url"),
                "providerPublishTime": n.get("providerPublishTime") or n.get("published_at"),
            }
        )
    return out

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
