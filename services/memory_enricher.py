# services/memory_enricher.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Small, safe stopword list (expand later). Keep domain words.
_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","can","could","did","do","does","for","from",
    "had","has","have","he","her","hers","him","his","how","i","if","in","into","is","it","its",
    "just","like","me","my","of","on","or","our","ours","please","said","say","she","so","some",
    "that","the","their","them","then","there","these","they","this","to","today","tomorrow",
    "was","we","were","what","when","where","which","who","why","will","with","would","you","your","yours",
    "vozlia",
}

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

@dataclass
class EnrichedTurn:
    cleaned_text: str
    keywords: List[str]
    tags: List[str]
    facts: Dict[str, str]  # e.g. {"favorite_color":"green"}

def _normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    return t

def strip_filler(text: str) -> Tuple[str, List[str]]:
    """Return (cleaned_text, keywords). Keeps meaningful tokens; removes common filler/stopwords."""
    raw = (text or "").strip()
    toks = [_normalize_token(m.group(0)) for m in _WORD_RE.finditer(raw)]
    kws = [t for t in toks if t and t not in _STOPWORDS and not t.isdigit()]
    # de-dupe while preserving order
    seen=set()
    out=[]
    for t in kws:
        if t not in seen:
            seen.add(t)
            out.append(t)
    cleaned = " ".join(out)
    return cleaned, out

_FAV_COLOR_RE = re.compile(r"\b(?:my\s+)?favorite\s+color\b(?:\s+\w+){0,4}\s+(?:is|was)\s+([A-Za-z]+)\b", re.I)

def enrich_turn(text: str) -> EnrichedTurn:
    cleaned, keywords = strip_filler(text)
    tags: List[str] = []
    facts: Dict[str, str] = {}

    # keyword tags
    for kw in keywords[:25]:
        tags.append(f"kw:{kw}")

    # simple fact: favorite_color
    m = _FAV_COLOR_RE.search(text or "")
    if m:
        val = _normalize_token(m.group(1))
        if val:
            facts["favorite_color"] = val
            tags.append("fact_key:favorite_color")
            tags.append(f"fact:favorite_color={val}")

    return EnrichedTurn(cleaned_text=cleaned, keywords=keywords, tags=tags, facts=facts)
