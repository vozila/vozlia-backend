# services/memory_enricher.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# A small, conservative stopword list: removes filler but keeps meaning.
_STOPWORDS = {
    "a","an","the","and","or","but","if","then","so","to","of","in","on","for","with","at","by","from",
    "is","am","are","was","were","be","been","being",
    "i","me","my","mine","we","our","ours","you","your","yours","he","him","his","she","her","hers","they","them","their","theirs",
    "it","its","this","that","these","those",
    "what","which","who","whom","whose","where","when","why","how",
    "do","does","did","doing","done",
    "can","could","would","should","may","might","must","will","shall",
    "please","thanks","thank","hi","hello","hey",
    # filler / discourse
    "um","uh","like","actually","basically","literally","just","maybe","kinda","sorta","youknow","okay","ok","right",
}

_COLOR_WORDS = {
    "red","orange","yellow","green","blue","purple","violet","pink","black","white","gray","grey","brown","tan","gold","silver",
    "maroon","teal","cyan","magenta","navy","lime","beige",
}

def _tokenize(text: str) -> List[str]:
    low = (text or "").lower()
    # Keep simple word/number tokens
    return re.findall(r"[a-z0-9']+", low)

def strip_fillers(text: str) -> Tuple[str, List[str], List[str]]:
    """Return (clean_text, tokens, keywords)."""
    raw = (text or "").strip()
    toks = _tokenize(raw)

    # Normalize "you know" -> youknow
    toks2: List[str] = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == "you" and toks[i + 1] == "know":
            toks2.append("youknow")
            i += 2
        else:
            toks2.append(toks[i])
            i += 1

    # Remove stopwords
    kept = [t for t in toks2 if t and t not in _STOPWORDS]

    # Keywords: prefer 3+ chars, cap for storage
    kws = [t for t in kept if len(t) >= 3][:24]

    clean = " ".join(kept).strip()
    return clean, toks2, kws

def extract_facts(text: str) -> Dict[str, str]:
    """Very cheap fact extraction (expand later)."""
    raw = (text or "").strip()
    low = raw.lower()

    facts: Dict[str, str] = {}

    # Favorite color patterns
    m = re.search(r"\bfavorite\s+color\s+(?:is|was)\s+([a-zA-Z]+)\b", low)
    if m:
        val = m.group(1).strip().lower()
        if val:
            facts["favorite_color"] = val

    # Another common phrasing: "my favorite color today: green"
    m2 = re.search(r"\bfavorite\s+color\b[^a-zA-Z0-9]{0,10}([a-zA-Z]+)\b", low)
    if "favorite_color" not in facts and m2:
        cand = m2.group(1).strip().lower()
        if cand in _COLOR_WORDS:
            facts["favorite_color"] = cand

    return facts

def build_tags(*, role: str, keywords: List[str], facts: Dict[str, str]) -> List[str]:
    tags: List[str] = []
    tags.append(f"role:{role}")
    for kw in keywords[:20]:
        tags.append(f"kw:{kw}")
    for k, v in facts.items():
        tags.append(f"fact_key:{k}")
        tags.append(f"fact:{k}={v}")
        tags.append(k)
        tags.append(v)
    # de-dupe while preserving order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def is_memory_question(text: str) -> bool:
    low = (text or "").lower()
    # Memory question cues
    cues = [
        "what did i say",
        "what did i tell you",
        "remind me",
        "last time we talked",
        "previous call",
        "earlier you said",
        "did i say",
        "do you remember",
    ]
    if any(c in low for c in cues):
        return True
    # Specific fact questions
    if "favorite color" in low:
        return True
    return False

def parse_fact_query(text: str) -> str | None:
    low = (text or "").lower()
    if "favorite color" in low:
        return "favorite_color"
    return None
