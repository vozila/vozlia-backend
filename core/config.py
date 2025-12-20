"""Environment-backed configuration for Vozlia (Flow B MVP).

Behavior-neutral extraction from main_12-19-2025.py:
- Reads env vars at import time (same as before)
- Keeps the same defaults
"""

from __future__ import annotations

import os

from core.logging import logger

# ---------- Google / Gmail OAuth config ----------
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# For now: read-only Gmail scope (you can expand later)
GOOGLE_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"

# ---------- FSM router helper base URL ----------
VOZLIA_BACKEND_BASE_URL = os.getenv(
    "VOZLIA_BACKEND_BASE_URL",
    "https://vozlia-backend.onrender.com",
)

# ---------- OpenAI config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Keep same warning behavior as prior inline config block.
    logger.warning("OPENAI_API_KEY is not set. GPT / Realtime calls will fail.")

OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL",
    "gpt-4o-mini-realtime-preview-2024-12-17",
)

# Keep same URL composition
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

# Voice selection is validated in main (SUPPORTED_VOICES lives there)
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "coral")
