"""Environment-backed configuration for Vozlia (Flow B MVP).

Behavior-neutral extraction from main_12-19-2025.py:
- Reads env vars at import time (same as before)
- Keeps the same defaults
"""

from __future__ import annotations
from typing import Dict

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

# Default style for the whole system (warm|concise)
VOZLIA_DEFAULT_STYLE = os.getenv("VOZLIA_DEFAULT_STYLE", "warm").lower()

# Optional per-skill overrides (warm|concise)
VOZLIA_STYLE_EMAIL = os.getenv("VOZLIA_STYLE_EMAIL", "").lower()  # if empty -> default
VOZLIA_STYLE_CHITCHAT = os.getenv("VOZLIA_STYLE_CHITCHAT", "").lower()
VOZLIA_STYLE_DEFAULT = VOZLIA_DEFAULT_STYLE  # alias for clarity

# When true: even gratitude gets a response in concise mode
VOZLIA_CONCISE_ACKS = os.getenv("VOZLIA_CONCISE_ACKS", "0") == "1"

# --- OpenAI Realtime WS headers (used by vozlia_twilio/stream.py) -----------



def _build_openai_realtime_headers() -> Dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # required for Realtime beta endpoints
        headers["OpenAI-Beta"] = "realtime=v1"
    return headers

# Exported constant expected by vozlia_twilio/stream.py
OPENAI_REALTIME_HEADERS = _build_openai_realtime_headers()


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

# ---------- Debug toggles ----------
REALTIME_LOG_TEXT = os.getenv("REALTIME_LOG_TEXT", "0") == "1"
REALTIME_LOG_ALL_EVENTS = os.getenv("REALTIME_LOG_ALL_EVENTS", "0") == "1"

# ---------- Flow / routing flags ----------
SKILL_GATED_ROUTING = os.getenv("SKILL_GATED_ROUTING", "0") == "1"

# If true, allow OpenAI server to interrupt its own responses on VAD start
OPENAI_INTERRUPT_RESPONSE = os.getenv("OPENAI_INTERRUPT_RESPONSE", "0") == "1"

# ---------- Realtime session defaults ----------
# Audio formats used by Twilio Media Streams in your code (G.711 μ-law)
REALTIME_INPUT_AUDIO_FORMAT = os.getenv("REALTIME_INPUT_AUDIO_FORMAT", "g711_ulaw")
REALTIME_OUTPUT_AUDIO_FORMAT = os.getenv("REALTIME_OUTPUT_AUDIO_FORMAT", "g711_ulaw")

# Voice name for the realtime session (you currently call this m.VOICE_NAME in code)
VOICE_NAME = os.getenv("VOICE_NAME", OPENAI_REALTIME_VOICE or "coral")

# System prompt for the realtime session (expected by vozlia_twilio/stream.py)
REALTIME_SYSTEM_PROMPT = os.getenv(
    "REALTIME_SYSTEM_PROMPT",
    "You are Vozlia, a helpful real-time voice assistant. Be concise and natural.",
)


# VAD tuning (keep your current defaults)
REALTIME_VAD_THRESHOLD = float(os.getenv("REALTIME_VAD_THRESHOLD", "0.5"))
REALTIME_VAD_SILENCE_MS = int(os.getenv("REALTIME_VAD_SILENCE_MS", "600"))
REALTIME_VAD_PREFIX_MS = int(os.getenv("REALTIME_VAD_PREFIX_MS", "200"))

# ---------- Twilio audio pacing ----------
# If you already define these elsewhere, centralize them here and import.
BYTES_PER_FRAME = int(os.getenv("BYTES_PER_FRAME", "160"))          # 20ms @ 8kHz μ-law
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "0.02"))         # seconds per frame
PREBUFFER_BYTES = int(os.getenv("PREBUFFER_BYTES", "8000"))         # ~1s prebuffer default
MAX_TWILIO_BACKLOG_SECONDS = float(os.getenv("MAX_TWILIO_BACKLOG_SECONDS", "1.0"))

