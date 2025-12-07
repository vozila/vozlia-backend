import os
import json
import base64
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import websockets

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("vozlia")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_URL = os.getenv(
    "OPENAI_REALTIME_URL",
    "wss://api.openai.com/v1/realtime?model=gpt-4.1-realtime-preview",
)

INITIAL_SYSTEM_MESSAGE = (
    "You are Vozlia, a friendly voice AI receptionist for a small business. "
    "Keep answers concise, speak clearly, and pause often so callers can interrupt. "
    "If the caller interrupts you, immediately stop speaking and listen."
)

# -----------------------------------------------------------------------------
# Call state
# -----------------------------------------------------------------------------
@dataclass
class CallState:
    stream_sid_
