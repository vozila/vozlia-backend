# core/logging.py
"""
Centralized logging for Vozlia.

Goal: always emit Vozlia application logs to stdout in Render/Uvicorn.
This is intentionally defensive:
- sets root level from LOG_LEVEL
- ensures a StreamHandler to stdout exists
- ensures 'vozlia' logger is INFO by default and propagates to root
"""

from __future__ import annotations

import logging
import os
import sys


def _level() -> int:
    name = (os.getenv("LOG_LEVEL") or "INFO").upper().strip()
    return getattr(logging, name, logging.INFO)


LEVEL = _level()

# Root logger: make sure something prints to stdout
root = logging.getLogger()
root.setLevel(LEVEL)

# Ensure at least one stdout handler exists (Render captures stdout)
if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(LEVEL)
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(h)

# Vozlia app logger
logger = logging.getLogger("vozlia")
logger.setLevel(LEVEL)
logger.propagate = True
