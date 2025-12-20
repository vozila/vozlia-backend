"""Centralized logging for Vozlia (Flow B MVP).

Behavior-neutral extraction from main_12-19-2025.py:
- logging.basicConfig(level=logging.INFO)
- logger name: 'vozlia'
"""

from __future__ import annotations

import logging

# Keep behavior identical to previous inline setup.
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)
