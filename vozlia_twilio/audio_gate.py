# vozlia_twilio/audio_gate.py
from __future__ import annotations

import audioop
import math
from dataclasses import dataclass


def ulaw_dbfs(ulaw_bytes: bytes) -> float:
    """
    Estimate loudness in dBFS from Twilio μ-law bytes (8kHz).
    This is digital dBFS (relative to full scale), not SPL.
    """
    if not ulaw_bytes:
        return -120.0
    pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)  # 16-bit PCM
    rms = audioop.rms(pcm16, 2)
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)


@dataclass
class GateConfig:
    # Typical speech often lands around -35..-18 dBFS depending on gain.
    threshold_dbfs: float = -32.0
    min_open_ms: int = 160      # consecutive above-threshold time to open
    hangover_ms: int = 400      # time to stay open after dropping below threshold
    frame_ms: int = 20          # Twilio frame duration


class SpeechGate:
    """
    Opens only after sustained energy above threshold; closes after hangover.
    Designed for real-time: cheap per-frame update.
    """
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.is_open = False
        self.last_dbfs = -120.0
        self._above_ms = 0
        self._below_ms = 0

    def update_ulaw(self, ulaw_bytes: bytes) -> bool:
        dbfs = ulaw_dbfs(ulaw_bytes)
        self.last_dbfs = dbfs

        if dbfs >= self.cfg.threshold_dbfs:
            self._above_ms += self.cfg.frame_ms
            self._below_ms = 0
        else:
            self._below_ms += self.cfg.frame_ms
            if not self.is_open:
                self._above_ms = 0

        if not self.is_open and self._above_ms >= self.cfg.min_open_ms:
            self.is_open = True
            self._below_ms = 0

        if self.is_open and self._below_ms >= self.cfg.hangover_ms:
            self.is_open = False
            self._above_ms = 0
            self._below_ms = 0

        return self.is_open
