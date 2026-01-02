# vozlia_twilio/audio_gate.py
from __future__ import annotations

import math
from dataclasses import dataclass

# --- μ-law decode -----------------------------------------------------------
# We precompute a 256-entry lookup table so per-frame work is tiny.
# Twilio media payload is 8kHz 8-bit μ-law.

def _mulaw_decode_byte(uval: int) -> int:
    """
    Decode 8-bit μ-law to signed 16-bit PCM sample (int).
    Standard ITU-T G.711 μ-law decode.
    """
    uval = (~uval) & 0xFF
    sign = uval & 0x80
    exponent = (uval >> 4) & 0x07
    mantissa = uval & 0x0F

    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    if sign:
        sample = -sample
    # Clamp to int16 range (defensive)
    if sample > 32767:
        sample = 32767
    elif sample < -32768:
        sample = -32768
    return sample

_MULAW_DECODE_TABLE = [_mulaw_decode_byte(i) for i in range(256)]


def ulaw_dbfs(ulaw_bytes: bytes) -> float:
    """
    Estimate loudness in dBFS from Twilio μ-law bytes (8kHz).
    This is digital dBFS (relative to full scale), not SPL.

    Implementation avoids `audioop` (not present in Python 3.13 on Render).
    """
    if not ulaw_bytes:
        return -120.0

    # 20ms @ 8kHz = 160 samples, so this loop is small.
    # Sum-of-squares RMS in int domain
    n = len(ulaw_bytes)
    if n == 0:
        return -120.0

    sse = 0
    table = _MULAW_DECODE_TABLE
    for b in ulaw_bytes:
        s = table[b]
        sse += s * s

    mean_sq = sse / n
    if mean_sq <= 0:
        return -120.0

    rms = math.sqrt(mean_sq)
    # Convert to dBFS relative to int16 full scale
    dbfs = 20.0 * math.log10(rms / 32768.0)
    # Guard extremely small values
    if dbfs < -120.0:
        return -120.0
    return dbfs


@dataclass
class GateConfig:
    threshold_dbfs: float = -32.0
    min_open_ms: int = 160
    hangover_ms: int = 400
    frame_ms: int = 20


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
