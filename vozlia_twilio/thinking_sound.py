# vozlia_twilio/thinking_sound.py
"""Vozlia "thinking sound" earcon for PSTN calls.

We intentionally do NOT replicate any proprietary UI sounds.
This is a simple, subtle double-beep chime generated as 8kHz G.711 μ-law bytes.

Why:
- Callers can experience 2–4s latency while we do KB retrieval + LLM + TTS.
- A short earcon helps signal "I'm working on it" without adding hot-path compute.

Safety:
- No external dependencies.
- Clip is generated once at import time.
- Designed to be mixed into the existing Twilio outbound μ-law audio buffer.
"""

from __future__ import annotations

import math

THINKING_SAMPLE_RATE_HZ = 8000
# Twilio Media Streams uses 20ms frames at 8kHz => 160 samples/bytes per frame (μ-law is 1 byte per sample)
THINKING_FRAME_SAMPLES = 160

_BIAS = 0x84
_CLIP = 32635


def _linear16_to_mulaw(sample: int) -> int:
    """Encode a signed 16-bit PCM sample to 8-bit G.711 μ-law."""
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample

    if sample > _CLIP:
        sample = _CLIP

    sample = sample + _BIAS

    exponent = 7
    exp_mask = 0x4000
    while exponent > 0 and (sample & exp_mask) == 0:
        exp_mask >>= 1
        exponent -= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw


def _tone_pcm(freq_hz: float, dur_s: float, *, amp: float = 0.18) -> list[int]:
    n = int(THINKING_SAMPLE_RATE_HZ * dur_s)
    A = int(max(0.0, min(1.0, amp)) * 32767)
    out: list[int] = []
    for i in range(n):
        t = i / THINKING_SAMPLE_RATE_HZ
        out.append(int(A * math.sin(2.0 * math.pi * freq_hz * t)))
    return out


def _silence_pcm(dur_s: float) -> list[int]:
    n = int(THINKING_SAMPLE_RATE_HZ * dur_s)
    return [0] * n


def _pcm_to_ulaw(pcm: list[int]) -> bytes:
    return bytes(_linear16_to_mulaw(s) for s in pcm)


def build_thinking_clip_ulaw() -> bytes:
    """Return a short earcon clip as μ-law bytes.

    Pattern (0.80s total):
    - 120ms @ 520Hz
    -  60ms silence
    - 120ms @ 740Hz
    - 500ms silence
    """
    pcm = (
        _tone_pcm(520.0, 0.12)
        + _silence_pcm(0.06)
        + _tone_pcm(740.0, 0.12)
        + _silence_pcm(0.50)
    )
    ulaw = _pcm_to_ulaw(pcm)

    # Pad to whole frames (160 bytes) for clean streaming.
    pad = (-len(ulaw)) % THINKING_FRAME_SAMPLES
    if pad:
        ulaw += bytes([_linear16_to_mulaw(0)]) * pad  # μ-law silence (0xFF)

    return ulaw


THINKING_CLIP_ULAW: bytes = build_thinking_clip_ulaw()
THINKING_CLIP_DURATION_S: float = len(THINKING_CLIP_ULAW) / THINKING_SAMPLE_RATE_HZ
