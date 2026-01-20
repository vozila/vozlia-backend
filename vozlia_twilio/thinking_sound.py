# vozlia_twilio/thinking_sound.py
"""Vozlia "thinking sound" earcon for PSTN calls (8kHz G.711 μ-law).

We intentionally do NOT replicate any proprietary UI sounds.

Usage
- Set VOICE_THINKING_SOUND_TONE_ID:
  - 0 = legacy tone (default; preserves current behavior)
  - 1..20 = alternative tones
- Optional: VOICE_THINKING_SOUND_AMP (0.0–1.0, default 1.0)

Notes
- Clip is generated once at import time (no hot-path compute).
- Output is padded to whole 20ms frames (160 bytes) for clean Twilio streaming.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

THINKING_SAMPLE_RATE_HZ = 8000
THINKING_FRAME_SAMPLES = 160  # 20ms @ 8kHz
TOTAL_S = 0.80               # keep constant so stream.py assumptions stay stable

SegType = Literal["tone", "dual", "chirp", "silence"]
Segment = Tuple[SegType, float, float, float, float]  # type, a, b, dur_s, amp (unused for silence)


# ----------------------------
# μ-law conversion (no deps)
# ----------------------------

def _linear16_to_mulaw(sample: int) -> int:
    """Convert signed 16-bit PCM to 8-bit μ-law."""
    MU_LAW_MAX = 0x1FFF
    BIAS = 33
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    if sample > MU_LAW_MAX:
        sample = MU_LAW_MAX
    sample = sample + BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        mask >>= 1
        exponent -= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ----------------------------
# Waveform helpers
# ----------------------------

def _envelope(i: int, n: int, fade_n: int) -> float:
    if fade_n <= 0 or n <= 0:
        return 1.0
    if i < fade_n:
        return i / float(fade_n)
    tail = n - 1 - i
    if tail < fade_n:
        return tail / float(fade_n)
    return 1.0


def _tone(freq_hz: float, dur_s: float, amp: float, fade_ms: int = 8) -> List[int]:
    n = int(THINKING_SAMPLE_RATE_HZ * dur_s)
    fade_n = int(THINKING_SAMPLE_RATE_HZ * (fade_ms / 1000.0))
    A = int(_clamp(amp, 0.0, 1.0) * 32767)
    out: List[int] = []
    for i in range(n):
        t = i / THINKING_SAMPLE_RATE_HZ
        e = _envelope(i, n, fade_n)
        out.append(int(A * e * math.sin(2.0 * math.pi * freq_hz * t)))
    return out


def _dual(f1_hz: float, f2_hz: float, dur_s: float, amp: float, fade_ms: int = 10) -> List[int]:
    n = int(THINKING_SAMPLE_RATE_HZ * dur_s)
    fade_n = int(THINKING_SAMPLE_RATE_HZ * (fade_ms / 1000.0))
    A = int(_clamp(amp, 0.0, 1.0) * 32767)
    out: List[int] = []
    for i in range(n):
        t = i / THINKING_SAMPLE_RATE_HZ
        e = _envelope(i, n, fade_n)
        s = 0.5 * (math.sin(2.0 * math.pi * f1_hz * t) + math.sin(2.0 * math.pi * f2_hz * t))
        out.append(int(A * e * s))
    return out


def _chirp(f0_hz: float, f1_hz: float, dur_s: float, amp: float, fade_ms: int = 10) -> List[int]:
    n = int(THINKING_SAMPLE_RATE_HZ * dur_s)
    fade_n = int(THINKING_SAMPLE_RATE_HZ * (fade_ms / 1000.0))
    A = int(_clamp(amp, 0.0, 1.0) * 32767)
    out: List[int] = []
    phase = 0.0
    for i in range(n):
        e = _envelope(i, n, fade_n)
        frac = i / float(max(1, n - 1))
        f = f0_hz + (f1_hz - f0_hz) * frac
        phase += 2.0 * math.pi * (f / THINKING_SAMPLE_RATE_HZ)
        out.append(int(A * e * math.sin(phase)))
    return out


def _silence(dur_s: float) -> List[int]:
    return [0] * int(THINKING_SAMPLE_RATE_HZ * dur_s)


def _with_total_length(pcm: List[int]) -> List[int]:
    target = int(TOTAL_S * THINKING_SAMPLE_RATE_HZ)
    if len(pcm) < target:
        pcm.extend(_silence((target - len(pcm)) / THINKING_SAMPLE_RATE_HZ))
    elif len(pcm) > target:
        del pcm[target:]
    return pcm


def _pcm_to_ulaw(pcm: List[int]) -> bytes:
    return bytes(_linear16_to_mulaw(s) for s in pcm)


def _pad_to_frames(ulaw: bytes) -> bytes:
    pad = (-len(ulaw)) % THINKING_FRAME_SAMPLES
    if pad:
        ulaw += bytes([_linear16_to_mulaw(0)]) * pad
    return ulaw


# ----------------------------
# Presets
# ----------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    desc: str
    segments: List[Segment]


# Segment tuple is: (type, a, b, dur_s, amp)
# - tone:  a=freq_hz, b=0
# - dual:  a=f1_hz,   b=f2_hz
# - chirp: a=f0_hz,   b=f1_hz
# - silence: a=0, b=0, amp ignored

PRESETS: Dict[int, Preset] = {
    # 0 = legacy (preserves behavior)
    0: Preset(
        name="legacy",
        desc="Legacy double-beep (520Hz then 740Hz)",
        segments=[
            ("tone", 520.0, 0.0, 0.12, 0.18),
            ("silence", 0.0, 0.0, 0.06, 0.0),
            ("tone", 740.0, 0.0, 0.12, 0.18),
            ("silence", 0.0, 0.0, 0.50, 0.0),
        ],
    ),

    1: Preset("soft_chime_C5", "Single soft chime (C5)", [("tone", 523.25, 0.0, 0.22, 0.17), ("silence", 0, 0, 0.58, 0)]),
    2: Preset("soft_chime_A4", "Single soft chime (A4)", [("tone", 440.0, 0.0, 0.22, 0.17), ("silence", 0, 0, 0.58, 0)]),
    3: Preset("double_chime", "Two short beeps (C5 then E5)", [
        ("tone", 523.25, 0.0, 0.16, 0.16), ("silence", 0, 0, 0.08, 0),
        ("tone", 659.25, 0.0, 0.16, 0.16), ("silence", 0, 0, 0.40, 0),
    ]),
    4: Preset("double_low_high", "Two short beeps (G4 then G5)", [
        ("tone", 392.0, 0.0, 0.16, 0.16), ("silence", 0, 0, 0.08, 0),
        ("tone", 784.0, 0.0, 0.16, 0.14), ("silence", 0, 0, 0.40, 0),
    ]),
    5: Preset("two_tone_chime", "Two-tone simultaneous chime (A4+E5)", [("dual", 440.0, 660.0, 0.30, 0.16), ("silence", 0, 0, 0.50, 0)]),
    6: Preset("rising_chirp", "Rising chirp", [("chirp", 350.0, 950.0, 0.30, 0.14), ("silence", 0, 0, 0.50, 0)]),
    7: Preset("falling_chirp", "Falling chirp", [("chirp", 950.0, 350.0, 0.30, 0.14), ("silence", 0, 0, 0.50, 0)]),
    8: Preset("triple_beep_soft", "Three quick beeps", [
        ("tone", 660.0, 0.0, 0.10, 0.14), ("silence", 0, 0, 0.06, 0),
        ("tone", 660.0, 0.0, 0.10, 0.14), ("silence", 0, 0, 0.06, 0),
        ("tone", 660.0, 0.0, 0.10, 0.14), ("silence", 0, 0, 0.38, 0),
    ]),
    9: Preset("soft_tick", "Very short tick (high)", [("tone", 1200.0, 0.0, 0.06, 0.10), ("silence", 0, 0, 0.74, 0)]),
    10: Preset("soft_bloop", "Low bloop (downward)", [("chirp", 320.0, 180.0, 0.22, 0.18), ("silence", 0, 0, 0.58, 0)]),
    11: Preset("soft_pluck", "Plucky two-tone (E4+B4)", [("dual", 330.0, 495.0, 0.22, 0.15), ("silence", 0, 0, 0.58, 0)]),
    12: Preset("warm_chime", "Warm low chime (C4+G4)", [("dual", 262.0, 392.0, 0.28, 0.16), ("silence", 0, 0, 0.52, 0)]),
    13: Preset("glass_ping", "Glassy ping (high harmonic)", [("dual", 1047.0, 1568.0, 0.18, 0.09), ("silence", 0, 0, 0.62, 0)]),
    14: Preset("soft_ding", "Soft ding (F#5-ish)", [("tone", 740.0, 0.0, 0.18, 0.14), ("silence", 0, 0, 0.62, 0)]),
    15: Preset("bubble_up", "Bubble up (short rising)", [("chirp", 500.0, 1200.0, 0.22, 0.12), ("silence", 0, 0, 0.58, 0)]),
    16: Preset("bubble_down", "Bubble down (short falling)", [("chirp", 1200.0, 500.0, 0.22, 0.12), ("silence", 0, 0, 0.58, 0)]),
    17: Preset("pulse_pair", "Two pulses (octave)", [
        ("dual", 440.0, 880.0, 0.12, 0.12), ("silence", 0, 0, 0.10, 0),
        ("dual", 440.0, 880.0, 0.12, 0.12), ("silence", 0, 0, 0.46, 0),
    ]),
    18: Preset("soft_marimba", "Marimba-ish (octave)", [("dual", 294.0, 588.0, 0.18, 0.14), ("silence", 0, 0, 0.62, 0)]),
    19: Preset("subtle_fifth", "Subtle fifth (C5+G5)", [("dual", 523.25, 783.99, 0.20, 0.13), ("silence", 0, 0, 0.60, 0)]),
    20: Preset("chatty_chime", "Down-up pair (E5 then C5)", [
        ("tone", 659.25, 0.0, 0.14, 0.13), ("silence", 0, 0, 0.06, 0),
        ("tone", 523.25, 0.0, 0.14, 0.13), ("silence", 0, 0, 0.46, 0),
    ]),
}


def _build_pcm(tone_id: int, amp_scale: float) -> List[int]:
    preset = PRESETS.get(tone_id) or PRESETS[0]
    pcm: List[int] = []
    for typ, a, b, dur_s, amp in preset.segments:
        if dur_s <= 0:
            continue
        if typ == "silence":
            pcm.extend(_silence(dur_s))
        elif typ == "tone":
            pcm.extend(_tone(a, dur_s, amp * amp_scale))
        elif typ == "dual":
            pcm.extend(_dual(a, b, dur_s, amp * amp_scale))
        elif typ == "chirp":
            pcm.extend(_chirp(a, b, dur_s, amp * amp_scale))
    return _with_total_length(pcm)


def build_thinking_clip_ulaw() -> bytes:
    tone_id = _env_int("VOICE_THINKING_SOUND_TONE_ID", 0)
    amp_scale = _clamp(_env_float("VOICE_THINKING_SOUND_AMP", 1.0), 0.0, 1.0)
    pcm = _build_pcm(tone_id, amp_scale)
    return _pad_to_frames(_pcm_to_ulaw(pcm))


THINKING_TONE_ID: int = _env_int("VOICE_THINKING_SOUND_TONE_ID", 0)
_p = PRESETS.get(THINKING_TONE_ID) or PRESETS[0]
THINKING_TONE_NAME: str = _p.name
THINKING_TONE_DESC: str = _p.desc

THINKING_CLIP_ULAW: bytes = build_thinking_clip_ulaw()
THINKING_CLIP_DURATION_S: float = len(THINKING_CLIP_ULAW) / THINKING_SAMPLE_RATE_HZ
