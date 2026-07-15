"""Tests for audio_quality.assess_quality using synthetic clips built with the
same synth_voice() helper make_synthetic_data.py uses for pipeline smoke tests.
"""
import os
import tempfile

import numpy as np
import soundfile as sf

from audio_quality import assess_quality, quality_penalty, MIN_DURATION_S
from make_synthetic_data import synth_voice, SR


def _write_wav(signal, sr=SR):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, signal, sr)
    return path


def test_clean_clip_has_no_warnings():
    rng = np.random.default_rng(0)
    signal = synth_voice(rng, jitter_amount=0.1, shimmer_amount=0.1, silence_amount=0.0)
    path = _write_wav(signal)
    try:
        result = assess_quality(path)
        assert result["ok"] is True
        assert result["warnings"] == []
        assert result["duration_s"] > MIN_DURATION_S
    finally:
        os.unlink(path)


def test_short_clip_flagged():
    rng = np.random.default_rng(1)
    signal = synth_voice(rng, jitter_amount=0.1, shimmer_amount=0.1, silence_amount=0.0)
    short_signal = signal[: int(0.5 * SR)]  # well under MIN_DURATION_S
    path = _write_wav(short_signal)
    try:
        result = assess_quality(path)
        assert result["ok"] is False
        assert any("short" in w for w in result["warnings"])
    finally:
        os.unlink(path)


def test_mostly_silent_clip_flagged():
    silence = np.zeros(int(3.0 * SR), dtype=np.float32)
    path = _write_wav(silence)
    try:
        result = assess_quality(path)
        assert result["ok"] is False
        assert any("silence" in w or "empty" in w for w in result["warnings"])
    finally:
        os.unlink(path)


def test_clipped_clip_flagged():
    rng = np.random.default_rng(2)
    signal = synth_voice(rng, jitter_amount=0.1, shimmer_amount=0.1, silence_amount=0.0)
    clipped = np.clip(signal * 5.0, -1.0, 1.0)  # force hard clipping
    path = _write_wav(clipped)
    try:
        result = assess_quality(path)
        assert any("clipping" in w for w in result["warnings"])
    finally:
        os.unlink(path)


def test_quality_penalty_sums_known_warning_types():
    quality = {"warnings": ["Audio is clipping (x)", "Recording is short (y)"]}
    # 20 (clipping) + 15 (short), per the fixed weights in quality_penalty
    assert quality_penalty(quality) == 35


def test_quality_penalty_zero_when_clean():
    assert quality_penalty({"warnings": []}) == 0
