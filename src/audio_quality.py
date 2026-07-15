"""Objective recording-quality checks, measured from the audio itself.

The app's pre-screen asks the user whether their room is noisy; this module
replaces that self-report with measurement -- clipping, silence, duration, and
a rough SNR estimate -- so a genuinely bad recording is flagged even if the
user didn't think to mention it. Warn-only: it never blocks a recording, it
just surfaces problems and feeds a penalty into the reliability estimate.

Uses only librosa/numpy (already dependencies).
"""
import librosa
import numpy as np

# Thresholds -- deliberately lenient so only clearly-bad clips trip a warning.
MIN_DURATION_S = 1.5
MAX_CLIPPING_PCT = 1.0      # % of samples at/near full scale
MAX_SILENCE_PCT = 85.0      # % of the clip that is silence
MIN_SNR_DB = 10.0
TOP_DB = 30                 # matches features.py silence detection


def assess_quality(audio_path: str) -> dict:
    """Return objective quality metrics + warnings for a recording.

    Keys: duration_s, clipping_pct, silence_pct, snr_db, warnings (list[str]),
    ok (bool). snr_db is a coarse voiced-vs-silent energy ratio, not a
    calibrated SNR -- treat it as a relative indicator.
    """
    y, sr = librosa.load(audio_path, sr=None)
    warnings = []

    if y.size == 0:
        return {
            "duration_s": 0.0, "clipping_pct": 0.0, "silence_pct": 100.0,
            "snr_db": 0.0, "warnings": ["Recording is empty."], "ok": False,
        }

    duration_s = len(y) / sr

    peak = np.max(np.abs(y))
    # Normalize the clipping test to the clip's own peak so it works whether the
    # audio is full-scale float or quieter.
    threshold = 0.99 * peak if peak > 0 else 1.0
    clipping_pct = float(np.mean(np.abs(y) >= threshold) * 100.0)

    intervals = librosa.effects.split(y, top_db=TOP_DB)
    voiced_samples = int(sum(end - start for start, end in intervals))
    silence_pct = float((1 - voiced_samples / len(y)) * 100.0) if len(y) else 100.0

    # Coarse SNR: mean energy of voiced regions vs mean energy of the rest.
    voiced_mask = np.zeros(len(y), dtype=bool)
    for start, end in intervals:
        voiced_mask[start:end] = True
    voiced = y[voiced_mask]
    noise = y[~voiced_mask]
    if voiced.size and noise.size:
        voiced_power = float(np.mean(voiced ** 2))
        noise_power = float(np.mean(noise ** 2))
        snr_db = 10 * np.log10(voiced_power / noise_power) if noise_power > 0 else 40.0
    else:
        snr_db = 40.0  # essentially no measurable background -> treat as clean
    snr_db = float(np.clip(snr_db, -20.0, 40.0))

    if duration_s < MIN_DURATION_S:
        warnings.append(f"Recording is short ({duration_s:.1f}s) -- aim for ~5s of steady sound.")
    if clipping_pct > MAX_CLIPPING_PCT:
        warnings.append(f"Audio is clipping ({clipping_pct:.1f}% of samples) -- move back from the mic or lower the input level.")
    if silence_pct > MAX_SILENCE_PCT:
        warnings.append(f"Mostly silence ({silence_pct:.0f}%) -- make sure your voice was captured.")
    if snr_db < MIN_SNR_DB:
        warnings.append(f"Low signal-to-noise (~{snr_db:.0f} dB) -- background noise may skew the result.")

    return {
        "duration_s": round(duration_s, 2),
        "clipping_pct": round(clipping_pct, 2),
        "silence_pct": round(silence_pct, 1),
        "snr_db": round(snr_db, 1),
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }


def quality_penalty(quality: dict) -> int:
    """Points to subtract from the reliability estimate for measured problems.
    Fixed penalties, transparent and additive, mirroring compute_reliability."""
    penalty = 0
    for w in quality.get("warnings", []):
        if "clipping" in w:
            penalty += 20
        elif "silence" in w or "empty" in w:
            penalty += 30
        elif "short" in w:
            penalty += 15
        elif "signal-to-noise" in w:
            penalty += 15
    return penalty


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        print(path)
        for k, v in assess_quality(path).items():
            print(f"  {k}: {v}")
