"""Generate synthetic .wav files for smoke-testing the pipeline end-to-end.

Not real patient data -- just sine-wave-based audio with injected jitter/noise
so build_dataset.py / train.py / predict.py can be exercised before real
recordings are available. "als" files get more pitch/amplitude instability and
more silence injected to loosely mimic dysarthric/hypophonic speech; "control"
files are cleaner. This is only useful for testing that the code runs, not for
any conclusions about real acoustic patterns.
"""
import os
import numpy as np
import soundfile as sf

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SR = 16000
N_PER_CLASS = 8
DURATION = 3.0


def synth_voice(rng, jitter_amount, shimmer_amount, silence_amount, base_f0=120):
    n_samples = int(DURATION * SR)
    t = np.arange(n_samples) / SR

    f0_wobble = base_f0 + jitter_amount * base_f0 * rng.standard_normal(n_samples).cumsum() / n_samples * 50
    phase = 2 * np.pi * np.cumsum(f0_wobble) / SR

    amp_wobble = 1.0 + shimmer_amount * rng.standard_normal(n_samples) * 0.3
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 6))
    signal = amp_wobble * harmonics
    signal += 0.02 * rng.standard_normal(n_samples)  # mild background noise

    if silence_amount > 0:
        n_gaps = int(silence_amount * 10)
        gap_len = int(0.15 * SR)
        for _ in range(n_gaps):
            start = rng.integers(0, max(1, n_samples - gap_len))
            signal[start:start + gap_len] = 0.0

    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.7
    return signal.astype(np.float32)


def main():
    rng = np.random.default_rng(42)

    for label, subdir, jitter, shimmer, silence in [
        ("als", "als", 0.8, 0.8, 0.5),
        ("control", "control", 0.15, 0.15, 0.05),
    ]:
        folder = os.path.join(OUT_DIR, subdir)
        os.makedirs(folder, exist_ok=True)
        n_participants = N_PER_CLASS // 2
        for p in range(n_participants):
            participant_id = f"{subdir}P{p:02d}"
            for rec in range(2):  # 2 recordings per participant
                y = synth_voice(rng, jitter, shimmer, silence)
                fname = f"{participant_id}_{rec}.wav"
                sf.write(os.path.join(folder, fname), y, SR)
        print(f"Wrote {n_participants * 2} synthetic files to {folder}")


if __name__ == "__main__":
    main()
