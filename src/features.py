"""Extract acoustic features from a voice recording relevant to ALS-related dysarthria."""
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


def extract_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None)
    snd = parselmouth.Sound(audio_path)

    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call(
        [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    hnr = call(
        call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0
    )

    f0_values = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    mfcc_stds = mfccs.std(axis=1)

    intervals = librosa.effects.split(y, top_db=30)
    speech_duration = sum((end - start) for start, end in intervals) / sr
    total_duration = len(y) / sr
    pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0

    n_frames = pitch.get_number_of_frames()
    voiced_fraction = pitch.count_voiced_frames() / n_frames if n_frames > 0 else 0.0

    formant = call(snd, "To Formant (burg)", 0.0, 5, 5000, 0.025, 50)
    f1_vals, f2_vals = [], []
    for t in np.arange(0, total_duration, 0.01):
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        if f1 == f1:  # filters out NaN
            f1_vals.append(f1)
        if f2 == f2:
            f2_vals.append(f2)
    f1_mean = float(np.mean(f1_vals)) if f1_vals else np.nan
    f2_mean = float(np.mean(f2_vals)) if f2_vals else np.nan

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    features = {
        "jitter_local": jitter_local,
        "shimmer_local": shimmer_local,
        "hnr": hnr,
        "f0_mean": f0_values,
        "f0_std": f0_std,
        "pause_ratio": pause_ratio,
        "speech_rate_proxy": len(intervals) / total_duration if total_duration > 0 else 0,
        "voiced_fraction": voiced_fraction,
        "formant1_mean": f1_mean,
        "formant2_mean": f2_mean,
        "spectral_centroid_mean": float(spectral_centroid.mean()),
        "spectral_rolloff_mean": float(spectral_rolloff.mean()),
        "zcr_mean": float(zcr.mean()),
    }
    for i, (m, s) in enumerate(zip(mfcc_means, mfcc_stds)):
        features[f"mfcc{i}_mean"] = m
        features[f"mfcc{i}_std"] = s

    return features
