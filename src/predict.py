"""Run the trained model on a single audio file. Usage: python predict.py path/to/file.wav"""
import sys
import os
import joblib
import pandas as pd
from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "als_detector.joblib")


def main(audio_path: str):
    bundle = joblib.load(MODEL_PATH)
    pipeline, feature_names = bundle["pipeline"], bundle["feature_names"]

    feats = extract_features(audio_path)
    X = pd.DataFrame([feats])[feature_names]

    prob = pipeline.predict_proba(X)[0, 1]
    label = "ALS-consistent dysarthria pattern" if prob >= 0.5 else "Control-consistent pattern"
    print(f"{audio_path}: {label} (risk score: {prob:.2f})")
    print(
        "NOTE: this is an acoustic pattern-matching score from a research prototype, "
        "not a diagnosis. Dysarthria-like acoustic markers (jitter, shimmer, low HNR, "
        "reduced formant range) also occur with intoxication, Parkinson's disease, "
        "stroke, and other non-ALS causes -- this model cannot distinguish between them. "
        "An elevated score should prompt referral to a neurologist, not be treated as a result."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/file.wav")
        sys.exit(1)
    main(sys.argv[1])
