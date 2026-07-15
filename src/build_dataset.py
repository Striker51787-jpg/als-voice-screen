"""Walk data/raw/{als,control}, extract features per file, write data/features.csv."""
import os
import pandas as pd
from features import extract_features

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")


def main():
    rows = []
    for label, subdir in [(1, "als"), (0, "control")]:
        folder = os.path.join(RAW_DIR, subdir)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".wav", ".flac")):
                continue
            path = os.path.join(folder, fname)
            try:
                feats = extract_features(path)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
                continue
            feats["label"] = label
            feats["file"] = fname
            # Filenames are expected as "<participant_id>_<recording>.wav" so that
            # multiple recordings from the same person can be grouped during CV
            # (random k-fold splitting would otherwise leak a participant's voice
            # into both train and test folds and inflate reported accuracy).
            feats["participant_id"] = fname.split("_")[0] if "_" in fname else fname
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
