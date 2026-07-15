"""Extract features from all 8 VOC-ALS task recordings per participant
(phonationA/E/I/O/U, rhythmKA/PA/TA) and pivot into one wide feature row per
participant, labeled from data/VOC-ALS.xlsx. Writes data/features_multitask.csv.
"""
import os
import pandas as pd
from features import extract_features

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_audio", "all_tasks")
XLSX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "VOC-ALS.xlsx")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features_multitask.csv")

TASKS = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
         "rhythmKA", "rhythmPA", "rhythmTA"]


def main():
    meta = pd.read_excel(XLSX_PATH, sheet_name="VOC-ALS_Data", header=1)
    id_to_label = dict(zip(meta["ID"], meta["Category"]))

    per_participant = {}
    for fname in sorted(os.listdir(SOURCE_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        participant_id, task = fname.rsplit("_", 1)
        task = task.replace(".wav", "")
        if task not in TASKS:
            continue
        label = id_to_label.get(participant_id)
        if label not in ("ALS", "HC"):
            print(f"Skipping {fname}: unknown participant {participant_id}")
            continue

        try:
            feats = extract_features(os.path.join(SOURCE_DIR, fname))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        row = per_participant.setdefault(participant_id, {"participant_id": participant_id, "label": 1 if label == "ALS" else 0})
        for k, v in feats.items():
            row[f"{task}__{k}"] = v

    df = pd.DataFrame(per_participant.values())
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} participant rows ({df.shape[1] - 2} features) to {OUT_PATH}")


if __name__ == "__main__":
    main()
