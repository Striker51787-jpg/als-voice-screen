"""Sort downloaded VOC-ALS .wav files into data/raw/{als,control}/ using the
ground-truth Category column from data/VOC-ALS.xlsx (not just the CT/PZ
filename prefix, to be safe).

Usage:
    python sort_audio.py /path/to/downloaded/phonationA --task phonationA
"""
import argparse
import os
import shutil
import pandas as pd

XLSX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "VOC-ALS.xlsx")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_id_to_label():
    df = pd.read_excel(XLSX_PATH, sheet_name="VOC-ALS_Data", header=1)
    return dict(zip(df["ID"], df["Category"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="Folder containing downloaded .wav files")
    parser.add_argument("--task", default="phonationA", help="Task suffix to filter on, e.g. phonationA")
    args = parser.parse_args()

    id_to_label = load_id_to_label()
    os.makedirs(os.path.join(RAW_DIR, "als"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "control"), exist_ok=True)

    counts = {"als": 0, "control": 0, "skipped": 0}
    for fname in sorted(os.listdir(args.source_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        if f"_{args.task}." not in fname:
            continue
        participant_id = fname.split("_")[0]
        label = id_to_label.get(participant_id)
        if label == "ALS":
            dest = os.path.join(RAW_DIR, "als", fname)
            counts["als"] += 1
        elif label == "HC":
            dest = os.path.join(RAW_DIR, "control", fname)
            counts["control"] += 1
        else:
            print(f"Skipping {fname}: unknown participant ID '{participant_id}'")
            counts["skipped"] += 1
            continue
        shutil.copy2(os.path.join(args.source_dir, fname), dest)

    print(f"Sorted {counts['als']} ALS files, {counts['control']} control files, skipped {counts['skipped']}.")


if __name__ == "__main__":
    main()
