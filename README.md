# AI-Powered ALS Voice Detector (Research Prototype)

Detects ALS-consistent dysarthria patterns from voice recordings using acoustic
feature extraction + a gradient boosting classifier. Built for an internship
presentation — **not a diagnostic tool**.

## How it works
1. **Feature extraction** (`src/features.py`): jitter, shimmer, harmonics-to-noise
   ratio, pitch stats, voiced-frame fraction, formant means, spectral shape,
   pause ratio, and MFCCs — acoustic markers known to shift with ALS-related
   bulbar/speech motor decline.
2. **Dataset build** (`src/build_dataset.py`): walks `data/raw/als/` and
   `data/raw/control/`, extracts features per recording, writes `data/features.csv`.
   Expects filenames like `<participant_id>_<recording>.wav` so recordings can be
   grouped by participant.
3. **Training** (`src/train.py`): compares logistic regression, random forest, and
   gradient boosting, each calibrated and evaluated with **participant-grouped**
   5-fold CV (`StratifiedGroupKFold`) so a person's voice never appears in both
   the train and test side of a fold. Picks the best model by ROC-AUC and saves
   the trained pipeline to `models/als_detector.joblib`.
4. **Inference** (`src/predict.py`): runs the trained model on a single new
   recording, prints a risk score, and prints a disclaimer about non-ALS
   confounds (intoxication, Parkinson's, stroke).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Getting data
This repo does not include any patient data. Dataset used: **VOC-ALS**
(1224 recordings, 153 participants — 102 ALS, 51 healthy controls), published
in *Scientific Data* (Nature). Hosted on **Synapse** (DOI 10.7303/syn53009474)
— access only requires a free self-service Synapse account (no approval wait),
not a gated data access request as initially assumed.

Synapse provides 8 task recordings per participant (`phonationA/E/I/O/U`,
`rhythmKA/PA/TA`) plus `VOC-ALS.xlsx` with diagnosis labels, demographics, and
precomputed acoustic summary stats.

`src/sort_audio.py` sorts downloaded recordings into `data/raw/{als,control}/`
using the ground-truth `Category` column from `VOC-ALS.xlsx` (not just the
`CT`/`PZ` filename prefix):
```bash
python src/sort_audio.py /path/to/downloaded/phonationA --task phonationA
```

## Running the pipeline (single task: phonationA)
```bash
python src/build_dataset.py
python src/train.py
python src/predict.py path/to/new_recording.wav
```

## Running the multi-task pipeline (all 8 tasks, recommended)
Extracts and concatenates features across all 8 VOC-ALS tasks per participant
(one row per person instead of per recording). This performed meaningfully
better than any single task alone.
```bash
python src/build_multitask_dataset.py   # reads data/raw_audio/all_tasks/
python src/train_multitask.py
```

## Tabular baseline (no audio needed)
`src/train_tabular.py` trains directly on the spreadsheet's precomputed
acoustic features (jitter/shimmer/HNR/F0 per task) plus Age/Sex — useful as a
fast sanity check before processing raw audio. Note it deliberately excludes
ALSFRS-R/FVC%/OnsetRegion/etc., which are `-` placeholders for every healthy
control and would otherwise leak the label directly.
```bash
python src/train_tabular.py
```

## Viewing it in a browser
`src/app.py` is a Streamlit UI with two tabs:
- **Full assessment (recommended)**: walks through all 8 VOC-ALS speech tasks
  (5 sustained vowels + 3 syllable-repetition tasks), record or upload each,
  then scores with the validated multi-task model (AUC ≈ 0.70). Requires
  `models/als_multitask_detector.joblib` (run `build_multitask_dataset.py` +
  `train_multitask.py` first).
- **Quick single-clip screen**: one recording, scored with the weaker
  single-task model — clearly labeled in the UI as lower confidence, since it
  tends to over-predict "ALS-consistent." Requires `models/als_detector.joblib`.

Both tabs open with a **one-question-at-a-time pre-screen** (alcohol/sedatives,
congestion severity, non-ALS speech conditions, non-native language, noisy
recording environment, smoking, fatigue — some yes/no, some a 1-10 severity
slider). It doesn't block the test — it computes a transparent 0-100%
"reliability estimate" (simple weighted point deductions per answer, not a
trained model) and shows which factors lowered it, then that summary is
displayed alongside the eventual result so a compromised recording isn't
mistaken for a clean one.

The browser will prompt for microphone permission the first time you record.
```bash
streamlit run src/app.py
```
This opens a local browser tab (default `http://localhost:8501`).

## Smoke-testing without real data
`src/make_synthetic_data.py` generates fake sine-wave-based `.wav` files into
`data/raw/{als,control}/` so you can verify the full pipeline runs end-to-end
before you have real recordings. It proves nothing about real acoustic
patterns -- delete the generated files (and `data/features.csv`,
`models/als_detector.joblib`) before working with real data.
```bash
python src/make_synthetic_data.py
python src/build_dataset.py
python src/train.py
```

## Real results (VOC-ALS, 5-fold CV)
| Approach | Best model | ROC-AUC | Notes |
|---|---|---|---|
| Single task (phonationA only) | gradient_boosting | ~0.62 | Collapsed to predicting "ALS" for nearly everyone — looked OK on AUC, useless in practice (0% recall on controls) |
| Tabular (spreadsheet's precomputed features) | gradient_boosting | ~0.65 | Same collapse issue |
| **Multi-task (all 8 tasks combined)** | **logistic_regression** | **~0.70** | Best result; some real signal on controls (14% recall), still far from screening-grade |

AUC ~0.70 is "fair" discrimination by typical standards, not "good" (most
clinical screening tools target >0.8 before pilot consideration). With n=153,
treat any single AUC number as having a wide confidence interval, not a fixed
result. Bootstrapped 95% CI (`src/bootstrap_auc.py`): **[0.61, 0.78]** —
confirms the model beats chance (interval excludes 0.5) but the uncertainty
is wide.

## Rigorous evaluation (`src/evaluate.py`)
The 5-fold CV number above is mildly optimistic — the model type and feature
count `k` were chosen on the same CV that reports the score. `evaluate.py`
removes that optimism:
- **Nested CV** (inner loop re-runs the model-selection procedure, outer loop
  scores it on unseen data): ROC-AUC **0.72**, balanced accuracy 0.55 — an
  unbiased estimate of the whole pipeline including selection.
- **Locked held-out test set** (20%, n=31, scored exactly once): ROC-AUC
  **0.76**. Consistent with the CV number; the small test set means wide
  uncertainty, so read it alongside the bootstrap CI, not instead of it.
- **Operating points** on the held-out set: at the default 0.50 cutoff,
  sensitivity 0.81 / specificity 0.50 (the model over-flags ALS). A
  screening-oriented threshold tuned for ~0.85 sensitivity on the *train* split
  collapsed to flagging everyone on this tiny test set (specificity 0.0) — an
  honest illustration of how weak the specificity is at this sample size, not a
  usable operating point.
- **Calibration curve** saved to `models/calibration_curve.png`; full metrics in
  `models/evaluation_report.json`.

```bash
python src/evaluate.py
```

## External validation

**TORGO** (the obvious English dysarthria corpus) was ruled out: it doesn't
provide the 8 specific VOC-ALS tasks the production model needs, lumps ALS
speakers with cerebral-palsy speakers without clean per-speaker labels, is
English (VOC-ALS is Italian — a domain shift), and has only ~15 speakers.
Running the frozen 8-task model on it would measure task/language mismatch,
not generalization.

Instead, we found a genuinely compatible independent dataset: the
**Minsk2020 ALS database** (31 ALS patients, 33 healthy controls, Belarus,
different recording hardware and population than VOC-ALS) — Vashkevich M.,
Rushkevich Yu., *"Classification of ALS patients based on acoustic analysis
of sustained vowel phonations"*, Biomedical Signal Processing and Control,
2021 ([doi.org/10.1016/j.bspc.2020.102350](https://doi.org/10.1016/j.bspc.2020.102350),
[github.com/Mak-Sim/Minsk2020_ALS_database](https://github.com/Mak-Sim/Minsk2020_ALS_database),
GPL-3.0). It shares 2 of our 8 tasks (sustained vowels /a/ and /i/), so a
**phonationA + phonationI-only** model — trained *purely* on VOC-ALS, never
shown a single Minsk recording — could be scored on it as a true
cross-population, cross-equipment external test (script:
`src/train_two_task_external_val.py`; raw audio isn't redistributed here —
see the script for how to fetch it from the source above).

**Result**: external ROC-AUC **0.77** (95% CI [0.64, 0.88], n=64) for logistic
regression — comparable to, even slightly above, VOC-ALS's own 5-fold CV
estimate for the same 2-task model (0.66), and close to the full 8-task
model's held-out AUC (0.76). This is real evidence the acoustic signal isn't
just an artifact of one dataset's recording setup or population.

**Important caveat**: this only held for logistic regression. Random forest
and gradient boosting both collapsed on the external set (predicted every
participant as ALS, AUC 0.50–0.64, no real discrimination) — they appear to
overfit to VOC-ALS-specific feature-value thresholds that don't transfer.
This is itself a useful finding: it's a point in favor of preferring the
simpler linear model in production, not just here.

Two honest limits: this only validates 2 of the production model's 8 tasks
(Minsk doesn't have audio for the other 6), and n=64 external participants
is small, hence the wide confidence interval.

## Extra tooling
- **`src/audio_quality.py`** — objective recording-quality checks (clipping,
  silence, duration, rough SNR) measured from the audio itself. The app runs
  these on every clip, warns on problems, and folds a penalty into the
  reliability estimate — measurement, not just the user's self-report.
- **`src/report.py`** — optional plain-language explanation of a result, with
  guardrails so it never asserts a diagnosis or invents numbers. Tries three
  tiers in order, so it always produces something and works with or without
  any setup:
  1. **Anthropic API** (`claude-opus-4-8`, paid) if `ANTHROPIC_API_KEY` is set.
  2. **Local Ollama** (free, fully offline) if a server is running at
     `localhost:11434`. One-time setup:
     ```bash
     brew install ollama
     brew services start ollama
     ollama pull llama3.2:1b
     ```
     No API key, no cost, no internet required after the model is downloaded
     (~1.3GB). Being a small 1B-parameter model, it follows the guardrails
     more loosely than Claude — still cautious and non-diagnostic, just less
     precise about repeating every caveat.
  3. **Deterministic template** — always available, zero setup.
- **`src/export_result.py`** — builds a one-page PDF summary of a result
  (score, gauge, reliability, flagged factors, disclaimer) via matplotlib, no
  new dependency. Wired into the app as a "Download result summary" button.

## Running tests
```bash
pip install pytest  # already in requirements.txt
pytest tests/
```
Covers `audio_quality.assess_quality` (clipping/silence/duration detection on
synthetic clips), `app.compute_reliability`/`reliability_band` (pre-screen
scoring arithmetic), `report.generate_report`'s fallback chain (degrades to
the template with no API key and no Ollama), and `evaluate.py`'s threshold/
confusion-matrix helpers. Fast (~3s) — no real audio or model retraining
involved. This suite caught a real bug during development: `assess_quality`
was silently failing to flag a fully silent recording because
`librosa.effects.split` has no reference peak to compare against on an
all-zero signal — fixed with an explicit near-zero-peak check.

## Important caveats (for the presentation)
- Small dataset sizes (~100s of subjects) mean results should be reported with
  confidence intervals, not as a finished diagnostic.
- This is a screening-pattern classifier, not a clinical diagnostic device —
  no FDA clearance, not validated prospectively.
- Acoustic features alone can't distinguish ALS from other causes of dysarthria
  (e.g., intoxication, Parkinson's, stroke) — frame results as "ALS-consistent
  speech pattern," not "has ALS." `predict.py` now prints this caveat directly.
- CV is grouped by participant (`StratifiedGroupKFold`), not by file. With a
  file-level random split, multiple recordings from the same person can leak
  across train/test and make accuracy look better than it really is.
