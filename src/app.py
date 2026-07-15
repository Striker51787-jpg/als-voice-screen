"""Streamlit UI for the ALS voice-pattern research prototype.

Primary flow: answer a one-at-a-time pre-screen (see below), then record (or
upload) all 8 VOC-ALS speech tasks and score with the multi-task model
(validated AUC ~0.70, 95% CI [0.61, 0.78] -- see README). A secondary "quick
screen" tab offers a single-clip score using the weaker single-task model,
clearly labeled as lower confidence.

Run with: streamlit run src/app.py
"""
import os
import tempfile

import joblib
import librosa
import pandas as pd
import streamlit as st

from features import extract_features
from audio_quality import assess_quality, quality_penalty
from report import generate_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MULTITASK_MODEL_PATH = os.path.join(MODELS_DIR, "als_multitask_detector.joblib")
SINGLE_MODEL_PATH = os.path.join(MODELS_DIR, "als_detector.joblib")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "presentation", "figures")

TASKS = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
         "rhythmKA", "rhythmPA", "rhythmTA"]

TASK_INSTRUCTIONS = {
    "phonationA": 'Say "ahhh" and hold it steady for about 5 seconds.',
    "phonationE": 'Say "ehhh" and hold it steady for about 5 seconds.',
    "phonationI": 'Say "eeee" and hold it steady for about 5 seconds.',
    "phonationO": 'Say "ohhh" and hold it steady for about 5 seconds.',
    "phonationU": 'Say "oooo" and hold it steady for about 5 seconds.',
    "rhythmKA": 'Repeat "ka-ka-ka-ka..." as fast and evenly as you can for about 5 seconds.',
    "rhythmPA": 'Repeat "pa-pa-pa-pa..." as fast and evenly as you can for about 5 seconds.',
    "rhythmTA": 'Repeat "ta-ta-ta-ta..." as fast and evenly as you can for about 5 seconds.',
}

DISCLAIMER = (
    "This score comes from an acoustic pattern-matching model, not a diagnosis. "
    "Things like a raspy voice, intoxication, Parkinson's, or a stroke can produce similar "
    "acoustic markers (jitter, shimmer, low HNR), and this model has no way to tell them apart. "
    "If you get a high score, talk to a neurologist. Don't treat this as a result on its own."
)

# Pre-screen questions, asked one at a time. Each flags a condition known to
# alter voice acoustics in ways the model can't distinguish from ALS (see
# README). "bool" questions are yes/no; "scale" questions are 1-10 severity.
# Penalties are a transparent, hand-picked weighting -- not a validated
# clinical instrument or a model output, just simple arithmetic so the
# "reliability estimate" below is auditable rather than a black box.
PRESCREEN_QUESTIONS = [
    dict(key="alcohol", type="bool", penalty=25,
         text="Have you had alcohol or a sedating medication in the last 4 hours?"),
    dict(key="congestion", type="scale", max_penalty=20,
         text="How congested or sore is your throat right now?"),
    dict(key="other_condition", type="bool", penalty=40,
         text="Do you have a diagnosed speech condition unrelated to ALS "
              "(stutter, cleft palate, prior stroke, Parkinson's, etc.)?"),
    dict(key="non_native", type="bool", penalty=15,
         text="Are you speaking in a language you're not a native speaker of?"),
    dict(key="noisy_room", type="scale", max_penalty=15,
         text="How noisy or echo-prone is your recording environment?"),
    dict(key="smoked", type="bool", penalty=10,
         text="Have you smoked in the last hour?"),
    dict(key="fatigue", type="scale", max_penalty=10,
         text="How tired do you feel right now?"),
]


def compute_reliability(answers: dict):
    """Turn pre-screen answers into a 0-100 reliability estimate + reasons.

    Simple weighted penalty sum, not a trained model -- bool questions
    subtract a fixed penalty when flagged, scale questions subtract a
    fraction of their max penalty proportional to severity (1 = none, 10 =
    full penalty). Floored at 15 so the number never implies "zero chance
    this result means anything."
    """
    penalty_total = 0.0
    reasons = []
    for q in PRESCREEN_QUESTIONS:
        val = answers.get(q["key"])
        if q["type"] == "bool":
            if val:
                penalty_total += q["penalty"]
                reasons.append(q["text"])
        else:
            if val and val > 1:
                penalty_total += (val - 1) / 9 * q["max_penalty"]
                if val >= 6:
                    reasons.append(f"{q['text']} (rated {val}/10)")
    reliability = max(15, round(100 - penalty_total))
    return reliability, reasons


def reliability_band(pct):
    if pct >= 85:
        return "High confidence"
    if pct >= 65:
        return "Moderate confidence"
    if pct >= 40:
        return "Low confidence"
    return "Very low confidence"


def run_prescreen_wizard(prefix: str):
    """One-question-at-a-time pre-screen. Returns (reliability, reasons) once
    complete, or None while still in progress (caller should st.stop() the
    rest of that tab's content until this returns non-None)."""
    step_key, answers_key, done_key = f"{prefix}_step", f"{prefix}_answers", f"{prefix}_prescreen_done"
    st.session_state.setdefault(step_key, 0)
    st.session_state.setdefault(answers_key, {})
    st.session_state.setdefault(done_key, False)

    if st.session_state[done_key]:
        return compute_reliability(st.session_state[answers_key])

    st.subheader("Before you start")
    step = st.session_state[step_key]
    total = len(PRESCREEN_QUESTIONS)

    if step < total:
        q = PRESCREEN_QUESTIONS[step]
        st.caption(f"Question {step + 1} of {total}")
        st.progress(step / total)

        if q["type"] == "bool":
            choice = st.radio(q["text"], ["No", "Yes"], key=f"{prefix}_{q['key']}_widget", horizontal=True)
            current_val = choice == "Yes"
        else:
            current_val = st.slider(f"{q['text']} (1 = not at all, 10 = severe)", 1, 10, 1,
                                     key=f"{prefix}_{q['key']}_widget")

        cols = st.columns(2)
        if step > 0 and cols[0].button("Back", key=f"{prefix}_back"):
            st.session_state[step_key] -= 1
            st.rerun()
        if cols[1].button("Next", key=f"{prefix}_next"):
            st.session_state[answers_key][q["key"]] = current_val
            st.session_state[step_key] += 1
            st.rerun()
        return None

    reliability, reasons = compute_reliability(st.session_state[answers_key])
    band = reliability_band(reliability)
    st.metric("Estimated screen reliability", f"{reliability}%", help=band)
    if reasons:
        st.write("What's lowering it:")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("Nothing flagged — good conditions for this screen.")
    st.caption(
        "This is a simple weighted estimate from your answers (fixed point deductions per question), "
        "not a validated clinical instrument or a model prediction."
    )
    if st.button("Continue to recording", key=f"{prefix}_continue"):
        st.session_state[done_key] = True
        st.rerun()
    return None


def show_prescreen_summary(reliability, reasons):
    band = reliability_band(reliability)
    msg = f"Pre-screen reliability estimate: {reliability}% ({band})."
    if reasons:
        msg += " Flagged: " + "; ".join(reasons) + ". Take the score below with that in mind."
    st.warning(msg) if reliability < 65 else st.info(msg)


def render_step_tracker(done_flags):
    """Row of numbered circles (filled once that task's clip is captured)
    instead of a plain "N of 8 done" line."""
    circles = ""
    for i, done in enumerate(done_flags):
        bg = "#1d4ed8" if done else "#e5e7eb"
        fg = "white" if done else "#6b7280"
        label = "&#10003;" if done else str(i + 1)
        circles += (
            f'<div style="width:30px;height:30px;border-radius:50%;display:flex;'
            f'align-items:center;justify-content:center;background:{bg};color:{fg};'
            f'font-size:13px;font-weight:600;">{label}</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:8px;margin:10px 0 18px 0;">{circles}</div>',
        unsafe_allow_html=True,
    )


def render_gauge(prob):
    """Horizontal zoned gauge (control-like / uncertain / ALS-consistent)
    with a marker at the actual score, instead of a bare progress bar."""
    pct = min(max(prob, 0.0), 1.0) * 100
    st.markdown(
        f"""
        <div style="position:relative;height:34px;border-radius:8px;overflow:visible;
                    background:linear-gradient(to right, #16a34a 0%, #16a34a 40%,
                    #e5e7eb 40%, #e5e7eb 60%, #dc2626 60%, #dc2626 100%);margin:10px 0 2px 0;">
          <div style="position:absolute;top:-5px;left:{pct}%;transform:translateX(-50%);
                      width:4px;height:44px;background:#111827;border-radius:2px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:#6b7280;">
          <span>Control-like</span><span>Uncertain</span><span>ALS-consistent</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_model(path):
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["feature_names"], bundle.get("model_name", "unknown")


def extract_from_bytes(audio_bytes: bytes):
    """Returns (features_dict, quality_dict, tmp_path). Caller unlinks tmp_path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        quality = assess_quality(tmp_path)
        return extract_features(tmp_path), quality, tmp_path
    except Exception:
        os.unlink(tmp_path)
        raise


def show_result(prob, model_name, feature_row, reliability=None, reasons=None,
                quality_summary=None, key_prefix="result"):
    st.subheader("Result")
    label = "ALS-consistent pattern" if prob >= 0.5 else "Control-consistent pattern"
    st.metric("Score", f"{prob:.2f}", help="0 = control-like, 1 = ALS-consistent")
    st.write(f"{label} (model: `{model_name}`)")
    render_gauge(prob)
    st.info(DISCLAIMER)
    with st.expander("Extracted features"):
        st.dataframe(feature_row.T.rename(columns={0: "value"}))

    if reliability is not None:
        report_key = f"{key_prefix}_report"
        if st.button("Explain this result in plain language", key=f"{key_prefix}_explain"):
            with st.spinner("Writing a plain-language explanation..."):
                st.session_state[report_key] = generate_report(
                    prob, label, reliability, reasons or [], quality_summary
                )
        if report_key in st.session_state:
            st.write(st.session_state[report_key])


st.set_page_config(page_title="ALS Voice Screen", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 760px; }
    h1 { font-weight: 650; letter-spacing: -0.02em; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    section[data-testid="stSidebar"] h3 { margin-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this tool")
    st.write(
        "A research project screening for ALS-consistent speech patterns from "
        "voice recordings, built on the VOC-ALS dataset (153 participants)."
    )
    st.warning("Not a diagnostic tool. See the note under every result.")
    st.markdown("### Model performance")
    st.markdown(
        "**Full assessment** (8 tasks)\n"
        "- Held-out ROC-AUC: **0.76**\n"
        "- Nested-CV estimate: 0.72\n"
        "- 95% CI (5-fold CV): [0.61, 0.78]"
    )
    st.markdown(
        "**Quick screen** (1 clip)\n"
        "- ROC-AUC: 0.60–0.65\n"
        "- Weaker — over-flags ALS"
    )
    st.caption("See the Model Insights tab for the figures behind these numbers.")

st.title("ALS Voice Screen")

tab_multi, tab_quick, tab_insights = st.tabs(
    ["Full assessment (8 tasks)", "Quick screen (1 clip)", "Model Insights"]
)

with tab_multi:
    bundle = load_model(MULTITASK_MODEL_PATH)
    if bundle is None:
        st.error(
            "No multi-task model found at `models/als_multitask_detector.joblib`. "
            "Run `python src/build_multitask_dataset.py` then `python src/train_multitask.py` first."
        )
    else:
        pipeline, feature_names, model_name = bundle
        st.write(
            "The validated model (see sidebar for numbers). Needs 8 short "
            "recordings — go through each one below."
        )

        prescreen_result = run_prescreen_wizard("multi")
        if prescreen_result is not None:
            multi_reliability, multi_reasons = prescreen_result
            st.divider()

            task_features = {}
            worst_quality_penalty = 0
            done_flags = []
            for task in TASKS:
                task_done = False
                with st.expander(f"{task}", expanded=False):
                    st.write(TASK_INSTRUCTIONS[task])
                    recording = st.audio_input("Record", key=f"record_{task}")
                    uploaded = st.file_uploader("...or upload a .wav", type=["wav", "flac"], key=f"upload_{task}")
                    audio_bytes = uploaded.read() if uploaded is not None else (recording.read() if recording is not None else None)
                    if audio_bytes is not None:
                        try:
                            feats, quality, tmp_path = extract_from_bytes(audio_bytes)
                            os.unlink(tmp_path)
                            for k, v in feats.items():
                                task_features[f"{task}__{k}"] = v
                            task_done = True
                            worst_quality_penalty = max(worst_quality_penalty, quality_penalty(quality))
                            if quality["warnings"]:
                                for w in quality["warnings"]:
                                    st.warning(w)
                            else:
                                st.success("Got it (recording quality looks fine).")
                        except Exception as e:
                            st.error(f"Couldn't process that one: {e}")
                done_flags.append(task_done)

            completed = sum(done_flags)
            render_step_tracker(done_flags)
            st.caption(f"{completed} of {len(TASKS)} done.")
            if completed == len(TASKS):
                if st.button("Run the analysis"):
                    X = pd.DataFrame([task_features])[feature_names]
                    prob = float(pipeline.predict_proba(X)[0, 1])
                    # Fold measured recording quality into the reliability estimate.
                    adj_reliability = max(15, multi_reliability - worst_quality_penalty)
                    adj_reasons = list(multi_reasons)
                    if worst_quality_penalty > 0:
                        adj_reasons.append("measured recording-quality problems (see per-task warnings)")
                    st.session_state["multi_result"] = {
                        "prob": prob, "X": X, "reliability": adj_reliability, "reasons": adj_reasons,
                    }
                    st.session_state.pop("multi_result_report", None)  # invalidate old explanation
            else:
                st.info("Fill in all 8 above to run the analysis.")

            if "multi_result" in st.session_state:
                r = st.session_state["multi_result"]
                show_prescreen_summary(r["reliability"], r["reasons"])
                show_result(r["prob"], model_name, r["X"], reliability=r["reliability"],
                            reasons=r["reasons"], key_prefix="multi_result")

with tab_quick:
    bundle = load_model(SINGLE_MODEL_PATH)
    if bundle is None:
        st.error(
            "No single-task model found at `models/als_detector.joblib`. "
            "Run `python src/build_dataset.py` then `python src/train.py` first."
        )
    else:
        pipeline, feature_names, model_name = bundle
        st.warning(
            "Heads up: this is a quick demo with the weaker single-clip model (see sidebar) — "
            "use the full assessment tab if you want a number that means anything."
        )
        prescreen_result = run_prescreen_wizard("quick")
        if prescreen_result is not None:
            quick_reliability, quick_reasons = prescreen_result
            st.divider()

            st.write('Say "ahhh" and hold it steady for about 5 seconds.')
            recording = st.audio_input("Record a voice sample", key="quick_record")
            uploaded = st.file_uploader("...or upload a .wav", type=["wav", "flac"], key="quick_upload")
            audio_bytes = uploaded.read() if uploaded is not None else (recording.read() if recording is not None else None)

            if audio_bytes is not None:
                feats, quality, tmp_path = extract_from_bytes(audio_bytes)
                try:
                    y, sr = librosa.load(tmp_path, sr=None)
                    X = pd.DataFrame([feats])[feature_names]
                    prob = pipeline.predict_proba(X)[0, 1]
                    for w in quality["warnings"]:
                        st.warning(w)
                    adj_reliability = max(15, quick_reliability - quality_penalty(quality))
                    adj_reasons = list(quick_reasons)
                    if quality["warnings"]:
                        adj_reasons.append("measured recording-quality problems (see warnings above)")
                    quality_summary = "; ".join(quality["warnings"]) if quality["warnings"] else "no issues detected"
                    show_prescreen_summary(adj_reliability, adj_reasons)
                    show_result(prob, model_name, X, reliability=adj_reliability,
                                reasons=adj_reasons, quality_summary=quality_summary,
                                key_prefix="quick_result")
                    st.subheader("Waveform")
                    st.line_chart(pd.DataFrame({"amplitude": y[:: max(1, len(y) // 2000)]}))
                finally:
                    os.unlink(tmp_path)

with tab_insights:
    st.write(
        "The evaluation figures behind the numbers in the sidebar — generated straight from "
        "`models/evaluation_report.json` and the saved model, so they can't drift from what "
        "was actually measured. Regenerate with `python src/generate_figures.py`."
    )

    figures = [
        ("model_comparison.png", "Why 8 tasks instead of 1",
         "Combining all 8 speech tasks clearly beats scoring from a single sustained vowel or from "
         "the spreadsheet's precomputed features alone. The two blue bars are the multi-task model "
         "under two different honesty checks — an unbiased nested-CV estimate and a locked held-out "
         "test set that was never touched during model selection."),
        ("roc_curve.png", "How good is the model, really",
         "ROC-AUC 0.76 on data the model never saw during training or selection. That's \"fair,\" "
         "not \"good\" — for comparison, 0.50 is a coin flip and most clinical screening tools aim "
         "for 0.80+. The nested-CV estimate (0.72) is shown alongside it as a sanity check."),
        ("calibration_curve.png", "Do the scores mean what they say",
         "If the model says 0.70, does that group turn out to be ALS about 70% of the time? This "
         "checks that — points near the diagonal mean the score is trustworthy as a probability, not "
         "just a ranking. With only 31 held-out participants, treat the exact shape loosely."),
        ("confusion_matrices.png", "The sensitivity/specificity trade-off is a choice",
         "At the default 0.5 cutoff the model catches 81% of ALS cases while flagging half of "
         "healthy controls too. Pushing the threshold down catches everyone with ALS but flags "
         "every control as well — there's no free lunch, just a deliberate trade-off depending on "
         "what a screen is for."),
        ("feature_importance.png", "What the model is actually looking at",
         "Only one of the top features (pause ratio) is a classic clinical marker. Most of the rest "
         "are MFCC statistics — generic vocal-timbre coefficients. With only 51 healthy controls, "
         "that's a real risk the model is partly fitting incidental recording/voice-quality "
         "differences rather than disease-specific patterns. See the README for the full writeup."),
    ]

    for filename, heading, caption in figures:
        path = os.path.join(FIGURES_DIR, filename)
        st.subheader(heading)
        if os.path.exists(path):
            st.image(path)
            st.caption(caption)
        else:
            st.info(f"`{filename}` not found — run `python src/generate_figures.py` from the `src/` directory first.")
        st.divider()
