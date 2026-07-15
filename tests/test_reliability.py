"""Tests for the pre-screen reliability scoring in app.py -- pure arithmetic,
no Streamlit runtime needed (see conftest.py for the import-safety note)."""
from app import compute_reliability, reliability_band, PRESCREEN_QUESTIONS


def test_no_flags_gives_full_reliability():
    reliability, reasons = compute_reliability({})
    assert reliability == 100
    assert reasons == []


def test_every_question_maxed_floors_at_15_with_all_reasons():
    answers = {}
    for q in PRESCREEN_QUESTIONS:
        answers[q["key"]] = True if q["type"] == "bool" else 10
    reliability, reasons = compute_reliability(answers)
    assert reliability == 15
    assert len(reasons) == len(PRESCREEN_QUESTIONS)


def test_single_bool_flag_applies_only_its_penalty():
    # alcohol question has a fixed penalty of 25 (see app.py PRESCREEN_QUESTIONS)
    reliability, reasons = compute_reliability({"alcohol": True})
    assert reliability == 75
    assert len(reasons) == 1


def test_low_scale_answer_below_threshold_not_listed_as_reason():
    # fatigue max_penalty=10; a rating of 2 shouldn't cross the >=6 reason threshold
    reliability, reasons = compute_reliability({"fatigue": 2})
    assert reasons == []
    assert reliability < 100  # still applies a small penalty


def test_high_scale_answer_is_listed_as_reason():
    reliability, reasons = compute_reliability({"fatigue": 8})
    assert any("fatigue" in r.lower() or "tired" in r.lower() for r in reasons)


def test_reliability_band_thresholds():
    assert reliability_band(100) == "High confidence"
    assert reliability_band(85) == "High confidence"
    assert reliability_band(84) == "Moderate confidence"
    assert reliability_band(65) == "Moderate confidence"
    assert reliability_band(64) == "Low confidence"
    assert reliability_band(40) == "Low confidence"
    assert reliability_band(39) == "Very low confidence"
    assert reliability_band(15) == "Very low confidence"
