"""Tests for report.generate_report's fallback chain -- confirms it degrades
gracefully to the deterministic template when neither the Anthropic API nor a
local Ollama server is available, and never raises."""
import report


def test_falls_through_to_template_when_no_credentials_and_no_ollama(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    # Point Ollama at a port nothing listens on so the request fails fast.
    monkeypatch.setattr(report, "OLLAMA_URL", "http://127.0.0.1:1/api/generate")

    result = report.generate_report(
        0.67, "ALS-consistent pattern", 55,
        flags=["congestion (rated 7/10)"], quality_summary="low signal-to-noise (~8 dB)",
    )

    expected = report._template(
        0.67, "ALS-consistent pattern", 55,
        ["congestion (rated 7/10)"], "low signal-to-noise (~8 dB)",
    )
    assert result == expected


def test_template_mentions_score_and_reliability_and_flags():
    text = report._template(0.30, "Control-consistent pattern", 90, [], None)
    assert "0.30" in text
    assert "90%" in text


def test_template_flags_low_reliability_explicitly():
    text = report._template(0.5, "ALS-consistent pattern", 20, [], None)
    assert "especially uncertain" in text


def test_via_ollama_returns_none_when_unreachable(monkeypatch):
    monkeypatch.setattr(report, "OLLAMA_URL", "http://127.0.0.1:1/api/generate")
    facts = report._facts(0.5, "Control-consistent pattern", 100, [], None)
    assert report._via_ollama(facts, timeout_s=2) is None


def test_via_anthropic_returns_none_without_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    facts = report._facts(0.5, "Control-consistent pattern", 100, [], None)
    assert report._via_anthropic(facts) is None
