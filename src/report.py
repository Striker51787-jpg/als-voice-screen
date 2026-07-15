"""Plain-language explanation of a screening result.

Turns the numbers the app already computed (score, reliability, flags, recording
quality) into a few careful sentences a non-expert can read. Three tiers, tried
in order, so the app always produces something and never requires payment:

  1. Anthropic API (claude-opus-4-8) if ANTHROPIC_API_KEY/AUTH_TOKEN is set.
  2. Local Ollama (http://localhost:11434, model `llama3.2:1b`) if a server is
     running -- free, fully offline, no account needed. Install with
     `brew install ollama && brew services start ollama && ollama pull llama3.2:1b`.
  3. Deterministic template -- always available, no network or install required.

Guardrails: whichever tier runs, the model is given ONLY the numbers below and
told never to assert a diagnosis, invent probabilities, or add medical claims --
it explains what the score means and repeats the referral/confound caveat. It is
not a medical device and its output is not advice.
"""
import json
import os
import urllib.error
import urllib.request

ANTHROPIC_MODEL = "claude-opus-4-8"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = (
    "You explain the output of an ALS voice-SCREENING research prototype to a "
    "non-expert. You are NOT a doctor and this is NOT a diagnosis. Rules you must "
    "follow: (1) Explain only the numbers you are given -- never invent a "
    "probability, a diagnosis, or any medical claim. (2) The 'score' is how much "
    "the voice pattern statistically resembles the ALS group in a small (n=153) "
    "research dataset, NOT a probability of having ALS. (3) Always remind the "
    "reader that dysarthria-like acoustic markers also come from fatigue, "
    "congestion, intoxication, Parkinson's, or stroke, and that an elevated score "
    "should prompt a conversation with a neurologist, not be treated as a result. "
    "(4) If reliability is low, say the result is especially uncertain. Keep it to "
    "about 4 short sentences, plain and calm, no jargon, no bullet points."
)


def _template(score, label, reliability, flags, quality_summary):
    parts = [
        f"Your score is {score:.2f} out of 1, which the tool reads as \"{label}.\" "
        "That number reflects how much your voice pattern resembles the ALS group "
        "in a small research dataset — it is not a probability that you have ALS.",
    ]
    if reliability < 65:
        parts.append(
            f"The reliability estimate for this session is low ({reliability}%), so "
            "treat the result as especially uncertain."
        )
    else:
        parts.append(f"The reliability estimate for this session is {reliability}%.")
    if flags:
        parts.append("Flagged factors that can skew the result: " + "; ".join(flags) + ".")
    parts.append(
        "The same acoustic markers also come from fatigue, a cold, intoxication, "
        "Parkinson's, or stroke — this tool can't tell them apart. If the score is "
        "elevated, talk to a neurologist; don't treat this as a diagnosis."
    )
    return " ".join(parts)


def _facts(score, label, reliability, flags, quality_summary):
    return (
        f"score={score:.2f} (0=control-like, 1=ALS-consistent)\n"
        f"label={label}\n"
        f"session_reliability_estimate={reliability}% (higher is better)\n"
        f"flagged_factors={flags or 'none'}\n"
        f"recording_quality={quality_summary or 'not assessed'}\n"
        "model_context=multi-task acoustic classifier, held-out ROC-AUC ~0.70-0.76, "
        "n=153 (102 ALS / 51 controls), Italian-language VOC-ALS dataset."
    )


def _via_anthropic(facts):
    """Returns explanation text, or None if unavailable/failed."""
    try:
        import anthropic
    except ImportError:
        return None
    # Read credentials from the environment only; never prompt the user for a key.
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": "Explain this screening result to the person who took it:\n\n" + facts,
            }],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception:
        # Network error, auth failure, rate limit -- fall through to the next tier.
        return None


def _via_ollama(facts, timeout_s=20):
    """Returns explanation text from a local Ollama server, or None if it's not
    running / not reachable / errors out. Free, fully offline -- no API key."""
    prompt = (
        SYSTEM_PROMPT + "\n\nExplain this screening result to the person who took it:\n\n" + facts
    )
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "").strip()
        return text or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        # Server not running, model not pulled, or timed out -- fall through.
        return None


def generate_report(score, label, reliability, flags=None, quality_summary=None):
    """Return a plain-language explanation string. Tries the Anthropic API, then
    a local Ollama server, then a deterministic template -- always returns
    something and never raises."""
    flags = flags or []
    facts = _facts(score, label, reliability, flags, quality_summary)

    text = _via_anthropic(facts)
    if text:
        return text

    text = _via_ollama(facts)
    if text:
        return text

    return _template(score, label, reliability, flags, quality_summary)


if __name__ == "__main__":
    print("--- via whichever tier is available (Anthropic key / local Ollama / template) ---")
    print(generate_report(0.67, "ALS-consistent pattern", 55,
                          flags=["congestion (rated 7/10)"],
                          quality_summary="low signal-to-noise (~8 dB)"))
