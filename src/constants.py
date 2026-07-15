"""Shared text constants used by both app.py and export_result.py.

Split out to avoid a circular import (app.py needs build_result_pdf from
export_result.py, which needs this disclaimer text).
"""

DISCLAIMER = (
    "This score comes from an acoustic pattern-matching model, not a diagnosis. "
    "Things like a raspy voice, intoxication, Parkinson's, or a stroke can produce similar "
    "acoustic markers (jitter, shimmer, low HNR), and this model has no way to tell them apart. "
    "If you get a high score, talk to a neurologist. Don't treat this as a result on its own."
)
