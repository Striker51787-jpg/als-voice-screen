"""Build a one-page PDF summary of a screening result, for a user to download
as a leave-behind. Pure matplotlib (already a dependency) -- no new package.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import DISCLAIMER


def build_result_pdf(prob, label, reliability, reasons, model_name) -> bytes:
    """Returns PDF bytes for a single-page result summary."""
    reasons = reasons or []
    fig = plt.figure(figsize=(6.5, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.96, "ALS Voice Screen -- Result Summary", ha="center",
              fontsize=16, fontweight="bold")
    fig.text(0.5, 0.925, "Research prototype -- not a diagnostic tool", ha="center",
              fontsize=10, style="italic", color="#555555")

    # Score
    fig.text(0.5, 0.85, f"{prob:.2f}", ha="center", fontsize=48, fontweight="bold",
              color="#1d4ed8")
    fig.text(0.5, 0.79, label, ha="center", fontsize=13)
    fig.text(0.5, 0.765, f"Model: {model_name}", ha="center", fontsize=9, color="#777777")

    # Zoned gauge, same concept as the app's render_gauge().
    gauge_ax = fig.add_axes([0.12, 0.71, 0.76, 0.035])
    gauge_ax.set_xlim(0, 1)
    gauge_ax.set_ylim(0, 1)
    gauge_ax.axvspan(0.0, 0.4, color="#16a34a")
    gauge_ax.axvspan(0.4, 0.6, color="#e5e7eb")
    gauge_ax.axvspan(0.6, 1.0, color="#dc2626")
    gauge_ax.axvline(min(max(prob, 0.0), 1.0), color="black", linewidth=3)
    gauge_ax.set_xticks([])
    gauge_ax.set_yticks([])
    for spine in gauge_ax.spines.values():
        spine.set_visible(False)
    fig.text(0.12, 0.695, "Control-like", fontsize=8, color="#555555")
    fig.text(0.5, 0.695, "Uncertain", fontsize=8, color="#555555", ha="center")
    fig.text(0.88, 0.695, "ALS-consistent", fontsize=8, color="#555555", ha="right")

    # Reliability
    fig.text(0.08, 0.64, "Session reliability estimate", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.61, f"{reliability}%", fontsize=20, color="#1d4ed8")

    y = 0.56
    if reasons:
        fig.text(0.08, y, "Factors that may have affected this session:", fontsize=10,
                  fontweight="bold")
        y -= 0.03
        for r in reasons:
            fig.text(0.1, y, f"- {r}", fontsize=9, wrap=True)
            y -= 0.03
    else:
        fig.text(0.08, y, "No factors flagged in the pre-screen.", fontsize=9, color="#555555")
        y -= 0.03

    # Disclaimer -- reused verbatim from the app, not restated/paraphrased.
    y -= 0.03
    fig.text(0.08, y, "Important context", fontsize=10, fontweight="bold")
    y -= 0.03
    fig.text(0.08, y, DISCLAIMER, fontsize=9, wrap=True, va="top",
              bbox=dict(boxstyle="round", facecolor="#f3f4f6", edgecolor="#d1d5db"))

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()
