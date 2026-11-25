"""Create the publication architecture diagram for the SA AML simulator."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .paths import IMAGE_RESULTS_DIR, ensure_project_dirs


CURRENT_STATS = {
    "accounts": "35,000",
    "transactions": "1,441,479",
    "legitimate_accounts": "88.57%",
    "suspicious_accounts": "11.43%",
    "suspicious_transactions": "7.15%",
    "median": "27",
    "p95": "127",
    "maximum": "220",
    "cross_border": "15.06%",
    "fatf": "6.66%",
}


COLORS = {
    "blue": "#e8f1fb",
    "green": "#e8f5e9",
    "orange": "#fff4df",
    "red": "#fde9e7",
    "purple": "#f1eafe",
    "border": "#263238",
    "ink": "#162235",
    "muted": "#52606d",
    "green_text": "#147a35",
    "red_text": "#c62828",
    "purple_text": "#5420b4",
    "blue_text": "#1959a6",
}


def _rounded_box(ax, x, y, width, height, text, face, *, fontsize=10.2,
                 edge="#2f2f2f", weight="normal", radius=0.018,
                 text_color=None, linespacing=1.12, text_x=None, text_y=None):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        text_x if text_x is not None else x + width / 2,
        text_y if text_y is not None else y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color or COLORS["ink"],
        linespacing=linespacing,
        zorder=3,
    )


def _circle_icon(ax, x, y, letter, face):
    ax.add_patch(Circle((x, y), 0.19, facecolor=face, edgecolor="none", zorder=4))
    ax.text(x, y, letter, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", zorder=5)


def _arrow(ax, start, end, *, color="#202020", dashed=False, width=1.7,
           curve=0.0):
    style = "-|>"
    connection = f"arc3,rad={curve}"
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=width,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=connection,
        zorder=1,
    ))


def _label(ax, x, y, text, color, *, fontsize=9.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color)


def _metric_line(ax, x, y, label, value, *, fontsize=8.5):
    ax.text(x, y, label, ha="right", va="center", fontsize=fontsize,
            color=COLORS["muted"])
    ax.text(x + 0.02, y, value, ha="left", va="center", fontsize=fontsize,
            fontweight="bold", color=COLORS["ink"])


def generate_architecture_figure(output_path: str | Path | None = None,
                                 *, show: bool = False,
                                 stats: dict[str, str] | None = None) -> Path:
    """Save the detailed decision-tree architecture figure."""
    ensure_project_dirs()
    stats = {**CURRENT_STATS, **(stats or {})}
    output = Path(output_path) if output_path else IMAGE_RESULTS_DIR / "00_framework" / "simulation_architecture.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(18, 11), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(9.0, 10.72, "SA AML SIMULATION: DECISION TREE & VALIDATION LOGIC",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=COLORS["ink"])

    # Draw connectors first so every panel remains visually clean and readable.
    _arrow(ax, (9.0, 9.75), (9.0, 9.62))
    _arrow(ax, (9.0, 8.95), (9.0, 8.82))
    _arrow(ax, (9.0, 8.25), (9.0, 7.25))
    _arrow(ax, (9.0, 6.25), (9.0, 6.15))
    _arrow(ax, (9.0, 5.55), (9.0, 4.58))
    _arrow(ax, (9.0, 3.75), (9.0, 3.48))
    _arrow(ax, (3.2, 9.0), (5.32, 9.26), color=COLORS["blue_text"], dashed=True)
    _arrow(ax, (3.2, 5.95), (5.32, 5.88), color=COLORS["blue_text"], dashed=True)
    _arrow(ax, (5.4, 8.52), (4.28, 8.13), curve=0.03)
    _arrow(ax, (12.6, 8.52), (13.68, 8.13), curve=-0.03)
    _arrow(ax, (4.28, 7.35), (5.0, 7.1), curve=-0.03)
    _arrow(ax, (13.68, 7.35), (13.0, 7.1), curve=0.03)
    _arrow(ax, (6.1, 5.55), (4.45, 5.42), curve=0.03)
    _arrow(ax, (11.9, 5.55), (13.55, 5.42), curve=-0.03)
    _arrow(ax, (15.0, 7.65), (14.68, 7.82), color=COLORS["purple_text"], dashed=True)
    _arrow(ax, (16.25, 7.0), (16.25, 3.16), color=COLORS["purple_text"], dashed=True)
    _arrow(ax, (16.25, 3.16), (12.6, 3.16), color=COLORS["purple_text"], dashed=True)

    # Main process path.
    _rounded_box(ax, 6.35, 9.75, 5.3, 0.55,
                 "START: Simulation parameters loaded", "#e2f6d9",
                 fontsize=10.5, edge="#188038", weight="bold", text_x=9.15)
    _circle_icon(ax, 6.78, 10.02, ">", "#188038")

    _rounded_box(ax, 5.4, 8.95, 7.2, 0.65,
                 f"STEP 1: Generate {stats['accounts']} monitored\n"
                 "account profiles\n"
                 "Observation window: 2024-01-01 to 2026-03-31",
                 COLORS["orange"], fontsize=9.1, weight="bold", text_x=9.2,
                 text_y=9.29)
    _circle_icon(ax, 5.86, 9.28, "A", "#b26a16")

    _rounded_box(ax, 5.4, 8.25, 7.2, 0.55,
                 "DECISION 1: Is the account suspicious?", COLORS["orange"],
                 fontsize=10.5, weight="bold", text_x=9.2)
    _circle_icon(ax, 5.86, 8.52, "!", "#b26a16")

    _rounded_box(ax, 5.0, 6.25, 8.0, 0.95,
                 "STEP 2: Schedule transactions\n"
                 "Overdispersed Gamma-Poisson process\n"
                 f"{stats['transactions']} total transactions",
                 COLORS["orange"], fontsize=9.2, weight="bold", text_x=9.35,
                 text_y=6.82)
    _circle_icon(ax, 5.48, 6.74, "T", "#b26a16")
    ax.text(9.15, 6.39,
            f"Median/profile: {stats['median']}  |  p95/profile: {stats['p95']}  |  "
            f"Maximum/profile: {stats['maximum']}",
            ha="center", va="center", fontsize=8.4, color=COLORS["muted"], zorder=3)

    _rounded_box(ax, 5.4, 5.55, 7.2, 0.6,
                 "DECISION 2: Select behaviour scenario\n(Legitimate or suspicious)",
                 COLORS["orange"], fontsize=10.2, weight="bold", text_x=9.2)
    _circle_icon(ax, 5.86, 5.85, "B", "#b26a16")

    _rounded_box(ax, 5.2, 3.75, 7.6, 0.8,
                 "STEP 3: Payment type, amount,\n"
                 "counterparty and jurisdiction routing",
                 COLORS["orange"], fontsize=9.8, weight="bold", text_x=9.25,
                 text_y=4.25)
    _circle_icon(ax, 5.68, 4.15, "P", "#b26a16")
    ax.text(9.05, 3.91,
            f"Transaction suspicious share: {stats['suspicious_transactions']}  |  "
            f"Cross-border share: {stats['cross_border']}",
            ha="center", va="center", fontsize=8.5, color=COLORS["muted"], zorder=3)

    _rounded_box(ax, 5.45, 2.75, 7.1, 0.72,
                 "STEP 4: Public synthetic outputs\n"
                 "58 raw fields | profile and transaction records\n"
                 "Scenario metadata | validation benchmarks",
                 COLORS["purple"], fontsize=8.8, weight="bold",
                 edge=COLORS["purple_text"], text_x=9.2)
    _circle_icon(ax, 5.92, 3.11, "O", COLORS["purple_text"])

    # Account context and scenario weighting panels.
    _rounded_box(ax, 0.5, 8.05, 2.65, 1.85,
                 "ACCOUNT CONTEXT\n\nCustomer profile\nGeography\nEmployment\nBusiness status",
                 COLORS["blue"], fontsize=9.2, weight="bold",
                 edge=COLORS["blue_text"], text_x=1.85, text_y=8.88)
    _circle_icon(ax, 0.88, 9.57, "A", COLORS["blue_text"])

    _rounded_box(ax, 0.5, 4.9, 2.65, 2.15,
                 "SCENARIO WEIGHTING\n\nLegitimate:\nincome type\n\nSuspicious:\ntypology weight",
                 COLORS["blue"], fontsize=9.0, weight="bold",
                 edge=COLORS["blue_text"], text_x=1.85, text_y=5.88)
    _circle_icon(ax, 0.88, 6.63, "S", COLORS["blue_text"])

    _label(ax, 4.28, 8.43, f"{stats['legitimate_accounts']} Legitimate", COLORS["green_text"], fontsize=9.1)
    _label(ax, 13.68, 8.43, f"{stats['suspicious_accounts']} Suspicious", COLORS["red_text"], fontsize=9.1)

    _rounded_box(ax, 3.35, 7.35, 1.85, 0.78,
                 "LEGITIMATE\naccount profile\nEmployment | sector",
                 COLORS["green"], fontsize=7.45, edge="#188038", text_x=4.48)
    _circle_icon(ax, 3.68, 7.75, "L", "#188038")

    _rounded_box(ax, 12.8, 7.35, 1.85, 0.78,
                 "SUSPICIOUS\naccount profile\nPEP | sectoral context",
                 COLORS["red"], fontsize=7.15, edge=COLORS["red_text"], text_x=13.96)
    _circle_icon(ax, 13.12, 7.75, "!", COLORS["red_text"])

    _rounded_box(ax, 3.35, 4.65, 2.2, 0.78,
                 "11 NORMAL\nBEHAVIOURS\nSalary | grants\nInformal | retail",
                 COLORS["green"], fontsize=7.0, edge="#188038", text_x=4.72)
    _circle_icon(ax, 3.68, 5.04, "N", "#188038")

    _rounded_box(ax, 12.45, 4.65, 2.55, 0.78,
                 "22 AML\nTYPOLOGIES\nMule | structuring\nCross-border | shell\nCrypto | procurement",
                 COLORS["red"], fontsize=6.45, edge=COLORS["red_text"], text_x=14.16,
                 text_y=5.03, linespacing=1.02)
    _circle_icon(ax, 12.82, 5.04, "A", COLORS["red_text"])

    # Context inputs and output feedback.
    _rounded_box(ax, 15.0, 7.0, 2.5, 3.0,
                 f"CONTEXTUAL INPUTS\n\nCustomer profile\nSocioeconomic status\nGeography\nPayment context\nJurisdiction\nPEP / intermediary\nVirtual assets\n\nFATF exposure\nshare: {stats['fatf']}",
                 COLORS["purple"], fontsize=8.0, weight="bold",
                 edge=COLORS["purple_text"], text_x=16.35, text_y=8.47)
    _circle_icon(ax, 15.4, 9.6, "C", COLORS["purple_text"])

    # Legend and validation snapshot.
    ax.add_patch(FancyBboxPatch((2.05, 1.92), 13.9, 0.42,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                facecolor="white", edgecolor="#7b8794", linewidth=1.1, zorder=2))
    ax.text(2.45, 2.13, "Legend:", ha="left", va="center", fontsize=9.8,
            fontweight="bold", color=COLORS["ink"], zorder=3)
    legend = [("Start/End", "#e2f6d9", "#188038"),
              ("Process / Decision", COLORS["orange"], "#b26a16"),
              ("Legitimate Path", COLORS["green"], "#188038"),
              ("Suspicious Path", COLORS["red"], COLORS["red_text"]),
              ("Outputs / Validation", COLORS["purple"], COLORS["purple_text"])]
    x = 3.8
    for label, face, edge in legend:
        ax.add_patch(FancyBboxPatch((x, 2.05), 0.3, 0.18,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=face, edgecolor=edge, linewidth=1.0, zorder=2))
        ax.text(x + 0.4, 2.13, label, ha="left", va="center", fontsize=8.6,
                fontweight="bold", color=COLORS["ink"], zorder=3)
        x += 2.35

    ax.add_patch(FancyBboxPatch((0.5, 0.42), 17.0, 1.18,
                                boxstyle="round,pad=0.03,rounding_size=0.02",
                                facecolor="#123b63", edgecolor="#0b5fa5", linewidth=1.5, zorder=2))
    ax.text(0.9, 1.36, "Validation snapshot:", ha="left", va="center",
            fontsize=10.8, fontweight="bold", color="white", zorder=3)
    first = [
        f"{stats['accounts']}\nmonitored profiles",
        f"{stats['transactions']}\ntransactions",
        f"legitimate share\n{stats['legitimate_accounts']}",
        f"suspicious share\n{stats['suspicious_accounts']}",
        f"transaction suspicious\nshare: {stats['suspicious_transactions']}",
        f"FATF exposure\nshare: {stats['fatf']}",
    ]
    second = [
        "22 AML typologies",
        "11 normal behaviours",
        f"median/profile: {stats['median']}",
        f"p95/profile: {stats['p95']}",
        f"maximum/profile: {stats['maximum']}",
        f"cross-border share: {stats['cross_border']}",
    ]
    for idx, text in enumerate(first):
        x = 4.05 + idx * 2.55
        ax.text(x, 1.35, text, ha="center", va="center", fontsize=8.8,
                fontweight="bold", color="white", zorder=3)
        if idx < len(first) - 1:
            ax.plot([x + 1.27, x + 1.27], [1.1, 1.54], color="#78b4e5", linewidth=1.0, zorder=3)
    for idx, text in enumerate(second):
        x = 2.65 + idx * 2.55
        ax.text(x, 0.77, text, ha="center", va="center", fontsize=8.8,
                fontweight="bold", color="white", zorder=3)
        if idx < len(second) - 1:
            ax.plot([x + 1.27, x + 1.27], [0.58, 0.96], color="#78b4e5", linewidth=1.0, zorder=3)

    fig.savefig(output, dpi=300, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


if __name__ == "__main__":
    print(generate_architecture_figure())
