"""Create the publication architecture diagram for the SA AML simulator."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from .paths import IMAGE_RESULTS_DIR, ensure_project_dirs


CURRENT_STATS = {
    "accounts": "35,000",
    "transactions": "1,412,358",
    "legitimate_accounts": "88.57%",
    "suspicious_accounts": "11.43%",
    "suspicious_transactions": "2.16%",
    "median": "27",
    "p95": "126",
    "maximum": "220",
    "cross_border": "14.78%",
    "fatf": "5.91%",
}


COLORS = {
    "blue": "#dcecff",
    "green": "#def3dd",
    "orange": "#fff0d2",
    "red": "#f9d9dc",
    "purple": "#eadfff",
    "border": "#2f2f2f",
    "ink": "#182235",
    "muted": "#4b5563",
    "green_text": "#137333",
    "red_text": "#d93025",
    "purple_text": "#4b17a8",
    "blue_text": "#124b9a",
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
    )


def _circle_icon(ax, x, y, letter, face):
    ax.add_patch(Circle((x, y), 0.19, facecolor=face, edgecolor="none"))
    ax.text(x, y, letter, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")


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
                                 *, show: bool = False) -> Path:
    """Save the detailed decision-tree architecture figure."""
    ensure_project_dirs()
    output = Path(output_path) if output_path else IMAGE_RESULTS_DIR / "00_framework" / "simulation_architecture.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.8, 9.45), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 9.45)
    ax.axis("off")

    ax.text(8.4, 9.16, "SA AML SIMULATION: DECISION TREE & VALIDATION LOGIC",
            ha="center", va="center", fontsize=19, fontweight="bold",
            color=COLORS["ink"])

    # Start and central process path.
    _rounded_box(ax, 6.0, 8.53, 4.8, 0.43,
                 "START: Simulation Parameters Loaded", "#e2f6d9",
                 fontsize=10.0, edge="#198b2e", weight="bold", text_x=8.85)
    _circle_icon(ax, 6.42, 8.745, ">", "#159447")

    _rounded_box(ax, 6.0, 7.62, 4.8, 0.58,
                 f"STEP 1: Generate {CURRENT_STATS['accounts']} Receiver Accounts\n"
                 "Observation window: 2024-01-01 to 2026-03-31",
                 COLORS["orange"], fontsize=9.0, weight="bold", text_x=8.85)
    _circle_icon(ax, 6.42, 7.91, "A", "#a96718")

    _rounded_box(ax, 6.0, 6.94, 4.8, 0.48,
                 "DECISION 1: Is Suspicious Account?", COLORS["orange"],
                 fontsize=10.2, weight="bold", text_x=8.85)
    _circle_icon(ax, 6.42, 7.18, "!", "#a96718")

    _rounded_box(ax, 5.35, 5.0, 6.1, 0.9,
                 "STEP 2: Schedule Transactions\n"
                 "(Overdispersed Gamma-Poisson process)\n"
                 f"{CURRENT_STATS['transactions']} total transactions",
                 COLORS["orange"], fontsize=8.6, weight="bold", text_x=8.9,
                 text_y=5.64)
    _circle_icon(ax, 5.77, 5.67, "T", "#a96718")
    ax.text(8.45, 5.13,
            f"Median/account: {CURRENT_STATS['median']}  |  "
            f"p95/account: {CURRENT_STATS['p95']}  |  "
            f"Maximum/account: {CURRENT_STATS['maximum']}",
            ha="center", va="center", fontsize=7.9, color=COLORS["muted"])

    _rounded_box(ax, 6.0, 4.35, 4.8, 0.5,
                 "DECISION 2: Behaviour Scenario?\n(Legitimate vs Suspicious)",
                 COLORS["orange"], fontsize=10.0, weight="bold", text_x=8.85)
    _circle_icon(ax, 6.42, 4.60, "B", "#a96718")

    _rounded_box(ax, 6.48, 2.98, 4.15, 0.85,
                 "STEP 3: Payment Type, Amount,\nCounterparty & Jurisdiction Routing",
                 COLORS["orange"], fontsize=8.9, weight="bold", text_x=8.85,
                 text_y=3.58)
    _circle_icon(ax, 6.83, 3.58, "P", "#a96718")
    ax.text(8.55, 3.14,
            f"Transaction-level suspicious share: {CURRENT_STATS['suspicious_transactions']}\n"
            f"Cross-border transaction share: {CURRENT_STATS['cross_border']}",
            ha="center", va="center", fontsize=8.4, color=COLORS["muted"])

    _rounded_box(ax, 5.9, 2.18, 5.0, 0.62,
                 "STEP 4: Public Synthetic Outputs and Validation\n"
                 "58 raw fields | account and transaction outputs\n"
                 "scenario metadata | validation benchmarks",
                 COLORS["purple"], fontsize=8.0, weight="bold",
                 edge="#4b17a8", text_x=8.95)
    _circle_icon(ax, 6.32, 2.49, "O", "#5420b4")

    # Account decision panel.
    _rounded_box(ax, 0.45, 6.55, 2.55, 1.75,
                 "ACCOUNT\nDECISION\n\n- Customer Type\n- Geography\n- Employment\n- Business Status",
                 COLORS["blue"], fontsize=9.5, weight="bold",
                 edge="#124b9a")
    _circle_icon(ax, 0.83, 7.88, "A", "#124b9a")
    _arrow(ax, (3.0, 7.65), (5.95, 7.88), color="#1456b8", dashed=True)

    _label(ax, 4.3, 7.32, f"{CURRENT_STATS['legitimate_accounts']} Legitimate", COLORS["green_text"])
    _label(ax, 12.35, 7.32, f"{CURRENT_STATS['suspicious_accounts']} Suspicious", COLORS["red_text"])
    _arrow(ax, (6.0, 7.1), (4.42, 6.54), curve=0.03)
    _arrow(ax, (10.8, 7.1), (12.38, 6.54), curve=-0.03)

    _rounded_box(ax, 2.75, 5.78, 2.35, 0.65,
                 "Legitimate Account Profile:\n- Employment\n- Sector | Customer type",
                 COLORS["green"], fontsize=7.0, edge="#198b2e", text_x=4.18)
    _circle_icon(ax, 3.05, 6.10, "L", "#159447")

    _rounded_box(ax, 11.65, 5.78, 2.35, 0.65,
                 "Suspicious Account Profile:\n- PEP / intermediary\n- Sectoral and institutional context",
                 COLORS["red"], fontsize=6.7, edge="#d93025", text_x=13.1)
    _circle_icon(ax, 11.95, 6.10, "!", "#d93025")

    # Scenario selection panels.
    _rounded_box(ax, 0.45, 3.86, 2.55, 1.52,
                 "SCENARIO\nDECISION\n\nIf Legitimate:\nWeighted by Income Type\nIf Suspicious:\nWeighted by Typology Weight",
                 COLORS["blue"], fontsize=8.7, weight="bold",
                 edge="#124b9a")
    _circle_icon(ax, 0.83, 5.16, "S", "#124b9a")
    _arrow(ax, (3.0, 4.62), (5.95, 4.60), color="#1456b8", dashed=True)

    _label(ax, 4.65, 4.44, "11 Normal Behaviour Categories", COLORS["green_text"], fontsize=8.2)
    _label(ax, 12.55, 4.44, "22 AML Scenarios / Typologies", COLORS["red_text"], fontsize=8.2)
    _arrow(ax, (6.0, 4.60), (4.95, 4.10), curve=0.03)
    _arrow(ax, (10.8, 4.60), (11.75, 4.10), curve=-0.03)

    _rounded_box(ax, 2.75, 3.18, 2.95, 0.84,
                 "11 Normal Behaviour Categories:\nSalary | Grants | Informal Business\nRemittances | Retail Payments | ...",
                 COLORS["green"], fontsize=7.4, edge="#198b2e", text_x=4.42)
    _circle_icon(ax, 3.08, 3.60, "N", "#159447")

    _rounded_box(ax, 11.15, 3.18, 3.05, 0.84,
                 "22 AML Scenarios / Typologies:\nStructuring | Mule Accounts\nCross-Border | Procurement | Shell\nCrypto Off-Ramp | ...",
                 COLORS["red"], fontsize=7.0, edge="#d93025", text_x=12.85)
    _circle_icon(ax, 11.5, 3.60, "A", "#d93025")

    # Context input panel and connectors.
    _rounded_box(ax, 14.15, 6.1, 2.25, 2.2,
                 "CONTEXTUAL\nSIMULATION INPUTS\n\n- Customer profile\n- Socioeconomic status\n- Geography\n- Payment context\n- Jurisdiction\n- PEP / intermediary context\n- Virtual-asset exposure\n\nFATF contextual\nexposure share: 5.91%",
                 COLORS["purple"], fontsize=7.5, weight="bold",
                 edge="#4b17a8", text_x=15.35)
    _circle_icon(ax, 14.42, 7.82, "C", "#5420b4")
    _arrow(ax, (14.15, 6.75), (13.98, 6.18), color="#5420b4", dashed=True)
    _arrow(ax, (15.05, 6.1), (15.05, 2.55), color="#5420b4", dashed=True)
    _arrow(ax, (15.05, 2.55), (10.92, 2.55), color="#5420b4", dashed=True)

    # Central arrows.
    _arrow(ax, (8.4, 8.53), (8.4, 8.2))
    _arrow(ax, (8.4, 7.62), (8.4, 7.43))
    _arrow(ax, (8.4, 6.94), (8.4, 5.92))
    _arrow(ax, (8.4, 5.0), (8.4, 4.85))
    _arrow(ax, (8.4, 4.35), (8.4, 3.64))
    _arrow(ax, (8.4, 3.02), (8.4, 2.8))

    # Legend and validation snapshot.
    ax.add_patch(FancyBboxPatch((2.2, 1.72), 12.4, 0.38,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                facecolor="white", edgecolor="#9aa0a6", linewidth=1.0))
    ax.text(2.65, 1.91, "Legend:", ha="left", va="center", fontsize=8.8,
            fontweight="bold", color=COLORS["ink"])
    legend = [("Start/End", "#dff3df", "#198b2e"),
              ("Process / Decision", COLORS["orange"], "#a96718"),
              ("Legitimate Path", COLORS["green"], "#198b2e"),
              ("Suspicious Path", COLORS["red"], "#d93025"),
              ("Outputs & Validation", COLORS["purple"], "#5420b4")]
    x = 3.95
    for label, face, edge in legend:
        ax.add_patch(FancyBboxPatch((x, 1.83), 0.28, 0.16,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + 0.38, 1.91, label, ha="left", va="center", fontsize=7.9,
                color=COLORS["ink"])
        x += 2.1

    ax.add_patch(FancyBboxPatch((0.45, 0.45), 15.9, 1.0,
                                boxstyle="round,pad=0.03,rounding_size=0.02",
                                facecolor="#eef7ff", edgecolor="#1473e6", linewidth=1.2))
    ax.text(0.85, 1.2, "Validation snapshot:", ha="left", va="center",
            fontsize=9.2, fontweight="bold", color="#0a56b5")
    first = [
        f"{CURRENT_STATS['accounts']}\nreceiver accounts",
        f"{CURRENT_STATS['transactions']}\ntransactions",
        f"legitimate-account\nshare: {CURRENT_STATS['legitimate_accounts']}",
        f"suspicious-account\nshare: {CURRENT_STATS['suspicious_accounts']}",
        f"transaction-level\nsuspicious share: {CURRENT_STATS['suspicious_transactions']}",
        f"FATF contextual\nexposure share: {CURRENT_STATS['fatf']}",
    ]
    second = [
        "22 AML scenarios",
        "11 normal behaviour categories",
        f"median/account: {CURRENT_STATS['median']}",
        f"p95/account: {CURRENT_STATS['p95']}",
        f"maximum/account: {CURRENT_STATS['maximum']}",
        f"cross-border share: {CURRENT_STATS['cross_border']}",
    ]
    for idx, text in enumerate(first):
        x = 3.7 + idx * 2.15
        ax.text(x, 1.2, text, ha="center", va="center", fontsize=7.1,
                color=COLORS["ink"])
        if idx < len(first) - 1:
            ax.plot([x + 1.08, x + 1.08], [0.98, 1.4], color="#a9c7e9", linewidth=0.8)
    for idx, text in enumerate(second):
        x = 2.35 + idx * 2.45
        ax.text(x, 0.73, text, ha="center", va="center", fontsize=7.1,
                color=COLORS["ink"])
        if idx < len(second) - 1:
            ax.plot([x + 1.22, x + 1.22], [0.55, 0.91], color="#a9c7e9", linewidth=0.8)

    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


if __name__ == "__main__":
    print(generate_architecture_figure())
