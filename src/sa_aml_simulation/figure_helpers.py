"""Figure output and polishing helpers for the SA AML simulation notebook.

This module keeps plot-folder routing, split-panel export and label-cleaning
logic outside the notebook so result sections stay easier to explain.
"""

import os
import calendar
import sys
import random
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd


# Matplotlib is only used for local plots; feel free to comment out if running headless.
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from .paths import IMAGE_RESULTS_DIR, ensure_project_dirs

ensure_project_dirs()


# Notebook plots are simulation validation assets, so save them by default.
# Override with AML_SIM_EXPORT_INSIGHTS=0 if you only want to display plots.
# AML_SIM_PLOTS_DIR controls the root plot folder; category subfolders are
# created automatically below it.
PLOTS_DIR = Path(os.getenv("AML_SIM_PLOTS_DIR", str(IMAGE_RESULTS_DIR)))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_INSIGHT_PLOTS = os.getenv("AML_SIM_EXPORT_INSIGHTS", "1") != "0"
publication_figures_saved_to = []

IMAGE_CATEGORY_DIRS = {
    "framework": "00_framework",
    "eda_overview": "01_eda_overview",
    "temporal_sequence": "02_temporal_sequence",
    "payments_amounts": "03_payments_amounts",
    "aml_typologies": "04_aml_typologies",
    "normal_behaviour": "05_normal_behaviour",
    "customer_profiles": "06_customer_profiles",
    "geography": "07_geography",
    "informality": "08_informality",
    "context_diagnostics": "09_context_diagnostics",
}

PLOT_FILENAME_CATEGORY = {
    "00_dataset_overview_counts.png": "eda_overview",
    "eda_transactions_per_monitored_account.png": "eda_overview",
    "eda_amount_distribution_by_laundering_flag.png": "eda_overview",
    "eda_class_distribution_pie.png": "eda_overview",
    "eda_amount_scatter_by_label.png": "eda_overview",
    "eda_monthly_transactions_by_label.png": "temporal_sequence",
    "eda_weekly_transactions_by_label.png": "temporal_sequence",
    "eda_daily_transactions_by_label.png": "temporal_sequence",
    "01_label_imbalance_transaction_and_account.png": "eda_overview",
    "02_receiver_account_sequence_lengths.png": "temporal_sequence",
    "03_active_days_vs_account_transaction_count.png": "temporal_sequence",
    "04_monthly_transaction_volume_by_label.png": "temporal_sequence",
    "05_monthly_context_shares.png": "temporal_sequence",
    "18_sequential_timing_and_rapid_repeat.png": "temporal_sequence",
    "19_account_transaction_density_by_label.png": "temporal_sequence",
    "00_transaction_account_boxplots.png": "temporal_sequence",
    "06_amount_and_payment_mix_by_label.png": "payments_amounts",
    "eda_payment_type_summary_heatmap.png": "payments_amounts",
    "eda_monthly_alerts_by_payment_type.png": "payments_amounts",
    "laundering_alerts_per_account.png": "aml_typologies",
    "07_aml_scenario_transaction_mix.png": "aml_typologies",
    "08_aml_scenario_fingerprint_heatmap.png": "aml_typologies",
    "31_network_ponzi_scheme_collections.png": "aml_typologies",
    "32_network_mule_pass_through.png": "aml_typologies",
    "33_network_cross_border_layering.png": "aml_typologies",
    "09_normal_behaviour_mix.png": "normal_behaviour",
    "10_normal_behaviour_fingerprint_heatmap.png": "normal_behaviour",
    "28_social_grant_type_distribution.png": "normal_behaviour",
    "29_social_grant_amount_by_type.png": "normal_behaviour",
    "30_social_grant_payment_day_pattern.png": "normal_behaviour",
    "11_employment_status_accounts_and_aml_rate.png": "customer_profiles",
    "12_sector_context_aml_rate.png": "customer_profiles",
    "13_age_gender_aml_rate_heatmap.png": "customer_profiles",
    "14_education_level_aml_rate.png": "customer_profiles",
    "21_income_band_accounts_and_aml_rate.png": "customer_profiles",
    "22_income_band_transaction_mix_by_label.png": "customer_profiles",
    "23_employment_sector_accounts_and_aml_rate.png": "customer_profiles",
    "24_employment_sector_transaction_mix_by_label.png": "customer_profiles",
    "25_customer_type_accounts_and_aml_rate.png": "customer_profiles",
    "26_risk_indicator_share_by_account_label.png": "customer_profiles",
    "27_income_employment_aml_rate_heatmap.png": "customer_profiles",
    "15_nationality_cross_border_bubble.png": "geography",
    "16_cross_border_country_exposure.png": "geography",
    "16b_sa_province_aml_exposure_map.png": "geography",
    "16c_cross_border_country_risk_map.png": "geography",
    "17_informal_business_registration_cash_share.png": "informality",
    "20_context_feature_correlation_heatmap.png": "context_diagnostics",
}

for category_dir in IMAGE_CATEGORY_DIRS.values():
    (PLOTS_DIR / category_dir).mkdir(parents=True, exist_ok=True)


SPLIT_FIGURE_OUTPUTS = {
    "00_dataset_overview_counts.png": [
        (0, "00a_transaction_row_counts_by_label.png"),
        (1, "00b_receiver_account_counts_by_label.png"),
    ],
    "00_transaction_account_boxplots.png": [
        (0, "00c_transaction_amount_boxplot_by_label.png"),
        (1, "00d_account_sequence_length_boxplot_by_label.png"),
    ],
    "01_label_imbalance_transaction_and_account.png": [
        (0, "01a_transaction_label_imbalance.png"),
        (1, "01b_receiver_account_label_imbalance.png"),
    ],
    "eda_monthly_transactions_by_label.png": [
        (0, "eda_monthly_transactions_non_laundering.png"),
        (1, "eda_monthly_transactions_laundering.png"),
    ],
    "eda_weekly_transactions_by_label.png": [
        (0, "eda_weekly_transactions_non_laundering.png"),
        (1, "eda_weekly_transactions_laundering.png"),
    ],
    "eda_daily_transactions_by_label.png": [
        (0, "eda_daily_transactions_non_laundering.png"),
        (1, "eda_daily_transactions_laundering.png"),
    ],
    "06_amount_and_payment_mix_by_label.png": [
        (0, "06a_amount_distribution_by_label.png"),
        (1, "06b_payment_rail_mix_by_label.png"),
    ],
    "11_employment_status_accounts_and_aml_rate.png": [
        (0, "11a_employment_status_account_count.png"),
        (1, "11b_employment_status_aml_rate.png"),
    ],
    "16_cross_border_country_exposure.png": [
        (0, "16a_cross_border_country_transaction_count.png"),
        (1, "16d_cross_border_country_suspicious_rate.png"),
    ],
    "17_informal_business_registration_cash_share.png": [
        (0, "17a_business_registration_account_count.png"),
        (1, "17b_business_registration_cash_share.png"),
    ],
    "18_sequential_timing_and_rapid_repeat.png": [
        (0, "18a_time_since_previous_transaction_boxplot.png"),
        (1, "18b_rapid_repeat_share_by_aml_scenario.png"),
    ],
    "21_income_band_accounts_and_aml_rate.png": [
        (0, "21a_income_band_account_count.png"),
        (1, "21b_income_band_aml_rate.png"),
    ],
    "23_employment_sector_accounts_and_aml_rate.png": [
        (0, "23a_employment_sector_account_count.png"),
        (1, "23b_employment_sector_aml_rate.png"),
    ],
    "25_customer_type_accounts_and_aml_rate.png": [
        (0, "25a_customer_type_account_count.png"),
        (1, "25b_customer_type_aml_rate.png"),
    ],
}
SAVE_COMBINED_PANEL_PLOTS = os.getenv("AML_SIM_SAVE_COMBINED_PLOTS", "0") != "0"


def _save_single_axis_panel(fig, ax, output_path, dpi=180):
    other_axes = [other_ax for other_ax in fig.axes if other_ax is not ax]
    original_visibility = [other_ax.get_visible() for other_ax in other_axes]
    for other_ax in other_axes:
        other_ax.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    # Expand slightly so titles, tick labels and bar annotations are not clipped.
    bbox = bbox.expanded(1.08, 1.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox)
    for other_ax, visible in zip(other_axes, original_visibility):
        other_ax.set_visible(visible)


def plot_output_path(filename):
    path = Path(filename)
    if path.parent != Path("."):
        return PLOTS_DIR / path
    category_key = PLOT_FILENAME_CATEGORY.get(path.name, "context_diagnostics")
    return PLOTS_DIR / IMAGE_CATEGORY_DIRS[category_key] / path.name


def _set_count_axis(ax, axis="x"):
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    target.set_major_locator(MaxNLocator(nbins=6, integer=True))


def _set_percent_axis(ax, axis="x", scale=1.0):
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(PercentFormatter(xmax=scale, decimals=0))
    target.set_major_locator(MaxNLocator(nbins=6))


def _clean_axis_labels(ax):
    replacements = {
        "Percent of transactions": "Share of transactions (%)",
        "Percent of accounts": "Share of accounts (%)",
        "Percent within label": "Transactions within label (%)",
        "Suspicious account rate": "Suspicious account rate (%)",
        "Suspicious transaction rate": "Suspicious transaction rate (%)",
        "Average cross-border share": "Average cross-border share (%)",
        "Cross-border share": "Cross-border share (%)",
        "Rapid repeat share": "Rapid repeat share (%)",
        "Cash share": "Cash share (%)",
        "Share of accounts": "Share of accounts (%)",
        "Share": "Share (%)",
        "Rate": "Suspicious account rate (%)",
    }
    for getter, setter in ((ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)):
        label = getter()
        if not label:
            continue
        cleaned = replacements.get(label, label)
        if cleaned != label:
            setter(cleaned)


def _humanise_label(text):
    return str(text).replace("_", " ").replace("  ", " ").strip()


def _clean_tick_and_legend_labels(ax):
    for axis_name in ("x", "y"):
        labels = ax.get_xticklabels() if axis_name == "x" else ax.get_yticklabels()
        texts = [label.get_text() for label in labels]
        cleaned = [_humanise_label(text) for text in texts]
        if any(new_text != old_text for old_text, new_text in zip(texts, cleaned)):
            rotations = [label.get_rotation() for label in labels]
            ticks = ax.get_xticks() if axis_name == "x" else ax.get_yticks()
            if axis_name == "x":
                ax.set_xticks(ticks)
                ax.set_xticklabels(cleaned)
                for label, rotation in zip(ax.get_xticklabels(), rotations):
                    label.set_rotation(rotation)
            else:
                ax.set_yticks(ticks)
                ax.set_yticklabels(cleaned)
                for label, rotation in zip(ax.get_yticklabels(), rotations):
                    label.set_rotation(rotation)
    legend = ax.get_legend()
    if legend is not None:
        for text_obj in legend.get_texts():
            text = text_obj.get_text()
            cleaned = _humanise_label(text)
            if cleaned and cleaned != text:
                text_obj.set_text(cleaned)


def _bar_rectangles(ax):
    rects = []
    for patch in ax.patches:
        if not isinstance(patch, Rectangle):
            continue
        width = float(patch.get_width())
        height = float(patch.get_height())
        if not np.isfinite(width) or not np.isfinite(height):
            continue
        if width <= 0 or height <= 0:
            continue
        rects.append(patch)
    return rects


def _infer_bar_orientation(ax, rects):
    x_label = ax.get_xlabel().strip().lower()
    y_label = ax.get_ylabel().strip().lower()
    axis_words = ["account", "transaction", "alert", "count", "rate", "share", "percent", "%"]
    if any(word in x_label for word in axis_words):
        return "h"
    if any(word in y_label for word in axis_words):
        return "v"
    widths = np.array([patch.get_width() for patch in rects], dtype=float)
    heights = np.array([patch.get_height() for patch in rects], dtype=float)
    return "h" if np.nanmedian(widths) > np.nanmedian(heights) else "v"


def _axis_is_percent(ax, orientation, max_value):
    axis_label = ax.get_xlabel() if orientation == "h" else ax.get_ylabel()
    all_text = " ".join([axis_label, ax.get_ylabel(), ax.get_xlabel(), ax.get_title()]).lower()
    has_percent_language = any(token in all_text for token in ["%", "percent", "rate", "share"])
    if not has_percent_language:
        return False, 1.0
    scale = 100.0 if max_value > 1.5 else 1.0
    return True, scale


def _add_bar_labels(ax, rects, orientation, percent=False, scale=1.0, decimals=1):
    values = [patch.get_width() if orientation == "h" else patch.get_height() for patch in rects]
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return
    max_value = max([abs(value) for value in values] + [1.0])
    pad = max_value * 0.015

    def fmt(value):
        if percent:
            return f"{(float(value) / scale) * 100:.{decimals}f}%"
        return f"{float(value):,.0f}"

    for patch in rects:
        value = patch.get_width() if orientation == "h" else patch.get_height()
        if not np.isfinite(value):
            continue
        if orientation == "h":
            x = patch.get_x() + patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(x + pad, y, fmt(value), va="center", ha="left", fontsize=7, color="#222222")
        else:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + patch.get_height()
            ax.text(x, y + pad, fmt(value), va="bottom", ha="center", fontsize=7, color="#222222")

    if orientation == "h":
        left, right = ax.get_xlim()
        desired_right = max(values) + max_value * 0.20
        if right < desired_right:
            ax.set_xlim(left, desired_right)
    else:
        bottom, top = ax.get_ylim()
        desired_top = max(values) + max_value * 0.20
        if top < desired_top:
            ax.set_ylim(bottom, desired_top)


def _axis_has_manual_text(ax):
    """Return True when an axis already contains intentional text annotations."""
    return any(text.get_visible() and text.get_text().strip() for text in ax.texts)


def polish_plot_figure(fig=None):
    fig = fig or plt.gcf()
    for ax in fig.axes:
        _clean_axis_labels(ax)
        _clean_tick_and_legend_labels(ax)
        rects = _bar_rectangles(ax)
        # Avoid unreadable labels on histograms or highly stacked charts.
        if 1 <= len(rects) <= 40 and not _axis_has_manual_text(ax):
            orientation = _infer_bar_orientation(ax, rects)
            max_value = max(
                [patch.get_width() if orientation == "h" else patch.get_height() for patch in rects]
                + [1.0]
            )
            is_percent, scale = _axis_is_percent(ax, orientation, max_value)
            if is_percent:
                _set_percent_axis(ax, axis="x" if orientation == "h" else "y", scale=scale)
            else:
                _set_count_axis(ax, axis="x" if orientation == "h" else "y")
            _add_bar_labels(ax, rects, orientation, percent=is_percent, scale=scale)
        else:
            x_label = ax.get_xlabel().lower()
            y_label = ax.get_ylabel().lower()
            if "(%)" in x_label or "rate" in x_label or "share" in x_label:
                _set_percent_axis(ax, axis="x", scale=1.0)
            if "(%)" in y_label or "rate" in y_label or "share" in y_label:
                _set_percent_axis(ax, axis="y", scale=1.0)
            if any(token in y_label for token in ["transactions", "accounts", "alerts", "count"]):
                _set_count_axis(ax, axis="y")


def finish_plot(filename=None, dpi=180):
    fig = plt.gcf()
    polish_plot_figure(fig)
    plt.tight_layout()
    if filename and EXPORT_INSIGHT_PLOTS:
        output_path = plot_output_path(filename)
        split_outputs = SPLIT_FIGURE_OUTPUTS.get(Path(filename).name, [])
        if split_outputs:
            for axis_index, panel_filename in split_outputs:
                if axis_index < len(fig.axes):
                    panel_path = output_path.parent / panel_filename
                    _save_single_axis_panel(fig, fig.axes[axis_index], panel_path, dpi=dpi)
                    publication_figures_saved_to.append(str(panel_path))
            if SAVE_COMBINED_PANEL_PLOTS:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
                publication_figures_saved_to.append(str(output_path))
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            publication_figures_saved_to.append(str(output_path))
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except Exception:
        shell_name = ""
    default_show = "1" if shell_name == "ZMQInteractiveShell" else "0"
    if os.getenv("AML_SIM_SHOW_PLOTS", default_show) != "0":
        plt.show()
    else:
        plt.close(fig)
