"""Regenerate simulation validation figures and insight tables from saved data.

Examples
--------
Regenerate figures from the default internal audit source:
    python scripts/save_figures.py

Use a specific audit CSV:
    python scripts/save_figures.py --input-csv results/insights/feature_audit/SA_SAML_D_full_audit.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("AML_SIM_SHOW_PLOTS", "0")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_step(message: str) -> None:
    print(f"[figures] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate SA AML simulation figures and insight tables.")
    parser.add_argument("--input-csv", type=str, help="Path to the full raw simulation CSV.")
    parser.add_argument("--output-prefix", type=str, help="Output prefix used for insight table workbook names.")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip older exploratory plots.")
    parser.add_argument("--skip-extended", action="store_true", help="Skip extended customer-profile plots.")
    parser.add_argument("--no-tables", action="store_true", help="Skip insight table CSV/Excel export.")
    parser.add_argument("--no-plots", action="store_true", help="Display/compute tables but do not save plots.")
    return parser.parse_args()


def apply_env_overrides(args: argparse.Namespace) -> None:
    if args.output_prefix:
        os.environ["AML_SIM_OUTPUT_PREFIX"] = args.output_prefix
    if args.no_tables:
        os.environ["AML_SIM_EXPORT_TABLES"] = "0"
    if args.no_plots:
        os.environ["AML_SIM_EXPORT_INSIGHTS"] = "0"


def main() -> None:
    args = parse_args()
    from src.sa_aml_simulation.logging_utils import tee_run_log
    from src.sa_aml_simulation.paths import FIGURE_LOG_DIR

    run_name = args.output_prefix or os.getenv("AML_SIM_OUTPUT_PREFIX")
    if not run_name and args.input_csv:
        run_name = Path(args.input_csv).stem
    run_name = run_name or "figures"

    with tee_run_log(FIGURE_LOG_DIR, "figures", run_name, sys.argv[1:]) as log_path:
        print_step("Reading command-line options.")
        print_step(f"Run log: {log_path.resolve()}")
        print_step("Applying command-line options to figure configuration.")
        apply_env_overrides(args)

        print_step("Importing figure pipeline.")
        from src.sa_aml_simulation.figure_pipeline import generate_figures

        print_step("Starting figure and insight-table generation.")
        result = generate_figures(
            input_csv=args.input_csv,
            include_legacy=not args.skip_legacy,
            include_extended=not args.skip_extended,
        )
        print_step("Figure script complete.")
        print_step(f"Figures tracked: {len(result['publication_figures_saved_to'])}")
        print_step(f"Insight tables: {len(result['publication_tables'])}")


if __name__ == "__main__":
    main()
