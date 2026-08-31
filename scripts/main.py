"""Run the SA AML simulation without opening the notebook.

Examples
--------
Run the full configured simulation:
    python scripts/main.py

Run a small smoke test:
    python scripts/main.py --n-accounts 500 --output-prefix smoke_test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_step(message: str) -> None:
    print(f"[main] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the South African AML simulation.")
    parser.add_argument("--n-accounts", type=int, help="Number of monitored receiver accounts.")
    parser.add_argument("--max-tx-per-account", type=int, help="Maximum generated transactions per account.")
    parser.add_argument("--output-prefix", type=str, help="Prefix for exported CSV/Excel files.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--start-date", type=str, help="Simulation start date, e.g. 2024-01-01.")
    parser.add_argument("--end-date", type=str, help="Simulation end date, e.g. 2026-03-31.")
    parser.add_argument("--no-files", action="store_true", help="Run simulation but do not export CSV/Excel files.")
    parser.add_argument("--no-excel", action="store_true", help="Skip the final Excel workbook export.")
    parser.add_argument("--export-engineered", action="store_true", help="Also export optional engineered features.")
    return parser.parse_args()


def apply_env_overrides(args: argparse.Namespace) -> None:
    if args.n_accounts is not None:
        os.environ["AML_SIM_N_ACCOUNTS"] = str(args.n_accounts)
    if args.max_tx_per_account is not None:
        os.environ["AML_SIM_MAX_TX_PER_ACCOUNT"] = str(args.max_tx_per_account)
    if args.output_prefix:
        os.environ["AML_SIM_OUTPUT_PREFIX"] = args.output_prefix
    if args.seed is not None:
        os.environ["AML_SIM_SEED"] = str(args.seed)
    if args.start_date:
        os.environ["AML_SIM_START_DATE"] = args.start_date
    if args.end_date:
        os.environ["AML_SIM_END_DATE"] = args.end_date
    if args.no_files:
        os.environ["AML_SIM_EXPORT_FILES"] = "0"
    if args.no_excel:
        os.environ["AML_SIM_EXPORT_EXCEL"] = "0"
    if args.export_engineered:
        os.environ["AML_SIM_EXPORT_ENGINEERED"] = "1"


def main() -> None:
    args = parse_args()
    from src.sa_aml_simulation.logging_utils import tee_run_log
    from src.sa_aml_simulation.paths import MAIN_SIMULATION_LOG_DIR

    run_name = args.output_prefix or os.getenv("AML_SIM_OUTPUT_PREFIX", "simulation")
    with tee_run_log(MAIN_SIMULATION_LOG_DIR, "main_simulation", run_name, sys.argv[1:]) as log_path:
        print_step("Reading command-line options.")
        print_step(f"Run log: {log_path.resolve()}")
        print_step("Applying command-line options to simulation configuration.")
        apply_env_overrides(args)

        print_step("Importing simulation pipeline.")
        from src.sa_aml_simulation.simulation_pipeline import run_simulation

        print_step("Starting simulation run.")
        result = run_simulation()
        print_step("Simulation script complete.")
        print_step(f"Rows generated: {len(result['df_final']):,}")
        print_step(f"Output prefix: {result['output_prefix']}")


if __name__ == "__main__":
    main()
