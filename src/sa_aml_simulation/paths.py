"""Project paths and output-folder setup for the SA AML simulation."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root whether the notebook runs from root or notebooks/."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "sa_aml_simulation").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / os.getenv("AML_SIM_RESULTS_DIR", "results")
SIMULATED_DATA_DIR = Path(os.getenv("AML_SIM_DATA_DIR", str(RESULTS_DIR / "simulated_data")))
IMAGE_RESULTS_DIR = Path(os.getenv("AML_SIM_PLOTS_DIR", str(RESULTS_DIR / "images")))
INSIGHTS_DIR = Path(os.getenv("AML_SIM_INSIGHTS_DIR", str(RESULTS_DIR / "insights")))
TABLE_RESULTS_DIR = Path(os.getenv("AML_SIM_TABLES_DIR", str(INSIGHTS_DIR)))
FEATURE_AUDIT_DIR = Path(os.getenv("AML_SIM_FEATURE_AUDIT_DIR", str(INSIGHTS_DIR / "feature_audit")))
FEATURE_AUDIT_FIGURE_DIR = IMAGE_RESULTS_DIR / "09_context_diagnostics" / "feature_audit"
LOGS_DIR = PROJECT_ROOT / os.getenv("AML_SIM_LOGS_DIR", "logs")

# The public dataset is intentionally flat: one clean CSV and one schema file.
FULL_SAML_DIR = SIMULATED_DATA_DIR
FINAL_RESULTS_DIR = INSIGHTS_DIR
TABLE_CSV_DIR = TABLE_RESULTS_DIR
MAIN_SIMULATION_LOG_DIR = LOGS_DIR / "main_simulation"
FIGURE_LOG_DIR = LOGS_DIR / "figures"


def ensure_project_dirs() -> None:
    """Create the standard folder layout used by the notebook."""
    for path in [
        NOTEBOOKS_DIR,
        SRC_DIR,
        SCRIPTS_DIR,
        RESULTS_DIR,
        SIMULATED_DATA_DIR,
        IMAGE_RESULTS_DIR,
        INSIGHTS_DIR,
        LOGS_DIR,
        MAIN_SIMULATION_LOG_DIR,
        FIGURE_LOG_DIR,
        TABLE_RESULTS_DIR,
        FEATURE_AUDIT_DIR,
        FEATURE_AUDIT_FIGURE_DIR,
        FULL_SAML_DIR,
        FINAL_RESULTS_DIR,
        TABLE_CSV_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path | str) -> str:
    """Format a path relative to the project root when possible."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
