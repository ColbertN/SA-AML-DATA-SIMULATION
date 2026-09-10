"""Scriptable figure and insight-table pipeline for the SA AML simulation.

Use this module when simulation outputs already exist and you only want to
regenerate validation plots and insight tables.
"""

from __future__ import annotations

import calendar
import os
import random
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

try:
    from IPython.display import display
except Exception:  # pragma: no cover - plain script fallback
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

from src.sa_aml_simulation.config import (
    DEFAULT_AML_ACCOUNT_SHARE_TARGET,
    DEFAULT_MAX_TX_PER_ACCOUNT,
    DEFAULT_N_RECEIVER_ACCOUNTS,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_END,
    DEFAULT_SIMULATION_START,
    DEFAULT_SUSPICIOUS_TX_PROB_SCALE,
)
from src.sa_aml_simulation.paths import (
    FEATURE_AUDIT_DIR,
    FULL_SAML_DIR,
    SIMULATED_DATA_DIR,
    TABLE_CSV_DIR,
    ensure_project_dirs,
)
from src.sa_aml_simulation.figure_helpers import (
    PLOTS_DIR,
    EXPORT_INSIGHT_PLOTS,
    publication_figures_saved_to,
    finish_plot,
    _set_count_axis,
    _set_percent_axis,
)


def _default_input_csv(output_prefix: str | None = None) -> Path:
    prefix = output_prefix or os.getenv("AML_SIM_OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX)
    preferred_path = FEATURE_AUDIT_DIR / f"{prefix}_full_audit.csv"
    if preferred_path.exists():
        return preferred_path
    return preferred_path


def _read_simulation_csv(input_csv: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(input_csv) if input_csv else _default_input_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Simulation CSV not found: {csv_path}. Run scripts/main.py first, "
            "or pass --input-csv to scripts/save_figures.py."
        )
    print(f"Loading simulation data from {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def _reconstruct_receiver_accounts(df_final: pd.DataFrame) -> pd.DataFrame:
    account_rows = df_final.sort_values(["Account_id", "Transaction_id"]).drop_duplicates("Account_id").copy()
    if "Account_is_laundering" in df_final.columns:
        account_labels = (
            pd.to_numeric(df_final["Account_is_laundering"], errors="coerce")
            .fillna(0)
            .astype(int)
            .groupby(df_final["Account_id"])
            .max()
        )
        account_rows["Account_is_laundering"] = account_rows["Account_id"].map(account_labels).fillna(0).astype(int)
    # Account_id is the monitored receiver-account key. The first transaction
    # row may have a different Receiver_account when the account sends funds,
    # so profile joins must use the monitored account itself.
    account_rows["Receiver_account"] = account_rows["Account_id"]
    account_rows = account_rows.drop(columns=["Account_id"], errors="ignore")
    return account_rows.reset_index(drop=True)


def generate_figures(input_csv: str | Path | None = None, include_legacy: bool = True, include_extended: bool = True) -> dict:
    """Regenerate validation figures and insight tables from saved simulation data."""
    def _progress(message: str) -> None:
        print(f"[figures] {message}", flush=True)

    _progress("Preparing folders and reading figure configuration.")
    ensure_project_dirs()

    # Context constants and scenario catalogues from the simulation definition.
    # ============================================================
    # PARAMETERS AND CONTEXT TABLES
    # ============================================================

    # The full-scale default uses more monitored receiver accounts and covers
    # 2024-01-01 through 2026-03-31. Override these with environment variables
    # when running quick smoke tests or larger experiments.
    N_RECEIVER_ACCOUNTS = int(os.getenv("AML_SIM_N_ACCOUNTS", str(DEFAULT_N_RECEIVER_ACCOUNTS)))
    AML_ACCOUNT_SHARE_TARGET = float(os.getenv("AML_SIM_ACCOUNT_SHARE", str(DEFAULT_AML_ACCOUNT_SHARE_TARGET)))
    SIMULATION_START = os.getenv("AML_SIM_START_DATE", DEFAULT_SIMULATION_START)
    SIMULATION_END = os.getenv("AML_SIM_END_DATE", DEFAULT_SIMULATION_END)
    START_DATE = datetime.fromisoformat(SIMULATION_START)
    if os.getenv("AML_SIM_DAYS"):
        DAYS = int(os.getenv("AML_SIM_DAYS"))
        END_DATE = START_DATE + timedelta(days=DAYS - 1)
    else:
        END_DATE = datetime.fromisoformat(SIMULATION_END)
        DAYS = (END_DATE.date() - START_DATE.date()).days + 1
    END_DATETIME = END_DATE.replace(hour=23, minute=59, second=59)
    SEED = int(os.getenv("AML_SIM_SEED", str(DEFAULT_RANDOM_SEED)))
    FORCE_ALL_AML_SCENARIOS = os.getenv("AML_SIM_FORCE_SCENARIO_COVERAGE", "1") != "0"
    MAX_TX_PER_ACCOUNT = int(os.getenv("AML_SIM_MAX_TX_PER_ACCOUNT", str(DEFAULT_MAX_TX_PER_ACCOUNT)))
    SUSPICIOUS_TX_PROB_SCALE = float(os.getenv("AML_SIM_SUSPICIOUS_TX_PROB_SCALE", str(DEFAULT_SUSPICIOUS_TX_PROB_SCALE)))
    assert DAYS > 0, "Simulation end date must be on or after the start date."

    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    _progress(
        "Figure configuration loaded: "
        f"default window {START_DATE.date()} to {END_DATE.date()}, "
        f"seed {SEED}, legacy={'yes' if include_legacy else 'no'}, "
        f"extended={'yes' if include_extended else 'no'}."
    )

    # ------------------------------------------------------------
    # Jurisdiction context
    # ------------------------------------------------------------
    # FATF lists are time-dependent. These lists were checked against FATF public
    # statements dated 13 February 2026. Refresh them before a final experiment.
    SA = "South Africa"

    SADC_CORRIDORS = [
        "Angola", "Botswana", "Democratic Republic of Congo", "Eswatini",
        "Lesotho", "Malawi", "Mauritius", "Mozambique", "Namibia",
        "Seychelles", "Tanzania", "Zambia", "Zimbabwe"
    ]

    FATF_CALL_FOR_ACTION = [
        "Democratic People's Republic of Korea", "Iran", "Myanmar"
    ]

    FATF_INCREASED_MONITORING = [
        "Algeria", "Angola", "Bolivia", "Bulgaria", "Cameroon",
        "Cote d'Ivoire", "Democratic Republic of Congo", "Haiti", "Kenya",
        "Kuwait", "Lao PDR", "Lebanon", "Monaco", "Namibia", "Nepal",
        "Papua New Guinea", "South Sudan", "Syria", "Venezuela", "Vietnam",
        "Virgin Islands (UK)", "Yemen"
    ]

    # Other countries are included as trade, family or VASP corridors. They are not
    # treated as FATF high-risk unless they appear in the two FATF lists above.
    OTHER_RELEVANT_COUNTRIES = [
        "United Arab Emirates", "China", "India", "United Kingdom", "United States"
    ]

    ALL_COUNTRIES = sorted(set(
        [SA] + SADC_CORRIDORS + FATF_CALL_FOR_ACTION +
        FATF_INCREASED_MONITORING + OTHER_RELEVANT_COUNTRIES
    ))

    CURRENCY_MAP = {
        "South Africa": "ZAR", "Angola": "AOA", "Botswana": "BWP",
        "Democratic Republic of Congo": "CDF", "Eswatini": "SZL",         
        "Lesotho": "LSL", "Malawi": "MWK", "Mauritius": "MUR",
        "Mozambique": "MZN", "Namibia": "NAD", "Seychelles": "SCR",
        "Tanzania": "TZS", "Zambia": "ZMW", "Zimbabwe": "USD",
        "Democratic People's Republic of Korea": "KPW", "Iran": "IRR",
        "Myanmar": "MMK", "Algeria": "DZD", "Bolivia": "BOB",
        "Bulgaria": "BGN", "Cameroon": "XAF", "Cote d'Ivoire": "XOF",
        "Haiti": "HTG", "Kenya": "KES", "Kuwait": "KWD",
        "Lao PDR": "LAK", "Lebanon": "LBP", "Monaco": "EUR",
        "Nepal": "NPR", "Papua New Guinea": "PGK", "South Sudan": "SSP",
        "Syria": "SYP", "Venezuela": "VES", "Vietnam": "VND",
        "Virgin Islands (UK)": "USD", "Yemen": "YER",
        "United Arab Emirates": "AED", "China": "CNY", "India": "INR",
        "United Kingdom": "GBP", "United States": "USD"
    }

    RISK_TIER_SCORE = {
        "Domestic": 0,
        "SADC corridor": 1,
        "Other international": 1,
        "FATF Increased Monitoring": 2,
        "FATF Call for Action": 3,
    }

    PAYMENT_TYPES = [
        "Cash Deposit", "Cash Withdrawal", "Electronic Transfer", "RTC", "PayShap",
        "Card Payment", "Cheque", "International Wire"
    ]

    LAUNDERING_TYPES = [
        "Structuring", "Mule Account", "Intermediary-Based Flows",
        "Procurement-Linked Laundering", "Shell-Entity Behaviour",
        "Cross-Border Layering", "Crypto Off-Ramp"
    ]

    CUSTOMER_TYPE_WEIGHTS = {
        "Formal Employee": 0.25,
        "Informal Trader": 0.16,
        "Grant Recipient": 0.14,
        "Cash-Intensive SME": 0.10,
        "Migrant Worker / Remittance User": 0.09,
        "Student / Youth Account": 0.08,
        "Public Sector Employee": 0.07,
        "Business Owner": 0.06,
        "Procurement Vendor": 0.03,
        "Corporate Entity": 0.015,
        "NPO / Community Organisation": 0.005,
    }

    PROVINCE_WEIGHTS = {
        "Gauteng": 0.25,
        "KwaZulu-Natal": 0.18,
        "Eastern Cape": 0.13,
        "Western Cape": 0.12,
        "Limpopo": 0.10,
        "Mpumalanga": 0.08,
        "North West": 0.07,
        "Free State": 0.05,
        "Northern Cape": 0.02,
    }

    INCOME_RANGES = {
        "No regular income": (0, 2200),
        "Grant-linked": (350, 2600),
        "Low / irregular": (1200, 7000),
        "Low formal": (4500, 12000),
        "Middle": (12000, 35000),
        "Upper-middle": (35000, 85000),
        "Business turnover": (25000, 260000),
    }

    LEGIT_LAMBDA = {
        "Salary and household payments": 14,
        "Grant-linked inflows and survival payments": 9,
        "Informal registered business cash cycle": 20,
        "Informal unregistered business cash cycle": 22,
        "Spaza / township retail cash cycle": 24,
        "Taxi or transport cash cycle": 24,
        "Low-value high-frequency retail payments": 22,
        "Remittance corridor activity": 13,
        "Stokvel / rotating savings contributions": 8,
        "Business operating payments": 20,
        "Irregular income / gig work": 10,
    }

    # Scenario catalogue used both by the generator and by the reference notebook.
    # Source keys are defined in aml_simulation_references.ipynb. They keep the
    # simulation auditable without implying that every transaction in a category is
    # criminal in the real world.
    AML_SCENARIO_CATALOG = {
        "Structuring / Smurfing Cash Deposits": {
            "code": "SA_AML_001", "weight": 1.18, "lambda_add": 42, "tx_prob": 0.30,
            "predicate_offence": "Tax evasion, corruption, narcotics or other cash proceeds",
            "stage": "Placement", "risk_sector": "Cash-intensive retail / informal economy",
            "source_keys": "FIC-TYPOLOGIES; FIC-REPORTING-OBLIGATIONS",
            "indicator": "Repeated cash deposits just below the cash threshold reporting line",
        },
        "Cash Courier / Bulk Cash Movement": {
            "code": "SA_AML_002", "weight": 0.62, "lambda_add": 30, "tx_prob": 0.24,
            "predicate_offence": "Cash-intensive organised crime proceeds",
            "stage": "Placement", "risk_sector": "Cash courier / informal cash movement",
            "source_keys": "FATF-SA-MER-2021; FIC-TYPOLOGIES",
            "indicator": "Large cash movement inconsistent with customer profile",
        },
        "Mule Account / Rapid Pass-Through": {
            "code": "SA_AML_003", "weight": 1.12, "lambda_add": 55, "tx_prob": 0.34,
            "predicate_offence": "Fraud, cybercrime or organised-crime account takeover",
            "stage": "Layering", "risk_sector": "Retail individual accounts",
            "source_keys": "FIC-CASE-STUDIES-2019; FIC-TYPOLOGIES",
            "indicator": "Many-to-one inflows followed by rapid withdrawals or onward transfers",
        },
        "Cross-Border Layering via SADC Corridor": {
            "code": "SA_AML_004", "weight": 1.00, "lambda_add": 42, "tx_prob": 0.28,
            "predicate_offence": "Regional laundering or remittance-channel misuse",
            "stage": "Layering", "risk_sector": "Money remittance / SADC corridor",
            "source_keys": "FATF-SA-MER-2021; WB-RPW-DATA",
            "indicator": "Frequent SADC corridor transfers inconsistent with income profile",
        },
        "High-Risk Jurisdiction Layering": {
            "code": "SA_AML_005", "weight": 0.70, "lambda_add": 34, "tx_prob": 0.25,
            "predicate_offence": "Cross-border layering through FATF-monitored jurisdictions",
            "stage": "Layering", "risk_sector": "International funds transfer",
            "source_keys": "FATF-LISTS-2026-02; FATF-SA-DELIST-2025",
            "indicator": "Routing through FATF call-for-action or increased-monitoring jurisdictions",
        },
        "Intermediary / Trust Account Abuse": {
            "code": "SA_AML_006", "weight": 0.78, "lambda_add": 38, "tx_prob": 0.26,
            "predicate_offence": "Corruption, fraud or proceeds hidden through professional intermediaries",
            "stage": "Layering", "risk_sector": "Legal practitioner / trust account",
            "source_keys": "FIC-TYPOLOGIES; FIC-LEGAL-RMCP-2025",
            "indicator": "Pass-through activity using intermediary or trust-style accounts",
        },
        "Procurement Corruption / Tender Kickback": {
            "code": "SA_AML_007", "weight": 0.92, "lambda_add": 32, "tx_prob": 0.24,
            "predicate_offence": "Corruption, bribery, procurement fraud or state-capture style abuse",
            "stage": "Integration", "risk_sector": "Public procurement / public sector",
            "source_keys": "FATF-SA-MER-2021; ZONDO-PART-1-2022; FIC-TYPOLOGIES",
            "indicator": "Public-sector or connected-vendor flows inconsistent with profile",
        },
        "Shell or Front Company Flow": {
            "code": "SA_AML_008", "weight": 0.82, "lambda_add": 28, "tx_prob": 0.22,
            "predicate_offence": "Fraud, corruption or tax evasion concealed by front companies",
            "stage": "Layering", "risk_sector": "Opaque company / front business",
            "source_keys": "FATF-BO-LEGAL-PERSONS-2023; FIC-TYPOLOGIES",
            "indicator": "High-value flows through entities with opaque beneficial ownership",
        },
        "Complex Legal Structure / Beneficial Ownership Obfuscation": {
            "code": "SA_AML_009", "weight": 0.58, "lambda_add": 26, "tx_prob": 0.21,
            "predicate_offence": "Corruption or fraud hidden through complex ownership structures",
            "stage": "Layering", "risk_sector": "Trusts, legal arrangements and companies",
            "source_keys": "FATF-BO-LEGAL-ARRANGEMENTS-2024; FIC-BO-OBLIGATIONS-2026",
            "indicator": "Complex ownership, unrelated counterparties and unexplained foreign flows",
        },
        "Crypto Asset Off-Ramp": {
            "code": "SA_AML_010", "weight": 0.72, "lambda_add": 34, "tx_prob": 0.25,
            "predicate_offence": "Virtual-asset laundering, fraud or cybercrime proceeds",
            "stage": "Layering / Integration", "risk_sector": "Crypto asset service provider",
            "source_keys": "FIC-CASP; FATF-VA-RED-FLAGS-2020",
            "indicator": "Virtual-asset off-ramp followed by bank transfer or cash-out",
        },
        "Ponzi / Investment Scheme Collections": {
            "code": "SA_AML_011", "weight": 0.72, "lambda_add": 46, "tx_prob": 0.28,
            "predicate_offence": "Fraudulent investment scheme",
            "stage": "Placement / Layering", "risk_sector": "Retail investor collections",
            "source_keys": "FIC-TYPOLOGIES",
            "indicator": "Multiple unrelated investor deposits in similar amounts over a short period",
        },
        "Fraudulent Company or Invoice Scam": {
            "code": "SA_AML_012", "weight": 0.82, "lambda_add": 38, "tx_prob": 0.27,
            "predicate_offence": "Business email compromise, invoice fraud or company scam",
            "stage": "Layering", "risk_sector": "Corporate payment / retail business accounts",
            "source_keys": "FIC-TYPOLOGIES; LNRS-AFRICA-FINCRIME-2022",
            "indicator": "Victim-company inflows followed by rapid onward payments",
        },
        "Cybercrime Proceeds": {
            "code": "SA_AML_013", "weight": 0.78, "lambda_add": 48, "tx_prob": 0.30,
            "predicate_offence": "Cybercrime, phishing or account takeover",
            "stage": "Layering", "risk_sector": "Digital retail payments",
            "source_keys": "FIC-TYPOLOGIES; LNRS-AFRICA-FINCRIME-2022",
            "indicator": "Fast digital transfers inconsistent with normal account use",
        },
        "Narcotics Cash Proceeds": {
            "code": "SA_AML_014", "weight": 0.62, "lambda_add": 40, "tx_prob": 0.26,
            "predicate_offence": "Drug trafficking",
            "stage": "Placement", "risk_sector": "Cash-intensive retail",
            "source_keys": "FIC-TYPOLOGIES; FATF-SA-MER-2021",
            "indicator": "High cash frequency and unexplained retail-style cash deposits",
        },
        "Illegal Gold or Precious Metals": {
            "code": "SA_AML_015", "weight": 0.56, "lambda_add": 30, "tx_prob": 0.23,
            "predicate_offence": "Illegal mining, gold smuggling or precious-metal crime",
            "stage": "Placement / Integration", "risk_sector": "Precious metals and stones",
            "source_keys": "FIC-TYPOLOGIES; FATF-SA-MER-2021",
            "indicator": "Precious-metal trade payments and high-value cash or cross-border flows",
        },
        "Environmental or Wildlife Crime Proceeds": {
            "code": "SA_AML_016", "weight": 0.45, "lambda_add": 22, "tx_prob": 0.20,
            "predicate_offence": "Environmental, wildlife or natural-resource crime",
            "stage": "Layering", "risk_sector": "Environmental / export-linked flows",
            "source_keys": "FIC-TYPOLOGIES; UNODC-IFF-2011",
            "indicator": "Export, logistics or cash flows inconsistent with stated activity",
        },
        "Stolen Student Loan or Grant Abuse": {
            "code": "SA_AML_017", "weight": 0.52, "lambda_add": 38, "tx_prob": 0.26,
            "predicate_offence": "Student-loan, grant or benefit fraud",
            "stage": "Placement / Layering", "risk_sector": "Students, youth and grant-linked accounts",
            "source_keys": "FIC-TYPOLOGIES; STATSSA-GHS-2024",
            "indicator": "Grant or student-funding inflows followed by atypical cash-out or transfers",
        },
        "Casino or Gambling Laundering": {
            "code": "SA_AML_018", "weight": 0.52, "lambda_add": 30, "tx_prob": 0.22,
            "predicate_offence": "Integration through gambling value instruments or winnings",
            "stage": "Integration", "risk_sector": "Casino / gambling",
            "source_keys": "FIC-CASE-STUDIES-2019; FATF-SA-MER-2021",
            "indicator": "Cash or electronic movement linked to gambling counterparties",
        },
        "Property / Real Estate Integration": {
            "code": "SA_AML_019", "weight": 0.55, "lambda_add": 20, "tx_prob": 0.19,
            "predicate_offence": "Integration of illicit funds into immovable property",
            "stage": "Integration", "risk_sector": "Property / real estate",
            "source_keys": "FIC-PROPERTY-TFS-2025; FATF-SA-MER-2021",
            "indicator": "High-value property deposits or rental flows inconsistent with profile",
        },
        "Motor Vehicle or High-Value Goods Purchase": {
            "code": "SA_AML_020", "weight": 0.54, "lambda_add": 24, "tx_prob": 0.21,
            "predicate_offence": "Integration through movable high-value goods",
            "stage": "Integration", "risk_sector": "Motor vehicle / high-value goods",
            "source_keys": "FIC-MOTOR-VEHICLE-2025",
            "indicator": "High-value goods purchase using cash or unusual third-party payments",
        },
        "Trade-Based Import-Export Layering": {
            "code": "SA_AML_021", "weight": 0.62, "lambda_add": 26, "tx_prob": 0.22,
            "predicate_offence": "Trade-based money laundering, fraud or customs/tax abuse",
            "stage": "Layering", "risk_sector": "Import/export and trade finance",
            "source_keys": "FATF-SA-MER-2021; UNODC-IFF-2011",
            "indicator": "International trade payments inconsistent with customer profile",
        },
        "NPO / Community Organisation Misuse": {
            "code": "SA_AML_022", "weight": 0.42, "lambda_add": 22, "tx_prob": 0.18,
            "predicate_offence": "Misuse of community or non-profit structures",
            "stage": "Layering", "risk_sector": "NPO / community organisation",
            "source_keys": "FATF-SA-MER-2021; FIC-TF-NRA-2024",
            "indicator": "Donation-like flows with foreign or unrelated counterparties",
        },
    }

    LAUNDERING_TYPES = list(AML_SCENARIO_CATALOG.keys())
    # Suspicious accounts keep their ordinary customer activity. Only a minority
    # of their transactions receive the typology overlay so account labels are not
    # trivially recoverable from transaction volume or a single transaction row.
    TYPOLOGY_TX_PROB = {
        name: float(np.clip(meta["tx_prob"] * SUSPICIOUS_TX_PROB_SCALE, 0.04, 0.16))
        for name, meta in AML_SCENARIO_CATALOG.items()
    }
    TYPOLOGY_TX_PROB["Normal"] = 0.00

    NORMAL_BEHAVIOUR_CATALOG = {
        "Salary and household payments": {
            "source_keys": "STATSSA-QLFS-Q4-2025; SARB-PAYMENTS-REPORT-2023",
            "indicator": "Salary inflows and household merchant, debit-card, PayShap and cash-out payments",
        },
        "Grant-linked inflows and survival payments": {
            "source_keys": "STATSSA-GHS-2024; SARB-PAYMENTS-REPORT-2023",
            "indicator": "Social grant inflows and low-value survival spending",
        },
        "Informal registered business cash cycle": {
            "source_keys": "STATS-SA-SESE-2023; STATS-SA-INFORMALITY-2026",
            "indicator": "Small business cash/card turnover with some registration or records",
        },
        "Informal unregistered business cash cycle": {
            "source_keys": "STATS-SA-SESE-2023; STATS-SA-INFORMALITY-2026; STANDARD-BANK-INFORMAL-2025",
            "indicator": "Non-VAT or unregistered business cash cycle using personal accounts",
        },
        "Spaza / township retail cash cycle": {
            "source_keys": "STATS-SA-SESE-2023; STANDARD-BANK-INFORMAL-2025",
            "indicator": "Township retail, spaza, street trading and supplier cash cycle",
        },
        "Taxi or transport cash cycle": {
            "source_keys": "STATS-SA-SESE-2023; STATS-SA-INFORMALITY-2026",
            "indicator": "Cash-heavy transport revenue and fuel, repairs or owner payments",
        },
        "Low-value high-frequency retail payments": {
            "source_keys": "SARB-PAYMENTS-REPORT-2023",
            "indicator": "Frequent low-value consumer payments through digital and card rails",
        },
        "Remittance corridor activity": {
            "source_keys": "WB-RPW-DATA; FATF-SA-MER-2021",
            "indicator": "Family remittances through SADC corridors and money-transfer channels",
        },
        "Stokvel / rotating savings contributions": {
            "source_keys": "FATF-SA-MER-2021; SARB-PAYMENTS-REPORT-2023",
            "indicator": "Group savings contributions and periodic withdrawals",
        },
        "Business operating payments": {
            "source_keys": "STATS-SA-SESE-2023; SARB-PA-BANKING-SRA-2025",
            "indicator": "Supplier, customer, payroll and operating transactions",
        },
        "Irregular income / gig work": {
            "source_keys": "STATS-SA-INFORMALITY-2026; STATSSA-QLFS-Q4-2025",
            "indicator": "Irregular wages, gig income, airtime/data, transport and cash use",
        },
    }

    CUSTOMER_TYPE_WEIGHTS = {
        "Formal Employee": 0.24,
        "Informal Trader": 0.17,
        "Grant Recipient": 0.14,
        "Cash-Intensive SME": 0.10,
        "Migrant Worker / Remittance User": 0.09,
        "Student / Youth Account": 0.08,
        "Public Sector Employee": 0.07,
        "Business Owner": 0.06,
        "Procurement Vendor": 0.03,
        "Corporate Entity": 0.015,
        "NPO / Community Organisation": 0.005,
    }

    PROVINCE_WEIGHTS = {
        "Gauteng": 0.25,
        "KwaZulu-Natal": 0.18,
        "Eastern Cape": 0.13,
        "Western Cape": 0.12,
        "Limpopo": 0.10,
        "Mpumalanga": 0.08,
        "North West": 0.07,
        "Free State": 0.05,
        "Northern Cape": 0.02,
    }

    INCOME_RANGES = {
        "No regular income": (0, 2200),
        "Grant-linked": (350, 2600),
        "Low / irregular": (1200, 7000),
        "Low formal": (4500, 12000),
        "Middle": (12000, 35000),
        "Upper-middle": (35000, 85000),
        "Business turnover": (25000, 260000),
    }

    # ------------------------------------------------------------


    _progress("Loading simulation CSV input.")
    df_final = _read_simulation_csv(input_csv)
    df = df_final
    RAW_SIMULATION_COLUMNS = list(df_final.columns)
    _progress(f"Loaded {len(df_final):,} transaction rows and {df_final['Account_id'].nunique():,} receiver accounts.")

    _progress("Reconstructing account-level summary from transaction rows.")
    receiver_accounts = _reconstruct_receiver_accounts(df_final)

    # Use observed dates from the file so figures match the loaded simulation run.
    observed_dates = pd.to_datetime(df["Date"])
    START_DATE = observed_dates.min().to_pydatetime()
    END_DATE = observed_dates.max().to_pydatetime()
    DAYS = int((observed_dates.max().date() - observed_dates.min().date()).days + 1)

    scenario_catalog_df = (
        pd.DataFrame.from_dict(AML_SCENARIO_CATALOG, orient="index")
        .reset_index()
        .rename(columns={"index": "AML_scenario_category"})
    )
    normal_behavior_catalog_df = (
        pd.DataFrame.from_dict(NORMAL_BEHAVIOUR_CATALOG, orient="index")
        .reset_index()
        .rename(columns={"index": "Normal_behavior_category"})
    )

    # Recreate notebook-only diagnostic features needed by the plots.
    # Date and time normalisation used by the plots and sequence features below.
    _progress("Rebuilding diagnostic fields required by validation plots.")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'].astype(str), format="%H:%M:%S").dt.time

    df['fulldatetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    )

    df.columns


    laundering_stats = df[df['Is_laundering'] == 1]['Amount'].agg(['max', 'mean', 'min'])

    normal_stats = df[df['Is_laundering'] == 0]['Amount'].agg(['max', 'mean', 'min'])

    print("Laundering Transactions Stats:\n", laundering_stats)
    print("\nNormal Transactions Stats:\n", normal_stats)


    # ========== 4. Feature Engineering ==========
    # These variables are deliberately separated from the label. They are context
    # features a model may learn from, not rules that automatically imply laundering.

    df['year'] = df['fulldatetime'].dt.year
    df['month'] = df['fulldatetime'].dt.month
    df['day'] = df['fulldatetime'].dt.day
    df['weekday'] = df['fulldatetime'].dt.dayofweek
    df['hour'] = df['fulldatetime'].dt.hour
    df['year_month'] = df['fulldatetime'].dt.to_period('M').astype(str)
    df['quarter'] = df['fulldatetime'].dt.to_period('Q').astype(str)

    # Account lifecycle context for sequence modelling. Account_open_date can be
    # before the observation window, while First_observed_transaction_date marks the
    # start of the simulated sequence for that receiver account.
    df['Account_open_date'] = pd.to_datetime(df['Account_open_date'])
    df['First_observed_transaction_date'] = pd.to_datetime(df['First_observed_transaction_date'])
    df['account_age_at_transaction_days'] = (
        df['fulldatetime'].dt.normalize() - df['Account_open_date']
    ).dt.days.clip(lower=0)
    df['account_observation_age_days'] = (
        df['fulldatetime'].dt.normalize() - df['First_observed_transaction_date']
    ).dt.days.clip(lower=0)
    df['account_lifecycle_segment'] = np.select(
        [
            df['Opened_for_laundering_flag'].eq(1),
            df['Opened_during_observation_flag'].eq(1),
            df['Account_age_days'].lt(120),
            df['Account_age_days'].ge(730),
        ],
        [
            'New account opened for laundering scenario',
            'New legitimate / newly observed account',
            'Recently opened account',
            'Established account',
        ],
        default='Mature account'
    )

    # Cyclical encoding for periodic features.
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # Country and currency context.
    df['currency_mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
    df['fatf_call_for_action_sender'] = df['Sender_bank_location'].isin(FATF_CALL_FOR_ACTION).astype(int)
    df['fatf_call_for_action_receiver'] = df['Receiver_bank_location'].isin(FATF_CALL_FOR_ACTION).astype(int)
    df['fatf_increased_monitoring_sender'] = df['Sender_bank_location'].isin(FATF_INCREASED_MONITORING).astype(int)
    df['fatf_increased_monitoring_receiver'] = df['Receiver_bank_location'].isin(FATF_INCREASED_MONITORING).astype(int)
    df['sadc_sender'] = df['Sender_bank_location'].isin(SADC_CORRIDORS).astype(int)
    df['sadc_receiver'] = df['Receiver_bank_location'].isin(SADC_CORRIDORS).astype(int)

    # Backwards-compatible high-risk flags for earlier experiments. Here high-risk
    # means either FATF call-for-action or increased-monitoring exposure.
    df['high_risk_sender'] = ((df['fatf_call_for_action_sender'] == 1) | (df['fatf_increased_monitoring_sender'] == 1)).astype(int)
    df['high_risk_receiver'] = ((df['fatf_call_for_action_receiver'] == 1) | (df['fatf_increased_monitoring_receiver'] == 1)).astype(int)

    # Cash and structuring context. South African CTR reporting applies to cash
    # received or issued above R49,999.99; near-threshold cash activity is modelled
    # as a risk feature, not a label by itself.
    df['cash_transaction'] = df['Payment_type'].isin(['Cash Deposit', 'Cash Withdrawal']).astype(int)
    df['large_cash_flag'] = ((df['cash_transaction'] == 1) & (df['Amount'] > 49_999.99)).astype(int)
    df['cash_threshold_near_miss'] = ((df['cash_transaction'] == 1) & df['Amount'].between(40_000, 49_999.99)).astype(int)
    df['round_amount_flag'] = ((df['Amount'] % 1000).round(2).isin([0.00])).astype(int)

    # International funds transfer report context. FIC guidance notes cross-border
    # electronic funds transactions above R19,999.99 as a reporting stream.
    df['iftr_reportable_cross_border'] = (
        (df['Cross_border_transaction'] == 1) &
        df['Payment_type'].isin(['International Wire', 'Electronic Transfer', 'RTC']) &
        (df['Amount'] > 19_999.99)
    ).astype(int)

    # Time-between-transaction features for sequential account-level modelling.
    df = df.sort_values(['Account_id', 'fulldatetime', 'Transaction_id']).reset_index(drop=True)
    df['prev_time'] = df.groupby('Account_id')['fulldatetime'].shift(1)
    df['time_since_last'] = (df['fulldatetime'] - df['prev_time']).dt.total_seconds().fillna(0)
    df['rapid_repeat_transaction'] = ((df['time_since_last'] > 0) & (df['time_since_last'] <= 15 * 60)).astype(int)

    # Rolling statistics per monitored account.
    df['rolling_amt_mean'] = df.groupby('Account_id')['Amount'].transform(lambda x: x.rolling(5, 1).mean())
    df['rolling_amt_std'] = df.groupby('Account_id')['Amount'].transform(lambda x: x.rolling(5, 1).std()).fillna(0)
    df['account_daily_tx_count'] = df.groupby(['Account_id', 'Date'])['Transaction_id'].transform('count')

    # Account aggregates are useful for graph, sequence and behavioural models.
    account_group = df.groupby('Account_id')
    df['account_tx_count'] = account_group['Transaction_id'].transform('count')
    df['account_total_amount'] = account_group['Amount'].transform('sum')
    df['account_mean_amount'] = account_group['Amount'].transform('mean')
    df['account_unique_counterparties'] = account_group['Counterparty_account'].transform('nunique')
    df['account_cross_border_share'] = account_group['Cross_border_transaction'].transform('mean')
    df['account_cash_share'] = account_group['cash_transaction'].transform('mean')
    df['account_fatf_exposure_share'] = account_group['Jurisdiction_risk_tier'].transform(
        lambda x: x.isin(['FATF Increased Monitoring', 'FATF Call for Action']).mean()
    )
    df['account_first_tx_datetime'] = account_group['fulldatetime'].transform('min')
    df['account_last_tx_datetime'] = account_group['fulldatetime'].transform('max')
    df['account_observed_span_days'] = (
        df['account_last_tx_datetime'] - df['account_first_tx_datetime']
    ).dt.days.add(1).clip(lower=1)
    df['account_tx_per_active_month'] = (
        df['account_tx_count'] / np.maximum(df['Active_observation_days'] / 30.4375, 1 / 30.4375)
    )
    df['account_tx_per_observed_month'] = (
        df['account_tx_count'] / np.maximum(df['account_observed_span_days'] / 30.4375, 1 / 30.4375)
    )
    df['account_generated_expected_ratio'] = (
        df['Generated_transaction_count'] / np.maximum(df['Expected_transaction_count'], 0.1)
    )

    # A transparent context score helps inspect scenarios. It should not be used as
    # a hard-coded AML rule in evaluation because it is built from known simulation factors.
    df['transaction_context_risk_score'] = np.clip(
        8 * df['currency_mismatch'] +
        14 * df['Cross_border_transaction'] +
        9 * df['SADC_corridor_transaction'] +
        22 * (df['Jurisdiction_risk_tier'] == 'FATF Increased Monitoring').astype(int) +
        35 * (df['Jurisdiction_risk_tier'] == 'FATF Call for Action').astype(int) +
        18 * df['cash_threshold_near_miss'] +
        8 * df['rapid_repeat_transaction'] +
        10 * df['Intermediary_account_flag'] +
        10 * df['Virtual_asset_exposure'] +
        12 * df['Procurement_vendor'] +
        12 * df['Public_sector_counterparty'] +
        12 * df['PEP_related_transaction'] +
        8 * (df['Beneficial_ownership_opacity'] > 0.60).astype(int) +
        6 * df['Informal_business_flag'] +
        8 * df['Uses_personal_account_for_business'] +
        9 * df['High_risk_sector_flag'] +
        10 * df['High_risk_person_flag'] +
        10 * (df['KYC_strength'].isin(['Weak / incomplete', 'Enhanced required'])).astype(int),
        0,
        100
    )

    # Avoid duplicating a multi-million-row dataframe in memory. Engineered fields remain
    # available for notebook diagnostics, but the public raw export is limited to
    # RAW_SIMULATION_COLUMNS captured before feature engineering.
    df_final = df
    ENGINEERED_FEATURE_COLUMNS = [
        col for col in df_final.columns
        if col not in RAW_SIMULATION_COLUMNS
    ]
    print("Engineered diagnostic columns created in notebook only:", len(ENGINEERED_FEATURE_COLUMNS))


    if include_legacy:
        tx_per_receiver = df.groupby('Account_id').size()

        plt.figure(figsize=(8,5))
        tx_per_receiver.hist(bins=50)
        plt.xlabel("Number of Transactions")
        plt.ylabel("Number of Monitored Accounts")
        plt.title("Distribution of Transactions per Monitored Account")
        finish_plot("eda_transactions_per_monitored_account.png")


        plt.figure(figsize=(10,6))
        sns.boxplot(x='Is_laundering', y='Amount', data=df)
        plt.xticks([0,1], ['Non-Laundering', 'Laundering'])
        plt.title("Transaction Amount Distribution by Laundering Flag")
        finish_plot("eda_amount_distribution_by_laundering_flag.png")


        class_distribution = df['Is_laundering'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.pie(class_distribution, labels=['Non-Laundering Transactions', 'Suspicious Transactions'], autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])

        plt.title('Class Distribution')
        plt.axis('equal')

        finish_plot("eda_class_distribution_pie.png")


        # --- Monthly Transactions ---
        monthly_transactions = df.groupby(['month', 'Is_laundering']).size().unstack(fill_value=0)

        avg_monthly_non = monthly_transactions[0].mean()
        avg_monthly_lau = monthly_transactions[1].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # removed sharey=True

        # Non-Laundering
        axes[0].plot(monthly_transactions.index, monthly_transactions[0], color='blue', label='Non-Laundering')
        axes[0].axhline(y=avg_monthly_non, color='blue', linestyle='--', alpha=0.7,
                        label=f'Avg Non-Laundering ({avg_monthly_non:.2f})')
        axes[0].set_title("Monthly Transactions - Non-Laundering")
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Number of Transactions")
        axes[0].legend()
        axes[0].grid(True)

        # Laundering
        axes[1].plot(monthly_transactions.index, monthly_transactions[1], color='red', label='Laundering')
        axes[1].axhline(y=avg_monthly_lau, color='red', linestyle='--', alpha=0.7,
                        label=f'Avg Laundering ({avg_monthly_lau:.2f})')
        axes[1].set_title("Monthly Transactions - Laundering")
        axes[1].set_xlabel("Month")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        finish_plot("eda_monthly_transactions_by_label.png")


        # --- Weekly Transactions ---
        weekly_transactions = df.groupby(['weekday', 'Is_laundering']).size().unstack(fill_value=0)

        avg_weekly_non = weekly_transactions[0].mean()
        avg_weekly_lau = weekly_transactions[1].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # removed sharey=True

        # Non-Laundering
        axes[0].plot(weekly_transactions.index, weekly_transactions[0], color='blue', label='Non-Laundering')
        axes[0].axhline(y=avg_weekly_non, color='blue', linestyle='--', alpha=0.7,
                        label=f'Avg Non-Laundering ({avg_weekly_non:.2f})')
        axes[0].set_title("Weekly Transactions - Non-Laundering")
        axes[0].set_xlabel("Weekday")
        axes[0].set_ylabel("Number of Transactions")
        axes[0].legend()
        axes[0].grid(True)

        # Laundering
        axes[1].plot(weekly_transactions.index, weekly_transactions[1], color='red', label='Laundering')
        axes[1].axhline(y=avg_weekly_lau, color='red', linestyle='--', alpha=0.7,
                        label=f'Avg Laundering ({avg_weekly_lau:.2f})')
        axes[1].set_title("Weekly Transactions - Laundering")
        axes[1].set_xlabel("Weekday")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        finish_plot("eda_weekly_transactions_by_label.png")


        # Group by day and laundering flag
        daily_transactions = df.groupby(['day', 'Is_laundering']).size().unstack(fill_value=0)

        # Compute averages for both
        avg_non_laundering = daily_transactions[0].mean()
        avg_laundering = daily_transactions[1].mean()

        # Create side-by-side plots (independent scales)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Non-Laundering ---
        axes[0].plot(daily_transactions.index, daily_transactions[0], color='blue', label='Non-Laundering')
        axes[0].axhline(y=avg_non_laundering, color='blue', linestyle='--', alpha=0.7,
                        label=f'Avg Non-Laundering ({avg_non_laundering:.2f})')
        axes[0].set_title("Daily Transactions - Non-Laundering")
        axes[0].set_xlabel("Day")
        axes[0].set_ylabel("Number of Transactions")
        axes[0].legend()
        axes[0].grid(True)

        # --- Laundering ---
        axes[1].plot(daily_transactions.index, daily_transactions[1], color='red', label='Laundering')
        axes[1].axhline(y=avg_laundering, color='red', linestyle='--', alpha=0.7,
                        label=f'Avg Laundering ({avg_laundering:.2f})')
        axes[1].set_title("Daily Transactions - Laundering")
        axes[1].set_xlabel("Day")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        finish_plot("eda_daily_transactions_by_label.png")



        # Separate the data
        laundering_data = df[df['Is_laundering'] == 1]
        non_laundering_data = df[df['Is_laundering'] == 0]

        plt.figure(figsize=(12, 6))

        #  Plot Laundering FIRST (so it sits underneath)
        plt.scatter(
            laundering_data.index,
            laundering_data['Amount'],
            color='red',
            label='Laundering',
            alpha=0.3,
            s=5   # small marker size
        )

        # Plot Non-Laundering LAST (so it sits on top)
        plt.scatter(
            non_laundering_data.index,
            non_laundering_data['Amount'],
            color='blue',
            label='Non-Laundering',
            alpha=0.3,
            s=5
        )

        plt.title('Laundering vs Non-Laundering Transaction Amounts')
        plt.xlabel('Transaction Index')
        plt.ylabel('Transaction Amount')

        #  Important for skewed financial data
        plt.yscale('log')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        finish_plot("eda_amount_scatter_by_label.png")


        # ============================================================
        # PAYMENT TYPE SUMMARY HEATMAP
        # ============================================================

        # ------------------------------------------------------------
        # 1. Create summary table
        # ------------------------------------------------------------
        total_amount_pivot = pd.pivot_table(
            df,
            index="Payment_type",
            values="Amount",
            aggfunc=np.sum
        )

        laundering_count_pivot = (
            df[df["Is_laundering"] == 1]
            .groupby("Payment_type")
            .size()
            .to_frame("Laundering_Count")
        )

        normal_count_pivot = (
            df[df["Is_laundering"] == 0]
            .groupby("Payment_type")
            .size()
            .to_frame("Normal_Count")
        )

        combined_pivot = total_amount_pivot.join(
            [laundering_count_pivot, normal_count_pivot],
            how="outer"
        ).fillna(0)

        # Optional: rename Amount column nicely
        combined_pivot = combined_pivot.rename(columns={"Amount": "Total_Amount"})

        # ------------------------------------------------------------
        # 2. Sort for better readability
        # ------------------------------------------------------------
        combined_pivot = combined_pivot.sort_values(
            by="Total_Amount",
            ascending=False
        )

        # ------------------------------------------------------------
        # 3. Create annotation labels with original values
        # ------------------------------------------------------------
        annot_labels = combined_pivot.copy()

        annot_labels["Total_Amount"] = annot_labels["Total_Amount"].apply(
            lambda x: f"{x:,.0f}"
        )

        annot_labels["Laundering_Count"] = annot_labels["Laundering_Count"].apply(
            lambda x: f"{int(x):,}"
        )

        annot_labels["Normal_Count"] = annot_labels["Normal_Count"].apply(
            lambda x: f"{int(x):,}"
        )

        # ------------------------------------------------------------
        # 4. Normalise each column for heatmap colouring
        #    This prevents Amount from overpowering the count columns
        # ------------------------------------------------------------
        heatmap_data = combined_pivot.copy()

        heatmap_data = heatmap_data.apply(
            lambda col: (col - col.min()) / (col.max() - col.min())
            if col.max() != col.min() else col,
            axis=0
        )

        # ------------------------------------------------------------
        # 5. Plot heatmap
        # ------------------------------------------------------------
        plt.figure(figsize=(11, max(5, 0.45 * len(combined_pivot))))

        sns.heatmap(
            heatmap_data,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Column-normalised intensity"}
        )

        plt.title(
            "Payment Type Summary: Total Amount, Laundering Count and Normal Count",
            fontsize=14,
            fontweight="bold",
            pad=15
        )

        plt.xlabel("")
        plt.ylabel("Payment Type")

        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()
        finish_plot("eda_payment_type_summary_heatmap.png")


        df['Date'] = pd.to_datetime(df['Date'])
        grouped_data = df.groupby(['Date', 'Payment_type']).agg({'Is_laundering': 'sum'}).reset_index()
        grouped_data['Month_Year'] = grouped_data['Date'].dt.to_period('M')
        monthly_alerts = grouped_data.groupby(['Month_Year', 'Payment_type']).agg({'Is_laundering': 'sum'}).reset_index()

        pivot_data = monthly_alerts.pivot(index='Month_Year', columns='Payment_type', values='Is_laundering')

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        pivot_data.plot(kind='bar', ax=ax, stacked=True, colormap='tab10')
        pivot_data = monthly_alerts.pivot(index='Month_Year', columns='Payment_type', values='Is_laundering')

        plt.title('Number of Alerts Per Month Split by Payment Type', fontsize=14)
        plt.xlabel('Month-Year', fontsize=12)
        plt.ylabel('Number of Alerts', fontsize=12)
        plt.xticks(rotation=45)

        plt.legend(title='Payment Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        finish_plot("eda_monthly_alerts_by_payment_type.png")


        # --- Plot 7: Laundering Alerts per Monitored Account ---
        alerts_per_account = df[df['Is_laundering'] == 1].groupby('Account_id').size()
        import matplotlib.patches as mpatches

        plt.figure(figsize=(10, 4))
        sns.histplot(alerts_per_account, bins=20, color='crimson', kde=True)
        plt.title("Number of Laundering Alerts per Monitored Account")
        plt.xlabel("Number of Alerts")
        plt.ylabel("Count")

        legend_patch = mpatches.Patch(color='crimson', label='Laundering Alerts')
        plt.legend(handles=[legend_patch])

        plt.tight_layout()
        finish_plot("laundering_alerts_per_account.png")


    # Main validation sections.
    # Publication-grade insight dashboard for the South African AML simulation.
    # The tables and plots below are designed to make the synthetic data auditable:
    # they show what normal behaviour looks like, how suspicious overlays differ,
    # and whether receiver-account sequences are heterogeneous enough for later
    # sequential modelling.

    sns.set_theme(style="whitegrid", context="notebook")
    # Full-scale runs can contain millions of rows. Avoid copying df_final.
    analysis_df = df_final
    DASHBOARD_PLOT_SAMPLE_ROWS = int(os.getenv("AML_SIM_DASHBOARD_SAMPLE_ROWS", "250000"))
    analysis_df["Label"] = pd.Categorical(
        np.where(analysis_df["Is_laundering"] == 1, "Suspicious AML", "Legitimate baseline")
    )
    analysis_df["log_amount"] = np.log10(analysis_df["Amount"].clip(lower=1))
    analysis_df["time_since_last_hours"] = analysis_df["time_since_last"] / 3600
    analysis_df["year_month_dt"] = pd.to_datetime(analysis_df["year_month"] + "-01")

    EXPORT_INSIGHT_PLOTS = os.getenv("AML_SIM_EXPORT_INSIGHTS", "1") != "0"
    publication_figures_saved_to = []


    def show_table(title, table):
        print("\n" + title)
        display(table)


    def rate_table(group_col, source_df=None, min_count=1):
        frame = analysis_df if source_df is None else source_df
        table = (
            frame.groupby(group_col)
            .agg(
                transactions=("Transaction_id", "count"),
                suspicious_rate=("Is_laundering", "mean"),
                median_amount=("Amount", "median"),
                avg_amount=("Amount", "mean"),
                cash_share=("cash_transaction", "mean"),
                cross_border_share=("Cross_border_transaction", "mean"),
                rapid_repeat_share=("rapid_repeat_transaction", "mean"),
                avg_context_risk=("transaction_context_risk_score", "mean"),
            )
            .query("transactions >= @min_count")
            .sort_values("transactions", ascending=False)
        )
        return table


    PROVINCE_TILE_LAYOUT = {
        "Limpopo": (3.85, 4.95, 1.45, 0.88),
        "Mpumalanga": (5.30, 4.05, 1.25, 0.86),
        "Gauteng": (4.35, 3.95, 0.82, 0.72),
        "North West": (2.85, 3.85, 1.35, 0.86),
        "Free State": (3.75, 2.75, 1.45, 0.92),
        "KwaZulu-Natal": (5.10, 2.45, 1.38, 1.08),
        "Northern Cape": (1.20, 2.25, 1.85, 1.60),
        "Western Cape": (1.15, 0.80, 1.55, 0.98),
        "Eastern Cape": (2.85, 0.95, 1.95, 1.00),
    }

    COUNTRY_COORDS = {
        "South Africa": (24.0, -29.0),
        "Angola": (17.9, -11.2),
        "Botswana": (24.7, -22.3),
        "Democratic Republic of Congo": (23.7, -2.9),
        "Eswatini": (31.5, -26.5),
        "Lesotho": (28.2, -29.6),
        "Malawi": (34.3, -13.3),
        "Mauritius": (57.6, -20.2),
        "Mozambique": (35.5, -18.7),
        "Namibia": (17.1, -22.1),
        "Seychelles": (55.5, -4.7),
        "Tanzania": (34.9, -6.4),
        "Zambia": (27.8, -13.1),
        "Zimbabwe": (29.2, -19.0),
        "Democratic People's Republic of Korea": (127.0, 40.3),
        "Iran": (53.7, 32.4),
        "Myanmar": (96.2, 21.9),
        "Algeria": (1.7, 28.0),
        "Bolivia": (-64.7, -16.3),
        "Bulgaria": (25.5, 42.7),
        "Cameroon": (12.4, 5.0),
        "Cote d'Ivoire": (-5.5, 7.5),
        "Haiti": (-72.3, 19.0),
        "Kenya": (37.9, 0.2),
        "Kuwait": (47.5, 29.3),
        "Lao PDR": (102.5, 19.9),
        "Lebanon": (35.9, 33.9),
        "Monaco": (7.4, 43.7),
        "Nepal": (84.1, 28.4),
        "Papua New Guinea": (143.9, -6.3),
        "South Sudan": (31.3, 6.9),
        "Syria": (38.9, 35.0),
        "Venezuela": (-66.6, 6.4),
        "Vietnam": (108.3, 14.1),
        "Virgin Islands (UK)": (-64.6, 18.4),
        "Yemen": (48.5, 15.6),
        "United Arab Emirates": (54.4, 24.3),
        "China": (104.2, 35.9),
        "India": (78.9, 22.0),
        "United Kingdom": (-3.4, 55.4),
        "United States": (-98.6, 39.8),
    }

    RISK_TIER_COLORS = {
        "SADC corridor": "#2A9D8F",
        "FATF Increased Monitoring": "#F4A261",
        "FATF Call for Action": "#D62828",
        "Other corridor": "#457B9D",
        "Domestic": "#A8A8A8",
    }


    def modal_value(series, default="Other corridor"):
        mode = series.dropna().mode()
        return mode.iloc[0] if len(mode) else default


    def make_province_exposure_map(account_summary):
        province_summary = (
            account_summary.groupby("Province")
            .agg(
                accounts=("Account_id", "count"),
                suspicious_account_rate=("Account_is_laundering", "mean"),
                median_tx_count=("tx_count", "median"),
                avg_cash_share=("cash_share", "mean"),
                avg_cross_border_share=("cross_border_share", "mean"),
            )
            .reindex(PROVINCE_TILE_LAYOUT.keys())
            .fillna(0)
        )
        max_rate = max(float(province_summary["suspicious_account_rate"].max()), 0.01)
        norm = Normalize(vmin=0, vmax=max_rate)
        cmap = plt.get_cmap("Reds")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlim(0.55, 7.05)
        ax.set_ylim(0.45, 6.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("South African Provincial AML Exposure Map", fontsize=15, fontweight="bold", pad=12)
        ax.text(
            0.58,
            0.55,
            "Tile cartogram: colour = suspicious account rate; text = accounts and median sequence length",
            fontsize=8.5,
            color="#4b5563",
        )
        for province, (x, y, w, h) in PROVINCE_TILE_LAYOUT.items():
            row = province_summary.loc[province]
            rate = float(row["suspicious_account_rate"])
            accounts = int(row["accounts"])
            median_tx = float(row["median_tx_count"])
            face = cmap(norm(rate))
            patch = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.035,rounding_size=0.08",
                facecolor=face,
                edgecolor="#2f2f2f",
                linewidth=1.15,
            )
            ax.add_patch(patch)
            short_name = province.replace("KwaZulu-Natal", "KZN").replace("Northern Cape", "N. Cape").replace("Western Cape", "W. Cape").replace("Eastern Cape", "E. Cape")
            text_color = "white" if norm(rate) > 0.62 else "#1f2933"
            ax.text(x + w / 2, y + h * 0.62, short_name, ha="center", va="center", fontsize=9.4, fontweight="bold", color=text_color)
            ax.text(x + w / 2, y + h * 0.36, f"{rate:.1%} AML\nn={accounts:,}, med={median_tx:.0f}", ha="center", va="center", fontsize=7.6, color=text_color, linespacing=1.05)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        cbar.set_label("Suspicious account rate (%)")
        finish_plot("16b_sa_province_aml_exposure_map.png")
        return province_summary


    def make_cross_border_country_risk_map(source_df):
        country_map_summary = (
            source_df[source_df["Cross_border_transaction"].eq(1)]
            .groupby("Counterparty_country")
            .agg(
                transactions=("Transaction_id", "count"),
                suspicious_rate=("Is_laundering", "mean"),
                median_amount=("Amount", "median"),
                iftr_reportable_share=("iftr_reportable_cross_border", "mean"),
                risk_tier=("Jurisdiction_risk_tier", modal_value),
            )
            .sort_values("transactions", ascending=False)
            .reset_index()
        )
        country_map_summary["lon"] = country_map_summary["Counterparty_country"].map(lambda value: COUNTRY_COORDS.get(value, (np.nan, np.nan))[0])
        country_map_summary["lat"] = country_map_summary["Counterparty_country"].map(lambda value: COUNTRY_COORDS.get(value, (np.nan, np.nan))[1])
        plot_df = country_map_summary.dropna(subset=["lon", "lat"]).copy()
        if plot_df.empty:
            return country_map_summary
        africa_outline = np.array([
            [-17, 36], [-5, 36], [8, 35], [23, 32], [34, 31], [45, 18],
            [51, 8], [48, -10], [42, -25], [32, -35], [18, -35],
            [7, -30], [-4, -18], [-12, -2], [-17, 12], [-17, 36],
        ])
        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.fill(africa_outline[:, 0], africa_outline[:, 1], color="#f4f1ea", ec="#888888", lw=0.9, zorder=0)
        ax.scatter([24.0], [-29.0], marker="*", s=280, c="#111827", label="South Africa", zorder=4)
        ax.text(17.8, -33.0, "South Africa", fontsize=8.2, fontweight="bold")
        max_tx = max(float(plot_df["transactions"].max()), 1.0)
        plot_df["marker_size"] = 60 + 520 * np.sqrt(plot_df["transactions"] / max_tx)
        for tier in ["SADC corridor", "FATF Increased Monitoring", "FATF Call for Action", "Other corridor"]:
            tier_df = plot_df[plot_df["risk_tier"].eq(tier)]
            if tier_df.empty:
                continue
            ax.scatter(
                tier_df["lon"], tier_df["lat"], s=tier_df["marker_size"],
                c=RISK_TIER_COLORS.get(tier, "#457B9D"),
                edgecolor="white", linewidth=0.9, alpha=0.88, label=tier, zorder=3,
            )
        label_df = plot_df[
            plot_df["risk_tier"].isin(["FATF Call for Action", "FATF Increased Monitoring"])
            | plot_df["transactions"].rank(method="first", ascending=False).le(12)
        ].copy()
        for _, row in label_df.iterrows():
            ax.annotate(
                row["Counterparty_country"],
                xy=(row["lon"], row["lat"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6.8,
                color="#1f2933",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.70),
                arrowprops=dict(arrowstyle="-", lw=0.45, color="#666666", shrinkA=0, shrinkB=4),
            )
        ax.set_title("Cross-Border and FATF Contextual Exposure Map", fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(min(-25, float(plot_df["lon"].min()) - 8), max(60, float(plot_df["lon"].max()) + 8))
        ax.set_ylim(min(-38, float(plot_df["lat"].min()) - 6), max(52, float(plot_df["lat"].max()) + 6))
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        ax.text(
            ax.get_xlim()[0] + 1,
            ax.get_ylim()[0] + 1.4,
            "Marker size = simulated transaction count; colour = jurisdiction tier used by the simulator. Coordinates are country centroids.",
            fontsize=8.2,
            color="#4b5563",
        )
        finish_plot("16c_cross_border_country_risk_map.png")
        return country_map_summary


    account_summary = (
        analysis_df.groupby("Account_id")
        .agg(
            tx_count=("Transaction_id", "count"),
            suspicious_tx_share=("Is_laundering", "mean"),
            total_amount=("Amount", "sum"),
            median_amount=("Amount", "median"),
            cash_share=("cash_transaction", "mean"),
            cross_border_share=("Cross_border_transaction", "mean"),
            rapid_repeat_share=("rapid_repeat_transaction", "mean"),
            unique_counterparties=("Counterparty_account", "nunique"),
            first_tx=("fulldatetime", "min"),
            last_tx=("fulldatetime", "max"),
            avg_context_risk=("transaction_context_risk_score", "mean"),
            account_lifecycle_segment=("account_lifecycle_segment", "first"),
        )
        .reset_index()
        .merge(
            receiver_accounts,
            left_on="Account_id",
            right_on="Receiver_account",
            how="left",
        )
    )
    if "Account_is_laundering" in account_summary.columns:
        account_summary["Account_is_laundering"] = (
            pd.to_numeric(account_summary["Account_is_laundering"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    if "Account_is_laundering" in receiver_accounts.columns:
        receiver_accounts["Account_is_laundering"] = (
            pd.to_numeric(receiver_accounts["Account_is_laundering"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    account_summary["Label"] = np.where(account_summary["Account_is_laundering"] == 1, "Suspicious AML account", "Legitimate account")
    account_summary["observed_span_days"] = (account_summary["last_tx"] - account_summary["first_tx"]).dt.days.add(1).clip(lower=1)
    account_summary["tx_per_active_month"] = account_summary["tx_count"] / np.maximum(account_summary["Active_observation_days"] / 30.4375, 1 / 30.4375)

    publication_tables = {}
    _progress("Building publication/insight tables and overview diagnostics.")
    publication_tables["simulation_kpis"] = pd.DataFrame([
        {"metric": "Observation start", "value": str(START_DATE.date())},
        {"metric": "Observation end", "value": str(END_DATE.date())},
        {"metric": "Observation days", "value": DAYS},
        {"metric": "Unique receiver accounts", "value": analysis_df["Account_id"].nunique()},
        {"metric": "Transactions", "value": len(analysis_df)},
        {"metric": "Suspicious transaction share", "value": round(float(analysis_df["Is_laundering"].mean()), 4)},
        {"metric": "Suspicious account share", "value": round(float(receiver_accounts["Account_is_laundering"].mean()), 4)},
        {"metric": "AML scenarios represented", "value": int(analysis_df.loc[analysis_df["Is_laundering"].eq(1), "AML_scenario_category"].nunique())},
        {"metric": "Normal behaviour categories represented", "value": int(analysis_df.loc[analysis_df["Is_laundering"].eq(0), "Normal_behavior_category"].nunique())},
        {"metric": "Median transactions per account", "value": int(account_summary["tx_count"].median())},
        {"metric": "95th percentile transactions per account", "value": int(account_summary["tx_count"].quantile(0.95))},
        {"metric": "Cross-border transaction share", "value": round(float(analysis_df["Cross_border_transaction"].mean()), 4)},
        {"metric": "FATF contextual exposure share", "value": round(float(analysis_df["Jurisdiction_risk_tier"].isin(["FATF Increased Monitoring", "FATF Call for Action"]).mean()), 4)},
    ])
    show_table("Simulation validation KPIs", publication_tables["simulation_kpis"])


    # Dataset-level overview before detailed diagnostics.
    _progress("Creating overview tables and dataset summary figures.")
    transaction_total = int(len(analysis_df))
    account_total = int(account_summary["Account_id"].nunique())
    publication_tables["dataset_overview"] = pd.DataFrame([
        {"Metric": "Receiver accounts", "Value": f"{account_total:,}"},
        {"Metric": "Transaction rows", "Value": f"{transaction_total:,}"},
        {"Metric": "Average transactions per account", "Value": f"{transaction_total / account_total:.2f}"},
        {"Metric": "Median transactions per account", "Value": f"{account_summary['tx_count'].median():,.0f}"},
        {"Metric": "95th percentile transactions per account", "Value": f"{account_summary['tx_count'].quantile(0.95):,.0f}"},
        {"Metric": "Maximum transactions per account", "Value": f"{account_summary['tx_count'].max():,.0f}"},
        {"Metric": "Legitimate transaction rows", "Value": f"{int((analysis_df['Is_laundering'] == 0).sum()):,}"},
        {"Metric": "Suspicious AML transaction rows", "Value": f"{int((analysis_df['Is_laundering'] == 1).sum()):,}"},
        {"Metric": "Legitimate accounts", "Value": f"{int((account_summary['Account_is_laundering'] == 0).sum()):,}"},
        {"Metric": "Suspicious AML accounts", "Value": f"{int((account_summary['Account_is_laundering'] == 1).sum()):,}"},
        {"Metric": "Transaction-level suspicious share", "Value": f"{analysis_df['Is_laundering'].mean() * 100:.2f}%"},
        {"Metric": "Account-level suspicious share", "Value": f"{account_summary['Account_is_laundering'].mean() * 100:.2f}%"},
    ])
    show_table("Generated dataset overview", publication_tables["dataset_overview"])

    publication_tables["transaction_descriptive_stats"] = (
        analysis_df.groupby("Label", observed=True)
        .agg(
            transactions=("Amount", "size"),
            mean_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            median_amount=("Amount", "median"),
            p25_amount=("Amount", lambda s: s.quantile(0.25)),
            p75_amount=("Amount", lambda s: s.quantile(0.75)),
            p95_amount=("Amount", lambda s: s.quantile(0.95)),
            max_amount=("Amount", "max"),
            cross_border_share=("Cross_border_transaction", "mean"),
        )
        .reset_index()
    )
    show_table("Transaction-level descriptive statistics", publication_tables["transaction_descriptive_stats"])

    publication_tables["account_sequence_descriptive_stats"] = (
        account_summary.groupby("Label", observed=True)
        .agg(
            accounts=("tx_count", "size"),
            mean_tx=("tx_count", "mean"),
            std_tx=("tx_count", "std"),
            median_tx=("tx_count", "median"),
            p25_tx=("tx_count", lambda s: s.quantile(0.25)),
            p75_tx=("tx_count", lambda s: s.quantile(0.75)),
            p95_tx=("tx_count", lambda s: s.quantile(0.95)),
            max_tx=("tx_count", "max"),
            mean_active_days=("Active_observation_days", "mean"),
        )
        .reset_index()
    )
    show_table("Account-level sequence descriptive statistics", publication_tables["account_sequence_descriptive_stats"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, data, label_col, title in [
        (axes[0], analysis_df, "Label", "Transaction Rows by Label"),
        (axes[1], account_summary, "Label", "Receiver Accounts by Label"),
    ]:
        counts = data[label_col].value_counts().sort_index()
        colors = ["#4C78A8" if "Legitimate" in str(idx) else "#E45756" for idx in counts.index]
        bars = ax.barh(counts.index.astype(str), counts.values, color=colors)
        total = counts.sum()
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        _set_count_axis(ax, axis="x")
        for bar, value in zip(bars, counts.values):
            ax.text(
                value + total * 0.014,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,.0f} ({value / total * 100:.2f}%)",
                va="center",
                ha="left",
                fontsize=8,
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            )
        ax.set_xlim(0, max(counts.values) * 1.28)
    finish_plot("00_dataset_overview_counts.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    amount_box_df = analysis_df.loc[:, ["Label", "Amount"]].dropna().copy()
    amount_box_df["Label"] = amount_box_df["Label"].astype(str)
    if len(amount_box_df) > DASHBOARD_PLOT_SAMPLE_ROWS:
        per_label_sample = max(DASHBOARD_PLOT_SAMPLE_ROWS // max(amount_box_df["Label"].nunique(), 1), 1)
        sampled_amount_parts = [
            part.sample(min(len(part), per_label_sample), random_state=SEED)
            for _, part in amount_box_df.groupby("Label", sort=False)
        ]
        amount_box_df = pd.concat(sampled_amount_parts, ignore_index=True)
    sns.boxplot(data=amount_box_df, x="Label", y="Amount", ax=axes[0], palette=["#4C78A8", "#E45756"], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Amount (R, log scale)")
    for tick, label in enumerate([item.get_text() for item in axes[0].get_xticklabels()]):
        label_amounts = analysis_df.loc[analysis_df["Label"].astype(str).eq(label), "Amount"]
        median = label_amounts.median()
        mean = label_amounts.mean()
        axes[0].text(tick, median * 1.20, f"Median R{median:,.0f}\nMean R{mean:,.0f}", ha="center", va="bottom", fontsize=8)
    sns.boxplot(data=account_summary, x="Label", y="tx_count", ax=axes[1], palette=["#4C78A8", "#E45756"], showfliers=False)
    axes[1].set_title("Transactions per Receiver Account")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Generated transactions per account")
    for tick, label in enumerate([item.get_text() for item in axes[1].get_xticklabels()]):
        median = account_summary.loc[account_summary["Label"].astype(str).eq(label), "tx_count"].median()
        axes[1].text(tick, median + 5, f"Median {median:,.0f}", ha="center", va="bottom", fontsize=8)
    finish_plot("00_transaction_account_boxplots.png")

    show_table(
        "Scenario catalogue used by the generator",
        scenario_catalog_df[["AML_scenario_category", "code", "predicate_offence", "stage", "risk_sector", "source_keys"]].sort_values("code"),
    )
    show_table(
        "Normal socioeconomic behaviour catalogue",
        normal_behavior_catalog_df[["Normal_behavior_category", "source_keys", "indicator"]].sort_values("Normal_behavior_category"),
    )


    # 1. Class imbalance at transaction and receiver-account levels.
    _progress("Creating class-imbalance and account-sequence figures.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    tx_class = analysis_df["Label"].value_counts(normalize=True).mul(100).rename("percent").reset_index()
    tx_class.columns = ["label", "percent"]
    sns.barplot(data=tx_class, x="label", y="percent", ax=axes[0], palette=["#4C78A8", "#E45756"])
    axes[0].set_title("Transaction Label Imbalance")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Share of transactions (%)")
    axes[0].tick_params(axis="x", rotation=15)
    acct_class = account_summary["Label"].value_counts(normalize=True).mul(100).rename("percent").reset_index()
    acct_class.columns = ["label", "percent"]
    sns.barplot(data=acct_class, x="label", y="percent", ax=axes[1], palette=["#4C78A8", "#E45756"])
    axes[1].set_title("Receiver Account Label Imbalance")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of accounts (%)")
    axes[1].tick_params(axis="x", rotation=15)
    finish_plot("01_label_imbalance_transaction_and_account.png")

    # 2. Heterogeneous sequence length by account and account lifecycle.
    plt.figure(figsize=(11, 5))
    sns.histplot(
        data=account_summary,
        x="tx_count",
        hue="Label",
        bins=55,
        element="step",
        stat="density",
        common_norm=False,
        log_scale=(True, False),
        palette={"Legitimate account": "#4C78A8", "Suspicious AML account": "#E45756"},
    )
    plt.title("Receiver Account Sequence Lengths")
    plt.xlabel("Transactions per receiver account, log scale")
    plt.ylabel("Density")
    finish_plot("02_receiver_account_sequence_lengths.png")

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=account_summary.sample(min(len(account_summary), 4000), random_state=SEED),
        x="Active_observation_days",
        y="tx_count",
        hue="Label",
        size="cash_share",
        sizes=(12, 160),
        alpha=0.55,
        palette={"Legitimate account": "#4C78A8", "Suspicious AML account": "#E45756"},
    )
    plt.yscale("log")
    plt.title("Active Observation Window vs Receiver Account Transaction Volume")
    plt.xlabel("Active observation days")
    plt.ylabel("Transactions per receiver account, log scale")
    finish_plot("03_active_days_vs_account_transaction_count.png")

    publication_tables["account_lifecycle_summary"] = (
        account_summary.groupby("account_lifecycle_segment")
        .agg(
            accounts=("Account_id", "count"),
            suspicious_account_rate=("Account_is_laundering", "mean"),
            median_tx_count=("tx_count", "median"),
            avg_tx_per_active_month=("tx_per_active_month", "mean"),
            avg_cross_border_share=("cross_border_share", "mean"),
            avg_cash_share=("cash_share", "mean"),
        )
        .sort_values("accounts", ascending=False)
    )
    show_table("Account lifecycle and sequence-length summary", publication_tables["account_lifecycle_summary"])


    # 3. Temporal coverage from 2024 through March 2026.
    _progress("Creating temporal coverage figures.")
    monthly_label = (
        analysis_df.groupby(["year_month_dt", "Label"])
        .size()
        .rename("transactions")
        .reset_index()
    )
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=monthly_label, x="year_month_dt", y="transactions", hue="Label", marker="o", palette=["#4C78A8", "#E45756"])
    plt.title("Monthly Transaction Volume by Label")
    plt.xlabel("Month")
    plt.ylabel("Transactions")
    plt.xticks(rotation=45)
    finish_plot("04_monthly_transaction_volume_by_label.png")

    monthly_rates = (
        analysis_df.groupby("year_month_dt")
        .agg(
            suspicious_rate=("Is_laundering", "mean"),
            cross_border_share=("Cross_border_transaction", "mean"),
            cash_share=("cash_transaction", "mean"),
            avg_amount=("Amount", "mean"),
        )
        .reset_index()
    )
    plt.figure(figsize=(13, 5))
    sns.lineplot(data=monthly_rates, x="year_month_dt", y="suspicious_rate", marker="o", color="#E45756", label="Suspicious rate")
    sns.lineplot(data=monthly_rates, x="year_month_dt", y="cross_border_share", marker="o", color="#72B7B2", label="Cross-border share")
    sns.lineplot(data=monthly_rates, x="year_month_dt", y="cash_share", marker="o", color="#F2CF5B", label="Cash share (%)")
    plt.title("Monthly Context Shares")
    plt.xlabel("Month")
    plt.ylabel("Share (%)")
    plt.xticks(rotation=45)
    finish_plot("05_monthly_context_shares.png")


    # 4. Legitimate baseline vs suspicious behaviour by amount and payment rail.
    _progress("Creating normal-versus-suspicious transaction behaviour figures.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    amount_plot_df = analysis_df[["Label", "log_amount"]]
    if len(amount_plot_df) > DASHBOARD_PLOT_SAMPLE_ROWS:
        amount_plot_df = amount_plot_df.sample(DASHBOARD_PLOT_SAMPLE_ROWS, random_state=SEED)
    sns.violinplot(data=amount_plot_df, x="Label", y="log_amount", inner="quartile", ax=axes[0], palette=["#4C78A8", "#E45756"])
    axes[0].set_title("Amount Distribution by Label")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("log10(amount)")
    top_payment_types = analysis_df["Payment_type"].value_counts().head(12).index
    payment_mix = (
        pd.crosstab(
            analysis_df.loc[analysis_df["Payment_type"].isin(top_payment_types), "Payment_type"],
            analysis_df.loc[analysis_df["Payment_type"].isin(top_payment_types), "Label"],
            normalize="columns",
        )
        .mul(100)
        .sort_values("Suspicious AML", ascending=True)
    )
    payment_mix.plot(kind="barh", ax=axes[1], color=["#4C78A8", "#E45756"])
    axes[1].set_title("Top Payment Rails by Label")
    axes[1].set_xlabel("Transactions within label (%)")
    axes[1].set_ylabel("")
    finish_plot("06_amount_and_payment_mix_by_label.png")

    publication_tables["label_comparison"] = (
        analysis_df.groupby("Label")
        .agg(
            transactions=("Transaction_id", "count"),
            median_amount=("Amount", "median"),
            avg_amount=("Amount", "mean"),
            cash_share=("cash_transaction", "mean"),
            cross_border_share=("Cross_border_transaction", "mean"),
            rapid_repeat_share=("rapid_repeat_transaction", "mean"),
            iftr_reportable_share=("iftr_reportable_cross_border", "mean"),
            avg_context_risk=("transaction_context_risk_score", "mean"),
        )
    )
    show_table("Legitimate baseline vs suspicious AML behaviour", publication_tables["label_comparison"])


    # 5. AML scenario diagnostics.
    _progress("Creating AML scenario mix and fingerprint figures.")
    publication_tables["aml_scenario_metrics"] = rate_table("AML_scenario_category", analysis_df[analysis_df["Is_laundering"].eq(1)])
    show_table("AML scenario metrics", publication_tables["aml_scenario_metrics"])

    plt.figure(figsize=(11, 8))
    scenario_counts = analysis_df.loc[analysis_df["Is_laundering"].eq(1), "AML_scenario_category"].value_counts().sort_values()
    scenario_counts.plot(kind="barh", color="#E45756")
    plt.title("Suspicious Transaction Mix by AML Scenario")
    plt.xlabel("Transactions")
    plt.ylabel("")
    finish_plot("07_aml_scenario_transaction_mix.png")

    scenario_heat = publication_tables["aml_scenario_metrics"][
        ["median_amount", "cash_share", "cross_border_share", "rapid_repeat_share", "avg_context_risk"]
    ].copy()
    scenario_heat["median_amount"] = np.log10(scenario_heat["median_amount"].clip(lower=1))
    scenario_heat = (scenario_heat - scenario_heat.mean()) / scenario_heat.std(ddof=0).replace(0, 1)
    plt.figure(figsize=(9, max(7, 0.35 * len(scenario_heat))))
    sns.heatmap(scenario_heat, cmap="vlag", center=0, linewidths=0.4)
    plt.title("AML Scenario Behaviour Fingerprints, Standardised")
    plt.xlabel("Metric")
    plt.ylabel("")
    finish_plot("08_aml_scenario_fingerprint_heatmap.png")


    # 5b. Scenario-level relationship network diagnostics.
    _progress("Creating selected scenario relationship-network figures.")
    def _compact_account_label(account_value):
        try:
            account_int = int(account_value)
            return f"A{account_int % 100000:05d}"
        except Exception:
            return str(account_value)


    def _format_rand_short(value):
        value = float(value)
        if value >= 1_000_000:
            return f"R{value / 1_000_000:.1f}m"
        if value >= 1_000:
            return f"R{value / 1_000:.0f}k"
        return f"R{value:.0f}"


    def _yes_no_flag(value):
        try:
            return "Yes" if int(value) == 1 else "No"
        except Exception:
            return "Yes" if bool(value) else "No"


    def _safe_profile_value(profile, column, fallback="Unknown"):
        if profile is None or column not in profile.index:
            return fallback
        value = profile.get(column, fallback)
        if pd.isna(value):
            return fallback
        return value


    def _monitored_account_profile_lines(account_id):
        profile_rows = receiver_accounts[receiver_accounts["Receiver_account"].eq(int(account_id))]
        profile = profile_rows.iloc[0] if not profile_rows.empty else None
        income_band = _safe_profile_value(profile, "Income_band")
        monthly_income = _safe_profile_value(profile, "Monthly_income_estimate", np.nan)
        if pd.notna(monthly_income):
            income_value = f"{income_band} ({_format_rand_short(monthly_income)}/mo)"
        else:
            income_value = income_band

        grant_linked = _yes_no_flag(_safe_profile_value(profile, "Grant_linked", 0))
        cross_border = _yes_no_flag(_safe_profile_value(profile, "Has_cross_border_links", 0))
        high_risk_sector = _yes_no_flag(_safe_profile_value(profile, "High_risk_sector_flag", 0))
        pep_or_high_risk = _yes_no_flag(
            max(
                int(_safe_profile_value(profile, "Is_PEP", 0)),
                int(_safe_profile_value(profile, "High_risk_person_flag", 0)),
            )
        )

        return [
            f"Profile: monitored account {_compact_account_label(account_id)}",
            f"Province: {_safe_profile_value(profile, 'Province')} | Customer: {_safe_profile_value(profile, 'Customer_type')}",
            f"Employment: {_safe_profile_value(profile, 'Employment_status')} | Sector: {_safe_profile_value(profile, 'Employment_sector')}",
            f"Income: {income_value} | Grant-linked: {grant_linked}",
            f"Cash: {_safe_profile_value(profile, 'Cash_intensity_band')} | Cross-border profile: {cross_border}",
            f"High-risk sector: {high_risk_sector} | PEP/high-risk person: {pep_or_high_risk}",
        ]


    def _spread_positions(nodes, x_value, y_top=1.55, y_bottom=-1.55):
        nodes = list(nodes)
        if not nodes:
            return {}
        if len(nodes) == 1:
            y_values = [0.0]
        else:
            y_values = np.linspace(y_top, y_bottom, len(nodes))
        return {node: (x_value, float(y)) for node, y in zip(nodes, y_values)}


    def _scenario_network_transactions(scenario_name, max_edges=16):
        scenario_df = analysis_df[
            analysis_df["Is_laundering"].eq(1)
            & analysis_df["AML_scenario_category"].eq(scenario_name)
        ].copy()
        if scenario_df.empty:
            return scenario_df, None

        account_stats = (
            scenario_df.groupby("Account_id")
            .agg(
                tx_count=("Transaction_id", "count"),
                unique_counterparties=("Counterparty_account", "nunique"),
                inflow_share=("Direction", lambda s: float(s.eq("Inflow").mean())),
                outflow_share=("Direction", lambda s: float(s.eq("Outflow").mean())),
                cross_border_share=("Cross_border_transaction", "mean"),
                total_amount=("Amount", "sum"),
            )
        )
        if account_stats.empty:
            return scenario_df.iloc[0:0], None

        score = account_stats["tx_count"] + account_stats["unique_counterparties"] * 2
        if scenario_name == "Mule Account / Rapid Pass-Through":
            score = score + (account_stats["inflow_share"].clip(0, 1) * account_stats["outflow_share"].clip(0, 1) * 30)
        elif scenario_name == "Cross-Border Layering via SADC Corridor":
            score = score + account_stats["cross_border_share"].fillna(0) * 30
        elif scenario_name == "Ponzi / Investment Scheme Collections":
            score = score + account_stats["inflow_share"].fillna(0) * 20

        account_id = score.sort_values(ascending=False).index[0]
        account_df = scenario_df[scenario_df["Account_id"].eq(account_id)].sort_values("fulldatetime").copy()
        if account_df.empty:
            return account_df, account_id

        if scenario_name == "Ponzi / Investment Scheme Collections":
            inflows = account_df[account_df["Direction"].eq("Inflow")].sort_values("Amount", ascending=False).head(max_edges - 4)
            outflows = account_df[account_df["Direction"].eq("Outflow")].sort_values("Amount", ascending=False).head(4)
            selected = pd.concat([inflows, outflows], ignore_index=False)
        elif scenario_name == "Mule Account / Rapid Pass-Through":
            half = max_edges // 2
            inflows = account_df[account_df["Direction"].eq("Inflow")].sort_values("fulldatetime").head(half)
            outflows = account_df[account_df["Direction"].eq("Outflow")].sort_values("fulldatetime").head(max_edges - len(inflows))
            selected = pd.concat([inflows, outflows], ignore_index=False)
            if len(selected) < max_edges:
                selected = pd.concat([selected, account_df.drop(selected.index).head(max_edges - len(selected))])
        else:
            cross = account_df[account_df["Cross_border_transaction"].eq(1)].sort_values("fulldatetime").head(max_edges)
            selected = cross if not cross.empty else account_df.head(max_edges)

        selected = selected.sort_values("fulldatetime").head(max_edges).copy()
        return selected, account_id


    def _draw_scenario_network(scenario_name, filename, max_edges=16):
        selected, account_id = _scenario_network_transactions(scenario_name, max_edges=max_edges)
        if selected.empty or account_id is None:
            print(f"No network diagnostic generated for {scenario_name}: no matching suspicious transactions.")
            return None

        rows = []
        main_node = f"account:{int(account_id)}"
        node_info = {
            main_node: {
                "label": f"Monitored\n{_compact_account_label(account_id)}",
                "role": "main",
                "country": "South Africa",
            }
        }

        for row in selected.itertuples(index=False):
            if row.Direction == "Inflow":
                source_id = f"account:{int(row.Sender_account)}"
                target_id = main_node
                role = "source"
            else:
                source_id = main_node
                target_id = f"account:{int(row.Receiver_account)}"
                role = "destination"

            counterparty_node = source_id if role == "source" else target_id
            counterparty_country = getattr(row, "Counterparty_country", "South Africa")
            counterparty_type = getattr(row, "Counterparty_type", "Counterparty")
            country_suffix = "" if counterparty_country == "South Africa" else f"\n{counterparty_country}"
            node_info.setdefault(
                counterparty_node,
                {
                    "label": f"{_compact_account_label(counterparty_node.split(':')[1])}{country_suffix}",
                    "role": role,
                    "country": counterparty_country,
                    "counterparty_type": counterparty_type,
                },
            )
            rows.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "direction": row.Direction,
                    "payment_type": row.Payment_type,
                    "amount": float(row.Amount),
                    "counterparty_country": counterparty_country,
                    "counterparty_type": counterparty_type,
                    "purpose": getattr(row, "Transaction_purpose_category", "Scenario movement"),
                    "time": row.fulldatetime,
                    "cross_border": int(getattr(row, "Cross_border_transaction", 0)),
                }
            )

        edge_df = pd.DataFrame(rows)
        edge_summary = (
            edge_df.groupby(["source", "target", "direction", "payment_type", "counterparty_country"], dropna=False)
            .agg(
                transactions=("amount", "size"),
                total_amount=("amount", "sum"),
                median_amount=("amount", "median"),
                first_time=("time", "min"),
                last_time=("time", "max"),
                cross_border=("cross_border", "max"),
            )
            .reset_index()
            .sort_values(["transactions", "total_amount"], ascending=False)
            .head(max_edges)
        )

        source_nodes = edge_summary.loc[edge_summary["target"].eq(main_node), "source"].drop_duplicates().tolist()
        destination_nodes = edge_summary.loc[edge_summary["source"].eq(main_node), "target"].drop_duplicates().tolist()
        positions = {main_node: (0.0, 0.0)}
        positions.update(_spread_positions(source_nodes, -2.5))
        positions.update(_spread_positions(destination_nodes, 2.5))

        for node in node_info:
            positions.setdefault(node, (-2.5, 0.0) if node != main_node else (0.0, 0.0))

        fig, ax = plt.subplots(figsize=(12.6, 8.0))
        ax.set_facecolor("#fbfbfb")
        max_total = max(float(edge_summary["total_amount"].max()), 1.0)

        for edge_idx, edge in edge_summary.reset_index(drop=True).iterrows():
            source_pos = positions[edge["source"]]
            target_pos = positions[edge["target"]]
            colour = "#2E7D32" if edge["direction"] == "Inflow" else "#C62828"
            if int(edge["cross_border"]) == 1:
                colour = "#6A3D9A"
            width = 1.0 + 2.8 * min(float(edge["total_amount"]) / max_total, 1.0)
            rad = 0.08 if edge_idx % 2 == 0 else -0.08
            ax.annotate(
                "",
                xy=target_pos,
                xytext=source_pos,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colour,
                    lw=width,
                    alpha=0.72,
                    shrinkA=22,
                    shrinkB=22,
                    connectionstyle=f"arc3,rad={rad}",
                ),
            )
            midpoint = ((source_pos[0] + target_pos[0]) / 2, (source_pos[1] + target_pos[1]) / 2)
            label = f"n={int(edge['transactions'])} | {edge['payment_type']} | {_format_rand_short(edge['total_amount'])}"
            ax.text(
                midpoint[0],
                midpoint[1] + (0.10 if edge_idx % 2 == 0 else -0.10),
                label,
                fontsize=7.0,
                ha="center",
                va="center",
                color="#263238",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d0d0d0", alpha=0.86),
            )

        for node, (x_value, y_value) in positions.items():
            role = node_info.get(node, {}).get("role", "counterparty")
            if role == "main":
                face, edge, size = "#1F77B4", "#0D3B66", 2600
                text_colour = "white"
            elif role == "source":
                face, edge, size = "#DFF2DF", "#2E7D32", 1450
                text_colour = "#173B1A"
            else:
                face, edge, size = "#FDE0DD", "#C62828", 1450
                text_colour = "#5A1717"
            if node_info.get(node, {}).get("country") != "South Africa" and role != "main":
                face, edge = "#E9D8FD", "#6A3D9A"
                text_colour = "#3D1766"
            ax.scatter([x_value], [y_value], s=size, c=face, edgecolors=edge, linewidths=1.7, zorder=5)
            ax.text(
                x_value,
                y_value,
                node_info.get(node, {}).get("label", node),
                ha="center",
                va="center",
                fontsize=7.5 if role != "main" else 8.2,
                fontweight="bold" if role == "main" else "normal",
                color=text_colour,
                zorder=6,
            )

        inflow_count = int((selected["Direction"] == "Inflow").sum())
        outflow_count = int((selected["Direction"] == "Outflow").sum())
        cross_border_share = float(selected["Cross_border_transaction"].mean()) if "Cross_border_transaction" in selected.columns else 0.0
        title = f"{scenario_name}\nSynthetic account relationship diagnostic"
        subtitle = (
            f"Main account {_compact_account_label(account_id)} | displayed transactions: {len(selected):,} | "
            f"inflows: {inflow_count:,} | outflows: {outflow_count:,} | cross-border: {cross_border_share * 100:.1f}%"
        )
        ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
        ax.text(0.5, 0.955, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9.2, color="#374151")
        ax.text(-2.5, 1.95, "Source accounts", ha="center", fontsize=9, fontweight="bold", color="#2E7D32")
        ax.text(2.5, 1.95, "Destination accounts", ha="center", fontsize=9, fontweight="bold", color="#C62828")

        profile_text = "\n".join(_monitored_account_profile_lines(account_id))
        ax.text(
            0.0,
            -1.62,
            profile_text,
            ha="center",
            va="top",
            fontsize=8.4,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.42", fc="#F3F8FF", ec="#1F77B4", lw=1.0, alpha=0.96),
        )
        ax.set_xlim(-3.35, 3.35)
        ax.set_ylim(-2.42, 2.18)
        ax.axis("off")
        finish_plot(filename)

        return {
            "scenario": scenario_name,
            "main_account": int(account_id),
            "displayed_transactions": int(len(selected)),
            "displayed_edges": int(len(edge_summary)),
            "unique_displayed_counterparties": int(edge_df[["source", "target"]].stack().nunique() - 1),
            "inflow_transactions": inflow_count,
            "outflow_transactions": outflow_count,
            "cross_border_share": round(cross_border_share, 4),
            "median_displayed_amount": round(float(selected["Amount"].median()), 2),
            "total_displayed_amount": round(float(selected["Amount"].sum()), 2),
        }


    scenario_network_specs = [
        ("Ponzi / Investment Scheme Collections", "31_network_ponzi_scheme_collections.png"),
        ("Mule Account / Rapid Pass-Through", "32_network_mule_pass_through.png"),
        ("Cross-Border Layering via SADC Corridor", "33_network_cross_border_layering.png"),
    ]
    scenario_network_rows = []
    for scenario_name, filename in scenario_network_specs:
        row = _draw_scenario_network(scenario_name, filename, max_edges=16)
        if row is not None:
            scenario_network_rows.append(row)

    if scenario_network_rows:
        publication_tables["scenario_network_examples"] = pd.DataFrame(scenario_network_rows)
        show_table("Scenario relationship network examples", publication_tables["scenario_network_examples"])


    # 6. Normal socioeconomic behaviour diagnostics.
    _progress("Creating normal socioeconomic behaviour figures.")
    publication_tables["normal_behaviour_metrics"] = (
        analysis_df.groupby("Normal_behavior_category")
        .agg(
            transactions=("Transaction_id", "count"),
            suspicious_rate=("Is_laundering", "mean"),
            median_amount=("Amount", "median"),
            avg_amount=("Amount", "mean"),
            cash_share=("cash_transaction", "mean"),
            cross_border_share=("Cross_border_transaction", "mean"),
            rapid_repeat_share=("rapid_repeat_transaction", "mean"),
            avg_context_risk=("transaction_context_risk_score", "mean"),
        )
        .drop(index="Suspicious overlay", errors="ignore")
        .sort_values("transactions", ascending=False)
    )
    show_table("Normal socioeconomic behaviour metrics", publication_tables["normal_behaviour_metrics"])

    plt.figure(figsize=(11, 5))
    normal_counts = (
        analysis_df["Normal_behavior_category"]
        .value_counts()
        .drop(labels=["Suspicious overlay"], errors="ignore")
        .sort_values()
    )
    normal_counts.plot(kind="barh", color="#4C78A8")
    plt.title("Legitimate Socioeconomic Behaviour Mix")
    plt.xlabel("Transactions")
    plt.ylabel("")
    finish_plot("09_normal_behaviour_mix.png")

    normal_heat = publication_tables["normal_behaviour_metrics"][
        ["median_amount", "cash_share", "cross_border_share", "rapid_repeat_share", "avg_context_risk"]
    ].copy()
    normal_heat["median_amount"] = np.log10(normal_heat["median_amount"].clip(lower=1))
    normal_heat = (normal_heat - normal_heat.mean()) / normal_heat.std(ddof=0).replace(0, 1)
    plt.figure(figsize=(9, max(5, 0.38 * len(normal_heat))))
    sns.heatmap(normal_heat, cmap="crest", linewidths=0.4)
    plt.title("Normal Behaviour Fingerprints, Standardised")
    plt.xlabel("Metric")
    plt.ylabel("")
    finish_plot("10_normal_behaviour_fingerprint_heatmap.png")


    # 7. Social grant diagnostics.
    _progress("Creating social grant diagnostics, when grant rows are available.")
    if "Social_grant_type" in analysis_df.columns:
        grant_type_df = analysis_df[analysis_df["Social_grant_type"].astype(str).ne("Not applicable")].copy()
    else:
        grant_type_df = analysis_df.iloc[0:0].copy()
    grant_inflow_df = (
        grant_type_df[
            grant_type_df["Direction"].eq("Inflow")
            if "Direction" in grant_type_df.columns
            else grant_type_df.index.to_series().isin(grant_type_df.index)
        ].copy()
        if not grant_type_df.empty
        else grant_type_df.copy()
    )
    if not grant_type_df.empty:
        publication_tables["social_grant_type_summary"] = (
            grant_type_df.groupby("Social_grant_type")
            .agg(
                grant_linked_transactions=("Transaction_id", "count"),
                social_grant_inflows=("Direction", lambda x: int(x.eq("Inflow").sum())),
                median_amount=("Amount", "median"),
                mean_amount=("Amount", "mean"),
                cash_share=("cash_transaction", "mean"),
            )
            .sort_values("grant_linked_transactions", ascending=False)
        )
        show_table("Social grant type summary", publication_tables["social_grant_type_summary"])

    if not grant_inflow_df.empty:
        grant_order = grant_inflow_df["Social_grant_type"].value_counts().index.tolist()

        plt.figure(figsize=(9, 4.8))
        grant_counts = grant_inflow_df["Social_grant_type"].value_counts().reindex(grant_order).sort_values()
        grant_counts.plot(kind="barh", color="#59A14F")
        plt.title("Social Grant Inflows by Grant Type")
        plt.xlabel("Transactions")
        plt.ylabel("")
        finish_plot("28_social_grant_type_distribution.png")

        plt.figure(figsize=(9, 4.8))
        grant_amounts = grant_inflow_df.groupby("Social_grant_type")["Amount"].median().reindex(grant_order).sort_values()
        grant_amounts.plot(kind="barh", color="#76B7B2")
        plt.title("Median Social Grant Inflow Amount by Grant Type")
        plt.xlabel("Median amount (R)")
        plt.ylabel("")
        finish_plot("29_social_grant_amount_by_type.png")

        plt.figure(figsize=(9, 4.8))
        grant_day_counts = pd.to_datetime(grant_inflow_df["Date"]).dt.day.value_counts().sort_index()
        grant_day_counts.plot(kind="bar", color="#F28E2B")
        plt.title("Social Grant Inflow Day-of-Month Pattern")
        plt.xlabel("Day of month")
        plt.ylabel("Transactions")
        finish_plot("30_social_grant_payment_day_pattern.png")


    # 8. Employment, sector, age, gender and education.
    _progress("Creating customer profile, employment, sector and demographic figures.")
    publication_tables["employment_status_account_summary"] = (
        account_summary.groupby("Employment_status")
        .agg(
            accounts=("Account_id", "count"),
            suspicious_account_rate=("Account_is_laundering", "mean"),
            median_tx_count=("tx_count", "median"),
            avg_tx_per_active_month=("tx_per_active_month", "mean"),
            avg_cash_share=("cash_share", "mean"),
            avg_cross_border_share=("cross_border_share", "mean"),
        )
        .sort_values("accounts", ascending=False)
    )
    show_table("Employment status account summary", publication_tables["employment_status_account_summary"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    employment_plot = publication_tables["employment_status_account_summary"].reset_index()
    sns.barplot(data=employment_plot, y="Employment_status", x="accounts", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Accounts by Employment Status")
    axes[0].set_xlabel("Accounts")
    axes[0].set_ylabel("")
    sns.barplot(data=employment_plot, y="Employment_status", x="suspicious_account_rate", ax=axes[1], color="#E45756")
    axes[1].set_title("Suspicious Account Rate by Employment Status")
    axes[1].set_xlabel("Suspicious account rate (%)")
    axes[1].set_ylabel("")
    finish_plot("11_employment_status_accounts_and_aml_rate.png")

    sector_summary = (
        account_summary.groupby("High_risk_sector_category")
        .agg(
            accounts=("Account_id", "count"),
            suspicious_account_rate=("Account_is_laundering", "mean"),
            median_tx_count=("tx_count", "median"),
            avg_cash_share=("cash_share", "mean"),
            avg_cross_border_share=("cross_border_share", "mean"),
        )
        .sort_values("accounts", ascending=False)
    )
    publication_tables["sector_summary"] = sector_summary
    show_table("Sector / risk-sector account summary", sector_summary)

    plt.figure(figsize=(11, 6))
    sector_plot = sector_summary.reset_index().sort_values("suspicious_account_rate")
    sns.barplot(data=sector_plot, y="High_risk_sector_category", x="suspicious_account_rate", color="#E45756")
    plt.title("Suspicious Account Rate by Sector Context")
    plt.xlabel("Suspicious account rate (%)")
    plt.ylabel("")
    finish_plot("12_sector_context_aml_rate.png")

    age_gender = account_summary.pivot_table(index="Age_band", columns="Gender", values="Account_is_laundering", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(age_gender, annot=True, fmt=".2%", cmap="Reds", linewidths=0.4)
    plt.title("Suspicious Account Rate by Age Band and Gender")
    plt.xlabel("Gender")
    plt.ylabel("Age band")
    finish_plot("13_age_gender_aml_rate_heatmap.png")

    education_summary = (
        account_summary.groupby("Education_level")
        .agg(
            accounts=("Account_id", "count"),
            suspicious_account_rate=("Account_is_laundering", "mean"),
            informal_business_share=("Informal_business_flag", "mean"),
            median_tx_count=("tx_count", "median"),
        )
        .sort_values("accounts", ascending=False)
    )
    publication_tables["education_summary"] = education_summary
    show_table("Education-level profile summary", education_summary)

    plt.figure(figsize=(10, 5))
    education_plot = education_summary.reset_index().sort_values("suspicious_account_rate")
    sns.barplot(data=education_plot, y="Education_level", x="suspicious_account_rate", color="#E45756")
    plt.title("Suspicious Account Rate by Education Level")
    plt.xlabel("Suspicious account rate (%)")
    plt.ylabel("")
    finish_plot("14_education_level_aml_rate.png")


    # 8. Nationality and cross-border exposure.
    _progress("Creating nationality, cross-border and geography figures.")
    nationality_summary = (
        account_summary.groupby("Nationality_country")
        .agg(
            accounts=("Account_id", "count"),
            suspicious_account_rate=("Account_is_laundering", "mean"),
            avg_cross_border_share=("cross_border_share", "mean"),
            migrant_worker_share=("Migrant_worker_flag", "mean"),
        )
        .query("accounts >= 10")
        .sort_values("accounts", ascending=False)
    )
    publication_tables["nationality_summary"] = nationality_summary
    show_table("Nationality and cross-border account summary", nationality_summary.head(20))

    plt.figure(figsize=(10, 6))
    nat_plot = nationality_summary.reset_index().head(20)
    sns.scatterplot(
        data=nat_plot,
        x="avg_cross_border_share",
        y="suspicious_account_rate",
        size="accounts",
        hue="Nationality_country",
        sizes=(40, 420),
        alpha=0.75,
    )
    plt.title("Nationality Context: Cross-Border Exposure vs Suspicious Account Rate")
    plt.xlabel("Average cross-border share (%)")
    plt.ylabel("Suspicious account rate (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_plot("15_nationality_cross_border_bubble.png")

    country_summary = (
        analysis_df[analysis_df["Cross_border_transaction"].eq(1)]
        .groupby("Counterparty_country")
        .agg(
            transactions=("Transaction_id", "count"),
            suspicious_rate=("Is_laundering", "mean"),
            median_amount=("Amount", "median"),
            iftr_reportable_share=("iftr_reportable_cross_border", "mean"),
        )
        .sort_values("transactions", ascending=False)
    )
    publication_tables["cross_border_country_summary"] = country_summary
    show_table("Cross-border counterparty country summary", country_summary.head(20))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    country_top = country_summary.head(15).reset_index()
    sns.barplot(data=country_top, y="Counterparty_country", x="transactions", ax=axes[0], color="#72B7B2")
    axes[0].set_title("Top Cross-Border Counterparty Countries")
    axes[0].set_xlabel("Transactions")
    axes[0].set_ylabel("")
    sns.barplot(data=country_top.sort_values("suspicious_rate"), y="Counterparty_country", x="suspicious_rate", ax=axes[1], color="#E45756")
    axes[1].set_title("Suspicious Rate in Top Cross-Border Countries")
    axes[1].set_xlabel("Suspicious transaction rate (%)")
    axes[1].set_ylabel("")
    finish_plot("16_cross_border_country_exposure.png")

    province_exposure_summary = make_province_exposure_map(account_summary)
    publication_tables["province_exposure_summary"] = province_exposure_summary
    show_table("Provincial exposure summary", province_exposure_summary)

    country_risk_map_summary = make_cross_border_country_risk_map(analysis_df)
    publication_tables["cross_border_country_risk_map_summary"] = country_risk_map_summary
    show_table("Cross-border country risk-map summary", country_risk_map_summary.head(25))


    # 9. Informal and unregistered business behaviour.
    _progress("Creating informal and unregistered business figures.")
    informal_summary = (
        account_summary.groupby("Business_registration_status")
        .agg(
            accounts=("Account_id", "count"),
            suspicious_account_rate=("Account_is_laundering", "mean"),
            median_tx_count=("tx_count", "median"),
            avg_daily_cash_turnover=("Estimated_daily_cash_turnover", "mean"),
            uses_personal_account_share=("Uses_personal_account_for_business", "mean"),
            keeps_records_share=("Keeps_business_records", "mean"),
            avg_cash_share=("cash_share", "mean"),
        )
        .sort_values("accounts", ascending=False)
    )
    publication_tables["informal_business_summary"] = informal_summary
    show_table("Informal and business registration profile summary", informal_summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    informal_plot = informal_summary.reset_index()
    sns.barplot(data=informal_plot, y="Business_registration_status", x="accounts", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Accounts by Business Registration Status")
    axes[0].set_xlabel("Accounts")
    axes[0].set_ylabel("")
    sns.barplot(data=informal_plot, y="Business_registration_status", x="avg_cash_share", ax=axes[1], color="#F2CF5B")
    axes[1].set_title("Average Cash Share by Registration Status")
    axes[1].set_xlabel("Cash share (%)")
    axes[1].set_ylabel("")
    finish_plot("17_informal_business_registration_cash_share.png")


    # 10. Sequential behaviour: speed, repetition and account density.
    _progress("Creating sequential timing and transaction-density figures.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    timing_plot_df = analysis_df.loc[
        analysis_df["time_since_last_hours"].between(0, 7 * 24),
        ["Label", "time_since_last_hours"]
    ]
    if len(timing_plot_df) > DASHBOARD_PLOT_SAMPLE_ROWS:
        timing_plot_df = timing_plot_df.sample(DASHBOARD_PLOT_SAMPLE_ROWS, random_state=SEED)
    sns.boxplot(
        data=timing_plot_df,
        x="Label",
        y="time_since_last_hours",
        ax=axes[0],
        palette=["#4C78A8", "#E45756"],
    )
    axes[0].set_yscale("symlog")
    axes[0].set_title("Time Since Previous Transaction")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Hours, symlog")
    rapid_by_scenario = (
        analysis_df[analysis_df["Is_laundering"].eq(1)]
        .groupby("AML_scenario_category")["rapid_repeat_transaction"]
        .mean()
        .sort_values()
    )
    rapid_by_scenario.plot(kind="barh", ax=axes[1], color="#E45756")
    axes[1].set_title("Rapid Repeat Share by AML Scenario")
    axes[1].set_xlabel("Rapid repeat share (%)")
    axes[1].set_ylabel("")
    finish_plot("18_sequential_timing_and_rapid_repeat.png")

    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=account_summary,
        x="Label",
        y="tx_per_active_month",
        palette=["#4C78A8", "#E45756"],
    )
    plt.yscale("log")
    plt.title("Receiver Account Transaction Density")
    plt.xlabel("")
    plt.ylabel("Transactions per active month, log scale")
    finish_plot("19_account_transaction_density_by_label.png")


    # 11. Correlation view for explainability of generated context variables.
    _progress("Creating generated-context correlation figure.")
    numeric_context_cols = [
        "Is_laundering", "Amount", "Cross_border_transaction", "SADC_corridor_transaction",
        "cash_transaction", "cash_threshold_near_miss", "iftr_reportable_cross_border",
        "rapid_repeat_transaction", "transaction_context_risk_score",
        "Informal_business_flag", "Uses_personal_account_for_business",
        "High_risk_sector_flag", "High_risk_person_flag", "Virtual_asset_exposure",
        "Procurement_vendor", "Public_sector_counterparty", "PEP_related_transaction",
        "account_tx_count", "account_cross_border_share", "account_cash_share",
    ]
    corr_df = analysis_df[numeric_context_cols]
    if len(corr_df) > DASHBOARD_PLOT_SAMPLE_ROWS:
        corr_df = corr_df.sample(DASHBOARD_PLOT_SAMPLE_ROWS, random_state=SEED)
    context_corr = corr_df.corr(numeric_only=True)
    publication_tables["context_correlation"] = context_corr
    plt.figure(figsize=(12, 10))
    sns.heatmap(context_corr, cmap="vlag", center=0, linewidths=0.25)
    plt.title("Correlation Among Generated AML Context Features")
    finish_plot("20_context_feature_correlation_heatmap.png")


    EXPORT_PUBLICATION_TABLES = os.getenv("AML_SIM_EXPORT_TABLES", "1") != "0"
    if EXPORT_PUBLICATION_TABLES:
        _progress("Exporting insight tables to CSV.")
        TABLE_CSV_DIR.mkdir(parents=True, exist_ok=True)
        exported_table_paths = []
        for table_name, table_value in publication_tables.items():
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(table_name)).strip("_")[:80]
            if isinstance(table_value, pd.Series):
                export_table = table_value.to_frame(name="value").reset_index()
            else:
                export_table = table_value.reset_index() if isinstance(table_value.index, pd.MultiIndex) or table_value.index.name else table_value.copy()
            csv_path = TABLE_CSV_DIR / f"{safe_name}.csv"
            export_table.to_csv(csv_path, index=False)
            exported_table_paths.append(csv_path)
        print(f"Insight table CSV files saved: {len(exported_table_paths)}")

    print("\nInsight plot files saved:")
    for path in publication_figures_saved_to:
        print(path)

    # Useful filtered dataframes for analysis examples:
    aml_examples_by_scenario = {
        scenario: analysis_df[analysis_df["AML_scenario_category"].eq(scenario)].head(10)
        for scenario in scenario_catalog_df["AML_scenario_category"]
    }
    normal_examples_by_behaviour = {
        behaviour: analysis_df[
            analysis_df["Normal_behavior_category"].eq(behaviour) &
            analysis_df["Is_laundering"].eq(0)
        ].head(10)
        for behaviour in normal_behavior_catalog_df["Normal_behavior_category"]
    }


    if include_extended:
        _progress("Creating extended customer-behaviour exploration figures.")
        # Extended Feature Exploration: Normal vs Suspicious Behaviour Patterns
        # This section is descriptive only. It compares simulated context attributes by
        # label so the simulation notes can explain how legitimate South African socioeconomic
        # activity differs from suspicious AML overlays.

        required_objects = ["analysis_df", "account_summary", "publication_tables", "finish_plot", "show_table"]
        available_objects = set(globals()) | set(locals())
        missing_objects = [name for name in required_objects if name not in available_objects]
        if missing_objects:
            raise RuntimeError(
                "Run the publication dashboard cell before this section. Missing: "
                + ", ".join(missing_objects)
            )

        extended_figures = []


        def finish_extended_plot(filename):
            before_count = len(publication_figures_saved_to)
            finish_plot(filename)
            if EXPORT_INSIGHT_PLOTS:
                extended_figures.extend(publication_figures_saved_to[before_count:])


        def ordered_category_values(series, preferred_order=None, top_n=None):
            values = series.dropna().astype(str)
            if preferred_order:
                present = [value for value in preferred_order if value in set(values)]
                remainder = [value for value in values.value_counts().index if value not in present]
                ordered = present + remainder
            else:
                ordered = list(values.value_counts().index)
            if top_n:
                ordered = ordered[:top_n]
            return ordered


        # 1. Income bands: account-level rate and transaction-level label mix.
        income_order = [
            "No formal income / grant only",
            "Below R3k",
            "R3k-R8k",
            "R8k-R15k",
            "R15k-R30k",
            "R30k-R60k",
            "R60k+",
            "Business variable",
        ]
        income_order = ordered_category_values(account_summary["Income_band"], income_order)

        publication_tables["income_band_account_summary"] = (
            account_summary.groupby("Income_band")
            .agg(
                accounts=("Account_id", "count"),
                suspicious_account_rate=("Account_is_laundering", "mean"),
                median_monthly_income=("Monthly_income_estimate", "median"),
                median_tx_count=("tx_count", "median"),
                avg_cash_share=("cash_share", "mean"),
                avg_cross_border_share=("cross_border_share", "mean"),
            )
            .reindex(income_order)
        )
        show_table("Income-band account summary", publication_tables["income_band_account_summary"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        income_plot = publication_tables["income_band_account_summary"].reset_index()
        sns.barplot(data=income_plot, y="Income_band", x="accounts", order=income_order, ax=axes[0], color="#4C78A8")
        axes[0].set_title("Accounts by Income Band")
        axes[0].set_xlabel("Accounts")
        axes[0].set_ylabel("")
        sns.barplot(data=income_plot, y="Income_band", x="suspicious_account_rate", order=income_order, ax=axes[1], color="#E45756")
        axes[1].set_title("Suspicious Account Rate by Income Band")
        axes[1].set_xlabel("Suspicious account rate (%)")
        axes[1].set_ylabel("")
        finish_extended_plot("21_income_band_accounts_and_aml_rate.png")

        income_label_mix = (
            pd.crosstab(analysis_df["Income_band"], analysis_df["Label"], normalize="columns")
            .mul(100)
            .reindex(income_order)
            .fillna(0)
        )
        publication_tables["income_band_label_mix"] = income_label_mix
        show_table("Income-band transaction mix within each label (%)", income_label_mix.round(2))

        plt.figure(figsize=(10, 6))
        income_label_mix.plot(kind="barh", color=["#4C78A8", "#E45756"], ax=plt.gca())
        plt.title("Income-Band Composition by Transaction Label")
        plt.xlabel("Transactions within label (%)")
        plt.ylabel("")
        finish_extended_plot("22_income_band_transaction_mix_by_label.png")


        # 2. Employment sectors: detailed sector context beyond the broad risk category.
        top_employment_sectors = ordered_category_values(account_summary["Employment_sector"], top_n=18)
        employment_sector_summary = (
            account_summary[account_summary["Employment_sector"].isin(top_employment_sectors)]
            .groupby("Employment_sector")
            .agg(
                accounts=("Account_id", "count"),
                suspicious_account_rate=("Account_is_laundering", "mean"),
                median_tx_count=("tx_count", "median"),
                avg_cash_share=("cash_share", "mean"),
                avg_cross_border_share=("cross_border_share", "mean"),
                informal_business_share=("Informal_business_flag", "mean"),
            )
            .sort_values("accounts", ascending=False)
        )
        publication_tables["employment_sector_summary"] = employment_sector_summary
        show_table("Employment-sector account summary", employment_sector_summary)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        sector_count_plot = employment_sector_summary.reset_index().sort_values("accounts", ascending=True)
        sns.barplot(data=sector_count_plot, y="Employment_sector", x="accounts", ax=axes[0], color="#4C78A8")
        axes[0].set_title("Top Employment Sectors by Account Count")
        axes[0].set_xlabel("Accounts")
        axes[0].set_ylabel("")
        sector_rate_plot = employment_sector_summary.reset_index().sort_values("suspicious_account_rate", ascending=True)
        sns.barplot(data=sector_rate_plot, y="Employment_sector", x="suspicious_account_rate", ax=axes[1], color="#E45756")
        axes[1].set_title("Suspicious Account Rate by Employment Sector")
        axes[1].set_xlabel("Suspicious account rate (%)")
        axes[1].set_ylabel("")
        finish_extended_plot("23_employment_sector_accounts_and_aml_rate.png")

        employment_sector_label_mix = (
            pd.crosstab(
                analysis_df.loc[analysis_df["Employment_sector"].isin(top_employment_sectors), "Employment_sector"],
                analysis_df.loc[analysis_df["Employment_sector"].isin(top_employment_sectors), "Label"],
                normalize="columns",
            )
            .mul(100)
            .fillna(0)
        )
        employment_sector_label_mix = employment_sector_label_mix.loc[
            employment_sector_label_mix.sum(axis=1).sort_values(ascending=True).index
        ]
        publication_tables["employment_sector_label_mix"] = employment_sector_label_mix
        show_table("Employment-sector transaction mix within each label (%)", employment_sector_label_mix.round(2))

        plt.figure(figsize=(10, max(6, 0.35 * len(employment_sector_label_mix))))
        employment_sector_label_mix.plot(kind="barh", color=["#4C78A8", "#E45756"], ax=plt.gca())
        plt.title("Employment-Sector Composition by Transaction Label")
        plt.xlabel("Transactions within label (%)")
        plt.ylabel("")
        finish_extended_plot("24_employment_sector_transaction_mix_by_label.png")


        # 3. Customer profiles: customer type and employment status compared by label.
        customer_type_summary = (
            account_summary.groupby("Customer_type")
            .agg(
                accounts=("Account_id", "count"),
                suspicious_account_rate=("Account_is_laundering", "mean"),
                median_tx_count=("tx_count", "median"),
                avg_cash_share=("cash_share", "mean"),
                avg_cross_border_share=("cross_border_share", "mean"),
            )
            .sort_values("accounts", ascending=False)
        )
        publication_tables["customer_type_summary"] = customer_type_summary
        show_table("Customer-type account summary", customer_type_summary)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        customer_plot = customer_type_summary.reset_index()
        sns.barplot(data=customer_plot, y="Customer_type", x="accounts", ax=axes[0], color="#4C78A8")
        axes[0].set_title("Accounts by Customer Type")
        axes[0].set_xlabel("Accounts")
        axes[0].set_ylabel("")
        sns.barplot(
            data=customer_plot.sort_values("suspicious_account_rate"),
            y="Customer_type",
            x="suspicious_account_rate",
            ax=axes[1],
            color="#E45756",
        )
        axes[1].set_title("Suspicious Account Rate by Customer Type")
        axes[1].set_xlabel("Suspicious account rate (%)")
        axes[1].set_ylabel("")
        finish_extended_plot("25_customer_type_accounts_and_aml_rate.png")

        profile_dims = ["Customer_type", "Employment_status", "Income_band", "Education_level"]
        profile_label_tables = {}
        for dim in profile_dims:
            profile_table = (
                analysis_df.groupby([dim, "Label"])
                .agg(
                    transactions=("Transaction_id", "count"),
                    median_amount=("Amount", "median"),
                    cash_share=("cash_transaction", "mean"),
                    cross_border_share=("Cross_border_transaction", "mean"),
                    rapid_repeat_share=("rapid_repeat_transaction", "mean"),
                    avg_context_risk=("transaction_context_risk_score", "mean"),
                )
                .reset_index()
                .sort_values([dim, "Label"])
            )
            profile_label_tables[dim] = profile_table
        publication_tables["profile_label_comparison_tables"] = profile_label_tables
        show_table("Customer type by label transaction metrics", profile_label_tables["Customer_type"])
        show_table("Income band by label transaction metrics", profile_label_tables["Income_band"])


        # 4. Context indicators: show which account-level risk-context attributes are
        # more common among suspicious accounts without treating them as deterministic.
        indicator_cols = [
            "Informal_business_flag",
            "Uses_personal_account_for_business",
            "High_risk_sector_flag",
            "High_risk_person_flag",
            "Is_PEP",
            "Procurement_vendor",
            "Intermediary_account_flag",
            "Virtual_asset_exposure",
            "Has_cross_border_links",
            "Targeted_financial_sanctions_match",
            "Adverse_media_flag",
        ]
        available_indicator_cols = [col for col in indicator_cols if col in account_summary.columns]
        indicator_by_label = (
            account_summary.groupby("Label")[available_indicator_cols]
            .mean()
            .T
            .sort_values("Suspicious AML account", ascending=True)
        )
        publication_tables["risk_indicator_share_by_label"] = indicator_by_label
        show_table("Risk-context indicator share by account label", indicator_by_label.round(4))

        plt.figure(figsize=(10, max(6, 0.35 * len(indicator_by_label))))
        indicator_by_label.plot(kind="barh", color=["#4C78A8", "#E45756"], ax=plt.gca())
        plt.title("Risk-Context Indicator Share by Account Label")
        plt.xlabel("Share of accounts")
        plt.ylabel("")
        finish_extended_plot("26_risk_indicator_share_by_account_label.png")


        # 5. Crossed profile view: where income and employment status interact.
        income_employment_heat = account_summary.pivot_table(
            index="Income_band",
            columns="Employment_status",
            values="Account_is_laundering",
            aggfunc="mean",
        ).reindex(income_order)
        plt.figure(figsize=(12, 6))
        sns.heatmap(income_employment_heat, annot=True, fmt=".1%", cmap="Reds", linewidths=0.4)
        plt.title("Suspicious Account Rate by Income Band and Employment Status")
        plt.xlabel("Employment status")
        plt.ylabel("Income band")
        finish_extended_plot("27_income_employment_aml_rate_heatmap.png")

        if extended_figures:
            print("\nExtended exploration plot files saved:")
            for path in extended_figures:
                print(path)
        else:
            print("\nExtended exploration plots displayed only because AML_SIM_EXPORT_INSIGHTS=0. By default they are saved to images/.")


    _progress(
        "Figure generation complete: "
        f"{len(publication_figures_saved_to):,} figure paths tracked, "
        f"{len(publication_tables) if 'publication_tables' in locals() else 0:,} tables prepared."
    )

    return {
        "df": df,
        "df_final": df_final,
        "receiver_accounts": receiver_accounts,
        "scenario_catalog_df": scenario_catalog_df,
        "normal_behavior_catalog_df": normal_behavior_catalog_df,
        "publication_tables": publication_tables if "publication_tables" in locals() else {},
        "publication_figures_saved_to": publication_figures_saved_to,
    }


def main() -> None:
    generate_figures()


if __name__ == "__main__":
    main()
