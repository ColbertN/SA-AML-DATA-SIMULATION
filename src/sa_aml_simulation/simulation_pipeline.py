"""Scriptable SA AML simulation pipeline.

This module mirrors the simulation logic from notebooks/african_data_simulation.ipynb
so the full data-generation workflow can be run without opening Jupyter.
"""

from __future__ import annotations

import calendar
import os
import random
import sys
import time
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
    BEHAVIOUR_ONLY_DIR,
    FINAL_RESULTS_DIR,
    FULL_SAML_DIR,
    OPTIONAL_ENGINEERED_DIR,
    PROJECT_ROOT,
    SIMULATED_DATA_DIR as DEFAULT_SIMULATED_DATA_DIR,
    ensure_project_dirs,
)
from src.sa_aml_simulation.figure_helpers import finish_plot


def run_simulation() -> dict:
    """Run the full simulation and export the configured data files.

    Environment variables such as AML_SIM_N_ACCOUNTS, AML_SIM_MAX_TX_PER_ACCOUNT,
    AML_SIM_EXPORT_FILES and AML_SIM_OUTPUT_PREFIX are honoured in the same way
    as the notebook. The returned dictionary contains the main dataframes and
    metadata for interactive use.
    """
    def _progress(message: str) -> None:
        print(f"[simulation] {message}", flush=True)

    _progress("Preparing folders and reading configuration.")
    ensure_project_dirs()

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
        "Configuration loaded: "
        f"{N_RECEIVER_ACCOUNTS:,} accounts, {START_DATE.date()} to {END_DATE.date()}, "
        f"max {MAX_TX_PER_ACCOUNT:,} transactions/account, seed {SEED}."
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
        "Debit Card", "Credit Card", "Cheque", "International Wire", "Social Grant",
        "Procurement EFT", "Crypto Exchange Off-Ramp"
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


    # Helper functions
    # ------------------------------------------------------------

    def weighted_choice(options, weights=None, size=None):
        options = list(options)
        if weights is None:
            return rng.choice(options, size=size)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        return rng.choice(options, p=weights, size=size)


    def draw_one(options, weights=None):
        return str(weighted_choice(options, weights=weights, size=None))


    def currency_for(country):
        return CURRENCY_MAP.get(country, "USD")


    def country_risk_tier(country):
        # FATF tiers take precedence over regional corridor status.
        if country in FATF_CALL_FOR_ACTION:
            return "FATF Call for Action"
        if country in FATF_INCREASED_MONITORING:
            return "FATF Increased Monitoring"
        if country == SA:
            return "Domestic"
        if country in SADC_CORRIDORS:
            return "SADC corridor"
        return "Other international"


    def combined_jurisdiction_tier(sender_country, receiver_country):
        sender_tier = country_risk_tier(sender_country)
        receiver_tier = country_risk_tier(receiver_country)
        return max([sender_tier, receiver_tier], key=lambda x: RISK_TIER_SCORE[x])


    def lognormal_amount(median, sigma, min_value=10, max_value=5_000_000):
        value = rng.lognormal(mean=np.log(median), sigma=sigma)
        return float(np.clip(value, min_value, max_value))


    def round_currency(value):
        return round(float(value), 2)


    def choose_profile_country(has_cross_border_links):
        # Most legitimate cross-border exposure in this study is regional SADC.
        roll = rng.random()
        if has_cross_border_links:
            if roll < 0.78:
                return draw_one(
                    SADC_CORRIDORS,
                    [0.06, 0.12, 0.06, 0.08, 0.13, 0.05, 0.03, 0.14, 0.10, 0.02, 0.05, 0.08, 0.08]
                )
            if roll < 0.95:
                return draw_one(FATF_INCREASED_MONITORING)
            return draw_one(FATF_CALL_FOR_ACTION)
        if roll < 0.965:
            return SA
        if roll < 0.992:
            return draw_one(SADC_CORRIDORS)
        return draw_one(FATF_INCREASED_MONITORING)


    def choose_transaction_country(acc, force_foreign=False, suspicious=False):
        if suspicious:
            roll = rng.random()
            if roll < 0.52:
                return draw_one(SADC_CORRIDORS)
            if roll < 0.88:
                return draw_one(FATF_INCREASED_MONITORING)
            if roll < 0.96:
                return draw_one(FATF_CALL_FOR_ACTION)
            return draw_one(OTHER_RELEVANT_COUNTRIES)
        if force_foreign or acc.Has_cross_border_links:
            return acc.Typical_foreign_counterparty_country if acc.Typical_foreign_counterparty_country != SA else draw_one(SADC_CORRIDORS)
        if rng.random() < 0.03:
            return draw_one(SADC_CORRIDORS)
        return SA


    def choose_employment_status(customer_type):
        table = {
            "Formal Employee": (["Employed"], [1]),
            "Informal Trader": (["Informal / Self-employed", "Unemployed"], [0.86, 0.14]),
            "Grant Recipient": (["Unemployed", "Pensioner", "Informal / Self-employed"], [0.56, 0.34, 0.10]),
            "Cash-Intensive SME": (["Business Owner", "Informal / Self-employed"], [0.62, 0.38]),
            "Migrant Worker / Remittance User": (["Employed", "Informal / Self-employed", "Unemployed"], [0.48, 0.34, 0.18]),
            "Student / Youth Account": (["Student", "Unemployed", "Employed"], [0.64, 0.24, 0.12]),
            "Public Sector Employee": (["Employed"], [1]),
            "Business Owner": (["Business Owner", "Informal / Self-employed"], [0.76, 0.24]),
            "Procurement Vendor": (["Business Owner", "Employed"], [0.82, 0.18]),
            "Corporate Entity": (["Business Entity"], [1]),
            "NPO / Community Organisation": (["NPO / Community"], [1]),
        }
        options, weights = table[customer_type]
        return draw_one(options, weights)


    def choose_sector(customer_type, employment_status):
        if customer_type == "Public Sector Employee":
            return draw_one(["Government", "Education", "Healthcare", "Municipality"], [0.42, 0.24, 0.20, 0.14])
        if customer_type == "Procurement Vendor":
            return draw_one(["Construction", "Security", "Health supplies", "Transport", "IT services"], [0.34, 0.18, 0.17, 0.16, 0.15])
        if customer_type == "Corporate Entity":
            return draw_one(["Import/export", "Real estate", "Construction", "Professional services", "Retail group"], [0.22, 0.22, 0.20, 0.18, 0.18])
        if customer_type == "Business Owner":
            return draw_one(["Retail", "Property services", "Motor vehicle dealer", "High-value goods", "Legal / trust services", "Casino / gambling", "Precious metals / stones", "Import/export", "Crypto / VASP"], [0.22, 0.10, 0.08, 0.08, 0.08, 0.05, 0.05, 0.18, 0.16])
        if customer_type == "Cash-Intensive SME":
            return draw_one(["Spaza / food trade", "Taxi / transport", "Hospitality", "Retail", "Personal services", "Second-hand goods"], [0.28, 0.22, 0.15, 0.17, 0.10, 0.08])
        if employment_status == "Informal / Self-employed":
            return draw_one(["Informal retail", "Taxi / transport", "Personal services", "Agriculture", "Domestic work", "Spaza / food trade"], [0.30, 0.20, 0.17, 0.13, 0.10, 0.10])
        if employment_status == "Employed":
            return draw_one(["Retail", "Mining", "Construction", "Financial services", "Manufacturing", "Security", "Healthcare"], [0.20, 0.14, 0.16, 0.13, 0.14, 0.12, 0.11])
        if employment_status == "Student":
            return "Education / student"
        if employment_status == "Pensioner":
            return "Retired"
        if employment_status == "Unemployed":
            return "Not employed"
        if employment_status == "NPO / Community":
            return "Community services"
        return "Other"


    def choose_income_band(customer_type, employment_status):
        if customer_type == "Grant Recipient":
            return "Grant-linked"
        if employment_status == "Unemployed":
            return "No regular income"
        if employment_status in ["Student", "Pensioner"]:
            return draw_one(["Grant-linked", "Low / irregular"], [0.55, 0.45])
        if customer_type in ["Cash-Intensive SME", "Business Owner", "Procurement Vendor", "Corporate Entity"]:
            return draw_one(["Business turnover", "Upper-middle", "Middle"], [0.62, 0.22, 0.16])
        if employment_status == "Informal / Self-employed":
            return draw_one(["Low / irregular", "Low formal", "Middle"], [0.54, 0.34, 0.12])
        return draw_one(["Low formal", "Middle", "Upper-middle"], [0.40, 0.47, 0.13])


    def monthly_income_from_band(income_band):
        low, high = INCOME_RANGES[income_band]
        if low == 0:
            return round_currency(rng.uniform(low, high))
        # A triangular draw gives more mass around realistic middle values.
        return round_currency(rng.triangular(low, (low + high) / 2, high))


    def cash_intensity(customer_type, employment_status, sector):
        base = {
            "Formal Employee": 0.30,
            "Informal Trader": 0.86,
            "Grant Recipient": 0.68,
            "Cash-Intensive SME": 0.82,
            "Migrant Worker / Remittance User": 0.58,
            "Student / Youth Account": 0.42,
            "Public Sector Employee": 0.22,
            "Business Owner": 0.52,
            "Procurement Vendor": 0.36,
            "Corporate Entity": 0.26,
            "NPO / Community Organisation": 0.48,
        }[customer_type]
        if "Taxi" in sector or "Informal" in sector or "Spaza" in sector:
            base += 0.10
        if employment_status in ["Unemployed", "Pensioner"]:
            base += 0.06
        return float(np.clip(base + rng.normal(0, 0.08), 0.03, 0.98))


    def cash_band(score):
        if score >= 0.75:
            return "Very high"
        if score >= 0.55:
            return "High"
        if score >= 0.30:
            return "Moderate"
        return "Low"


    def sample_age(customer_type, employment_status):
        if customer_type == "Student / Youth Account":
            return int(np.clip(rng.normal(23, 4), 18, 35))
        if employment_status == "Pensioner":
            return int(np.clip(rng.normal(67, 7), 55, 90))
        if customer_type == "Grant Recipient":
            return int(np.clip(rng.normal(43, 18), 18, 88))
        if employment_status == "Unemployed":
            return int(np.clip(rng.normal(34, 12), 18, 65))
        if employment_status in ["Business Owner", "Informal / Self-employed"]:
            return int(np.clip(rng.normal(41, 12), 18, 75))
        return int(np.clip(rng.normal(38, 11), 18, 72))


    def age_band(age):
        if age < 25:
            return "18-24"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        if age < 65:
            return "55-64"
        return "65+"


    def choose_nationality(customer_type, has_cross_border_links):
        if customer_type == "Migrant Worker / Remittance User" or has_cross_border_links:
            roll = rng.random()
            if roll < 0.58:
                return SA
            if roll < 0.92:
                return draw_one(["Zimbabwe", "Lesotho", "Mozambique", "Malawi", "Eswatini", "Namibia"], [0.30, 0.22, 0.18, 0.12, 0.10, 0.08])
            return draw_one(OTHER_RELEVANT_COUNTRIES + FATF_INCREASED_MONITORING)
        return SA if rng.random() < 0.965 else draw_one(SADC_CORRIDORS)


    def residency_status(nationality):
        if nationality == SA:
            return "Citizen / permanent resident"
        return draw_one(["Documented migrant", "Temporary worker", "Asylum/refugee status", "Undocumented / weak documentation"], [0.54, 0.24, 0.10, 0.12])


    def choose_primary_social_grant_type(customer_type, employment_status, age, grant_linked):
        if int(grant_linked) != 1:
            return "Not applicable"
        age = int(age)
        if age >= 60 or employment_status == "Pensioner":
            return draw_one(["old_age", "child_support", "grant_in_aid", "srd"], [0.70, 0.16, 0.08, 0.06])
        if age < 25 or customer_type == "Student / Youth Account":
            return draw_one(["srd", "child_support", "foster_child"], [0.50, 0.40, 0.10])
        if employment_status == "Unemployed":
            return draw_one(["child_support", "srd", "disability", "foster_child"], [0.54, 0.26, 0.12, 0.08])
        return draw_one(["child_support", "disability", "care_dependency", "foster_child", "srd"], [0.58, 0.14, 0.10, 0.08, 0.10])


    def choose_education(customer_type, employment_status):
        if customer_type == "Student / Youth Account":
            return draw_one(["Matric", "Tertiary in progress", "Other tertiary"], [0.42, 0.48, 0.10])
        if employment_status in ["Business Entity", "Business Owner"]:
            return draw_one(["Less than matric", "Matric", "Other tertiary", "Graduate"], [0.24, 0.42, 0.24, 0.10])
        if employment_status in ["Informal / Self-employed", "Unemployed"]:
            return draw_one(["Less than matric", "Matric", "Other tertiary"], [0.48, 0.40, 0.12])
        return draw_one(["Less than matric", "Matric", "Other tertiary", "Graduate"], [0.22, 0.42, 0.25, 0.11])


    def choose_channel_preference(cash_score, customer_type):
        if cash_score >= 0.75:
            return draw_one(["Branch/ATM cash first", "Agent/cash and mobile", "Mixed cash and digital"], [0.42, 0.33, 0.25])
        if customer_type == "Student / Youth Account":
            return draw_one(["Mobile app first", "PayShap/instant payments", "Card first"], [0.46, 0.36, 0.18])
        if customer_type == "Migrant Worker / Remittance User":
            return draw_one(["Money transfer operator", "Mobile/remittance app", "Branch/ATM cash first"], [0.42, 0.34, 0.24])
        return draw_one(["Card first", "Mobile app first", "Mixed cash and digital", "Branch/ATM cash first"], [0.28, 0.27, 0.30, 0.15])


    def choose_business_premises(row):
        if row.Informal_business_flag == 0 and row.Customer_type not in ["Business Owner", "Procurement Vendor", "Corporate Entity", "Cash-Intensive SME", "NPO / Community Organisation"]:
            return "Not business"
        if row.Informal_business_flag == 1:
            return draw_one(["Home-based", "Street stall", "Taxi rank / transport node", "Spaza or township shop", "Market", "Online/social media"], [0.24, 0.22, 0.16, 0.20, 0.12, 0.06])
        return draw_one(["Formal premises", "Home office", "Industrial site", "Professional office", "Online platform"], [0.46, 0.14, 0.16, 0.16, 0.08])


    def high_risk_sector_category(sector, customer_type):
        if sector in ["Real estate", "Property services"]:
            return "Property / real estate"
        if sector in ["Motor vehicle dealer", "High-value goods", "Second-hand goods"]:
            return "Motor vehicle / high-value goods"
        if sector in ["Legal / trust services", "Professional services"]:
            return "Legal / professional intermediary"
        if sector in ["Casino / gambling"]:
            return "Casino / gambling"
        if sector in ["Precious metals / stones"]:
            return "Precious metals / stones"
        if sector in ["Crypto / VASP"]:
            return "Crypto asset services"
        if sector in ["Import/export"]:
            return "Import/export trade"
        if sector in ["Construction", "Security", "Health supplies", "Transport", "IT services"] or customer_type == "Procurement Vendor":
            return "Public procurement adjacent"
        if "Taxi" in sector or "Spaza" in sector or "Informal" in sector:
            return "Cash-intensive informal economy"
        if customer_type == "NPO / Community Organisation":
            return "NPO / community organisation"
        return "Standard retail / household"


    def choose_legit_behaviour(row):
        if row.Customer_type == "Grant Recipient":
            return "Grant-linked inflows and survival payments"
        if row.Customer_type == "Student / Youth Account":
            return "Low-value high-frequency retail payments"
        if row.Customer_type == "Migrant Worker / Remittance User" or row.Has_cross_border_links:
            return "Remittance corridor activity"
        if row.Informal_business_flag == 1:
            if "Spaza" in row.Employment_sector or row.Business_premises_type == "Spaza or township shop":
                return "Spaza / township retail cash cycle"
            if "Taxi" in row.Employment_sector or row.Business_premises_type == "Taxi rank / transport node":
                return "Taxi or transport cash cycle"
            if row.Business_registered == 1:
                return "Informal registered business cash cycle"
            return "Informal unregistered business cash cycle"
        if row.Customer_type in ["Business Owner", "Procurement Vendor", "Corporate Entity", "NPO / Community Organisation", "Cash-Intensive SME"]:
            return "Business operating payments"
        if row.Employment_status in ["Unemployed", "Informal / Self-employed"]:
            return draw_one(["Irregular income / gig work", "Stokvel / rotating savings contributions"], [0.78, 0.22])
        return "Salary and household payments"


    def scenario_weight(row, scenario_name, base_weight):
        weight = float(base_weight)
        if row.Cash_intensity_score >= 0.65 and any(key in scenario_name for key in ["Structuring", "Cash Courier", "Narcotics"]):
            weight += 1.4
        if row.Employment_status in ["Student", "Unemployed"] and any(key in scenario_name for key in ["Mule", "Student", "Cybercrime"]):
            weight += 1.2
        if row.Has_cross_border_links and any(key in scenario_name for key in ["Cross-Border", "High-Risk Jurisdiction", "Trade-Based", "NPO"]):
            weight += 1.2
        if row.Profile_jurisdiction_risk_tier != "Domestic" and any(key in scenario_name for key in ["High-Risk Jurisdiction", "Cross-Border"]):
            weight += 1.4
        if row.Intermediary_account_flag and any(key in scenario_name for key in ["Intermediary", "Complex Legal"]):
            weight += 1.3
        if row.Procurement_vendor or row.Public_sector_exposure or row.Is_PEP:
            if any(key in scenario_name for key in ["Procurement", "Shell", "Complex Legal"]):
                weight += 1.5
        if row.Beneficial_ownership_opacity >= 0.60 and any(key in scenario_name for key in ["Shell", "Complex Legal", "Trade-Based"]):
            weight += 1.4
        if row.Virtual_asset_exposure and "Crypto" in scenario_name:
            weight += 1.8
        if row.High_risk_sector_category == "Property / real estate" and "Property" in scenario_name:
            weight += 1.6
        if row.High_risk_sector_category == "Motor vehicle / high-value goods" and "Motor Vehicle" in scenario_name:
            weight += 1.6
        if row.High_risk_sector_category == "Precious metals / stones" and "Gold" in scenario_name:
            weight += 1.4
        if row.High_risk_sector_category == "Casino / gambling" and "Casino" in scenario_name:
            weight += 1.4
        if row.Customer_type == "NPO / Community Organisation" and "NPO" in scenario_name:
            weight += 1.4
        if row.Informal_business_flag == 1 and any(key in scenario_name for key in ["Structuring", "Cash Courier", "Narcotics"]):
            weight += 0.8
        if row.High_risk_person_flag == 1 and any(key in scenario_name for key in ["Procurement", "High-Risk Jurisdiction", "Complex Legal"]):
            weight += 1.0
        return weight


    def choose_laundering_type(row):
        if row.Account_is_laundering == 0:
            return "Normal"
        weights = [scenario_weight(row, name, AML_SCENARIO_CATALOG[name]["weight"]) for name in LAUNDERING_TYPES]
        return draw_one(LAUNDERING_TYPES, weights)


    def random_date_between(start_dt, end_dt):
        start_dt = pd.to_datetime(start_dt).to_pydatetime()
        end_dt = pd.to_datetime(end_dt).to_pydatetime()
        if end_dt < start_dt:
            end_dt = start_dt
        span_days = max(0, (end_dt.date() - start_dt.date()).days)
        return (start_dt + timedelta(days=int(rng.integers(0, span_days + 1)))).date()


    def scenario_new_account_probability(row):
        # Some AML patterns often use newly opened or recently repurposed accounts.
        if row.Account_is_laundering == 0:
            return 0.0
        high_new_account_scenarios = [
            "Mule Account / Rapid Pass-Through",
            "Cybercrime Proceeds",
            "Stolen Student Loan or Grant Abuse",
            "Crypto Asset Off-Ramp",
            "Fraudulent Company or Invoice Scam",
        ]
        medium_new_account_scenarios = [
            "Structuring / Smurfing Cash Deposits",
            "Ponzi / Investment Scheme Collections",
            "High-Risk Jurisdiction Layering",
            "NPO / Community Organisation Misuse",
        ]
        if row.Laundering_type in high_new_account_scenarios:
            return 0.56
        if row.Laundering_type in medium_new_account_scenarios:
            return 0.34
        return 0.18


    def choose_account_open_date(row):
        if getattr(row, "Opened_for_laundering_flag", 0) == 1:
            # New laundering accounts are spread across the whole observation
            # window, with extra mass in the final six months.
            if rng.random() < 0.55:
                start = max(START_DATE, END_DATE - timedelta(days=180))
            else:
                start = START_DATE
            return random_date_between(start, END_DATE)

        # Normal accounts and established laundering accounts often pre-date the
        # observation window. Students and thin-file customers are more likely to
        # enter during the observation window.
        if row.Customer_type == "Student / Youth Account" and rng.random() < 0.38:
            return random_date_between(START_DATE, END_DATE)
        if row.Financial_inclusion_segment in ["Thin-file / underbanked", "Grant-card dependent"] and rng.random() < 0.16:
            return random_date_between(START_DATE, END_DATE)
        if row.Customer_type in ["Corporate Entity", "Procurement Vendor", "Business Owner"] and rng.random() < 0.10:
            return random_date_between(START_DATE, END_DATE)

        pre_observation_age_days = int(np.clip(rng.gamma(shape=2.8, scale=430), 30, 4200))
        return (START_DATE - timedelta(days=pre_observation_age_days)).date()


    def choose_first_observed_transaction_date(row):
        open_date = pd.to_datetime(row.Account_open_date).date()
        observation_start = max(open_date, START_DATE.date())
        max_delay = 10 if row.Opened_for_laundering_flag == 1 else 45
        delay = int(rng.integers(0, max_delay + 1))
        first_seen = observation_start + timedelta(days=delay)
        return min(first_seen, END_DATE.date())


    def account_activity_multiplier(row):
        # Overdispersion makes transaction counts account-specific while still
        # being anchored in the customer's normal or suspicious behaviour.
        multiplier = rng.lognormal(mean=0.0, sigma=0.42)
        behaviour_adjustment = {
            "Salary and household payments": 0.80,
            "Grant-linked inflows and survival payments": 0.62,
            "Informal registered business cash cycle": 1.18,
            "Informal unregistered business cash cycle": 1.34,
            "Spaza / township retail cash cycle": 1.46,
            "Taxi or transport cash cycle": 1.56,
            "Low-value high-frequency retail payments": 1.42,
            "Remittance corridor activity": 0.92,
            "Stokvel / rotating savings contributions": 0.66,
            "Business operating payments": 1.12,
            "Irregular income / gig work": 0.74,
        }.get(row.Legitimate_behaviour, 1.0)
        multiplier *= behaviour_adjustment
        multiplier *= 1 + 0.35 * row.Cash_intensity_score
        multiplier *= 1 + 0.20 * int(row.Uses_personal_account_for_business == 1)
        # Do not inflate transaction counts merely because an account is suspicious.
        # Suspicious behaviour is injected as a minority overlay inside the ordinary
        # account history, which avoids direct count-based target leakage.
        return float(np.clip(multiplier, 0.18, 1.70))


    def expected_transaction_count(row):
        base_lambda = LEGIT_LAMBDA[row.Legitimate_behaviour]
        active_years = max(row.Active_observation_days, 1) / 365.25
        return max(0.25, base_lambda * active_years * row.Account_activity_multiplier)


    def draw_account_transaction_count(acc):
        expected = max(float(acc.Expected_transaction_count), 0.25)
        # Gamma-Poisson mixture gives realistic variation: two accounts with the
        # same profile can still have different transaction counts.
        shape = 2.00 if acc.Legitimate_behaviour in [
            "Low-value high-frequency retail payments",
            "Taxi or transport cash cycle",
            "Spaza / township retail cash cycle",
        ] else 2.40
        account_lambda = rng.gamma(shape=shape, scale=expected / shape)
        count = int(rng.poisson(account_lambda))

        if acc.Legitimate_behaviour in ["Low-value high-frequency retail payments", "Taxi or transport cash cycle", "Spaza / township retail cash cycle"]:
            min_tx = 4
        elif acc.Legitimate_behaviour in ["Salary and household payments", "Business operating payments"]:
            min_tx = 2
        else:
            min_tx = 1
        count = max(min_tx, count)
        if count > MAX_TX_PER_ACCOUNT:
            upper_tail_floor = max(min_tx, int(MAX_TX_PER_ACCOUNT * rng.uniform(0.68, 0.90)))
            count = int(rng.integers(upper_tail_floor, MAX_TX_PER_ACCOUNT))
        return int(np.clip(count, 1, MAX_TX_PER_ACCOUNT))


    def account_active_datetime_bounds(acc=None):
        if acc is None:
            return START_DATE, END_DATETIME
        active_start = pd.to_datetime(acc.First_observed_transaction_date).to_pydatetime()
        active_days = max(1, int(acc.Active_observation_days))
        active_end = min(END_DATETIME, active_start + timedelta(days=active_days) - timedelta(seconds=1))
        if active_end < active_start:
            active_end = active_start
        return active_start, active_end


    def clamp_account_datetime(acc, tx_datetime):
        active_start, active_end = account_active_datetime_bounds(acc)
        if tx_datetime < active_start:
            tx_datetime = active_start
        elif tx_datetime > active_end:
            tx_datetime = active_end
        return tx_datetime.replace(microsecond=0)


    SECTOR_PAY_DAY_PROFILES = {
        "Government": ([15, 25, 30], [0.58, 0.26, 0.16]),
        "Municipality": ([15, 25, 30], [0.45, 0.36, 0.19]),
        "Education": ([22, 15, 25], [0.58, 0.24, 0.18]),
        "Healthcare": ([15, 25, 30], [0.42, 0.38, 0.20]),
        "Financial services": ([20, 25, 30], [0.55, 0.32, 0.13]),
        "Insurance": ([25, 20, 30], [0.52, 0.34, 0.14]),
        "Mining": ([30, 31, 25], [0.48, 0.34, 0.18]),
        "Retail": ([25, 30, 31], [0.55, 0.30, 0.15]),
        "Manufacturing": ([25, 30, 31], [0.48, 0.34, 0.18]),
        "Construction": ([25, 30, 31], [0.45, 0.35, 0.20]),
        "Security": ([25, 30, 31], [0.44, 0.36, 0.20]),
        "Telecommunications": ([25, 20, 30], [0.45, 0.36, 0.19]),
        "Professional services": ([25, 20, 30], [0.50, 0.30, 0.20]),
    }
    DEFAULT_PAY_DAY_PROFILE = ([25, 30, 31, 20, 15, 22], [0.38, 0.23, 0.14, 0.11, 0.08, 0.06])


    def sector_pay_day_profile(acc):
        sector = getattr(acc, "Employment_sector", "")
        if getattr(acc, "Public_sector_exposure", 0) == 1 and sector not in SECTOR_PAY_DAY_PROFILES:
            return ([15, 25, 30], [0.48, 0.32, 0.20])
        return SECTOR_PAY_DAY_PROFILES.get(sector, DEFAULT_PAY_DAY_PROFILE)


    def valid_month_pay_date(year, month, target_day):
        last_day = calendar.monthrange(int(year), int(month))[1]
        return min(int(target_day), last_day)


    def is_weekend(day_value):
        return pd.Timestamp(day_value).weekday() >= 5


    def following_business_day(day_value):
        day_value = pd.Timestamp(day_value).to_pydatetime().date()
        while day_value.weekday() >= 5:
            day_value = day_value + timedelta(days=1)
        return day_value


    def previous_business_day(day_value):
        day_value = pd.Timestamp(day_value).to_pydatetime().date()
        while day_value.weekday() >= 5:
            day_value = day_value - timedelta(days=1)
        return day_value


    def adjust_salary_pay_date(day_value):
        day_value = pd.Timestamp(day_value).to_pydatetime().date()
        if day_value.weekday() < 5:
            return day_value
        # In practice, employers differ: some advance weekend salaries to Friday,
        # while others clear on the following business day.
        return previous_business_day(day_value) if rng.random() < 0.72 else following_business_day(day_value)


    def adjust_debit_order_date(day_value):
        return following_business_day(day_value)


    def random_time_for_pay_cycle(kind="spend"):
        if kind == "salary_deposit":
            hour = int(draw_one([0, 1, 2, 3, 4, 5], [0.38, 0.24, 0.16, 0.10, 0.07, 0.05]))
            return hour, int(rng.integers(0, 60)), int(rng.integers(0, 60))
        if kind == "grant_deposit":
            hour = int(draw_one([4, 5, 6, 7, 8, 9, 10], [0.10, 0.16, 0.20, 0.20, 0.16, 0.12, 0.06]))
            return hour, int(rng.integers(0, 60)), int(rng.integers(0, 60))
        if kind == "debit_order":
            hour = int(draw_one([0, 1, 2, 3, 4, 5, 6], [0.34, 0.22, 0.16, 0.11, 0.08, 0.06, 0.03]))
            return hour, int(rng.integers(0, 30)), int(rng.integers(0, 60))
        hour = int(draw_one([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [0.04, 0.07, 0.10, 0.11, 0.11, 0.09, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02]))
        return hour, int(rng.integers(0, 60)), int(rng.integers(0, 60))


    def sample_salary_pay_cycle_datetime(acc, day_shift_max=0, timing="spend"):
        active_start, active_end = account_active_datetime_bounds(acc)
        active_days = max(1, (active_end.date() - active_start.date()).days + 1)
        month_starts = pd.date_range(active_start.date().replace(day=1), active_end.date().replace(day=1), freq="MS")
        if len(month_starts) == 0:
            hour, minute, second = random_time_for_pay_cycle(timing)
            return clamp_account_datetime(acc, active_start + timedelta(days=int(rng.integers(0, active_days)), hours=hour, minutes=minute, seconds=second))
        pay_days, pay_weights = sector_pay_day_profile(acc)
        for _ in range(30):
            month_start = pd.Timestamp(draw_one(list(month_starts))).to_pydatetime()
            pay_day = valid_month_pay_date(month_start.year, month_start.month, draw_one(pay_days, pay_weights))
            candidate_date = datetime(month_start.year, month_start.month, pay_day).date()
            if timing == "salary_deposit":
                candidate_date = adjust_salary_pay_date(candidate_date)
            elif timing == "debit_order":
                candidate_date = adjust_debit_order_date(candidate_date)
            shift = int(rng.integers(0, day_shift_max + 1)) if day_shift_max > 0 else 0
            hour, minute, second = random_time_for_pay_cycle(timing)
            candidate = datetime(candidate_date.year, candidate_date.month, candidate_date.day, hour, minute, second) + timedelta(days=shift)
            if active_start <= candidate <= active_end:
                return clamp_account_datetime(acc, candidate)
        hour, minute, second = random_time_for_pay_cycle(timing)
        return clamp_account_datetime(acc, active_start + timedelta(days=int(rng.integers(0, active_days)), hours=hour, minutes=minute, seconds=second))


    SOCIAL_GRANT_VALUES_BY_PERIOD = {
        "2024_q1": {"old_age": 2090, "disability": 2090, "care_dependency": 2090, "war_veteran": 2110, "foster_child": 1130, "child_support": 510, "grant_in_aid": 510, "srd": 350},
        "2024_25": {"old_age": 2185, "disability": 2185, "care_dependency": 2185, "war_veteran": 2205, "foster_child": 1180, "child_support": 530, "grant_in_aid": 530, "srd": 370},
        "2025_26": {"old_age": 2315, "disability": 2315, "care_dependency": 2315, "war_veteran": 2335, "foster_child": 1250, "child_support": 560, "grant_in_aid": 560, "srd": 370},
        "2026_27": {"old_age": 2400, "disability": 2400, "care_dependency": 2400, "war_veteran": 2420, "foster_child": 1295, "child_support": 580, "grant_in_aid": 580, "srd": 370},
    }


    def social_grant_period_key(day_value):
        day_value = pd.Timestamp(day_value).to_pydatetime().date()
        if day_value >= datetime(2026, 4, 1).date():
            return "2026_27"
        if day_value >= datetime(2025, 4, 1).date():
            return "2025_26"
        if day_value >= datetime(2024, 4, 1).date():
            return "2024_25"
        return "2024_q1"


    def nth_working_day(year, month, n):
        current = datetime(int(year), int(month), 1).date()
        seen = 0
        while True:
            if current.weekday() < 5:
                seen += 1
                if seen == int(n):
                    return current
            current = current + timedelta(days=1)


    def sample_social_grant_type(acc):
        primary_grant_type = getattr(acc, "Primary_social_grant_type", "Not applicable")
        if primary_grant_type != "Not applicable":
            return primary_grant_type
        age = int(getattr(acc, "Age", 40))
        if age >= 60:
            return draw_one(["old_age", "child_support", "grant_in_aid", "srd"], [0.68, 0.18, 0.08, 0.06])
        if age < 25:
            return draw_one(["srd", "child_support", "foster_child"], [0.48, 0.42, 0.10])
        return draw_one(["child_support", "srd", "disability", "care_dependency", "foster_child"], [0.56, 0.18, 0.12, 0.08, 0.06])


    def social_grant_payment_date(year, month, grant_type):
        if grant_type in ["old_age", "war_veteran"]:
            return nth_working_day(year, month, 2)
        if grant_type in ["disability", "care_dependency"]:
            return nth_working_day(year, month, 3)
        return nth_working_day(year, month, 3 if rng.random() < 0.78 else 4)


    def sample_social_grant_amount(acc, grant_type, day_value):
        values = SOCIAL_GRANT_VALUES_BY_PERIOD[social_grant_period_key(day_value)]
        if grant_type == "child_support":
            children = int(draw_one([1, 2, 3, 4], [0.56, 0.28, 0.12, 0.04]))
            amount = values["child_support"] * children
        else:
            amount = values[grant_type]
        if grant_type in ["old_age", "disability", "care_dependency"] and rng.random() < 0.08:
            amount += values["grant_in_aid"]
        return float(amount)


    def sample_social_grant_datetime(acc, grant_type=None, spend=False):
        active_start, active_end = account_active_datetime_bounds(acc)
        active_days = max(1, (active_end.date() - active_start.date()).days + 1)
        month_starts = pd.date_range(active_start.date().replace(day=1), active_end.date().replace(day=1), freq="MS")
        grant_type = grant_type or sample_social_grant_type(acc)
        if len(month_starts) == 0:
            hour, minute, second = random_time_for_pay_cycle("spend" if spend else "grant_deposit")
            return grant_type, clamp_account_datetime(acc, active_start + timedelta(days=int(rng.integers(0, active_days)), hours=hour, minutes=minute, seconds=second))
        for _ in range(30):
            month_start = pd.Timestamp(draw_one(list(month_starts))).to_pydatetime()
            grant_day = social_grant_payment_date(month_start.year, month_start.month, grant_type)
            offset = int(draw_one([0, 1, 2, 3, 4, 5, 6], [0.25, 0.22, 0.18, 0.14, 0.10, 0.07, 0.04])) if spend else 0
            hour, minute, second = random_time_for_pay_cycle("spend" if spend else "grant_deposit")
            candidate = datetime(grant_day.year, grant_day.month, grant_day.day, hour, minute, second) + timedelta(days=offset)
            if active_start <= candidate <= active_end:
                return grant_type, clamp_account_datetime(acc, candidate)
        hour, minute, second = random_time_for_pay_cycle("spend" if spend else "grant_deposit")
        return grant_type, clamp_account_datetime(acc, active_start + timedelta(days=int(rng.integers(0, active_days)), hours=hour, minutes=minute, seconds=second))


    def sample_datetime(behaviour, acc=None):
        active_start, active_end = account_active_datetime_bounds(acc)
        active_days = max(1, (active_end.date() - active_start.date()).days + 1)

        if behaviour == "Salary and household payments" and acc is not None and rng.random() < 0.42:
            return sample_salary_pay_cycle_datetime(acc, day_shift_max=3)

        day_offset = int(rng.integers(0, active_days))
        if behaviour in ["Business operating payments", "Salary and household payments"]:
            hour = int(weighted_choice(range(8, 18)))
        elif behaviour in ["Low-value high-frequency retail payments", "Taxi or transport cash cycle", "Spaza / township retail cash cycle"]:
            hour = int(weighted_choice(range(6, 23)))
        elif behaviour == "Remittance corridor activity":
            hour = int(weighted_choice(range(7, 22)))
        else:
            hour = int(weighted_choice(range(6, 22)))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        tx_datetime = active_start + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        return clamp_account_datetime(acc, tx_datetime) if acc is not None else min(tx_datetime, END_DATETIME)


    NORMAL_BURST_PROBABILITY = {
        "Low-value high-frequency retail payments": 0.22,
        "Taxi or transport cash cycle": 0.20,
        "Spaza / township retail cash cycle": 0.18,
        "Informal registered business cash cycle": 0.16,
        "Informal unregistered business cash cycle": 0.16,
        "Business operating payments": 0.10,
        "Remittance corridor activity": 0.08,
        "Irregular income / gig work": 0.08,
        "Grant-linked inflows and survival payments": 0.06,
        "Salary and household payments": 0.05,
        "Stokvel / rotating savings contributions": 0.04,
    }


    RAPID_SUSPICIOUS_TYPOLOGIES = {
        "Mule Account / Rapid Pass-Through",
        "Cybercrime Proceeds",
        "Fraudulent Company or Invoice Scam",
    }

    MODERATE_SUSPICIOUS_TYPOLOGIES = {
        "Structuring / Smurfing Cash Deposits",
        "Ponzi / Investment Scheme Collections",
        "Stolen Student Loan or Grant Abuse",
        "Crypto Asset Off-Ramp",
    }

    LAYERING_SUSPICIOUS_TYPOLOGIES = {
        "Cross-Border Layering via SADC Corridor",
        "High-Risk Jurisdiction Layering",
        "Intermediary / Trust Account Abuse",
        "Shell or Front Company Flow",
        "Complex Legal Structure / Beneficial Ownership Obfuscation",
        "Trade-Based Import-Export Layering",
    }

    SLOW_SUSPICIOUS_TYPOLOGIES = {
        "Procurement Corruption / Tender Kickback",
        "Property / Real Estate Integration",
        "Motor Vehicle or High-Value Goods Purchase",
        "Illegal Gold or Precious Metals",
        "Environmental or Wildlife Crime Proceeds",
        "NPO / Community Organisation Misuse",
        "Casino or Gambling Laundering",
        "Cash Courier / Bulk Cash Movement",
        "Narcotics Cash Proceeds",
    }


    def normal_gap_minutes(behaviour):
        if behaviour in ["Low-value high-frequency retail payments", "Taxi or transport cash cycle", "Spaza / township retail cash cycle"]:
            return float(np.clip(rng.lognormal(np.log(45), 0.85), 3, 360))
        if behaviour in ["Informal registered business cash cycle", "Informal unregistered business cash cycle", "Business operating payments"]:
            return float(np.clip(rng.lognormal(np.log(120), 0.90), 10, 720))
        if behaviour == "Remittance corridor activity":
            return float(np.clip(rng.lognormal(np.log(240), 0.80), 30, 1440))
        return float(np.clip(rng.lognormal(np.log(180), 0.95), 15, 1440))


    def sample_legitimate_datetime(acc, previous_legitimate_datetime=None):
        behaviour = acc.Legitimate_behaviour
        burst_probability = NORMAL_BURST_PROBABILITY.get(behaviour, 0.07)
        if previous_legitimate_datetime is not None and rng.random() < burst_probability:
            candidate = previous_legitimate_datetime + timedelta(minutes=normal_gap_minutes(behaviour))
            active_start, active_end = account_active_datetime_bounds(acc)
            if active_start <= candidate <= active_end:
                return clamp_account_datetime(acc, candidate)
        return sample_datetime(behaviour, acc)


    def suspicious_gap_minutes(typology):
        if typology in RAPID_SUSPICIOUS_TYPOLOGIES:
            return float(np.clip(rng.lognormal(np.log(12), 0.85), 2, 60))
        if typology in MODERATE_SUSPICIOUS_TYPOLOGIES:
            return float(np.clip(rng.lognormal(np.log(45), 0.90), 5, 240))
        if typology in LAYERING_SUSPICIOUS_TYPOLOGIES:
            return float(np.clip(rng.lognormal(np.log(240), 1.00), 20, 1440))
        if typology in SLOW_SUSPICIOUS_TYPOLOGIES:
            return float(np.clip(rng.lognormal(np.log(720), 1.05), 60, 4320))
        return float(np.clip(rng.lognormal(np.log(180), 1.00), 15, 1440))


    def suspicious_cluster_probability(typology):
        if typology in RAPID_SUSPICIOUS_TYPOLOGIES:
            return 0.84
        if typology in MODERATE_SUSPICIOUS_TYPOLOGIES:
            return 0.70
        if typology in LAYERING_SUSPICIOUS_TYPOLOGIES:
            return 0.54
        if typology in SLOW_SUSPICIOUS_TYPOLOGIES:
            return 0.34
        return 0.50


    def sample_suspicious_datetime(acc, typology, previous_suspicious_datetime=None):
        if previous_suspicious_datetime is not None and rng.random() < suspicious_cluster_probability(typology):
            candidate = previous_suspicious_datetime + timedelta(minutes=suspicious_gap_minutes(typology))
            active_start, active_end = account_active_datetime_bounds(acc)
            if active_start <= candidate <= active_end:
                return clamp_account_datetime(acc, candidate)

        tx_datetime = sample_datetime(acc.Legitimate_behaviour, acc)
        if typology in RAPID_SUSPICIOUS_TYPOLOGIES and rng.random() < 0.30:
            tx_datetime = tx_datetime.replace(
                hour=int(weighted_choice(range(17, 23))),
                minute=int(rng.integers(0, 60)),
                second=int(rng.integers(0, 60)),
            )
        return clamp_account_datetime(acc, tx_datetime)


    # ============================================================
    # GENERATE CUSTOMER / ACCOUNT PROFILES
    # ============================================================
    print("\n" + "="*100)
    print(" PHASE 1: GENERATING RECEIVER ACCOUNT PROFILES")
    print("="*100)
    print(f" Creating {N_RECEIVER_ACCOUNTS:,} receiver account profiles...")
    print(f"   - Target AML contamination: {AML_ACCOUNT_SHARE_TARGET*100:.1f}%")
    print(f"   - Simulation period: {START_DATE.date()} to {END_DATE.date()} ({DAYS} days)")
    print()

    import time
    phase1_start = time.time()

    receiver_accounts = pd.DataFrame({
        "Receiver_account": np.arange(N_RECEIVER_ACCOUNTS, dtype=int)
    })
    print(f"OK Base account table created")

    receiver_accounts["Customer_type"] = weighted_choice(
        list(CUSTOMER_TYPE_WEIGHTS.keys()),
        list(CUSTOMER_TYPE_WEIGHTS.values()),
        N_RECEIVER_ACCOUNTS
    )
    print(f"OK Customer types assigned (Formal Employee 25%, Informal Trader 16%, Grant Recipient 14%, ...)")
    receiver_accounts["Province"] = weighted_choice(
        list(PROVINCE_WEIGHTS.keys()),
        list(PROVINCE_WEIGHTS.values()),
        N_RECEIVER_ACCOUNTS
    )
    receiver_accounts["Employment_status"] = [
        choose_employment_status(customer_type)
        for customer_type in receiver_accounts["Customer_type"]
    ]
    print(f"   - Determining employment sectors and income bands...")
    receiver_accounts["Employment_sector"] = [
        choose_sector(customer_type, employment_status)
        for customer_type, employment_status in zip(
            receiver_accounts["Customer_type"], receiver_accounts["Employment_status"]
        )
    ]
    receiver_accounts["Income_band"] = [
        choose_income_band(customer_type, employment_status)
        for customer_type, employment_status in zip(
            receiver_accounts["Customer_type"], receiver_accounts["Employment_status"]
        )
    ]
    print(f"     OK Sectors and income bands determined")
    print(f"   - Generating monthly income estimates...")
    receiver_accounts["Monthly_income_estimate"] = [
        monthly_income_from_band(income_band)
        for income_band in receiver_accounts["Income_band"]
    ]
    print(f"     OK Monthly income estimates generated")
    print(f"   - Computing cash intensity profiles...")
    receiver_accounts["Cash_intensity_score"] = [
        cash_intensity(customer_type, employment_status, sector)
        for customer_type, employment_status, sector in zip(
            receiver_accounts["Customer_type"],
            receiver_accounts["Employment_status"],
            receiver_accounts["Employment_sector"]
        )
    ]
    receiver_accounts["Cash_intensity_band"] = receiver_accounts["Cash_intensity_score"].apply(cash_band)
    print(f"     OK Cash intensity computed (Low: {(receiver_accounts['Cash_intensity_band']=='Low').sum():,}, High: {(receiver_accounts['Cash_intensity_band']=='High').sum():,}, Very High: {(receiver_accounts['Cash_intensity_band']=='Very high').sum():,})")
    receiver_accounts["Grant_linked"] = (
        (receiver_accounts["Customer_type"] == "Grant Recipient") |
        (receiver_accounts["Employment_status"].isin(["Pensioner", "Unemployed"]) & (rng.random(N_RECEIVER_ACCOUNTS) < 0.35))
    ).astype(int)


    # Demographic and socioeconomic profile variables. These are synthetic and are
    print(f"   - Generating demographic profiles (gender, age, education, nationality)...")
    receiver_accounts["Gender"] = weighted_choice(
        ["Female", "Male", "Other / undisclosed"],
        [0.515, 0.475, 0.010],
        N_RECEIVER_ACCOUNTS
    )
    receiver_accounts["Age"] = [
        sample_age(customer_type, employment_status)
        for customer_type, employment_status in zip(
            receiver_accounts["Customer_type"], receiver_accounts["Employment_status"]
        )
    ]
    receiver_accounts["Age_band"] = receiver_accounts["Age"].apply(age_band)
    receiver_accounts["Education_level"] = [
        choose_education(customer_type, employment_status)
        for customer_type, employment_status in zip(
            receiver_accounts["Customer_type"], receiver_accounts["Employment_status"]
        )
    ]
    receiver_accounts["Urban_rural"] = weighted_choice(
        ["Metro / urban", "Township / peri-urban", "Small town", "Rural"],
        [0.42, 0.28, 0.14, 0.16],
        N_RECEIVER_ACCOUNTS
    )
    receiver_accounts["Township_or_informal_settlement_flag"] = (
        (receiver_accounts["Urban_rural"] == "Township / peri-urban") |
        ((receiver_accounts["Province"].isin(["Gauteng", "KwaZulu-Natal", "Western Cape"])) & (rng.random(N_RECEIVER_ACCOUNTS) < 0.14))
    ).astype(int)
    receiver_accounts["Financial_inclusion_segment"] = [
        draw_one(["Banked digital", "Banked cash-reliant", "Thin-file / underbanked", "Grant-card dependent"], [0.32, 0.30, 0.23, 0.15])
        if customer_type != "Grant Recipient" else draw_one(["Grant-card dependent", "Banked cash-reliant", "Thin-file / underbanked"], [0.58, 0.26, 0.16])
        for customer_type in receiver_accounts["Customer_type"]
    ]
    receiver_accounts["Channel_preference"] = [
        choose_channel_preference(cash_score, customer_type)
        for cash_score, customer_type in zip(
            receiver_accounts["Cash_intensity_score"], receiver_accounts["Customer_type"]
        )
    ]
    receiver_accounts["Primary_social_grant_type"] = [
        choose_primary_social_grant_type(customer_type, employment_status, age, grant_linked)
        for customer_type, employment_status, age, grant_linked in zip(
            receiver_accounts["Customer_type"], receiver_accounts["Employment_status"],
            receiver_accounts["Age"], receiver_accounts["Grant_linked"]
        )
    ]

    informal_candidate = (
        receiver_accounts["Customer_type"].isin(["Informal Trader", "Cash-Intensive SME"]) |
        (receiver_accounts["Employment_status"] == "Informal / Self-employed") |
        receiver_accounts["Employment_sector"].str.contains("Informal|Spaza|Taxi|Domestic|Agriculture", regex=True)
    )
    receiver_accounts["Informal_business_flag"] = (
        informal_candidate & (rng.random(N_RECEIVER_ACCOUNTS) < 0.82)
    ).astype(int)
    business_actor = (
        receiver_accounts["Informal_business_flag"].eq(1) |
        receiver_accounts["Customer_type"].isin(["Business Owner", "Procurement Vendor", "Corporate Entity", "Cash-Intensive SME", "NPO / Community Organisation"])
    )
    registration_probability = np.where(
        receiver_accounts["Informal_business_flag"].eq(1),
        0.20,
        np.where(receiver_accounts["Customer_type"].isin(["Corporate Entity", "Procurement Vendor"]), 0.86, 0.62)
    )
    receiver_accounts["Business_registered"] = (
        business_actor & (rng.random(N_RECEIVER_ACCOUNTS) < registration_probability)
    ).astype(int)
    receiver_accounts["Business_registration_status"] = np.select(
        [
            ~business_actor,
            receiver_accounts["Business_registered"].eq(1),
            receiver_accounts["Informal_business_flag"].eq(1),
        ],
        ["Not business", "Registered", "Unregistered informal"],
        default="Unregistered small business"
    )
    receiver_accounts["Has_trading_permit"] = (
        business_actor & (rng.random(N_RECEIVER_ACCOUNTS) < np.where(receiver_accounts["Informal_business_flag"].eq(1), 0.34, 0.78))
    ).astype(int)
    receiver_accounts["VAT_registered"] = (
        business_actor & receiver_accounts["Business_registered"].eq(1) &
        (rng.random(N_RECEIVER_ACCOUNTS) < np.where(receiver_accounts["Customer_type"].isin(["Corporate Entity", "Procurement Vendor"]), 0.80, 0.28))
    ).astype(int)
    receiver_accounts["Uses_personal_account_for_business"] = (
        business_actor & (rng.random(N_RECEIVER_ACCOUNTS) < np.where(receiver_accounts["Informal_business_flag"].eq(1), 0.74, 0.20))
    ).astype(int)
    receiver_accounts["Business_premises_type"] = receiver_accounts.apply(choose_business_premises, axis=1)
    receiver_accounts["Estimated_daily_cash_turnover"] = [
        round_currency(lognormal_amount(900 if informal else 3500, 0.95, 0, 180_000)) if is_business else 0.0
        for informal, is_business in zip(receiver_accounts["Informal_business_flag"], business_actor)
    ]
    receiver_accounts["Business_age_years"] = [
        round(float(np.clip(rng.gamma(2.0, 2.2), 0.1, 35)), 1) if is_business else 0.0
        for is_business in business_actor
    ]
    receiver_accounts["Keeps_business_records"] = (
        business_actor & (rng.random(N_RECEIVER_ACCOUNTS) < np.where(receiver_accounts["Business_registered"].eq(1), 0.70, 0.24))
    ).astype(int)
    receiver_accounts["Has_bank_loan_access"] = (
        business_actor & (rng.random(N_RECEIVER_ACCOUNTS) < np.where(receiver_accounts["Business_registered"].eq(1), 0.36, 0.08))
    ).astype(int)
    receiver_accounts["High_risk_sector_category"] = [
        high_risk_sector_category(sector, customer_type)
        for sector, customer_type in zip(receiver_accounts["Employment_sector"], receiver_accounts["Customer_type"])
    ]
    receiver_accounts["High_risk_sector_flag"] = (
        receiver_accounts["High_risk_sector_category"] != "Standard retail / household"
    ).astype(int)

    receiver_accounts["Has_cross_border_links"] = (
        (receiver_accounts["Customer_type"] == "Migrant Worker / Remittance User") |
        (rng.random(N_RECEIVER_ACCOUNTS) < np.where(receiver_accounts["Province"].isin(["Gauteng", "Limpopo", "Mpumalanga", "KwaZulu-Natal"]), 0.18, 0.08))
    ).astype(int)
    receiver_accounts["Typical_foreign_counterparty_country"] = [
        choose_profile_country(bool(flag))
        for flag in receiver_accounts["Has_cross_border_links"]
    ]
    receiver_accounts["Profile_jurisdiction_risk_tier"] = receiver_accounts[
        "Typical_foreign_counterparty_country"
    ].apply(country_risk_tier)

    receiver_accounts["Procurement_vendor"] = (
        (receiver_accounts["Customer_type"] == "Procurement Vendor") |
        ((receiver_accounts["Employment_sector"].isin(["Construction", "Security", "Health supplies", "IT services"])) & (rng.random(N_RECEIVER_ACCOUNTS) < 0.08))
    ).astype(int)
    receiver_accounts["Public_sector_exposure"] = (
        (receiver_accounts["Customer_type"] == "Public Sector Employee") |
        (receiver_accounts["Procurement_vendor"] == 1) |
        (rng.random(N_RECEIVER_ACCOUNTS) < 0.025)
    ).astype(int)

    pep_probability = (
        0.002 +
        0.011 * receiver_accounts["Public_sector_exposure"] +
        0.015 * receiver_accounts["Procurement_vendor"] +
        0.004 * (receiver_accounts["Customer_type"] == "Corporate Entity").astype(int)
    )
    receiver_accounts["Is_PEP"] = (rng.random(N_RECEIVER_ACCOUNTS) < pep_probability).astype(int)
    receiver_accounts["PEP_category"] = np.where(
        receiver_accounts["Is_PEP"] == 1,
        weighted_choice(
            ["Domestic prominent influential person", "Foreign PEP", "PEP family or associate"],
            [0.62, 0.12, 0.26],
            N_RECEIVER_ACCOUNTS
        ),
        "Not PEP"
    )

    receiver_accounts["Nationality_country"] = [
        choose_nationality(customer_type, bool(has_cross_border))
        for customer_type, has_cross_border in zip(
            receiver_accounts["Customer_type"], receiver_accounts["Has_cross_border_links"]
        )
    ]
    receiver_accounts["Residency_status"] = receiver_accounts["Nationality_country"].apply(residency_status)
    receiver_accounts["Migrant_worker_flag"] = (
        (receiver_accounts["Customer_type"] == "Migrant Worker / Remittance User") |
        (receiver_accounts["Residency_status"].isin(["Documented migrant", "Temporary worker", "Undocumented / weak documentation"]))
    ).astype(int)
    receiver_accounts["KYC_strength"] = np.select(
        [
            receiver_accounts["Residency_status"].eq("Undocumented / weak documentation"),
            receiver_accounts["Financial_inclusion_segment"].eq("Thin-file / underbanked"),
            receiver_accounts["Business_registration_status"].str.contains("Unregistered", regex=False),
        ],
        ["Weak / incomplete", "Basic", "Enhanced required"],
        default="Standard"
    )
    receiver_accounts["Targeted_financial_sanctions_match"] = (
        rng.random(N_RECEIVER_ACCOUNTS) < (
            0.0004 +
            0.0018 * receiver_accounts["Profile_jurisdiction_risk_tier"].isin(["FATF Call for Action", "FATF Increased Monitoring"]).astype(int)
        )
    ).astype(int)
    receiver_accounts["Adverse_media_flag"] = (
        rng.random(N_RECEIVER_ACCOUNTS) < (
            0.004 +
            0.016 * receiver_accounts["Is_PEP"] +
            0.010 * receiver_accounts["Procurement_vendor"] +
            0.006 * receiver_accounts["High_risk_sector_flag"]
        )
    ).astype(int)


    def derive_high_risk_person_type(row):
        if row.Targeted_financial_sanctions_match == 1:
            return "Targeted financial sanctions possible match"
        if row.PEP_category == "Foreign PEP":
            return "Foreign prominent public official"
        if row.PEP_category == "Domestic prominent influential person":
            return "Domestic prominent influential person"
        if row.PEP_category == "PEP family or associate":
            return "PEP immediate family / close associate"
        if row.Procurement_vendor == 1:
            return "Public procurement/vendor exposure"
        if row.Public_sector_exposure == 1:
            return "Public sector financial-flow exposure"
        if row.Adverse_media_flag == 1:
            return "Adverse media only"
        return "Standard retail customer"

    receiver_accounts["High_risk_person_type"] = receiver_accounts.apply(derive_high_risk_person_type, axis=1)
    receiver_accounts["High_risk_person_flag"] = (
        receiver_accounts["High_risk_person_type"] != "Standard retail customer"
    ).astype(int)

    receiver_accounts["Beneficial_ownership_opacity"] = np.clip(
        0.10 +
        0.24 * receiver_accounts["Procurement_vendor"] +
        0.34 * (receiver_accounts["Customer_type"] == "Corporate Entity").astype(int) +
        0.12 * (receiver_accounts["Customer_type"] == "NPO / Community Organisation").astype(int) +
        rng.normal(0, 0.10, N_RECEIVER_ACCOUNTS),
        0,
        1
    )
    receiver_accounts["Intermediary_account_flag"] = (
        rng.random(N_RECEIVER_ACCOUNTS) < (
            0.025 +
            0.055 * receiver_accounts["Procurement_vendor"] +
            0.060 * (receiver_accounts["Customer_type"].isin(["Corporate Entity", "NPO / Community Organisation"])).astype(int)
        )
    ).astype(int)
    receiver_accounts["Virtual_asset_exposure"] = (
        rng.random(N_RECEIVER_ACCOUNTS) < (
            0.025 +
            0.050 * (receiver_accounts["Customer_type"] == "Student / Youth Account").astype(int) +
            0.035 * (receiver_accounts["Customer_type"].isin(["Business Owner", "Corporate Entity"])).astype(int)
        )
    ).astype(int)
    receiver_accounts["Connected_party_count"] = rng.poisson(
        1 + 2 * receiver_accounts["Is_PEP"] + 2 * receiver_accounts["Procurement_vendor"] +
        1 * receiver_accounts["Intermediary_account_flag"]
    )
    receiver_accounts["Account_age_days"] = np.maximum(1, rng.gamma(shape=2.4, scale=420, size=N_RECEIVER_ACCOUNTS).astype(int))

    # This is a profile-level probability, not the final transaction label. It gives
    # the simulation controllable account-level prevalence while still conditioning
    # on socioeconomic and institutional risk context.
    profile_risk_score = (
        0.010 +
        0.024 * receiver_accounts["Cash_intensity_score"] +
        0.018 * receiver_accounts["Has_cross_border_links"] +
        0.038 * (receiver_accounts["Profile_jurisdiction_risk_tier"] == "FATF Increased Monitoring").astype(int) +
        0.075 * (receiver_accounts["Profile_jurisdiction_risk_tier"] == "FATF Call for Action").astype(int) +
        0.055 * receiver_accounts["Is_PEP"] +
        0.040 * receiver_accounts["Procurement_vendor"] +
        0.026 * receiver_accounts["Intermediary_account_flag"] +
        0.022 * receiver_accounts["Virtual_asset_exposure"] +
        0.022 * (receiver_accounts["Beneficial_ownership_opacity"] > 0.60).astype(int) +
        0.014 * receiver_accounts["Informal_business_flag"] +
        0.018 * receiver_accounts["Uses_personal_account_for_business"] +
        0.022 * receiver_accounts["High_risk_sector_flag"] +
        0.026 * receiver_accounts["High_risk_person_flag"] +
        0.018 * (receiver_accounts["KYC_strength"].isin(["Weak / incomplete", "Enhanced required"])).astype(int) +
        0.016 * (receiver_accounts["Account_age_days"] < 120).astype(int)
    )
    receiver_accounts["Account_laundering_probability"] = np.clip(
        profile_risk_score * (AML_ACCOUNT_SHARE_TARGET / profile_risk_score.mean()),
        0.001,
        0.70
    )
    receiver_accounts["Account_is_laundering"] = (
        rng.random(N_RECEIVER_ACCOUNTS) < receiver_accounts["Account_laundering_probability"]
    ).astype(int)

    if FORCE_ALL_AML_SCENARIOS and N_RECEIVER_ACCOUNTS >= len(LAUNDERING_TYPES):
        # Force at least one account per scenario in validation experiments. This keeps
        # the generated data filterable by category while preserving imbalanced
        # transaction labels at the row level.
        forced_idx = receiver_accounts["Account_laundering_probability"].sort_values(ascending=False).index[:len(LAUNDERING_TYPES)]
        receiver_accounts.loc[forced_idx, "Account_is_laundering"] = 1
    receiver_accounts["Legitimate_behaviour"] = receiver_accounts.apply(choose_legit_behaviour, axis=1)
    receiver_accounts["Laundering_type"] = receiver_accounts.apply(choose_laundering_type, axis=1)
    receiver_accounts["Scenario_forced_coverage_flag"] = 0

    if FORCE_ALL_AML_SCENARIOS:
        laundering_idx = receiver_accounts.index[receiver_accounts["Account_is_laundering"] == 1].tolist()
        for idx, scenario_name in zip(laundering_idx, LAUNDERING_TYPES):
            receiver_accounts.loc[idx, "Laundering_type"] = scenario_name
            receiver_accounts.loc[idx, "Scenario_forced_coverage_flag"] = 1

    receiver_accounts["Opened_for_laundering_flag"] = [
        int(row.Account_is_laundering == 1 and rng.random() < scenario_new_account_probability(row))
        for row in receiver_accounts.itertuples(index=False)
    ]
    receiver_accounts["Account_open_date"] = [
        choose_account_open_date(row)
        for row in receiver_accounts.itertuples(index=False)
    ]
    receiver_accounts["First_observed_transaction_date"] = [
        choose_first_observed_transaction_date(row)
        for row in receiver_accounts.itertuples(index=False)
    ]
    receiver_accounts["Opened_during_observation_flag"] = (
        pd.to_datetime(receiver_accounts["Account_open_date"]) >= START_DATE
    ).astype(int)
    receiver_accounts["Account_age_days"] = (
        END_DATE.date() - pd.to_datetime(receiver_accounts["Account_open_date"]).dt.date
    ).apply(lambda x: max(1, x.days)).astype(int)
    receiver_accounts["Active_observation_days"] = (
        END_DATE.date() - pd.to_datetime(receiver_accounts["First_observed_transaction_date"]).dt.date
    ).apply(lambda x: max(1, x.days + 1)).astype(int)
    receiver_accounts["Account_activity_multiplier"] = [
        account_activity_multiplier(row)
        for row in receiver_accounts.itertuples(index=False)
    ]
    receiver_accounts["Expected_transaction_count"] = [
        round(expected_transaction_count(row), 2)
        for row in receiver_accounts.itertuples(index=False)
    ]


    # ============================================================
    # TRANSACTION GENERATORS
    # ============================================================

    SENDER_POOL_SIZE = max(80_000, N_RECEIVER_ACCOUNTS * 3)
    EXTERNAL_ACCOUNT_OFFSET = SENDER_POOL_SIZE * 2

    TX_COLUMNS = [
        "Transaction_id", "Time", "Date", "Account_id", "Sender_account", "Receiver_account",
        "Counterparty_account", "Direction", "Amount", "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location", "Counterparty_country", "Payment_type",
        "Social_grant_type", "Counterparty_type", "Transaction_purpose_category", "Cross_border_transaction", "SADC_corridor_transaction", "Jurisdiction_risk_tier",
        "Customer_type", "Employment_status", "Employment_sector", "Income_band", "Monthly_income_estimate",
        "Province", "Cash_intensity_score", "Cash_intensity_band", "Grant_linked", "Public_sector_exposure",
        "Procurement_vendor", "Is_PEP", "PEP_category", "Beneficial_ownership_opacity",
        "Intermediary_account_flag", "Virtual_asset_exposure", "Connected_party_count", "Account_age_days",
        "Profile_jurisdiction_risk_tier", "Account_is_laundering", "Is_laundering", "Laundering_type",
        "Legitimate_behaviour", "Behaviour_component", "Typology_stage", "Suspicion_indicator",
        "Scenario_label", "CTR_threshold_near_miss", "Round_amount_flag", "Public_sector_counterparty",
        "PEP_related_transaction"
    ]

    def account_endpoints(account_id, direction, sender_country, receiver_country):
        counterparty_account = int(rng.integers(0, SENDER_POOL_SIZE))
        if direction == "Inflow":
            sender_account = counterparty_account
            receiver_account = int(account_id)
        else:
            sender_account = int(account_id)
            receiver_account = EXTERNAL_ACCOUNT_OFFSET + counterparty_account
        return sender_account, receiver_account, counterparty_account


    def household_spending_purpose(payment_type):
        if payment_type == "Cash Withdrawal":
            return draw_one([
                "Household cash withdrawal", "Taxi fare or transport cash", "Grocery cash purchase",
                "Airtime or data cash purchase", "Family support cash"
            ], [0.30, 0.22, 0.22, 0.16, 0.10])
        if payment_type in ["Debit Card", "Credit Card"]:
            return draw_one([
                "Grocery purchase", "Airtime or data purchase", "Fuel or transport purchase",
                "Pharmacy or healthcare purchase", "Clothing or household goods", "School or education expense"
            ], [0.30, 0.18, 0.16, 0.14, 0.12, 0.10])
        if payment_type == "PayShap":
            return draw_one([
                "Peer-to-peer household payment", "Airtime or data purchase", "Small merchant payment",
                "Family support payment", "Transport fare payment"
            ], [0.28, 0.22, 0.20, 0.18, 0.12])
        if payment_type in ["Electronic Transfer", "RTC"]:
            return draw_one([
                "Rent or accommodation payment", "Utilities or municipal payment", "Insurance or debit-order payment",
                "School fees", "Family support transfer", "Online retail payment"
            ], [0.22, 0.20, 0.18, 0.14, 0.14, 0.12])
        return "General household spending"


    def business_purpose(behaviour, direction, payment_type):
        if direction == "Inflow":
            if payment_type == "Cash Deposit":
                return "Daily cash takings deposit"
            if payment_type == "Debit Card":
                return "Customer card-sale settlement"
            if payment_type in ["PayShap", "Electronic Transfer", "RTC"]:
                return "Customer payment receipt"
            return "Business customer receipt"
        if payment_type == "Cash Withdrawal":
            return draw_one(["Supplier cash purchase", "Wages or casual labour cash", "Business cash float withdrawal"], [0.45, 0.30, 0.25])
        if payment_type == "Procurement EFT":
            return "Procurement or contract payment"
        if payment_type == "Crypto Exchange Off-Ramp":
            return "Virtual asset exchange transfer"
        return draw_one(["Supplier payment", "Stock purchase", "Rent or premises cost", "Payroll or wages", "Transport or fuel cost"], [0.32, 0.24, 0.16, 0.16, 0.12])


    def legitimate_purpose_category(behaviour, direction, payment_type, counterparty_type):
        if counterparty_type == "Debit order / service provider":
            return draw_one(["Insurance or debit-order payment", "Loan or credit repayment", "Cellphone contract debit order", "Subscription or service debit order"], [0.42, 0.26, 0.20, 0.12])
        if behaviour == "Grant-linked inflows and survival payments":
            return "Social grant inflow" if direction == "Inflow" else household_spending_purpose(payment_type)
        if behaviour in [
            "Informal registered business cash cycle", "Informal unregistered business cash cycle",
            "Spaza / township retail cash cycle", "Taxi or transport cash cycle"
        ]:
            return business_purpose(behaviour, direction, payment_type)
        if behaviour == "Remittance corridor activity":
            return "Family remittance inflow" if direction == "Inflow" else "Family remittance outflow"
        if behaviour == "Stokvel / rotating savings contributions":
            return "Stokvel payout received" if direction == "Inflow" else "Stokvel contribution"
        if behaviour == "Low-value high-frequency retail payments":
            return household_spending_purpose(payment_type) if direction == "Outflow" else "Small peer or refund inflow"
        if behaviour == "Business operating payments":
            return business_purpose(behaviour, direction, payment_type)
        if behaviour == "Salary and household payments":
            return "Salary or wage inflow" if direction == "Inflow" else household_spending_purpose(payment_type)
        if direction == "Inflow":
            return draw_one(["Gig income", "Casual wage", "Family support inflow", "Refund or repayment"], [0.36, 0.28, 0.24, 0.12])
        return household_spending_purpose(payment_type)


    def suspicious_purpose_category(typology, payment_type, counterparty_type):
        if payment_type in ["Cash Deposit", "Cash Withdrawal"]:
            return "Cash movement"
        if payment_type in ["International Wire", "Electronic Transfer", "RTC", "PayShap"]:
            return "Transfer or pass-through movement"
        if payment_type == "Procurement EFT":
            return "Procurement or supplier payment"
        if payment_type == "Crypto Exchange Off-Ramp":
            return "Virtual asset exchange transfer"
        if payment_type in ["Debit Card", "Credit Card"]:
            return "Merchant or high-value goods payment"
        return "Scenario-related financial movement"


    def finalise_transaction(acc, tx_id, tx_datetime, direction, amount, sender_country,
                             receiver_country, payment_type, counterparty_type,
                             transaction_purpose_category, is_laundering, laundering_type, behaviour_component,
                             typology_stage, suspicion_indicator, scenario_label,
                             public_sector_counterparty=0, social_grant_type="Not applicable"):
        sender_account, receiver_account, counterparty_account = account_endpoints(
            acc.Receiver_account, direction, sender_country, receiver_country
        )
        risk_tier = combined_jurisdiction_tier(sender_country, receiver_country)
        cross_border = int(sender_country != receiver_country)
        sadc_corridor = int((sender_country in SADC_CORRIDORS) or (receiver_country in SADC_CORRIDORS))
        amount = round_currency(amount)
        ctr_near_miss = int(payment_type in ["Cash Deposit", "Cash Withdrawal"] and 40_000 <= amount <= 49_999.99)
        round_amount_flag = int(abs(amount % 1000) <= 0.01 or abs(amount % 1000 - 1000) <= 0.01)
        pep_related = int(acc.Is_PEP == 1 or acc.Connected_party_count >= 4 or acc.PEP_category != "Not PEP")
        counterparty_country = sender_country if direction == "Inflow" else receiver_country

        return (
            int(tx_id), tx_datetime.time(), tx_datetime.date(), int(acc.Receiver_account),
            int(sender_account), int(receiver_account), int(counterparty_account), direction,
            amount, currency_for(sender_country), currency_for(receiver_country),
            sender_country, receiver_country, counterparty_country, payment_type,
            social_grant_type, counterparty_type, transaction_purpose_category, cross_border, sadc_corridor, risk_tier,
            acc.Customer_type, acc.Employment_status, acc.Employment_sector, acc.Income_band,
            float(acc.Monthly_income_estimate), acc.Province, float(acc.Cash_intensity_score),
            acc.Cash_intensity_band, int(acc.Grant_linked), int(acc.Public_sector_exposure),
            int(acc.Procurement_vendor), int(acc.Is_PEP), acc.PEP_category,
            float(acc.Beneficial_ownership_opacity), int(acc.Intermediary_account_flag),
            int(acc.Virtual_asset_exposure), int(acc.Connected_party_count), int(acc.Account_age_days),
            acc.Profile_jurisdiction_risk_tier, int(acc.Account_is_laundering), int(is_laundering),
            laundering_type, acc.Legitimate_behaviour, behaviour_component, typology_stage,
            suspicion_indicator, scenario_label, ctr_near_miss, round_amount_flag,
            int(public_sector_counterparty), pep_related
        )

    def maybe_add_legitimate_high_value_tail(amount, behaviour, direction, acc):
        """Add realistic non-AML high-value tails without changing payment type.

        This prevents the normal class from being unrealistically low-value while
        keeping ordinary grocery, airtime, transport and household transactions in
        the majority. The tail is intentionally probabilistic so amount alone does
        not become a clean label separator.
        """
        if behaviour == "Business operating payments" and rng.random() < 0.14:
            median = 52_000 if direction == "Outflow" else 34_000
            return max(float(amount), lognormal_amount(median, 0.95, 3_000, 650_000))
        if behaviour == "Salary and household payments":
            if direction == "Inflow" and rng.random() < 0.06:
                return float(np.clip(float(amount) * rng.uniform(1.35, 2.80), 1_000, 260_000))
            if direction == "Outflow" and rng.random() < 0.055:
                return max(float(amount), lognormal_amount(14_000, 0.95, 1_000, 180_000))
        if behaviour == "Remittance corridor activity" and rng.random() < 0.07:
            return max(float(amount), lognormal_amount(8_500, 0.95, 700, 120_000))
        if behaviour == "Stokvel / rotating savings contributions":
            tail_prob = 0.18 if direction == "Inflow" else 0.06
            if rng.random() < tail_prob:
                return max(float(amount), lognormal_amount(12_000, 1.00, 800, 180_000))
        if behaviour in ["Informal registered business cash cycle", "Informal unregistered business cash cycle", "Spaza / township retail cash cycle", "Taxi or transport cash cycle"] and rng.random() < 0.08:
            median = 13_500 if direction == "Outflow" else 10_500
            return max(float(amount), lognormal_amount(median, 0.98, 500, 160_000))
        if behaviour == "Irregular income / gig work" and direction == "Inflow" and rng.random() < 0.045:
            return max(float(amount), lognormal_amount(6_500, 1.00, 500, 85_000))
        return float(amount)


    def generate_legitimate_transaction(acc, tx_id, tx_datetime=None):
        behaviour = acc.Legitimate_behaviour
        if tx_datetime is None:
            tx_datetime = sample_legitimate_datetime(acc)
        sender_country = SA
        receiver_country = SA
        public_sector_counterparty = 0
        social_grant_type = "Not applicable"

        if behaviour == "Grant-linked inflows and survival payments":
            if rng.random() < 0.22:
                direction = "Inflow"
                grant_type, tx_datetime = sample_social_grant_datetime(acc, spend=False)
                social_grant_type = grant_type
                amount = sample_social_grant_amount(acc, grant_type, tx_datetime.date())
                payment_type = "Social Grant"
                counterparty_type = "Government grant payer"
                public_sector_counterparty = 1
            else:
                direction = "Outflow"
                if rng.random() < 0.72:
                    social_grant_type, tx_datetime = sample_social_grant_datetime(acc, spend=True)
                    amount = lognormal_amount(360, 0.85, 25, 4200)
                    payment_type = draw_one(["Cash Withdrawal", "Debit Card", "PayShap", "Electronic Transfer"], [0.44, 0.38, 0.13, 0.05])
                else:
                    amount = lognormal_amount(95, 0.78, 10, 1200)
                    payment_type = draw_one(["Debit Card", "PayShap", "Cash Withdrawal"], [0.42, 0.34, 0.24])
                counterparty_type = "Retail merchant"

        elif behaviour in ["Informal registered business cash cycle", "Informal unregistered business cash cycle", "Spaza / township retail cash cycle", "Taxi or transport cash cycle"]:
            if behaviour == "Taxi or transport cash cycle":
                inflow_weight = 0.66
                median_in, median_out = 1200, 850
                counterparty_type = "Transport customer / taxi association"
            elif behaviour == "Spaza / township retail cash cycle":
                inflow_weight = 0.62
                median_in, median_out = 850, 700
                counterparty_type = "Township retail customer"
            elif behaviour == "Informal unregistered business cash cycle":
                inflow_weight = 0.60
                median_in, median_out = 760, 620
                counterparty_type = "Informal customer"
            else:
                inflow_weight = 0.56
                median_in, median_out = 1250, 900
                counterparty_type = "Small business customer"

            if rng.random() < inflow_weight:
                direction = "Inflow"
                amount = lognormal_amount(median_in, 0.98, 25, 45_000)
                payment_type = draw_one(["Cash Deposit", "Debit Card", "PayShap", "Electronic Transfer"], [0.50, 0.20, 0.20, 0.10])
            else:
                direction = "Outflow"
                amount = lognormal_amount(median_out, 1.02, 20, 35_000)
                payment_type = draw_one(["Cash Withdrawal", "Electronic Transfer", "Debit Card", "PayShap"], [0.45, 0.25, 0.20, 0.10])
                counterparty_type = "Supplier or household"

        elif behaviour == "Remittance corridor activity":
            foreign_country = choose_transaction_country(acc, force_foreign=True, suspicious=False)
            if rng.random() < 0.50:
                direction = "Inflow"
                sender_country, receiver_country = foreign_country, SA
            else:
                direction = "Outflow"
                sender_country, receiver_country = SA, foreign_country
            amount = lognormal_amount(1600, 0.85, 100, 25_000)
            payment_type = draw_one(["International Wire", "Electronic Transfer", "PayShap", "RTC"], [0.36, 0.28, 0.24, 0.12])
            counterparty_type = "Family or remittance counterparty"

        elif behaviour == "Stokvel / rotating savings contributions":
            direction = draw_one(["Inflow", "Outflow"], [0.56, 0.44])
            amount = lognormal_amount(650, 0.85, 50, 35_000)
            payment_type = draw_one(["Electronic Transfer", "Cash Deposit", "PayShap", "Cash Withdrawal"], [0.34, 0.28, 0.24, 0.14])
            counterparty_type = "Stokvel or savings group"

        elif behaviour == "Low-value high-frequency retail payments":
            direction = draw_one(["Inflow", "Outflow"], [0.22, 0.78])
            amount = lognormal_amount(150, 0.90, 10, 3500)
            payment_type = draw_one(["Debit Card", "PayShap", "Cash Withdrawal", "Electronic Transfer"], [0.45, 0.30, 0.15, 0.10])
            counterparty_type = "Retail merchant or peer"

        elif behaviour == "Business operating payments":
            direction = draw_one(["Inflow", "Outflow"], [0.52, 0.48])
            amount = lognormal_amount(8500, 1.05, 120, 260_000)
            payment_type = draw_one(["Electronic Transfer", "RTC", "Debit Card", "Cash Deposit", "International Wire"], [0.48, 0.18, 0.17, 0.12, 0.05])
            counterparty_type = draw_one(["Customer", "Supplier", "Payroll", "Trade counterparty"], [0.38, 0.35, 0.15, 0.12])
            if (acc.Procurement_vendor or acc.Public_sector_exposure) and rng.random() < 0.10:
                payment_type = "Procurement EFT"
                counterparty_type = draw_one(["Public sector entity", "Supplier", "Municipal service provider"], [0.45, 0.35, 0.20])
                public_sector_counterparty = int(counterparty_type in ["Public sector entity", "Municipal service provider"])
            elif acc.Virtual_asset_exposure and rng.random() < 0.05:
                payment_type = "Crypto Exchange Off-Ramp"
                counterparty_type = "Virtual asset service provider"
            if rng.random() < 0.06 and acc.Has_cross_border_links:
                foreign_country = choose_transaction_country(acc, force_foreign=True, suspicious=False)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif behaviour == "Salary and household payments":
            if rng.random() < 0.18:
                direction = "Inflow"
                if rng.random() < 0.88:
                    tx_datetime = sample_salary_pay_cycle_datetime(acc, day_shift_max=0, timing="salary_deposit")
                amount = np.clip(rng.normal(acc.Monthly_income_estimate, max(acc.Monthly_income_estimate * 0.12, 250)), 1000, 180_000)
                payment_type = "Electronic Transfer"
                counterparty_type = "Employer"
                public_sector_counterparty = int(acc.Employment_sector in ["Government", "Municipality", "Education", "Healthcare"])
            else:
                direction = "Outflow"
                salary_spend_roll = rng.random()
                if salary_spend_roll < 0.24:
                    tx_datetime = sample_salary_pay_cycle_datetime(acc, day_shift_max=0, timing="debit_order")
                    amount = lognormal_amount(1350, 0.90, 120, 45_000)
                    payment_type = draw_one(["Electronic Transfer", "RTC"], [0.82, 0.18])
                    counterparty_type = "Debit order / service provider"
                elif salary_spend_roll < 0.62:
                    tx_datetime = sample_salary_pay_cycle_datetime(acc, day_shift_max=4, timing="spend")
                    amount = lognormal_amount(760, 0.95, 35, 28_000)
                    payment_type = draw_one(["Debit Card", "Cash Withdrawal", "Electronic Transfer", "PayShap", "Credit Card"], [0.38, 0.24, 0.18, 0.14, 0.06])
                    counterparty_type = "Household or merchant"
                else:
                    amount = lognormal_amount(170, 0.82, 10, 3500) if rng.random() < 0.62 else lognormal_amount(420, 0.95, 20, 12_000)
                    payment_type = draw_one(["Debit Card", "PayShap", "Cash Withdrawal", "Electronic Transfer", "Credit Card"], [0.42, 0.26, 0.16, 0.10, 0.06])
                    counterparty_type = "Household or merchant"

        else:
            direction = draw_one(["Inflow", "Outflow"], [0.46, 0.54])
            amount = lognormal_amount(650, 1.10, 20, 18_000)
            payment_type = draw_one(["Cash Deposit", "Cash Withdrawal", "PayShap", "Electronic Transfer", "Debit Card"], [0.24, 0.26, 0.24, 0.16, 0.10])
            counterparty_type = "Gig work or household"

        amount = maybe_add_legitimate_high_value_tail(amount, behaviour, direction, acc)

        transaction_purpose_category = legitimate_purpose_category(
            behaviour, direction, payment_type, counterparty_type
        )

        return finalise_transaction(
            acc, tx_id, tx_datetime, direction, amount, sender_country, receiver_country,
            payment_type, counterparty_type, transaction_purpose_category,
            0, "Normal", "legitimate_socioeconomic",
            "Legitimate", "Normal socioeconomic behaviour", behaviour,
            public_sector_counterparty=public_sector_counterparty,
            social_grant_type=social_grant_type
        )

    LOW_VALUE_OVERLAP_TYPOLOGIES = {
        "Mule Account / Rapid Pass-Through", "Cybercrime Proceeds",
        "Ponzi / Investment Scheme Collections", "Stolen Student Loan or Grant Abuse",
        "Crypto Asset Off-Ramp", "Fraudulent Company or Invoice Scam"
    }
    MEDIUM_VALUE_OVERLAP_TYPOLOGIES = {
        "Cross-Border Layering via SADC Corridor", "High-Risk Jurisdiction Layering",
        "Intermediary / Trust Account Abuse", "NPO / Community Organisation Misuse",
        "Narcotics Cash Proceeds", "Casino or Gambling Laundering"
    }
    HIGH_VALUE_INTEGRATION_TYPOLOGIES = {
        "Procurement Corruption / Tender Kickback", "Shell or Front Company Flow",
        "Complex Legal Structure / Beneficial Ownership Obfuscation",
        "Illegal Gold or Precious Metals", "Environmental or Wildlife Crime Proceeds",
        "Property / Real Estate Integration", "Motor Vehicle or High-Value Goods Purchase",
        "Trade-Based Import-Export Layering"
    }


    def maybe_add_suspicious_amount_overlap(amount, typology):
        """Keep AML typologies realistic while reducing amount-only separability."""
        if typology in LOW_VALUE_OVERLAP_TYPOLOGIES and rng.random() < 0.64:
            return min(float(amount), lognormal_amount(2_600, 1.05, 80, 45_000))
        if typology in MEDIUM_VALUE_OVERLAP_TYPOLOGIES and rng.random() < 0.48:
            return min(float(amount), lognormal_amount(7_500, 1.08, 250, 95_000))
        if typology in HIGH_VALUE_INTEGRATION_TYPOLOGIES and rng.random() < 0.24:
            return min(float(amount), lognormal_amount(18_000, 1.05, 900, 180_000))
        return float(amount)


    def generate_suspicious_transaction(acc, tx_id, tx_datetime=None):
        typology = acc.Laundering_type
        if tx_datetime is None:
            tx_datetime = sample_suspicious_datetime(acc, typology)
        sender_country = SA
        receiver_country = SA
        public_sector_counterparty = 0

        if typology == "Structuring / Smurfing Cash Deposits":
            direction = "Inflow"
            amount = rng.uniform(40_000, 49_999.99)
            payment_type = "Cash Deposit"
            counterparty_type = "Cash courier or related individual"

        elif typology == "Cash Courier / Bulk Cash Movement":
            direction = draw_one(["Inflow", "Outflow"], [0.64, 0.36])
            amount = lognormal_amount(58_000, 0.82, 18_000, 240_000)
            payment_type = draw_one(["Cash Deposit", "Cash Withdrawal"], [0.62, 0.38])
            counterparty_type = "Bulk cash courier or cash-intensive third party"

        elif typology == "Mule Account / Rapid Pass-Through":
            direction = draw_one(["Inflow", "Outflow"], [0.57, 0.43])
            amount = lognormal_amount(3800, 1.05, 120, 65_000)
            payment_type = draw_one(["PayShap", "RTC", "Electronic Transfer", "Cash Withdrawal"], [0.34, 0.26, 0.25, 0.15])
            counterparty_type = "Peer or mule network account"
            if rng.random() < 0.16:
                foreign_country = choose_transaction_country(acc, force_foreign=True, suspicious=False)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Cross-Border Layering via SADC Corridor":
            direction = draw_one(["Inflow", "Outflow"], [0.52, 0.48])
            amount = lognormal_amount(19_000, 1.20, 500, 650_000)
            payment_type = draw_one(["International Wire", "Electronic Transfer", "RTC"], [0.62, 0.24, 0.14])
            counterparty_type = "Foreign counterparty or regional corridor account"
            foreign_country = draw_one(SADC_CORRIDORS)
            if direction == "Inflow":
                sender_country = foreign_country
            else:
                receiver_country = foreign_country

        elif typology == "High-Risk Jurisdiction Layering":
            direction = draw_one(["Inflow", "Outflow"], [0.50, 0.50])
            amount = lognormal_amount(28_000, 1.18, 800, 900_000)
            payment_type = draw_one(["International Wire", "Electronic Transfer"], [0.76, 0.24])
            counterparty_type = "FATF-monitored foreign counterparty"
            foreign_country = draw_one(FATF_INCREASED_MONITORING + FATF_CALL_FOR_ACTION, [1] * len(FATF_INCREASED_MONITORING) + [1.6] * len(FATF_CALL_FOR_ACTION))
            if direction == "Inflow":
                sender_country = foreign_country
            else:
                receiver_country = foreign_country

        elif typology == "Intermediary / Trust Account Abuse":
            direction = draw_one(["Inflow", "Outflow"], [0.50, 0.50])
            amount = lognormal_amount(18_000, 1.08, 500, 450_000)
            payment_type = draw_one(["Electronic Transfer", "RTC", "International Wire"], [0.52, 0.28, 0.20])
            counterparty_type = "Intermediary, attorney trust or connected-party account"
            if rng.random() < 0.25:
                foreign_country = choose_transaction_country(acc, suspicious=True)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Procurement Corruption / Tender Kickback":
            direction = draw_one(["Inflow", "Outflow"], [0.68, 0.32])
            amount = float(rng.choice([18_000, 32_000, 45_000, 49_500, 75_000, 120_000])) if rng.random() < 0.54 else lognormal_amount(95_000, 1.05, 8_000, 1_500_000)
            payment_type = draw_one(["Procurement EFT", "Electronic Transfer", "RTC"], [0.66, 0.26, 0.08])
            counterparty_type = draw_one(["Public sector entity", "Connected vendor", "Subcontractor"], [0.50, 0.28, 0.22])
            public_sector_counterparty = int(counterparty_type == "Public sector entity" or direction == "Inflow")

        elif typology in ["Shell or Front Company Flow", "Complex Legal Structure / Beneficial Ownership Obfuscation"]:
            direction = draw_one(["Inflow", "Outflow"], [0.48, 0.52])
            amount = lognormal_amount(95_000, 1.18, 8_000, 2_500_000)
            payment_type = draw_one(["Electronic Transfer", "International Wire", "RTC"], [0.50, 0.36, 0.14])
            counterparty_type = "Shell company, trust or opaque beneficial-owner entity"
            if rng.random() < 0.44:
                foreign_country = choose_transaction_country(acc, suspicious=True)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Crypto Asset Off-Ramp":
            direction = draw_one(["Inflow", "Outflow"], [0.72, 0.28])
            amount = lognormal_amount(17_000, 0.95, 1_500, 140_000)
            payment_type = "Crypto Exchange Off-Ramp"
            counterparty_type = "Virtual asset service provider"
            if rng.random() < 0.35:
                foreign_country = draw_one(["South Africa", "Seychelles", "United Arab Emirates", "Kenya", "Namibia"], [0.45, 0.12, 0.18, 0.15, 0.10])
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Ponzi / Investment Scheme Collections":
            direction = draw_one(["Inflow", "Outflow"], [0.74, 0.26])
            amount = lognormal_amount(6200, 0.95, 500, 120_000)
            payment_type = draw_one(["Electronic Transfer", "RTC", "Cash Deposit"], [0.46, 0.34, 0.20])
            counterparty_type = "Retail investor or promoter account"

        elif typology == "Fraudulent Company or Invoice Scam":
            direction = draw_one(["Inflow", "Outflow"], [0.60, 0.40])
            amount = lognormal_amount(18_000, 1.10, 800, 420_000)
            payment_type = draw_one(["Electronic Transfer", "RTC", "International Wire"], [0.55, 0.34, 0.11])
            counterparty_type = draw_one(["Victim company", "Invoice-scam beneficiary", "Rapid onward-payment account"], [0.36, 0.34, 0.30])

        elif typology == "Cybercrime Proceeds":
            direction = draw_one(["Inflow", "Outflow"], [0.58, 0.42])
            amount = lognormal_amount(4800, 1.00, 100, 95_000)
            payment_type = draw_one(["PayShap", "RTC", "Electronic Transfer", "Cash Withdrawal"], [0.36, 0.28, 0.24, 0.12])
            counterparty_type = "Cybercrime victim, mule or cash-out account"

        elif typology == "Narcotics Cash Proceeds":
            direction = draw_one(["Inflow", "Outflow"], [0.70, 0.30])
            amount = lognormal_amount(12_500, 1.00, 300, 49_999)
            payment_type = draw_one(["Cash Deposit", "Cash Withdrawal", "Electronic Transfer"], [0.62, 0.22, 0.16])
            counterparty_type = "Cash-intensive retail front or courier"

        elif typology == "Illegal Gold or Precious Metals":
            direction = draw_one(["Inflow", "Outflow"], [0.55, 0.45])
            amount = lognormal_amount(62_000, 1.05, 4_000, 1_200_000)
            payment_type = draw_one(["Cash Deposit", "Electronic Transfer", "International Wire"], [0.34, 0.42, 0.24])
            counterparty_type = "Precious metals dealer or smuggling intermediary"
            if rng.random() < 0.40:
                foreign_country = draw_one(["Zimbabwe", "Mozambique", "United Arab Emirates", "Namibia", "Tanzania"], [0.30, 0.20, 0.24, 0.14, 0.12])
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Environmental or Wildlife Crime Proceeds":
            direction = draw_one(["Inflow", "Outflow"], [0.52, 0.48])
            amount = lognormal_amount(34_000, 1.15, 1_500, 850_000)
            payment_type = draw_one(["Electronic Transfer", "International Wire", "Cash Deposit"], [0.48, 0.30, 0.22])
            counterparty_type = "Environmental commodity, logistics or export counterparty"
            if rng.random() < 0.35:
                foreign_country = choose_transaction_country(acc, suspicious=True)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Stolen Student Loan or Grant Abuse":
            direction = draw_one(["Inflow", "Outflow"], [0.56, 0.44])
            amount = lognormal_amount(3200, 0.90, 200, 55_000)
            payment_type = draw_one(["Electronic Transfer", "PayShap", "Cash Withdrawal", "Debit Card"], [0.34, 0.30, 0.24, 0.12])
            counterparty_type = "Student funding beneficiary, grant payer or mule"
            public_sector_counterparty = int(direction == "Inflow" and rng.random() < 0.44)

        elif typology == "Casino or Gambling Laundering":
            direction = draw_one(["Inflow", "Outflow"], [0.50, 0.50])
            amount = lognormal_amount(15_000, 1.10, 500, 260_000)
            payment_type = draw_one(["Cash Deposit", "Cash Withdrawal", "Electronic Transfer", "Debit Card"], [0.30, 0.24, 0.30, 0.16])
            counterparty_type = "Casino, betting wallet or gambling value instrument"

        elif typology == "Property / Real Estate Integration":
            direction = draw_one(["Inflow", "Outflow"], [0.45, 0.55])
            amount = lognormal_amount(210_000, 1.00, 45_000, 3_500_000)
            payment_type = draw_one(["Electronic Transfer", "International Wire"], [0.78, 0.22])
            counterparty_type = "Property practitioner, conveyancer or rental counterparty"
            if rng.random() < 0.22:
                foreign_country = choose_transaction_country(acc, suspicious=True)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        elif typology == "Motor Vehicle or High-Value Goods Purchase":
            direction = draw_one(["Inflow", "Outflow"], [0.36, 0.64])
            amount = lognormal_amount(125_000, 0.90, 35_000, 1_400_000)
            payment_type = draw_one(["Electronic Transfer", "Cash Deposit", "Cash Withdrawal"], [0.58, 0.30, 0.12])
            counterparty_type = "Motor vehicle or high-value goods dealer"

        elif typology == "Trade-Based Import-Export Layering":
            direction = draw_one(["Inflow", "Outflow"], [0.48, 0.52])
            amount = lognormal_amount(135_000, 1.12, 15_000, 2_500_000)
            payment_type = draw_one(["International Wire", "Electronic Transfer"], [0.72, 0.28])
            counterparty_type = "Import/export or trade counterparty"
            foreign_country = choose_transaction_country(acc, suspicious=True)
            if direction == "Inflow":
                sender_country = foreign_country
            else:
                receiver_country = foreign_country

        elif typology == "NPO / Community Organisation Misuse":
            direction = draw_one(["Inflow", "Outflow"], [0.55, 0.45])
            amount = lognormal_amount(9_500, 1.12, 300, 240_000)
            payment_type = draw_one(["Electronic Transfer", "Cash Deposit", "International Wire", "PayShap"], [0.42, 0.25, 0.20, 0.13])
            counterparty_type = "Donation, community organisation or foreign donor counterparty"
            if rng.random() < 0.28:
                foreign_country = choose_transaction_country(acc, suspicious=True)
                if direction == "Inflow":
                    sender_country = foreign_country
                else:
                    receiver_country = foreign_country

        else:
            return generate_legitimate_transaction(acc, tx_id)

        amount = maybe_add_suspicious_amount_overlap(amount, typology)

        transaction_purpose_category = suspicious_purpose_category(typology, payment_type, counterparty_type)
        scenario_meta = AML_SCENARIO_CATALOG.get(typology, {})
        return finalise_transaction(
            acc, tx_id, tx_datetime, direction, amount, sender_country, receiver_country,
            payment_type, counterparty_type, transaction_purpose_category,
            1, typology, "suspicious_laundering",
            scenario_meta.get("stage", "Layering"),
            scenario_meta.get("indicator", "Scenario-specific suspicious behaviour"),
            typology,
            public_sector_counterparty=public_sector_counterparty
        )


    # ============================================================
    # GENERATE TRANSACTIONS
    # ============================================================

    transactions = []
    tx_id = 0

    generated_counts = []

    total_accounts_to_generate = len(receiver_accounts)
    progress_every = max(total_accounts_to_generate // 10, 1)
    _progress(f"Generating transaction histories for {total_accounts_to_generate:,} receiver accounts.")

    for account_number, acc in enumerate(receiver_accounts.itertuples(index=False), start=1):
        n_tx = draw_account_transaction_count(acc)
        generated_counts.append(n_tx)
        suspicious_tx_probability = TYPOLOGY_TX_PROB.get(acc.Laundering_type, 0.0)

        forced_suspicious_position = int(rng.integers(0, n_tx)) if acc.Account_is_laundering and n_tx > 0 else -1
        previous_legitimate_datetime = None
        previous_suspicious_datetime = None

        for tx_number in range(n_tx):
            force_suspicious_for_account = bool(
                acc.Account_is_laundering and tx_number == forced_suspicious_position
            )
            is_suspicious_tx = bool(
                force_suspicious_for_account or (acc.Account_is_laundering and rng.random() < suspicious_tx_probability)
            )
            if is_suspicious_tx:
                tx_datetime = sample_suspicious_datetime(acc, acc.Laundering_type, previous_suspicious_datetime)
                previous_suspicious_datetime = tx_datetime
                transactions.append(generate_suspicious_transaction(acc, tx_id, tx_datetime=tx_datetime))
            else:
                tx_datetime = sample_legitimate_datetime(acc, previous_legitimate_datetime)
                previous_legitimate_datetime = tx_datetime
                transactions.append(generate_legitimate_transaction(acc, tx_id, tx_datetime=tx_datetime))
            tx_id += 1
        if account_number % progress_every == 0 or account_number == total_accounts_to_generate:
            _progress(
                "Transaction histories generated for "
                f"{account_number:,}/{total_accounts_to_generate:,} accounts; rows so far: {tx_id:,}."
            )

    receiver_accounts["Generated_transaction_count"] = generated_counts


    # ============================================================
    # CREATE FINAL DATAFRAME
    # ============================================================

    _progress(f"Assembling transaction dataframe with {len(transactions):,} rows.")
    df = pd.DataFrame.from_records(transactions, columns=TX_COLUMNS)

    # Add scenario and socioeconomic context after transaction creation. The label
    # remains Is_laundering; these columns make scenario selection, stratified
    # evaluation and transparent documentation possible.
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

    scenario_maps = {
        field: {name: meta[field] for name, meta in AML_SCENARIO_CATALOG.items()}
        for field in ["code", "predicate_offence", "stage", "risk_sector", "source_keys", "indicator"]
    }
    normal_source_map = {name: meta["source_keys"] for name, meta in NORMAL_BEHAVIOUR_CATALOG.items()}
    normal_indicator_map = {name: meta["indicator"] for name, meta in NORMAL_BEHAVIOUR_CATALOG.items()}

    df["AML_scenario_category"] = np.where(df["Is_laundering"] == 1, df["Laundering_type"], "Normal")
    df["AML_scenario_code"] = df["Laundering_type"].map(scenario_maps["code"]).fillna("NORMAL")
    df["AML_predicate_offence"] = df["Laundering_type"].map(scenario_maps["predicate_offence"]).fillna("None - legitimate activity")
    df["AML_stage"] = df["Laundering_type"].map(scenario_maps["stage"]).fillna("Legitimate")
    df["AML_risk_sector"] = df["Laundering_type"].map(scenario_maps["risk_sector"]).fillna("Normal retail / household")
    df["AML_scenario_source_keys"] = df["Laundering_type"].map(scenario_maps["source_keys"]).fillna("Not applicable")
    df["Scenario_reference_key"] = np.where(df["Is_laundering"] == 1, df["AML_scenario_source_keys"], df["Legitimate_behaviour"].map(normal_source_map).fillna("Not applicable"))
    df["Scenario_detail"] = np.where(df["Is_laundering"] == 1, df["Laundering_type"].map(scenario_maps["indicator"]), df["Legitimate_behaviour"].map(normal_indicator_map))
    df["Normal_behavior_category"] = np.where(df["Is_laundering"] == 0, df["Legitimate_behaviour"], "Suspicious overlay")
    df["Normal_behavior_source_keys"] = df["Normal_behavior_category"].map(normal_source_map).fillna("Not applicable")

    ACCOUNT_PROFILE_COLUMNS = [
        "Gender", "Age", "Age_band", "Education_level", "Nationality_country",
        "Residency_status", "Migrant_worker_flag", "Urban_rural",
        "Township_or_informal_settlement_flag", "Financial_inclusion_segment",
        "Channel_preference", "Primary_social_grant_type", "KYC_strength", "Informal_business_flag",
        "Business_registered", "Business_registration_status", "Has_trading_permit",
        "VAT_registered", "Uses_personal_account_for_business", "Business_premises_type",
        "Estimated_daily_cash_turnover", "Business_age_years", "Keeps_business_records",
        "Has_bank_loan_access", "High_risk_sector_category", "High_risk_sector_flag",
        "High_risk_person_type", "High_risk_person_flag",
        "Targeted_financial_sanctions_match", "Adverse_media_flag",
        "Typical_foreign_counterparty_country", "Account_open_date",
        "First_observed_transaction_date", "Opened_during_observation_flag",
        "Opened_for_laundering_flag", "Scenario_forced_coverage_flag",
        "Active_observation_days", "Account_activity_multiplier", "Expected_transaction_count",
        "Generated_transaction_count",
    ]
    profile_lookup = receiver_accounts.set_index("Receiver_account")
    for column in ACCOUNT_PROFILE_COLUMNS:
        df[column] = df["Account_id"].map(profile_lookup[column])

    # Baseline SAML-D style view. The richer df keeps the context needed for
    # account-level and sequential modelling.
    df_baseline = df[[
        "Time", "Date", "Sender_account", "Receiver_account", "Amount",
        "Payment_currency", "Received_currency", "Sender_bank_location",
        "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type"
    ]].copy()

    df_final = df
    RAW_SIMULATION_COLUMNS = list(df_final.columns)

    VALIDATION_ONLY_COLUMNS = [
        "Account_is_laundering", "Laundering_type", "Behaviour_component",
        "Typology_stage", "Suspicion_indicator", "Scenario_label",
        "AML_scenario_category", "AML_scenario_code", "AML_predicate_offence",
        "AML_stage", "AML_risk_sector", "AML_scenario_source_keys",
        "Scenario_reference_key", "Scenario_detail", "Normal_behavior_category",
        "Normal_behavior_source_keys", "Opened_for_laundering_flag",
        "Scenario_forced_coverage_flag", "Account_laundering_probability",
        "Transaction_purpose_category"
    ]
    BEHAVIOUR_ONLY_COLUMNS = [
        col for col in RAW_SIMULATION_COLUMNS
        if col not in VALIDATION_ONLY_COLUMNS and col != "Is_laundering"
    ]

    print("Total Transactions:", len(df))
    print("Transaction-level suspicious %:", round(df["Is_laundering"].mean(), 4))
    print("Account-level laundering %:", round(receiver_accounts["Account_is_laundering"].mean(), 4))
    print("Simulation period:", START_DATE.date(), "to", END_DATE.date(), "days:", DAYS)
    print("Unique monitored receiver accounts:", receiver_accounts["Receiver_account"].nunique())
    print("Median generated transactions/account:", int(receiver_accounts["Generated_transaction_count"].median()))
    print("Suspicious scenario categories represented:", df.loc[df["Is_laundering"] == 1, "AML_scenario_category"].nunique())
    print("Normal behaviour categories represented:", df.loc[df["Is_laundering"] == 0, "Normal_behavior_category"].nunique())
    print("Cross-border transaction %:", round(df["Cross_border_transaction"].mean(), 4))
    print("FATF contextual exposure %:", round(df["Jurisdiction_risk_tier"].isin(["FATF Increased Monitoring", "FATF Call for Action"]).mean(), 4))
    print("Customer profile columns:", len(receiver_accounts.columns))
    print("Raw simulation columns:", len(RAW_SIMULATION_COLUMNS))
    print("Max transactions/account cap:", MAX_TX_PER_ACCOUNT)
    non_normal_overlay_probs = [v for k, v in TYPOLOGY_TX_PROB.items() if k != "Normal"]
    print("Suspicious overlay probability range:", round(min(non_normal_overlay_probs), 3), "to", round(max(non_normal_overlay_probs), 3))
    print("Validation-only / label-adjacent columns to exclude from behaviour experiments:", len([c for c in VALIDATION_ONLY_COLUMNS if c in df_final.columns]))


    # Date and time normalisation used by the plots and sequence features below.
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'].astype(str), format="%H:%M:%S").dt.time

    df['fulldatetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    )

    df.columns


    laundering_stats = df[df['Is_laundering'] == 1]['Amount'].agg(['max', 'mean', 'min'])

    normal_stats = df[df['Is_laundering'] == 0]['Amount'].agg(['max', 'mean', 'min'])

    _progress("Transaction amount sanity check:")
    print("Laundering Transactions Stats:\n", laundering_stats, flush=True)
    print("\nNormal Transactions Stats:\n", normal_stats, flush=True)


    # ========== 4. Feature Engineering ==========
    # These variables are deliberately separated from the label. They are context
    # features a model may learn from, not rules that automatically imply laundering.

    _progress("Creating diagnostic fields used for validation and export tables.")
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
    _progress(f"Diagnostic columns created outside the raw public export: {len(ENGINEERED_FEATURE_COLUMNS):,}.")


    # Final exports. The public transaction file is raw simulation output only:
    # no cyclical encodings, rolling statistics, account aggregates, sequence-density
    # fields or composite risk scores. The Excel workbook stores summary results;
    # Excel cannot hold the full-scale transaction table in one sheet.
    EXPORT_SIMULATION_FILES = os.getenv(
        "AML_SIM_EXPORT_FILES",
        os.getenv("AML_SIM_EXPORT_CSV", "1")
    ) != "0"
    EXPORT_EXCEL_RESULTS = os.getenv("AML_SIM_EXPORT_EXCEL", "1") != "0"
    EXPORT_ENGINEERED_FEATURES = os.getenv("AML_SIM_EXPORT_ENGINEERED", "0") != "0"
    OUTPUT_PREFIX = os.getenv("AML_SIM_OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX)
    _progress(
        "Preparing exports: "
        f"files={'yes' if EXPORT_SIMULATION_FILES else 'no'}, "
        f"excel={'yes' if EXPORT_EXCEL_RESULTS else 'no'}, "
        f"engineered={'yes' if EXPORT_ENGINEERED_FEATURES else 'no'}, "
        f"prefix={OUTPUT_PREFIX}."
    )
    DATA_OUTPUT_ROOT = Path(os.getenv("AML_SIM_DATA_DIR", str(DEFAULT_SIMULATED_DATA_DIR)))
    SIMULATED_DATA_DIR = DATA_OUTPUT_ROOT
    FULL_SAML_DIR = SIMULATED_DATA_DIR / "01_full_saml"
    BEHAVIOUR_ONLY_DIR = SIMULATED_DATA_DIR / "02_behaviour_only"
    FINAL_RESULTS_DIR = SIMULATED_DATA_DIR / "03_final_results_excel"
    OPTIONAL_ENGINEERED_DIR = SIMULATED_DATA_DIR / "04_optional_engineered"
    for output_dir in [FULL_SAML_DIR, BEHAVIOUR_ONLY_DIR, FINAL_RESULTS_DIR, OPTIONAL_ENGINEERED_DIR]:
        output_dir.mkdir(parents=True, exist_ok=True)
    EXCEL_SAMPLE_ROWS = int(os.getenv("AML_SIM_EXCEL_SAMPLE_ROWS", "5000"))
    EXCEL_MAX_ROWS = 1_048_576

    raw_export_columns = [col for col in RAW_SIMULATION_COLUMNS if col in df_final.columns]
    behaviour_export_columns = ["Is_laundering"] + [
        col for col in raw_export_columns
        if col not in VALIDATION_ONLY_COLUMNS and col != "Is_laundering"
    ]


    def percent(series):
        return (series * 100).round(2)


    def make_export_summary_tables():
        account_counts = df_final.groupby("Account_id", observed=True).size()
        account_level = df_final.drop_duplicates("Account_id")
        fatf_mask = df_final["Jurisdiction_risk_tier"].isin([
            "FATF Increased Monitoring", "FATF Call for Action"
        ])

        dataset_overview = pd.DataFrame([
            ("Receiver accounts", int(df_final["Account_id"].nunique())),
            ("Transaction rows", int(len(df_final))),
            ("Average transactions per account", round(float(len(df_final) / df_final["Account_id"].nunique()), 2)),
            ("Median transactions per account", int(account_counts.median())),
            ("95th percentile transactions per account", int(account_counts.quantile(0.95))),
            ("Maximum transactions per account", int(account_counts.max())),
            ("Legitimate transaction rows", int((df_final["Is_laundering"] == 0).sum())),
            ("Suspicious AML transaction rows", int((df_final["Is_laundering"] == 1).sum())),
            ("Legitimate accounts", int((account_level["Account_is_laundering"] == 0).sum())),
            ("Suspicious AML accounts", int((account_level["Account_is_laundering"] == 1).sum())),
            ("Transaction-level suspicious share", f"{df_final['Is_laundering'].mean() * 100:.2f}%"),
            ("Account-level suspicious share", f"{account_level['Account_is_laundering'].mean() * 100:.2f}%"),
        ], columns=["Metric", "Value"])

        validation_summary = pd.DataFrame([
            ("Receiver accounts", int(df_final["Account_id"].nunique())),
            ("Transactions", int(len(df_final))),
            ("Observation window", f"{START_DATE.date()} to {END_DATE.date()}"),
            ("Target suspicious-account share", f"{AML_ACCOUNT_SHARE_TARGET * 100:.2f}%"),
            ("Realised suspicious-account share", f"{account_level['Account_is_laundering'].mean() * 100:.2f}%"),
            ("Transaction-level suspicious share", f"{df_final['Is_laundering'].mean() * 100:.2f}%"),
            ("AML scenarios represented", int(df_final.loc[df_final["Is_laundering"] == 1, "AML_scenario_category"].nunique())),
            ("Normal behaviour categories represented", int(df_final.loc[df_final["Is_laundering"] == 0, "Normal_behavior_category"].nunique())),
            ("Median transactions per account", int(account_counts.median())),
            ("95th percentile transactions per account", int(account_counts.quantile(0.95))),
            ("Maximum transactions per account", int(account_counts.max())),
            ("Cross-border transaction share", f"{df_final['Cross_border_transaction'].mean() * 100:.2f}%"),
            ("FATF contextual exposure share", f"{fatf_mask.mean() * 100:.2f}%"),
            ("Raw simulation columns exported", int(len(raw_export_columns))),
            ("Raw behaviour-only columns exported", int(len(behaviour_export_columns))),
            ("Engineered notebook-only columns", int(len(ENGINEERED_FEATURE_COLUMNS))),
        ], columns=["Metric", "Value"])

        label_comparison = (
            df_final.groupby("Is_laundering", observed=True)
            .agg(
                transactions=("Transaction_id", "count"),
                median_amount=("Amount", "median"),
                cash_share=("cash_transaction", "mean"),
                cross_border_share=("Cross_border_transaction", "mean"),
                iftr_share=("iftr_reportable_cross_border", "mean"),
                rapid_repeat_share=("rapid_repeat_transaction", "mean"),
            )
            .reset_index()
        )
        label_comparison["Label"] = np.where(
            label_comparison["Is_laundering"] == 1,
            "Suspicious AML",
            "Legitimate baseline"
        )
        for col in ["cash_share", "cross_border_share", "iftr_share", "rapid_repeat_share"]:
            label_comparison[col] = percent(label_comparison[col])
        label_comparison = label_comparison[[
            "Label", "transactions", "median_amount", "cash_share",
            "cross_border_share", "iftr_share", "rapid_repeat_share"
        ]]

        suspicious_df = df_final[df_final["Is_laundering"] == 1]
        scenario_metrics = (
            suspicious_df.groupby("AML_scenario_category", observed=True)
            .agg(
                transactions=("Transaction_id", "count"),
                median_amount=("Amount", "median"),
                cash_share=("cash_transaction", "mean"),
                cross_border_share=("Cross_border_transaction", "mean"),
                iftr_share=("iftr_reportable_cross_border", "mean"),
                rapid_repeat_share=("rapid_repeat_transaction", "mean"),
            )
            .reset_index()
            .sort_values("transactions", ascending=False)
        )
        for col in ["cash_share", "cross_border_share", "iftr_share", "rapid_repeat_share"]:
            scenario_metrics[col] = percent(scenario_metrics[col])

        normal_df = df_final[df_final["Is_laundering"] == 0]
        normal_behaviour_metrics = (
            normal_df.groupby("Normal_behavior_category", observed=True)
            .agg(
                transactions=("Transaction_id", "count"),
                median_amount=("Amount", "median"),
                cash_share=("cash_transaction", "mean"),
                cross_border_share=("Cross_border_transaction", "mean"),
                rapid_repeat_share=("rapid_repeat_transaction", "mean"),
            )
            .reset_index()
            .sort_values("transactions", ascending=False)
        )
        for col in ["cash_share", "cross_border_share", "rapid_repeat_share"]:
            normal_behaviour_metrics[col] = percent(normal_behaviour_metrics[col])

        payment_type_metrics = (
            df_final.groupby("Payment_type", observed=True)
            .agg(
                transactions=("Transaction_id", "count"),
                suspicious_rate=("Is_laundering", "mean"),
                median_amount=("Amount", "median"),
                cross_border_share=("Cross_border_transaction", "mean"),
            )
            .reset_index()
            .sort_values("transactions", ascending=False)
        )
        for col in ["suspicious_rate", "cross_border_share"]:
            payment_type_metrics[col] = percent(payment_type_metrics[col])

        province_metrics = (
            account_level.groupby("Province", observed=True)
            .agg(
                accounts=("Account_id", "count"),
                suspicious_account_rate=("Account_is_laundering", "mean"),
                median_generated_tx=("Generated_transaction_count", "median"),
            )
            .reset_index()
            .sort_values("accounts", ascending=False)
        )
        province_metrics["suspicious_account_rate"] = percent(province_metrics["suspicious_account_rate"])

        cross_border_df = df_final[df_final["Cross_border_transaction"] == 1]
        country_metrics = (
            cross_border_df.groupby("Counterparty_country", observed=True)
            .agg(
                transactions=("Transaction_id", "count"),
                suspicious_rate=("Is_laundering", "mean"),
                median_amount=("Amount", "median"),
                iftr_share=("iftr_reportable_cross_border", "mean"),
            )
            .reset_index()
            .sort_values("transactions", ascending=False)
        )
        for col in ["suspicious_rate", "iftr_share"]:
            country_metrics[col] = percent(country_metrics[col])

        transaction_descriptive_stats = (
            df_final.assign(Label=np.where(df_final["Is_laundering"] == 1, "Suspicious AML", "Legitimate baseline"))
            .groupby("Label", observed=True)
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
        transaction_descriptive_stats["cross_border_share"] = percent(transaction_descriptive_stats["cross_border_share"])

        account_sequence_stats = (
            account_level.assign(Label=np.where(account_level["Account_is_laundering"] == 1, "Suspicious AML account", "Legitimate account"))
            .groupby("Label", observed=True)
            .agg(
                accounts=("Generated_transaction_count", "size"),
                mean_tx=("Generated_transaction_count", "mean"),
                std_tx=("Generated_transaction_count", "std"),
                median_tx=("Generated_transaction_count", "median"),
                p25_tx=("Generated_transaction_count", lambda s: s.quantile(0.25)),
                p75_tx=("Generated_transaction_count", lambda s: s.quantile(0.75)),
                p95_tx=("Generated_transaction_count", lambda s: s.quantile(0.95)),
                max_tx=("Generated_transaction_count", "max"),
                mean_active_days=("Active_observation_days", "mean"),
            )
            .reset_index()
        )

        account_sequence_quantiles = account_counts.quantile([0, .25, .5, .75, .9, .95, .99, 1]).reset_index()
        account_sequence_quantiles.columns = ["Quantile", "Transactions_per_account"]

        raw_columns_df = pd.DataFrame({"Raw_simulation_column": raw_export_columns})
        behaviour_columns_df = pd.DataFrame({"Raw_behaviour_only_column": behaviour_export_columns})
        engineered_columns_df = pd.DataFrame({"Notebook_engineered_column_not_in_raw_export": ENGINEERED_FEATURE_COLUMNS})

        return {
            "dataset_overview": dataset_overview,
            "validation_summary": validation_summary,
            "label_comparison": label_comparison,
            "transaction_stats": transaction_descriptive_stats,
            "account_sequence_stats": account_sequence_stats,
            "scenario_metrics": scenario_metrics,
            "normal_behaviour": normal_behaviour_metrics,
            "payment_type_metrics": payment_type_metrics,
            "province_metrics": province_metrics,
            "country_metrics": country_metrics,
            "sequence_quantiles": account_sequence_quantiles,
            "raw_columns": raw_columns_df,
            "behaviour_columns": behaviour_columns_df,
            "engineered_not_exported": engineered_columns_df,
        }


    if EXPORT_SIMULATION_FILES:
        raw_path = FULL_SAML_DIR / f"{OUTPUT_PREFIX}.csv"
        behaviour_path = BEHAVIOUR_ONLY_DIR / f"{OUTPUT_PREFIX}_behaviour_only.csv"
        excel_path = FINAL_RESULTS_DIR / f"{OUTPUT_PREFIX}_final_results.xlsx"
        engineered_path = OPTIONAL_ENGINEERED_DIR / f"{OUTPUT_PREFIX}_engineered_features.csv"

        df_final[raw_export_columns].to_csv(raw_path, index=False)
        df_final[behaviour_export_columns].to_csv(behaviour_path, index=False)
        print(f"Saved raw simulation dataset to {raw_path.resolve()}")
        print(f"Saved raw behaviour-only dataset to {behaviour_path.resolve()}")
        print(f"Raw simulation columns exported: {len(raw_export_columns)}")
        print(f"Raw behaviour-only columns exported: {len(behaviour_export_columns)}")

        if EXPORT_EXCEL_RESULTS:
            summary_tables = make_export_summary_tables()
            try:
                with pd.ExcelWriter(excel_path) as writer:
                    for sheet_name, table in summary_tables.items():
                        table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    sample_rows = min(EXCEL_SAMPLE_ROWS, EXCEL_MAX_ROWS - 1, len(df_final))
                    if sample_rows > 0:
                        df_final[raw_export_columns].head(sample_rows).to_excel(
                            writer,
                            sheet_name="raw_sample",
                            index=False,
                        )
                print(f"Saved Excel final-results workbook to {excel_path.resolve()}")
                print(f"Excel raw_sample rows: {min(EXCEL_SAMPLE_ROWS, EXCEL_MAX_ROWS - 1, len(df_final)):,}")
            except Exception as exc:
                print(f"Excel export failed: {exc}")
                print("Install openpyxl or xlsxwriter, then rerun the export cell to create the .xlsx workbook.")

        if EXPORT_ENGINEERED_FEATURES:
            engineered_export_columns = ["Transaction_id", "Account_id", "Is_laundering"] + [
                col for col in ENGINEERED_FEATURE_COLUMNS
                if col in df_final.columns
            ]
            df_final[engineered_export_columns].to_csv(engineered_path, index=False)
            print(f"Saved optional engineered diagnostics to {engineered_path.resolve()}")
    else:
        print("File export skipped. Set AML_SIM_EXPORT_FILES=1 to save raw CSV files and the Excel final-results workbook.")
        print(f"Dataset folders are prepared under {SIMULATED_DATA_DIR.resolve()}")

    _progress(
        "Simulation complete: "
        f"{df_final['Account_id'].nunique():,} accounts and {len(df_final):,} transaction rows generated."
    )

    return {
        "df": df,
        "df_final": df_final,
        "df_baseline": df_baseline,
        "receiver_accounts": receiver_accounts,
        "scenario_catalog_df": scenario_catalog_df,
        "normal_behavior_catalog_df": normal_behavior_catalog_df,
        "raw_simulation_columns": RAW_SIMULATION_COLUMNS,
        "behaviour_only_columns": BEHAVIOUR_ONLY_COLUMNS,
        "engineered_feature_columns": ENGINEERED_FEATURE_COLUMNS,
        "export_summary_tables": export_summary_tables if "export_summary_tables" in locals() else {},
        "output_prefix": OUTPUT_PREFIX if "OUTPUT_PREFIX" in locals() else DEFAULT_OUTPUT_PREFIX,
    }


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()
