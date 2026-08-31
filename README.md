# South African AML Data Simulation

This repository contains a context-aware synthetic anti-money laundering (AML)
data simulation framework for South African retail banking behaviour. The
framework is designed to generate synthetic transaction data that reflects both
ordinary socioeconomic behaviour and suspicious AML typologies in a South
African setting.

The goal is not to provide a trained detection model. The goal is to create a
reproducible data-generation pipeline that can later support AML modelling,
validation, explainability, calibration and benchmarking.

## What The Simulator Generates

The simulator creates a synthetic retail banking population with monitored
receiver accounts, account profiles, transaction histories and scenario labels.
It represents legitimate behaviour such as salary payments, household spending,
grant-linked flows, informal business cash cycles, spaza and township retail,
taxi and transport cash cycles, remittances, stokvels, business operating
payments and irregular income.

It also overlays suspicious AML scenarios such as structuring, mule accounts,
cross-border layering, procurement corruption, shell or front company flows,
beneficial ownership opacity, crypto off-ramping, Ponzi collections, cybercrime
proceeds, narcotics cash, illegal gold, property integration and high-value goods
purchases.

## Project Structure

```text
.
|-- notebooks/
|   |-- african_data_simulation.ipynb
|-- scripts/
|   |-- main.py
|   |-- save_figures.py
|-- src/
|   |-- sa_aml_simulation/
|       |-- config.py
|       |-- figure_helpers.py
|       |-- figure_pipeline.py
|       |-- logging_utils.py
|       |-- paths.py
|       |-- simulation_pipeline.py
|-- results/
|   |-- images/
|   |   |-- 00_framework/
|   |   |-- 01_eda_overview/
|   |   |-- 02_temporal_sequence/
|   |   |-- 03_payments_amounts/
|   |   |-- 04_aml_typologies/
|   |   |-- 05_normal_behaviour/
|   |   |-- 06_customer_profiles/
|   |   |-- 07_geography/
|   |   |-- 08_informality/
|   |   |-- 09_context_diagnostics/
|   |-- simulated_data/
|   |   |-- 01_full_saml/
|   |   |-- 02_behaviour_only/
|   |   |-- 03_final_results_excel/
|   |   |-- 04_optional_engineered/
|   |-- tables/
|       |-- csv/
|       |-- excel/
|-- logs/
|   |-- main_simulation/
|   |-- figures/
```

Large generated datasets under `results/simulated_data/` are ignored by Git
because full simulation CSV files can exceed GitHub's normal file-size limit.
They can be regenerated from the scripts or notebook.

## Simulation Flow

The core workflow follows the same logic whether it is run from the notebook or
from `scripts/main.py`.

### 1. Configuration And Setup

The run starts by reading shared configuration values from
`src/sa_aml_simulation/config.py`. These defaults define the output prefix, the
number of receiver accounts, the target suspicious-account share, the simulation
date range, the random seed, the maximum transaction cap per account and the
suspicious transaction probability scale.

The path setup in `src/sa_aml_simulation/paths.py` creates the required folder
layout for simulated data, figures, tables and logs. The script entry points
also create timestamped run logs so long simulations can be monitored and
audited after completion.

### 2. Account Population Generation

The simulator first creates monitored receiver accounts. Each account receives a
synthetic account identifier and contextual profile attributes such as customer
type, province, employment status, employment sector, income band, estimated
monthly income, age band, gender, education level, nationality, residency status
and migrant-worker context.

The account generator also assigns South African socioeconomic context,
including cash-intensity band, informal-business flag, business-registration
status, trading permit, VAT-registration status, business premises type,
personal-account business use, estimated daily cash turnover, access to bank
loans and whether business records are kept.

### 3. Risk Context Assignment

Before transactions are created, the simulator assigns contextual risk
attributes that influence laundering probability and scenario selection. These
include KYC strength, public-sector exposure, procurement-vendor status, PEP or
high-risk person context, high-risk sector category, beneficial-ownership
opacity, intermediary account flag, virtual-asset exposure, connected-party
count, adverse media flag, sanctions screening context and cross-border profile
links.

These fields are not deterministic laundering labels. They are context variables
that make some behaviours more likely while still allowing legitimate and
suspicious accounts to overlap.

### 4. Account Life Cycle And Transaction Scheduling

Each account receives an opening date, first observed transaction date, active
observation window, account age and activity multiplier. This prevents the data
from assuming that every account is active for the full period or has the same
number of transactions.

Transaction counts are generated with behaviour-conditioned overdispersion, so
some customers have very few transactions and others have many more. Suspicious
accounts are not automatically high-volume accounts, and legitimate accounts are
not automatically low-volume accounts.

### 5. Legitimate Behaviour Engine

For normal transactions, the simulator selects a legitimate behaviour category
and generates payment timing, transaction amounts, direction and payment rails
according to that behaviour. This includes salary timing, salary-week household
spending, grant-linked inflows and post-grant spending, low-value day-to-day
payments, remittances, informal cash cycles, taxi cash activity, stokvel
contributions and ordinary business operations.

This layer is important because real South African retail behaviour can be
irregular, cash-heavy or cross-border without being suspicious.

### 6. Suspicious Typology Engine

For accounts selected as laundering accounts, the simulator assigns one of the
AML scenarios. Suspicious transactions are then overlaid within the account's
transaction history. A suspicious account can still contain normal transactions,
which makes the data less obvious and more realistic.

Scenario assignment is context-weighted. For example, cash-intensive profiles
are more likely to produce cash-placement typologies, cross-border profiles are
more likely to produce layering typologies, procurement exposure increases
tender-related scenarios, and virtual-asset exposure increases crypto off-ramp
scenarios.

### 7. Payment And Jurisdiction Routing

Each transaction receives a payment type, amount, direction, counterparty type,
currency and location context. The payment rails include domestic electronic
transfers, RTC, PayShap, debit card, credit card, cash deposits, cash
withdrawals, international wires, social grants, procurement EFT and crypto
exchange off-ramp transactions.

Jurisdiction routing distinguishes domestic South African activity, SADC
corridors, FATF increased-monitoring jurisdictions, FATF call-for-action
jurisdictions and other foreign counterparties. Jurisdiction exposure is treated
as context, not automatic suspiciousness.

### 8. Raw Output Assembly

The final raw simulation dataset is exported to
`results/simulated_data/01_full_saml/`. A smaller behaviour-only version is
exported to `results/simulated_data/02_behaviour_only/`.

The raw public output intentionally excludes downstream modelling-engineered
features such as cyclical encodings, rolling statistics, account aggregates,
sequence-density fields and composite risk scores. Those fields may be rebuilt
inside the figure pipeline for validation, but they are not part of the raw
simulation CSV intended for reuse.

### 9. Figures And Insight Tables

The figure pipeline in `src/sa_aml_simulation/figure_pipeline.py` reads a saved
simulation CSV and regenerates validation figures and insight tables. It does
not rerun the simulation.

Figures are saved under `results/images/` by section:

- `00_framework`: framework and architecture visuals
- `01_eda_overview`: dataset size, class balance and basic EDA
- `02_temporal_sequence`: time coverage, sequence length and transaction timing
- `03_payments_amounts`: amount and payment-rail diagnostics
- `04_aml_typologies`: AML scenario metrics and relationship examples
- `05_normal_behaviour`: legitimate socioeconomic behaviour and grants
- `06_customer_profiles`: income, employment, customer type and demographics
- `07_geography`: province, nationality and cross-border exposure
- `08_informality`: informal and unregistered business behaviour
- `09_context_diagnostics`: context-correlation diagnostics

Insight tables are saved under `results/tables/csv/` and
`results/tables/excel/`.

### 10. Logging

Every script run creates a timestamped log:

- simulation logs: `logs/main_simulation/`
- figure-generation logs: `logs/figures/`

Each log records the run type, run name, start time, command arguments, all
printed progress steps, finish time, total runtime and final status.

## Running The Simulation

Run the full configured simulation:

```powershell
python scripts/main.py
```

Run a smaller smoke test:

```powershell
python scripts/main.py --n-accounts 500 --output-prefix smoke_test
```

Run without exporting files:

```powershell
python scripts/main.py --n-accounts 500 --output-prefix smoke_test --no-files
```

Skip the Excel workbook export:

```powershell
python scripts/main.py --no-excel
```

Export optional engineered diagnostics separately:

```powershell
python scripts/main.py --export-engineered
```

## Regenerating Figures From Existing Data

Regenerate figures and insight tables from the default simulation CSV:

```powershell
python scripts/save_figures.py
```

Use a specific simulation CSV:

```powershell
python scripts/save_figures.py --input-csv results/simulated_data/01_full_saml/SA_SAML_D.csv
```

Regenerate tables only:

```powershell
python scripts/save_figures.py --no-plots
```

Regenerate figures only:

```powershell
python scripts/save_figures.py --no-tables
```

## Useful Environment Variables

The simulator can be controlled without editing source code:

```powershell
$env:AML_SIM_N_ACCOUNTS="35000"
$env:AML_SIM_MAX_TX_PER_ACCOUNT="220"
$env:AML_SIM_ACCOUNT_SHARE="0.10"
$env:AML_SIM_START_DATE="2024-01-01"
$env:AML_SIM_END_DATE="2026-03-31"
$env:AML_SIM_SEED="42"
$env:AML_SIM_EXPORT_FILES="1"
$env:AML_SIM_EXPORT_EXCEL="1"
$env:AML_SIM_EXPORT_INSIGHTS="1"
$env:AML_SIM_EXPORT_TABLES="1"
$env:AML_SIM_DATA_DIR="results/simulated_data"
$env:AML_SIM_PLOTS_DIR="results/images"
```

Set `AML_SIM_EXPORT_INSIGHTS=0` when you want to run the simulation without
regenerating plots. Set `AML_SIM_EXPORT_TABLES=0` when you want to skip insight
table exports.

## Notes On Reproducibility

The default random seed is fixed, so the same configuration should reproduce the
same synthetic dataset. Changing the number of accounts, observation window,
transaction cap or suspicious-account target will produce a different run.

The generated data are synthetic. They should be used for research,
benchmarking, validation and method development, not as empirical estimates of
real South African laundering prevalence.
