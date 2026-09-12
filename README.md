# South African AML Data Simulation

This repository contains a context-aware synthetic anti-money laundering (AML)
data simulation framework for South African retail banking behaviour. It is
designed to generate reproducible synthetic transaction data that combines
ordinary South African socioeconomic activity with typology-grounded suspicious
AML behaviour.

The project does not provide a trained detection model. Its purpose is to create
a transparent simulation base that can be reused for AML research, validation,
scenario analysis, sequential modelling, explainability, calibration and
benchmarking in a South African context.

## Simulation Flow

<p align="center">
  <img src="results/images/00_framework/simulation_architecture.png" alt="South African retail AML simulation flow" width="100%">
</p>

The flow shows how the evidence base informs the synthetic account population,
customer and context profiles, normal socioeconomic activity, suspicious AML
typology overlays, transaction scheduling, payment and jurisdiction routing,
and the final synthetic outputs and validation evidence.

## Executive Summary

Open AML research is constrained because real banking data are confidential,
suspicious labels are rare, and operational alert labels do not provide complete
ground truth. This limitation is more serious in South Africa because local
retail behaviour differs from many international AML benchmark assumptions.
Normal customers may rely on cash, receive social grants, operate informal or
unregistered businesses, send regional remittances, use personal accounts for
small business turnover, receive irregular income, or transact around salary and
grant payment days. These patterns can look unusual if they are interpreted
without local context.

The simulator addresses this gap by creating a synthetic South African retail
banking environment where both normal and suspicious behaviour are generated
from documented context. The normal class includes salary and household
payments, grant-linked flows, township and spaza retail activity, taxi and
transport cash cycles, informal business turnover, remittances, stokvels,
business operating payments and low-value day-to-day spending. The suspicious
class overlays 22 AML typologies including structuring, mule accounts,
cross-border layering, procurement corruption, shell or front companies,
beneficial ownership opacity, crypto off-ramping, Ponzi collections, cybercrime,
narcotics cash, illegal gold, property integration and high-value goods
purchases.

The framework is intended to support South African and African financial-crime
research by reducing dependence on foreign behavioural assumptions and by
providing a reusable, inspectable dataset for work that cannot access real bank
records. Public inquiries and official reports on corruption, procurement abuse,
cross-border risk, financial intelligence reporting and organised crime provide
motivation, but the generated data remain fully synthetic and should not be read
as real prevalence estimates or as claims about any person, institution,
province or country.

## Current Generated Run

The latest saved run in the local results directory has the following summary:

| Metric | Value |
| --- | ---: |
| Observation period | 2024-01-01 to 2026-03-31 |
| Observation days | 821 |
| Receiver accounts | 35,000 |
| Final simulation columns | 58 |
| Full audit columns | 107 |
| Transaction rows | 1,412,358 |
| Average transactions per account | 40.35 |
| Median transactions per account | 27 |
| 95th percentile transactions per account | 126 |
| Maximum transactions per account | 220 |
| Legitimate transaction rows | 1,381,844 |
| Suspicious AML transaction rows | 30,514 |
| Legitimate accounts | 31,000 |
| Suspicious AML accounts | 4,000 |
| Transaction-level suspicious share | 2.16% |
| Account-level suspicious share | 11.43% |
| AML scenarios represented | 22 |
| Normal behaviour categories represented | 11 |
| Cross-border transaction share | 14.78% |
| FATF contextual exposure share | 5.91% |

Large generated datasets under `results/simulated_data/` are ignored by Git
because full simulation files can exceed GitHub's normal file-size limit. They
can be regenerated from the notebook or scripts.

## Selected Validation Views

The repository contains a larger set of individually saved diagnostics. The
plots below are selected views that explain the main construction choices and
can be inspected directly from the GitHub landing page.

<details open>
<summary><strong>Overview and account population</strong></summary>

<p align="center">
  <img src="results/images/01_eda_overview/01a_transaction_label_imbalance.png" alt="Transaction-level label balance" width="48%">
  <img src="results/images/01_eda_overview/01b_receiver_account_label_imbalance.png" alt="Account-level label balance" width="48%">
</p>

The transaction-level and account-level views are shown together because their
shares are intentionally different. Suspicious accounts retain ordinary
transactions, so the account label is not copied to every row in the account
history.
</details>

<details open>
<summary><strong>Account histories and transaction timing</strong></summary>

<p align="center">
  <img src="results/images/02_temporal_sequence/02_receiver_account_sequence_lengths.png" alt="Receiver-account sequence lengths" width="48%">
  <img src="results/images/02_temporal_sequence/03_active_days_vs_account_transaction_count.png" alt="Active days versus account transaction count" width="48%">
</p>

Account histories vary in both length and active observation period. This is a
central design feature: transaction count alone is not intended to identify a
suspicious account, while timing and sequence structure remain available for
later analysis.
</details>

<details open>
<summary><strong>Amounts, payment rails and inter-transaction timing</strong></summary>

<p align="center">
  <img src="results/images/03_payments_amounts/06a_amount_distribution_by_label.png" alt="Transaction amount distribution by label" width="48%">
  <img src="results/images/03_payments_amounts/06b_payment_rail_mix_by_label.png" alt="Payment rail mix by label" width="48%">
  <img src="results/images/02_temporal_sequence/18a_time_since_previous_transaction_boxplot.png" alt="Time since previous transaction" width="48%">
</p>

The amount and timing views show why the simulator uses overlapping classes.
Cash, electronic transfers and other operational rails occur in both normal and
suspicious activity. Suspicious typologies receive distinctive amount and
timing patterns, but no single payment rail is treated as a laundering label.
</details>

<details open>
<summary><strong>Normal behaviour and social grants</strong></summary>

<p align="center">
  <img src="results/images/05_normal_behaviour/10_normal_behaviour_fingerprint_heatmap.png" alt="Normal behaviour fingerprint heatmap" width="48%">
  <img src="results/images/05_normal_behaviour/30_social_grant_payment_day_pattern.png" alt="Social grant payment day pattern" width="48%">
</p>

Normal activity includes salary-linked spending, grants, informal trade,
remittances, stokvels, household payments and irregular income. Grant-linked
accounts can also contain non-grant transactions and can receive suspicious
overlays, so grant status is contextual rather than determinative.
</details>

<details open>
<summary><strong>AML typologies and relationship examples</strong></summary>

<p align="center">
  <img src="results/images/04_aml_typologies/08_aml_scenario_fingerprint_heatmap.png" alt="AML scenario fingerprint heatmap" width="48%">
  <img src="results/images/04_aml_typologies/31_network_ponzi_scheme_collections.png" alt="Ponzi or investment scheme relationship example" width="48%">
  <img src="results/images/04_aml_typologies/32_network_mule_pass_through.png" alt="Mule account pass-through relationship example" width="48%">
  <img src="results/images/04_aml_typologies/33_network_cross_border_layering.png" alt="Cross-border layering relationship example" width="48%">
</p>

The fingerprint heatmap compares the observable transaction patterns assigned
to the 22 AML scenarios. The three relationship diagnostics then illustrate
how selected scenarios can be inspected through repeated synthetic account
endpoints, direction, payment rail and cross-border context.
</details>

<details open>
<summary><strong>Customer and socioeconomic context</strong></summary>

<p align="center">
  <img src="results/images/06_customer_profiles/27_income_employment_aml_rate_heatmap.png" alt="Income and employment AML-rate heatmap" width="48%">
  <img src="results/images/06_customer_profiles/24_employment_sector_transaction_mix_by_label.png" alt="Employment-sector transaction mix by label" width="48%">
</p>

Customer type, employment, income, sector and demographic fields provide the
context needed to distinguish ordinary socioeconomic variation from suspicious
patterns. The plotted rates are simulation diagnostics, not real-world crime
prevalence estimates.
</details>

<details open>
<summary><strong>Geography and cross-border coverage</strong></summary>

<p align="center">
  <img src="results/images/07_geography/16b_sa_province_aml_exposure_map.png" alt="South African province exposure map" width="48%">
  <img src="results/images/07_geography/16c_cross_border_country_risk_map.png" alt="Cross-border country risk map" width="48%">
</p>

The geography views validate provincial coverage, SADC corridors and FATF
contextual tiers. They describe how the synthetic population is routed; they
must not be interpreted as maps of actual crime prevalence or jurisdictional
guilt.
</details>

<details open>
<summary><strong>Informality and contextual inputs</strong></summary>

<p align="center">
  <img src="results/images/08_informality/17b_business_registration_cash_share.png" alt="Cash share by business registration status" width="48%">
  <img src="results/images/09_context_diagnostics/20_context_feature_correlation_heatmap.png" alt="Context feature correlation heatmap" width="48%">
</p>

Informal and unregistered business activity is represented as a legitimate
possibility, including cash-heavy turnover. Context fields are generated as
probabilistic inputs and may occur in both legitimate and suspicious accounts.
</details>

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
|       |-- schema.py
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
|   |   |-- SA_SAML_D.csv
|   |   |-- SA_SAML_D_sample.csv
|   |   |-- SA_SAML_D_data_dictionary.csv
|   |   |-- SA_SAML_D_schema.json
|   |-- insights/
|       |-- *.csv
|       |-- feature_audit/
|-- logs/
|   |-- main_simulation/
|   |-- figures/
```

## Simulation Flow

The same simulation logic is available through the notebook and through
`scripts/main.py`.

### 1. Configuration And Setup

The run starts by reading shared defaults from
`src/sa_aml_simulation/config.py`. These define the output prefix, number of
receiver accounts, target suspicious-account share, observation window, random
seed, maximum transaction cap per account and suspicious transaction probability
scale.

`src/sa_aml_simulation/paths.py` creates the folder layout for data, figures,
insights and logs. Script runs also write timestamped logs so the user can see
which phase is running and how long each run takes.

### 2. Account Population Generation

The simulator creates monitored receiver accounts. Each account receives a
synthetic identity and profile variables including customer type, province,
employment status, employment sector, income band, estimated monthly income,
age, age band, gender, education level, nationality, residency status,
migrant-worker flag, settlement context, financial inclusion segment and
channel preference.

These variables are synthetic. They describe the simulated population and help
test whether AML behaviour can be analysed without treating socioeconomic
attributes as automatic suspiciousness.

### 3. Business, Informality And Cash Context

The account generator uses a compact business-context representation in the
clean export: `Informal_business_flag` and
`Business_registration_status`. Detailed registration, premises, permit,
turnover, record-keeping, loan-access and business-age fields remain internal
simulation context or audit metadata rather than being repeated in the public
analysis file.

This stage is important because informal and unregistered business activity is a
normal part of the South African retail environment. The simulator therefore
allows cash-heavy behaviour to exist in the legitimate class.

### 4. Risk Context Assignment

Before transactions are generated, accounts receive contextual risk variables:
KYC strength, public-sector exposure, procurement-vendor status, PEP category,
high-risk person context, high-risk sector category, beneficial ownership
opacity, intermediary account flag, virtual-asset exposure, connected-party
count, targeted financial sanctions screening context, adverse media flag,
profile-level foreign counterparty and jurisdiction risk tier.

These fields influence probability and scenario weighting. They are not
deterministic labels. For example, an account can be public-sector exposed,
grant-linked, cash-heavy or cross-border active and still be legitimate.

### 5. Account Life Cycle And Scheduling

Each account receives an account-open date, first observed transaction date,
active observation days, account age, activity multiplier, expected transaction
count and generated transaction count. Transaction counts vary by customer and
behaviour, with a current maximum cap of 220 transactions per account.

This prevents the dataset from becoming too obvious. Suspicious accounts are
not automatically high-volume accounts, and legitimate accounts are not
automatically low-volume accounts.

### 6. Legitimate Behaviour Engine

Normal transactions are generated from ordinary behavioural routines. The
simulator includes salary-day and grant-day timing, early-month and salary-week
household spending, debit-order-like payments, cash withdrawals, airtime/data,
transport, grocery-style purchases, remittances, informal business turnover and
business operating payments. These are represented through transaction timing,
amount, direction, counterparty type, payment rail and transaction purpose
category rather than through free-text transaction descriptions.

Free-text descriptions are intentionally excluded from the modelling-ready raw
data because references can be noisy or deliberately misleading. Behaviour is
therefore represented by structured fields and temporal/payment patterns.

### 7. Suspicious Typology Engine

Accounts selected as laundering accounts receive one AML typology. The typology
does not convert every transaction on that account into a suspicious
transaction. Suspicious events are overlaid inside otherwise ordinary account
activity, so a laundering account can still have salary, grocery, grant,
business, remittance or normal retail transactions.

Scenario assignment is context-weighted:

- cash-intensive profiles are more likely to produce cash placement scenarios;
- rapid digital profiles are more likely to produce mule, cybercrime or invoice
  scam behaviour;
- public-sector and procurement exposure increases tender-related behaviour;
- ownership opacity increases shell, front company and legal-structure
  scenarios;
- cross-border links increase regional and high-risk jurisdiction layering;
- virtual-asset exposure increases crypto off-ramp scenarios.

### 8. Payment And Jurisdiction Routing

Each transaction receives structured payment and jurisdiction fields:
transaction timestamp, sender account, receiver account, counterparty account,
inflow/outflow direction, amount, currency, payment type, counterparty type,
transaction purpose category, location context, cross-border flag, SADC corridor
flag, jurisdiction risk tier and contextual indicators. Grant source, credit-card
use, procurement context and virtual-asset context are represented as separate
indicator fields rather than being hidden inside the payment rail.

Payment rails include electronic transfer, RTC, PayShap, card payment, cash
deposit, cash withdrawal and international wire. Social grants use the
ordinary electronic-transfer rail and are identified by `Social_grant_indicator`
and `Social_grant_type`. Credit-card use is identified by
`Credit_card_indicator`, while procurement and virtual-asset activity use
`Procurement_payment_indicator` and `Virtual_asset_transaction_indicator`.
The earlier "card acquiring" label has been removed.

### 9. Raw Output Assembly

The simulator exports one complete synthetic transaction file, a public sample,
a data dictionary and separate CSV insight tables. The main public raw output
excludes downstream modelling-engineered
features such as cyclical encodings, rolling sequence statistics, account
aggregates, sequence-density fields and composite risk scores. Researchers can
create their own engineered features from the raw simulation output.

The public transaction record retains transaction identity, time, amount, payment
and routing fields, neutral counterparty roles, KYC and socioeconomic context,
geography, separate grant, credit, procurement and virtual-asset indicators, and
`Laundering_type` for scenario interpretation.
`Laundering_type`, `Scenario_detail`, `Scenario_reference_key` and
`Is_laundering` are label-adjacent fields and must be excluded from model
inputs; they are retained so users can audit and describe the simulated
scenarios. The export excludes baseline behaviour labels,
purpose metadata, internal activity multipliers, post-generation transaction
counts, reproducible diagnostic flags and scores, first-observed dates, and
duplicate fields such as age bands beside age, risk flags beside their
categories, and debit-card flags beside the card rail and credit-card
indicator. The public record contains the simulation observations and the
target/scenario interpretation fields, but no downstream sequence encodings,
rolling statistics, account aggregates or composite risk scores. Researchers
can derive those features from the retained raw fields for their own study
design. The separate sample CSV contains the same columns and is provided for
quick inspection only.

### Final 58-Column Data Dictionary

The final public file has exactly **58 columns**. The target and
scenario-adjacent fields are retained for transparent evaluation and
interpretation, but they must be excluded from predictive inputs when the task
is to predict `Is_laundering`. The tables below are the authoritative field
dictionary for the final export.

#### Labels And Scenario Interpretation

| Column | Meaning | Values or construction |
| --- | --- | --- |
| `Is_laundering` | Transaction-level ground-truth label generated by the simulator. | `0` legitimate baseline or `1` suspicious AML overlay. |
| `Laundering_type` | Scenario family assigned to a suspicious account or transaction. | Scenario name for AML rows; non-AML rows are not assigned an AML type. |
| `Scenario_reference_key` | Source and catalogue reference identifier for the simulated behaviour. | FIC/FATF/context source keys for AML rows; normal-behaviour source key for legitimate rows. |
| `Scenario_detail` | Short plain-language description of the observable pattern. | Examples include repeated near-threshold cash deposits, rapid onward movement, or ordinary household spending. |

#### Transaction Identity, Timing And Amount

| Column | Meaning | Values or construction |
| --- | --- | --- |
| `Transaction_id` | Unique synthetic transaction identifier. | Sequential simulation identifier; not a bank-account number. |
| `Time` | Time of day at which the transaction occurs. | `HH:MM:SS`; includes midnight salary/grant timing, working-day spending, and rapid AML episodes. |
| `Date` | Calendar date of the transaction. | Date within the configured observation window. |
| `Account_id` | Monitored receiver-account key used to group each account history. | Unique synthetic 9--11 digit account number. |
| `Sender_account` | Account from which the transaction is sent. | Unique synthetic 9--11 digit account number; may be the monitored account or an external counterparty. |
| `Receiver_account` | Account receiving the transaction. | Unique synthetic 9--11 digit account number; may be the monitored account or an external counterparty. |
| `Counterparty_account` | Counterparty account associated with the monitored transaction. | Reused synthetic account numbers allow repeated counterparties and network patterns without real identifiers. |
| `Direction` | Movement direction relative to the monitored account. | `Inflow` or `Outflow`. |
| `Amount` | Transaction value in the relevant payment currency. | Positive simulated amount generated from profile, behaviour, timing, rail, and scenario distributions. |

#### Payment Rails, Indicators And Jurisdiction

| Column | Meaning | Values or construction |
| --- | --- | --- |
| `Payment_currency` | Currency associated with the sending side. | Country-linked currencies such as `ZAR`, `USD`, `BWP`, `LSL`, and `MZN`. |
| `Received_currency` | Currency associated with the receiving side. | Country-linked currency; normally `ZAR` for domestic retail activity. |
| `Sender_bank_location` | Synthetic location of the sending bank or account. | South Africa, SADC, other international, or FATF-context countries. |
| `Receiver_bank_location` | Synthetic location of the receiving bank or account. | Same jurisdiction groups as the sender location. |
| `Counterparty_country` | Country associated with the counterparty. | South Africa, SADC corridors, other relevant countries, or FATF-risk tiers. |
| `Payment_type` | Operational payment rail only; it is not a scenario label. | `Cash Deposit`, `Cash Withdrawal`, `Electronic Transfer`, `RTC`, `PayShap`, `Card Payment`, or `International Wire`. |
| `Social_grant_indicator` | Indicates that the transaction is linked to a simulated social-grant flow. | `0` or `1`; grant activity uses ordinary payment rails. |
| `Social_grant_type` | Grant programme category linked to the transaction when applicable. | `old_age`, `disability`, `care_dependency`, `foster_child`, `child_support`, `grant_in_aid`, `srd`, or `Not applicable`. |
| `Credit_card_indicator` | Separates credit-card context from the operational card rail. | `0` or `1`; credit-card context can occur in either class. |
| `Procurement_payment_indicator` | Indicates a simulated procurement-related payment context. | `0` or `1`; does not by itself imply laundering. |
| `Virtual_asset_transaction_indicator` | Indicates a transaction associated with virtual-asset activity. | `0` or `1`; interpreted with payment and counterparty context. |
| `Counterparty_type` | Neutral functional role of the counterparty. | Household/peer, retail customer, business customer, business supplier, employer/payroll, public sector/grant payer, foreign counterparty, financial service provider, savings group, or trade/logistics counterparty. Typology-coded names are not exposed here. |
| `Cross_border_transaction` | Indicates whether the sending and receiving jurisdictions differ. | `0` domestic or `1` cross-border. |
| `SADC_corridor_transaction` | Indicates use of a SADC-related corridor. | `0` or `1`; corridor membership is separate from suspiciousness. |
| `Jurisdiction_risk_tier` | Contextual tier derived from simulated countries and FATF/SADC lists. | `Domestic`, `SADC corridor`, `Other international`, `FATF Increased Monitoring`, or `FATF Call for Action`. |

#### Customer And Socioeconomic Profile

| Column | Meaning | Values or construction |
| --- | --- | --- |
| `Customer_type` | Broad simulated retail or organisational customer segment. | `Formal Employee`, `Informal Trader`, `Grant Recipient`, `Cash-Intensive SME`, `Migrant Worker / Remittance User`, `Student / Youth Account`, `Public Sector Employee`, `Business Owner`, `Procurement Vendor`, `Corporate Entity`, or `NPO / Community Organisation`. |
| `Employment_status` | Employment or livelihood status used to condition normal activity. | `Employed`, `Informal / Self-employed`, `Unemployed`, `Pensioner`, `Student`, `Business Owner`, `Business Entity`, or `NPO / Community`. |
| `Employment_sector` | Sector associated with the customer profile. | Government, education, healthcare, retail, mining, construction, financial services, transport, informal retail, agriculture, legal/trust services, property, crypto/VASP, or import/export. |
| `Monthly_income_estimate` | Synthetic monthly income or business-turnover estimate. | Positive ZAR estimate drawn from the profile income band; not an observed bank value. |
| `Province` | South African province assigned to the monitored account. | Gauteng, KwaZulu-Natal, Eastern Cape, Western Cape, Limpopo, Mpumalanga, North West, Free State, or Northern Cape. |
| `Grant_linked` | Account-level context indicating a grant-linked customer profile. | `0` or `1`; it does not imply a suspicious account. |
| `Gender` | Synthetic demographic attribute. | `Female`, `Male`, or `Other / undisclosed`. |
| `Age` | Synthetic age in years. | Integer age generated from customer and employment context. |
| `Education_level` | Synthetic highest or current education category. | `Less than matric`, `Matric`, `Tertiary in progress`, `Other tertiary`, or `Graduate`. |
| `Nationality_country` | Synthetic nationality or country association. | Predominantly South Africa, with regional and international countries represented. |
| `Urban_rural` | Settlement context for the account profile. | `Metro / urban`, `Township / peri-urban`, `Small town`, or `Rural`. |
| `Township_or_informal_settlement_flag` | Indicates township or informal-settlement context. | `0` or `1`, derived from settlement and provincial context. |
| `Financial_inclusion_segment` | Synthetic access and usage segment. | `Banked digital`, `Banked cash-reliant`, `Thin-file / underbanked`, or `Grant-card dependent`. |
| `Channel_preference` | Typical access channel for the customer profile. | Card-first, mobile-app-first, mixed cash/digital, branch/ATM cash, agent/cash/mobile, or remittance-app usage. |

#### Business, Public-Sector And Risk Context

| Column | Meaning | Values or construction |
| --- | --- | --- |
| `Informal_business_flag` | Indicates participation in an informal-business context. | `0` or `1`; legitimate informal trade is explicitly represented. |
| `Business_registration_status` | Compact business-registration category. | `Not business`, `Registered`, `Unregistered informal`, or `Unregistered small business`. |
| `Public_sector_exposure` | Contextual relationship to public-sector activity. | `0` or `1`; may describe legitimate employment or vendor exposure. |
| `Procurement_vendor` | Indicates a simulated procurement-vendor relationship. | `0` or `1`; not a direct laundering label. |
| `PEP_category` | Politically exposed-person context where applicable. | `Not PEP`, `Domestic prominent influential person`, `Foreign PEP`, or `PEP family or associate`. |
| `Intermediary_account_flag` | Indicates intermediary or connected-account context. | `0` or `1`; contextual and not determinative. |
| `Virtual_asset_exposure` | Account-level exposure to virtual-asset services. | `0` or `1`. |
| `Connected_party_count` | Number of simulated connected-party relationships. | Non-negative integer generated from account context. |
| `Profile_jurisdiction_risk_tier` | Risk tier of the account's typical foreign-counterparty profile. | Domestic, SADC corridor, other international, FATF increased monitoring, or FATF call for action. |
| `Public_sector_counterparty` | Indicates a public-sector or grant-payer counterparty. | `0` or `1`; salary, grants, municipal services, and procurement can all appear here. |
| `High_risk_sector_category` | Broad sector context used for risk-aware analysis. | Standard retail/household, cash-intensive informal economy, public procurement adjacent, property/real estate, motor vehicle/high-value goods, legal/professional intermediary, casino/gambling, precious metals/stones, import/export trade, crypto asset services, or NPO/community organisation. |
| `High_risk_person_type` | Neutral profile-level person or exposure category. | Standard retail customer, public-sector financial-flow exposure, procurement/vendor exposure, domestic or foreign prominent public official, PEP family/associate, targeted-sanctions possible match, or adverse-media-only context. |
| `Targeted_financial_sanctions_match` | Synthetic screening-context indicator. | `0` or `1`; it is not a transaction-level finding. |
| `Adverse_media_flag` | Synthetic adverse-media screening context. | `0` or `1`; not proof of laundering. |
| `Typical_foreign_counterparty_country` | Typical foreign corridor associated with the profile. | South Africa for domestic profiles, or selected SADC, international, and FATF-context countries. |
| `Account_open_date` | Synthetic date on which the account was opened. | Date before or within the observation window, generated from lifecycle rules. |

### 10. Figures, Tables And Diagnostics

`scripts/save_figures.py` reads an existing simulation CSV and regenerates
figures and insight tables without rerunning the simulation. Figures are saved
individually by section so they can be used cleanly in presentations or LaTeX
documents.

Image folders:

- `00_framework`: framework and architecture visuals
- `01_eda_overview`: dataset size, class balance and basic EDA
- `02_temporal_sequence`: time coverage, sequence length and transaction timing
- `03_payments_amounts`: amount and payment-rail diagnostics
- `04_aml_typologies`: AML scenario metrics and relationship examples
- `05_normal_behaviour`: legitimate socioeconomic behaviour and grant analysis
- `06_customer_profiles`: income, employment, customer type and demographics
- `07_geography`: province, nationality and cross-border exposure
- `08_informality`: informal and unregistered business behaviour
- `09_context_diagnostics`: context-correlation diagnostics

Every insight table is saved as its own CSV file directly under
`results/insights/`. The files correspond to the summary views previously
presented together, including dataset overview, validation summary,
label comparison, transaction statistics, account sequence statistics, AML
scenario metrics, normal behaviour metrics, payment-rail metrics, province
metrics, country metrics, sequence quantiles, and public-column inventories.

`results/simulated_data/SA_SAML_D_sample.csv` is a small, GitHub-readable sample
of the public transaction record. It contains the same 58 columns as the full
CSV and is intended for quick inspection without downloading the large file.
`results/simulated_data/SA_SAML_D_data_dictionary.csv` documents every public
column's group, meaning, derivation rule, data type, role, and observed
categorical values. `SA_SAML_D_schema.json` provides the machine-readable schema.

The final public schema contains exactly 58 columns, including the target and
scenario-adjacent interpretation fields. The leakage audit examines the full
internal audit export, while `final_schema_inventory.csv` documents all 58
public columns. The schema dictionary records 45 categorical fields and 13
numeric, temporal, or identifier fields; the target is separately marked and
the internal audit retains the broader diagnostic view.

## AML Scenario Catalogue

The simulator represents all 22 AML typologies below. These scenarios are
synthetic design categories grounded in AML typology, regulatory and contextual
sources.

| Code | AML scenario | Stage | What the scenario represents |
| --- | --- | --- | --- |
| SA_AML_001 | Structuring / Smurfing Cash Deposits | Placement | Repeated cash deposits near threshold-sensitive ranges. |
| SA_AML_002 | Cash Courier / Bulk Cash Movement | Placement | Large or repeated cash movements inconsistent with profile. |
| SA_AML_003 | Mule Account / Rapid Pass-Through | Layering | Many-to-one inflows followed by rapid withdrawals or onward transfers. |
| SA_AML_004 | Cross-Border Layering via SADC Corridor | Layering | Regional transfers through SADC corridors that may resemble remittance misuse. |
| SA_AML_005 | High-Risk Jurisdiction Layering | Layering | Cross-border flows involving FATF call-for-action or increased-monitoring tiers. |
| SA_AML_006 | Intermediary / Trust Account Abuse | Layering | Pass-through behaviour using intermediary or trust-style account patterns. |
| SA_AML_007 | Procurement Corruption / Tender Kickback | Integration | Public-sector or connected-vendor flows linked to tender abuse typologies. |
| SA_AML_008 | Shell or Front Company Flow | Layering | High-value flows through opaque or front-entity style accounts. |
| SA_AML_009 | Complex Legal Structure / Beneficial Ownership Obfuscation | Layering | Ownership opacity, unrelated counterparties and complex entity movement. |
| SA_AML_010 | Crypto Asset Off-Ramp | Layering / Integration | Virtual-asset off-ramp followed by bank transfer or cash-out behaviour. |
| SA_AML_011 | Ponzi / Investment Scheme Collections | Placement / Layering | Multiple unrelated investor-like deposits into a collector account. |
| SA_AML_012 | Fraudulent Company or Invoice Scam | Layering | Victim-company inflows followed by rapid onward payments. |
| SA_AML_013 | Cybercrime Proceeds | Layering | Fast digital transfers and rapid repeat movement after cybercrime-like inflows. |
| SA_AML_014 | Narcotics Cash Proceeds | Placement | High cash frequency and unexplained retail-style cash deposits. |
| SA_AML_015 | Illegal Gold or Precious Metals | Placement / Integration | Precious-metal linked payments, cash movement or cross-border flows. |
| SA_AML_016 | Environmental or Wildlife Crime Proceeds | Layering | Export, logistics or resource-crime style proceeds movement. |
| SA_AML_017 | Stolen Student Loan or Grant Abuse | Placement / Layering | Student funding, grant or benefit abuse followed by atypical cash-out/transfers. |
| SA_AML_018 | Casino or Gambling Laundering | Integration | Cash or electronic movement linked to gambling-style counterparties. |
| SA_AML_019 | Property / Real Estate Integration | Integration | High-value property deposits, rentals or settlement-style flows. |
| SA_AML_020 | Motor Vehicle or High-Value Goods Purchase | Integration | High-value goods purchases through cash or unusual third-party payments. |
| SA_AML_021 | Trade-Based Import-Export Layering | Layering | International trade payments inconsistent with stated customer profile. |
| SA_AML_022 | NPO / Community Organisation Misuse | Layering | Donation-like flows involving unrelated or foreign counterparties. |

## Normal Behaviour Catalogue

The normal class is intentionally diverse. It is not designed as a perfectly
clean opposite of suspicious behaviour.

| Normal behaviour category | Simulation meaning |
| --- | --- |
| Salary and household payments | Salary receipts, debit-order-like expenses, groceries, airtime/data, transport, cash-out and routine household payments. |
| Grant-linked inflows and survival payments | Social grant receipts and low-value spending around grant payment periods. |
| Informal registered business cash cycle | Registered small-business turnover with cash, transfers and some record/formalisation indicators. |
| Informal unregistered business cash cycle | Unregistered small-business turnover, often using personal accounts and cash. |
| Spaza / township retail cash cycle | Township retail, spaza, street-trading and supplier-payment behaviour. |
| Taxi or transport cash cycle | Cash-heavy transport revenue, fuel, repairs and owner/operator payments. |
| Low-value high-frequency retail payments | Everyday small-value payments through debit card, PayShap, RTC, EFT and cash channels. |
| Remittance corridor activity | Family or worker remittances through SADC corridors and money-transfer-like rails. |
| Stokvel / rotating savings contributions | Group savings contributions and periodic withdrawals. |
| Business operating payments | Supplier, customer, payroll and operating transactions. |
| Irregular income / gig work | Irregular wages, casual work, gig income and small living expenses. |

## Demographics, Geography And Socioeconomic Coverage

The simulator includes demographic and socioeconomic fields because the South
African AML problem is context-sensitive. These fields make the population
inspectable and support later fairness, subgroup and false-positive analysis.

Demographic coverage in the clean export:

- age
- gender
- education level
- nationality

Geographic coverage:

- all 9 South African provinces
- metro, township/peri-urban, small-town and rural settlement context
- SADC corridor countries
- FATF call-for-action and increased-monitoring jurisdiction tiers
- selected other foreign counterparties used for trade, family and VASP-like
  corridors

Socioeconomic and sector coverage:

- formal employees
- informal traders
- grant recipients
- cash-intensive SMEs
- migrant workers and remittance users
- students and youth accounts
- public-sector employees
- business owners
- procurement vendors
- corporate entities
- NPO and community organisations
- cash-intensive informal economy sectors
- public procurement adjacent sectors
- property and real estate
- motor vehicle and high-value goods
- import/export trade
- crypto asset services
- legal/professional intermediary contexts
- casino/gambling
- precious metals/stones

Social grant fields include account-level grant context and transaction-level
grant types. Non-grant transactions are marked as non-grant rather than left
ambiguous. Grant-linked accounts can still have non-grant transactions, and a
grant-linked account can still receive a suspicious typology overlay through
another payment rail.

## Design Choices To Reduce Obvious Leakage

The simulator includes several protections against making the AML label too
easy to infer:

- Suspicious accounts retain ordinary customer behaviour.
- Only a minority of transactions on a suspicious account are labelled as
  suspicious AML transactions.
- Transaction volume overlaps between legitimate and suspicious accounts.
- The maximum transaction count is capped at 220 per account in the current
  configuration.
- Payment type is an operational rail, not a scenario label. Grant, credit-card,
  procurement and virtual-asset context are separate indicators.
- `Counterparty_type` is retained as a broad economic role in the public
  export. Explicit scenario wording such as mule, victim, shell company and
  FATF-monitored entity is removed from this field; those details remain only in
  validation metadata and scenario documentation.
- Business detail fields are reduced to `Informal_business_flag` and
  `Business_registration_status`. Permit, VAT, premises, turnover, business-age,
  record-keeping, loan-access and personal-account-use fields are excluded from
  the clean export.
- `High_risk_sector_category` and `High_risk_person_type` are retained as
  contextual categories; their redundant binary flags are excluded.
- Beneficial-ownership opacity is kept in the full audit metadata but excluded
  from the clean export because it is a label-adjacent risk-control field rather
  than an ordinary transaction observation.
- Migration-worker and residency-status fields are excluded from the clean
  export because they are difficult to verify consistently from ordinary bank
  transaction records.
- Amounts are drawn from a shared account- and rail-conditioned distribution for
  both classes. Scenario templates influence the tail and relationships, rather
  than creating a unique amount band.
- Suspicious events occur in imperfect short episodes. Mule, pass-through and
  related scenarios can produce directional sequences, while ordinary accounts
  also retain rapid bursts, round amounts and occasional high-value payments.
- `Is_laundering` is the target and `Laundering_type` is scenario metadata for
  interpretation; neither is a permissible predictor.
- PEP, public-sector, cash, cross-border and round-amount context can occur on
  legitimate transactions. These fields should be interpreted with the
  surrounding sequence rather than as automatic evidence of laundering.
- Cross-border, cash-heavy, informal, grant-linked and high-risk-sector context
  are risk context, not automatic suspiciousness.
- Free-text transaction descriptions are excluded from the raw modelling-ready
  dataset.
- Downstream engineered modelling features are excluded from the main raw CSV.

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
python scripts/save_figures.py --input-csv results/insights/feature_audit/SA_SAML_D_full_audit.csv
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
$env:AML_SIM_SAMPLE_ROWS="5000"
$env:AML_SIM_DATA_DIR="results/simulated_data"
$env:AML_SIM_PLOTS_DIR="results/images"
```

Set `AML_SIM_EXPORT_INSIGHTS=0` when running the simulation without regenerating
plots. Set `AML_SIM_EXPORT_TABLES=0` when skipping insight table exports.

## Logging

Script runs create timestamped logs:

- simulation logs: `logs/main_simulation/`
- figure-generation logs: `logs/figures/`

Each log records the run type, run name, start time, command arguments, printed
progress steps, finish time, runtime and final status.

## Reproducibility Notes

The default random seed is fixed. Running the same configuration should
reproduce the same synthetic dataset. Changing the number of accounts,
observation window, transaction cap, output prefix or suspicious-account target
will produce a different synthetic run.

The generated data are synthetic. They should be used for research,
benchmarking, validation and method development, not as empirical estimates of
real South African laundering prevalence.

## Limitations

This simulator is not a bank system, regulator system or law-enforcement
system. It does not claim to measure real crime rates. Labels are generated by
design, so downstream models recover simulated typology structure rather than
confirmed real-world criminal behaviour. Scenario weights are source-informed
and behaviourally motivated, but they are not calibrated against confidential
bank data.

Geographic outputs are diagnostic simulation views. Province and country plots
show synthetic exposure and routing context, not real risk rankings or crime
statistics. FATF jurisdiction lists and public-risk context should be refreshed
before any final dated publication.

## Evidence Base And References

The simulation design draws on regulatory, typology, socioeconomic, payment,
synthetic-data and public-record sources. Key sources include:

- Financial Action Task Force and ESAAMLG, "South Africa's measures to combat
  money laundering and terrorist financing: Mutual Evaluation Report," 2021:
  https://www.fatf-gafi.org/en/publications/Mutualevaluations/Mer-south-africa-2021.html
- International Monetary Fund, "South Africa: Detailed Assessment Report on
  Anti-Money Laundering and Combating the Financing of Terrorism," 2021:
  https://www.imf.org/en/Publications/CR/Issues/2021/10/06/South-Africa-Detailed-Assessment-Report-on-Anti-Money-Laundering-and-Combating-the-482069
- Financial Action Task Force, "High-risk and other monitored jurisdictions,"
  February 2026:
  https://www.fatf-gafi.org/en/countries/black-and-grey-lists.html
- Financial Action Task Force, "Jurisdictions under Increased Monitoring,"
  October 2025:
  https://www.fatf-gafi.org/en/publications/High-risk-and-other-monitored-jurisdictions/increased-monitoring-october-2025.html
- Financial Intelligence Centre, "What are my reporting obligations?":
  https://www.fic.gov.za/faqs/what-are-my-reporting-obligations/
- Financial Intelligence Centre, "The Financial Intelligence Centre":
  https://www.fic.gov.za/about/
- Financial Intelligence Centre, "FIC Annual Report 2024-2025":
  https://www.fic.gov.za/document/fic-annual-report-2024-2025/
- Financial Intelligence Centre, "Case studies: Money laundering typologies and
  indicators":
  https://www.fic.gov.za/document/case-studies-money-laundering-typologies-and-indicators/
- Financial Intelligence Centre, "Case studies and indicators collection":
  https://www.fic.gov.za/document/case-studies-and-indicators-collection/
- Financial Intelligence Centre, "PCC 51: Guidance on measures relating to
  FPPOs and DPIPs":
  https://www.fic.gov.za/document/pcc-51-fppo-dpip/
- Financial Intelligence Centre, "Crypto asset service providers":
  https://www.fic.gov.za/accountable-institut/crypto-asset-service-providers/
- Financial Intelligence Centre, "Estate agents and targeted financial
  sanctions obligations," 2025:
  https://www.fic.gov.za/2025/08/05/estate-agents-and-tfs/
- Financial Intelligence Centre, "Motor vehicle dealers face financial crime
  risks in the trade of high-value goods," 2025:
  https://www.fic.gov.za/2025/10/22/motor-vehicle-dealers-fic-act-obligations/
- Financial Intelligence Centre, "Drafting a risk management and compliance
  programme," 2025:
  https://www.fic.gov.za/2025/07/08/drafting-a-risk-management-and-compliance-programme/
- Financial Intelligence Centre, "What are the beneficial ownership obligations
  for business?", 2026:
  https://www.fic.gov.za/2026/03/09/what-are-the-beneficial-ownership-obligations-for-business/
- Financial Intelligence Centre, "National risk assessment: Terrorist financing
  national risk assessment 2024":
  https://www.fic.gov.za/document/national-risk-assessment-terrorist-financing-national-risk-assessment-2024/
- South African Reserve Bank, "Payments Study Report 2023":
  https://www.resbank.co.za/content/dam/sarb/publications/media-releases/2024/payments/SARB%20Payments%20Study%20Report%202023.pdf
- South African Reserve Bank, "South African Reserve Bank Payments Study Report
  Launch," 2024:
  https://www.resbank.co.za/en/home/publications/publication-detail-pages/media-releases/2024/south-african-reserve-bank-payments-study-report-launch
- South African Reserve Bank, "SARB Annual Report 2024/25":
  https://www.resbank.co.za/en/home/publications/publication-detail-pages/reports/annual-reports/2025/sarb-2024-25
- South African Reserve Bank Prudential Authority, "Sector risk assessment for
  South Africa's banking sector (2022-2024)":
  https://www.resbank.co.za/en/home/publications/publication-detail-pages/prudential-authority/pa-public-awareness/Communication/2025/sra-banks
- Statistics South Africa, "Quarterly Labour Force Survey Q4: 2025 media
  release":
  https://www.statssa.gov.za/publications/P0211/Media%20Release%20QLFS%20Q4%202025.pdf
- Statistics South Africa, "How informality shapes South Africa's employment
  landscape," 2026:
  https://www.statssa.gov.za/?p=19240
- Statistics South Africa, "Survey of Employers and Self-Employed 2023":
  https://isibaloweb.statssa.gov.za/pages/surveys/pss/sese/2023/sese2023.php
- Statistics South Africa, "General Household Survey 2024 release":
  https://www.gov.za/news/media-statements/statssa-releases-general-household-survey-2024-27-may-2025
- Statistics South Africa, "Mid-year population estimates, 2025":
  https://www.statssa.gov.za/?PPN=P0302&page_id=1854
- Standard Bank, "Township Informal Economy Report newsroom summary," 2025:
  https://www.standardbank.co.za/southafrica/news-and-media/newsroom/80-percent-of-township-businesses-unregistered-reveals-standard-bank-inaugural-informal-economy-report
- World Bank, "Remittance Prices Worldwide data catalog":
  https://datacatalog.worldbank.org/search/dataset/0037898/remittance-prices-worldwide
- World Bank, "Remittance Prices Worldwide: South Africa to Zimbabwe corridor":
  https://remittanceprices.worldbank.org/corridor/South-Africa/Zimbabwe
- World Bank, "Remittance Prices Worldwide: South Africa to Lesotho corridor":
  https://remittanceprices.worldbank.org/corridor/South%20Africa/Lesotho
- Financial Action Task Force, "The FATF Recommendations":
  https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html
- Financial Action Task Force, "Virtual Assets Red Flag Indicators of Money
  Laundering and Terrorist Financing," 2020:
  https://www.fatf-gafi.org/en/publications/Methodsandtrends/Virtual-assets-red-flag-indicators.html
- Financial Action Task Force, "Guidance on Beneficial Ownership of Legal
  Persons," 2023:
  https://www.fatf-gafi.org/content/fatf-gafi/en/publications/Fatfrecommendations/Guidance-Beneficial-Ownership-Legal-Persons.html
- Financial Action Task Force, "Guidance on Beneficial Ownership and
  Transparency of Legal Arrangements," 2024:
  https://www.fatf-gafi.org/content/fatf-gafi/en/publications/Fatfrecommendations/Guidance-Beneficial-Ownership-Transparency-Legal-Arrangements.html
- Financial Action Task Force, "National money laundering and terrorist
  financing risk assessment," 2013:
  https://www.fatf-gafi.org/en/publications/Methodsandtrends/Nationalmoneylaunderingandterroristfinancingriskassessment.html
- Financial Action Task Force, "Money Laundering National Risk Assessment
  Guidance," 2025:
  https://www.fatf-gafi.org/en/publications/Methodsandtrends/Money-Laundering-National-Risk-Assessment-Guidance.html
- Judicial Commission of Inquiry into State Capture, "State Capture Report,
  Part 1," 2022:
  https://archive.parliament.gov.za/items/7342f38b-7749-49db-8a1d-10a742b73edc
- United Nations Office on Drugs and Crime, "Estimating illicit financial flows
  resulting from drug trafficking and other transnational organized crimes,"
  2011:
  https://unodc.org/documents/data-and-analysis/Studies/Illicit_financial_flows_2011_web.pdf
- LexisNexis Risk Solutions, "Financial Crime in Africa White Paper," 2022:
  https://risk.lexisnexis.com/global/en/insights-resources/white-paper/financial-crime-in-africa
- IBM, "AMLSim: multi-agent simulator for anti-money laundering transaction
  data":
  https://github.com/IBM/AMLSim
- Scientific Data, "A synthetic data set to benchmark anti-money laundering
  methods," 2023:
  https://www.nature.com/articles/s41597-023-02569-2
- IBM Research and NeurIPS Datasets and Benchmarks, "Realistic Synthetic
  Financial Transactions for Anti-Money Laundering Models," 2023:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html
- B. Oztas, "Synthetic Transaction Monitoring Dataset AML (SAML-D)," Kaggle
  dataset and ICEBE 2023 paper:
  https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
- M. Weber et al., "Scalable Graph Learning for Anti-Money Laundering: A First
  Look," arXiv:1812.00076, 2018:
  https://arxiv.org/abs/1812.00076
- A. Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for
  Dynamic Graphs," AAAI, 2020:
  https://arxiv.org/abs/1902.10191
- The Alan Turing Institute, "Synthetic Data for Anti-Money Laundering":
  https://www.turing.ac.uk/research/research-projects/synthetic-data-anti-money-laundering
- The Presidency, "President Cyril Ramaphosa grants extension of Madlanga
  Commission report deadline," 9 July 2026:
  https://www.gov.za/news/media-statements/president-cyril-ramaphosa-grants-extension-madlanga-commission-report
- Madlanga Commission public-record archive, "Day 74, 10 March 2026":
  https://madlangacommission.co.za/hearings/day-74
- Madlanga Commission public-record archive, "About the commission":
  https://madlangacommission.co.za/about
