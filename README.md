# HC-MARL Hospital Resource Management

Hierarchical Constrained Multi-Agent Reinforcement Learning for equitable and
safe hospital resource allocation, built on the NHAMCS 2018–2022 ED-visit
dataset. This README documents what the codebase actually does, what was
fixed after an earlier AI-assisted pass, and how to reproduce every result.

## Quick start

Run from the repo root, in order, in the same shell (so the venv stays active):

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib joblib

# place the raw dataset at data/nhamcs_data_2018_22.csv first, then:
python3 -m src.data_processing   # cleans data, writes data/processed/{train,val,test}.csv
python3 -m src.triage_models     # baseline ESI classifiers + fairness check -> docs/
python3 -m src.hospital_env      # smoke-test the simulator alone
python3 -m src.train             # runs all 5 comparison methods, 3 seeds -> docs/ (~2-3 min)
python3 -m src.make_plots        # builds docs/training_convergence.png, docs/comparative_analysis.png
```

## Repository layout

```
hc_marl_hospital/
├── data/
│   ├── nhamcs_data_2018_22.csv     # raw dataset (not committed, see .gitignore)
│   └── processed/                  # train/val/test splits (generated)
├── docs/                           # all generated metrics, plots, result CSVs
├── src/
│   ├── data_processing.py          # cleaning, feature engineering, equity mapping, split
│   ├── triage_models.py            # baseline ESI classifiers + fairness check
│   ├── hospital_env.py             # 15-min-step ER/ICU/Ward simulator
│   ├── agents.py                   # MetaAgent (Evolution Strategies policy)
│   ├── controllers.py              # FCFS-Static / ESI-Proportional baselines
│   ├── constraints.py              # CMDP safety filter + equity penalty
│   ├── train.py                    # runs all 5 methods, 3 seeds, logs results
│   └── make_plots.py               # convergence + comparison charts from docs/*.csv
└── hospital_rl_project (1).docx    # IEEE paper draft (methodology only, no results yet)
```

## Setup

```bash
cd "hc_marl_hospital"
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib joblib
```

Place the raw dataset at `data/nhamcs_data_2018_22.csv`.

## Running the pipeline

```bash
python3 -m src.data_processing   # cleans data, writes data/processed/{train,val,test}.csv
python3 -m src.triage_models     # baseline ESI classifiers + fairness check -> docs/
python3 -m src.hospital_env      # smoke-test the simulator alone
python3 -m src.train             # runs all 5 comparison methods, 3 seeds -> docs/
python3 -m src.make_plots        # builds docs/training_convergence.png, docs/comparative_analysis.png
```

Full run time on a laptop CPU: data processing + triage models ≈ 15s,
`src.train` ≈ 2 minutes (750 episodes × 5 methods × ES perturbations).

## What each stage does

### 1. Data processing (`src/data_processing.py`)

- Cleans physiologically impossible vitals (e.g. heart rate outside 20–300)
  to `NaN`. Imputation is deliberately **not** done here — it happens later,
  fit only on the training split, so no validation/test statistics leak into
  training.
- Maps the raw `insurance` column into 5 equity groups: `Private`,
  `Medicaid`, `Medicare`, `SelfPay`, `Other/Unknown`.
- Engineers `shock_index`, `chronic_conditions`, `arrival_hour`,
  `ems_arrival_flag`, `injury_flag` — all information available at/before
  the triage decision, so none of it leaks the label.
- Stratified 70/15/15 split by ESI level: 40,686 / 8,719 / 8,719 rows.

### 2. Triage classifiers (`src/triage_models.py`)

Trains 7 baseline models (Logistic Regression, Decision Tree, Naive Bayes,
Random Forest, Gradient Boosting, MLP, Soft-Voting Ensemble) to predict ESI
1–5 from vitals + demographics-free features. `wait_time_minutes`,
`intervention_iv_fluids`, `race`, and `insurance` are excluded from the
feature set — the first two happen *after* triage (label leakage), the last
two are held out and used only for the fairness check.

Best model on the test set (`docs/triage_model_metrics.csv`): **Soft-Voting
Ensemble**, accuracy 0.524, macro-F1 0.328, weighted kappa 0.327, ROC-AUC
0.686.

Fairness check (`docs/fairness_under_triage_rates.csv`): the "under-triage
rate" is the fraction of true ESI 1–2 (critical) patients the best model
predicts as ESI 3–5. This is 60–77% across every insurance group — the
model is genuinely unreliable for critical patients on vitals alone, which
is the empirical motivation for the CMDP safety filter (see below) rather
than trusting a predictive model's output outright.

### 3. Hospital simulator (`src/hospital_env.py`)

A discrete-time (15-minute step) simulator of one hospital over one
simulated day (96 steps), built on real sampled NHAMCS patient records
(not synthetic random numbers).

- **Routing**: NHAMCS is ED-visit-level data with no admit-destination or
  length-of-stay field, so department routing is inferred from ESI acuity —
  ESI 1–2 → ICU, ESI 3 → Ward, ESI 4–5 → treated and discharged from the ER.
  This mirrors real triage practice but is a modeling assumption worth
  stating explicitly in the paper.
- **Length of stay**: also not observed in this dataset, so each department
  discharges occupied beds with a fixed per-step probability (ER fastest,
  ICU slowest).
- **Capacity**: hard-coded ratios (ER 1:4, ICU 1:2, Ward 1:6) plus a pool of
  6 floating nurses the Meta-Agent distributes.
- **Arrivals**: Poisson-sampled per step from the real empirical
  arrival-hour distribution, scaled to `daily_arrivals=900` — calibrated so
  the system is genuinely resource-constrained (at low volumes nurse
  allocation barely matters because capacity is always slack, which would
  make every method look identical).

### 4. Agents and safety/equity mechanisms

- `agents.py` — `MetaAgent`: a linear-softmax policy over hospital state
  (utilization pressure per department, boarding queue size, time of day)
  that outputs a floating-nurse allocation. Learns via antithetic
  **Evolution Strategies** (8 perturbation directions per update) — chosen
  over full PPO/backprop because it needs no gradient through the
  simulator and stays honestly within "plain NumPy RL."
- `controllers.py` — two rule-based baselines: `FCFSController` (static
  split proportional to base staffing, no adaptation) and
  `ESIProportionalController` (static split proportional to the historical
  ESI acuity mix).
- `constraints.py` — `ConstraintFilter` (hard CMDP check: rejects any
  proposed allocation that would push a department's patient-to-nurse
  ratio past its limit, falling back to base-only staffing) and
  `calculate_equity_penalty` (the soft `λ·Σ|E[T_d] − E[T_all]|` term from
  the paper's reward equation).

### 5. Training and comparison (`src/train.py`)

Runs 5 methods × 3 seeds × 250 episodes each on the training split:

| Method | Safety filter | Equity penalty (λ) |
|---|---|---|
| FCFS-Static | n/a (static) | 0 |
| ESI-Proportional | n/a (static) | 0 |
| HC-MARL-NoFilter | ✗ | 0 |
| HC-MARL-Phase1 | ✓ | 0 |
| HC-MARL-Phase2 | ✓ | 2.5 |

Latest results (`docs/comparison_summary.csv`, converged — mean of the last
10 episodes per seed):

| Method | Mean reward | Safety violation rate | Equity gap (steps) |
|---|---|---|---|
| ESI-Proportional | 172.8 | 0.149 | 0.255 |
| HC-MARL-NoFilter | 171.5 | 0.214 | 0.234 |
| HC-MARL-Phase1 (λ=0) | 170.3 | 0.126 | 0.247 |
| HC-MARL-Phase2 (λ=2.5) | 168.4 | 0.192 | **0.227 (lowest)** |
| FCFS-Static | 168.2 | **0.070 (lowest)** | 0.267 |

Reading this honestly:
- The CMDP filter works: adding it (NoFilter → Phase1) cuts the safety
  violation rate by ~40% (0.214 → 0.126) for a small reward cost.
- The equity penalty works: Phase2 has the lowest equity gap of all 5
  methods, but at the cost of a higher safety violation rate and lower
  reward than Phase1 — a real, reportable tension between the hard safety
  constraint and the soft equity objective, matching the paper's own
  stated research question in the Evaluation Metrics section.
- FCFS-Static is safest by construction (it never reacts, so it never
  overshoots) but has the worst equity outcome — the least adaptive policy
  is also the least fair one here.
- The learned HC-MARL policies are competitive with, but do not decisively
  beat, the best static heuristic (ESI-Proportional) on raw reward. With
  only 6 floating nurses and macro-decisions every 4 hours, there's limited
  room for a learned policy to out-throughput a sensible heuristic — its
  value-add is the safety/equity mechanism, not raw throughput. Be
  upfront about this in the paper rather than overclaiming.

Outputs: `docs/{method}_results.csv` (per-episode logs, for convergence
plots), `docs/comparison_summary.csv` (aggregated table for the paper).

## Known issues fixed from the earlier AI-generated pass

1. **Insurance mapping bug**: the raw `insurance` column contains
   `Medicaid/Public`, `Self_Pay`, `Unknown/Blank`, etc., but the original
   code only matched exact strings `Medicaid`/`Self-Pay`. Every real
   Medicaid and Self-Pay patient (24,865 of them) was silently dumped into
   `Other/Unknown`, so the fairness check only ever showed 3 of 5 groups.
   Fixed in `data_processing.py`.
2. **`src/train.py` was completely disconnected from the simulator.** It
   used `np.random.rand(5)` as fake state, hardcoded mock wait-time
   distributions unrelated to λ, and never called any policy-update method
   — nothing was learning. The "reward 90.50 / 79.03" numbers from an
   earlier run were an artifact of that fixed formula, not a trained
   policy. Rebuilt end-to-end (see above).
3. **`.gitignore`** referenced a dataset filename (`nhamcs_18_22.csv`) that
   didn't match the actual file (`nhamcs_data_2018_22.csv`), so the raw
   15MB CSV was about to get committed. Now `data/` is ignored entirely.
4. Triage models only used 5 basic vitals; added spo2, pain score, age,
   EMS-arrival flag (all legitimate pre-triage information) plus
   class-imbalance handling, which raised macro-F1 from ~0.20 to ~0.33.

## Note on the paper draft

`hospital_rl_project (1).docx` currently contains Title, Abstract,
Introduction, Literature Survey (6 refs — needs expanding to ~16 for the
2-page table), Motivation, and Proposed Methodology. **It does not yet
contain a written Results/Experimentation section** — that needs to be
written using the real numbers in `docs/` (triage metrics, fairness rates,
comparison summary, confusion matrix, convergence and comparison plots).
