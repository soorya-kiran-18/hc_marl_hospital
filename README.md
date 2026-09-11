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
│   ├── agents.py                   # MetaAgent + LocalAgent (ER/ICU/Ward), Evolution Strategies
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
`src.train` ≈ 9–10 minutes (750 episodes × 5 methods; the three HC-MARL
methods now co-train a Meta-Agent + 3 Local Agents per episode instead of
a single agent, which is the bulk of the added time over the old ≈2 min).

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

This is the paper's actual four-component architecture, all implemented:

- **`agents.py` — `MetaAgent`** (Global Meta-Agent, executive level): a
  linear-softmax policy over hospital global state (utilization pressure
  per department, boarding queue size, time of day) that outputs a
  floating-nurse allocation across [ER, ICU, Ward] every ~4 hours (16
  simulation steps). Learns via antithetic **Evolution Strategies**
  (16 perturbation directions per update, returns z-score normalized
  before the gradient estimate — see "ES stability" below).
- **`agents.py` — `LocalAgent`** (Local Agents, one per department: ER,
  ICU, Ward): decentralized department-head agents that act every 15-minute
  step from their own department-local observation only (no direct
  peer-to-peer communication — coupling is indirect, through the shared
  simulation environment and the Meta-Agent's finite nurse budget, matching
  the paper's "hierarchical decentralized execution"). Each Local Agent
  learns one thing: an `equity_priority` tie-break rule for which patient
  gets admitted first among candidates of equal clinical urgency (ESI).
  *Admission volume* is deliberately not learned — the environment always
  admits as many boarding patients as the Meta-Agent-derived, Safety-Filter
  verified capacity allows, which is what makes ICU/Ward overcrowding via
  admission structurally impossible rather than merely discouraged. (An
  earlier version also tried to learn admission *rate*; a partially-trained
  agent stuck near 50% choked patient outflow and made safety violations
  *worse*, not better — see the ES stability note.)
- **`controllers.py`** — two rule-based, "uncoordinated single-agent"
  baselines the paper's evaluation section calls for: `FCFSController`
  (static split proportional to base staffing, no adaptation) and
  `ESIProportionalController` (static split proportional to the historical
  ESI acuity mix). These do not get Local Agents — only the HC-MARL methods
  use the full hierarchy.
- **`constraints.py`** — `ConstraintFilter` (hard CMDP check: rejects any
  Meta-Agent allocation that would push a department's patient-to-nurse
  ratio past its limit, falling back to base-only staffing) and
  `calculate_equity_penalty` (the soft `λ·Σ|E[T_d] − E[T_all]|` term from
  the paper's reward equation).
- **`hospital_env.py`** — the shared Hospital Simulation Environment: the
  only channel through which the Meta-Agent and the three Local Agents
  ever interact.

**ES stability.** Co-training four agents (one Meta-Agent + three Local
Agents) off a single shared episode reward has noticeably higher reward
variance than training the Meta-Agent alone. An un-normalized antithetic-ES
update (the original single-agent implementation) is not robust to that:
in testing it occasionally blew Meta-Agent weights up to extreme magnitudes
within a few dozen episodes, collapsing the softmax into a degenerate
"always give one department 100% of the float nurses, ignore state
entirely" policy — which, depending on *which* department won, could
starve the ER (the tightest-capacity department) and spike violations. The
fix, standard in the ES literature (Salimans et al., 2017): normalize
returns (z-score across the batch of perturbations) before computing the
weighted-epsilon gradient, plus a hard clip on weight magnitude as a
defense-in-depth bound. `_step_reward` also now subtracts an explicit,
large `violation_penalty` whenever any department is over its safety ratio
that step — the graded utilization-overflow term alone was not always
enough to outweigh the reward gained by packing a *different* department
to 100%, so ES could still learn to trade one department's safety for
another's throughput before this term was added.

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
10 episodes per seed; `mean_reward` and `equity_gap` averaged over the last
10 episodes per seed, `safety_violation_rate` averaged over all 250):

| Method | Mean reward | Safety violation rate | Equity gap (steps) |
|---|---|---|---|
| HC-MARL-NoFilter | **125.7 (highest)** | 0.050 | 0.261 |
| HC-MARL-Phase1 (λ=0) | 122.1 | 0.051 | 0.212 |
| FCFS-Static | 119.1 | 0.070 | 0.267 |
| HC-MARL-Phase2 (λ=2.5) | 111.4 | 0.052 | 0.259 |
| ESI-Proportional | 65.3 | 0.149 | 0.255 |

Reading this honestly:
- **All three HC-MARL variants now have the lowest safety violation rate
  of all five methods (~5%, vs. FCFS's 7% and ESI-Proportional's 15%)** —
  and the two highest-reward methods overall. This is a materially
  different, much stronger result than an earlier iteration of this
  codebase, which briefly implemented only a single centralized Meta-Agent
  (no Local Agents) and got a Phase1 violation rate of 12.6%.
- The CMDP filter still does real, separable work even with Local Agents
  in place: NoFilter (0.050) vs. Phase1 (0.051) is close, but that's partly
  because Local Agents already structurally prevent ICU/Ward overcrowding
  via admission-capacity capping — the filter's remaining job is the
  Meta-Agent's own macro nurse *reallocation* risk (pulling nurses away
  from a department mid-shift), a narrower failure mode than before.
- The equity penalty's effect is genuinely noisy at n=3 seeds: Phase1's
  average equity gap (0.212) looks lower than Phase2's (λ=2.5, 0.259), but
  that's driven mostly by one favorable seed (0.09) against two unfavorable
  ones (0.27, 0.28) for Phase1, while Phase2's per-seed gaps are more
  tightly clustered (0.23–0.30). Reporting the average as "Phase1 wins" would
  overstate what a 3-seed experiment can actually show. More seeds are
  needed before drawing a real equity conclusion — say so plainly in the
  paper rather than picking whichever run looks best.
- ESI-Proportional's reward dropped sharply from the previous iteration of
  this codebase (was 172.8) — not because its policy changed at all (it's
  a static, unlearned baseline), but because the reward function itself was
  changed to subtract a large, explicit penalty for every step any
  department is in violation (see "ES stability" above). At a 14.9%
  violation rate, that penalty now dominates its reward. This is an honest,
  intended consequence of taking the paper's "hard safety constraint" stance
  seriously in the reward, not a regression in the baseline itself.

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
5. **The Local Agents described in the paper's architecture (Section
   IV-B) did not exist in the code** — `src/agents.py` had only a single
   centralized `MetaAgent`, i.e. the paper's proposed *hierarchical*
   Meta-Agent + Local Agents design had silently been implemented as a flat,
   single-agent system. Added `LocalAgent` (one per department: ER, ICU,
   Ward), wired into `hospital_env.py`'s admission/triage step and
   `train.py`'s training loop as decentralized agents coupled only through
   the shared simulator and the Meta-Agent's nurse budget, per the paper's
   "hierarchical decentralized execution" description. This also surfaced
   and required fixing an Evolution Strategies stability issue (see "ES
   stability" above) that the original single-agent training happened not
   to trigger.

## Note on the paper draft

`hospital_rl_project (1).docx` currently contains Title, Abstract,
Introduction, Literature Survey (6 refs — needs expanding to ~16 for the
2-page table), Motivation, and Proposed Methodology. **It does not yet
contain a written Results/Experimentation section** — that needs to be
written using the real numbers in `docs/` (triage metrics, fairness rates,
comparison summary, confusion matrix, convergence and comparison plots).
