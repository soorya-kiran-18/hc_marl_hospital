import numpy as np
import pandas as pd

EQUITY_GROUPS = ['Private', 'Medicaid', 'Medicare', 'SelfPay', 'Other/Unknown']


def load_patient_pool(data_path):
    """Load a processed split once into plain arrays for fast env reuse."""
    df = pd.read_csv(data_path)
    target_col = 'target_triage_acuity' if 'target_triage_acuity' in df.columns else 'triage_level'
    df = df.dropna(subset=[target_col])

    esi = df[target_col].astype(int).to_numpy()
    equity = df['equity_group'].where(df['equity_group'].isin(EQUITY_GROUPS), 'Other/Unknown').to_numpy() \
        if 'equity_group' in df.columns else np.array(['Other/Unknown'] * len(df))

    if 'arrival_hour' in df.columns:
        hour_counts = df['arrival_hour'].value_counts().reindex(range(24), fill_value=0)
        hourly_weights = (hour_counts / hour_counts.sum()).to_numpy()
    else:
        hourly_weights = np.ones(24) / 24

    return {'esi': esi, 'equity': equity, 'hourly_weights': hourly_weights}


class HospitalEnv:
    """
    Discrete-time (15-minute step) hospital patient-flow simulator driven by
    real NHAMCS patient records. One episode = one simulated 24h day
    (96 steps).

    NHAMCS is ED-visit-level data with no admit-destination or length-of-stay
    field, so two modeling choices approximate a full ER/ICU/Ward system:
      - Department routing is inferred from predicted ESI acuity (matching
        real triage practice): ESI 1-2 -> ICU, ESI 3 -> Ward, ESI 4-5 -> ER
        (treat-and-discharge).
      - Length of stay is not observed, so each department discharges
        occupied beds with a fixed per-step probability calibrated to give
        roughly realistic relative throughput (ER fastest, ICU slowest).
    """

    STEPS_PER_DAY = 96  # 24h / 15min
    DISCHARGE_PROB = {'ER': 0.15, 'ICU': 0.03, 'Ward': 0.06}

    def __init__(self, data_path="data/processed/train.csv", patient_pool=None,
                 daily_arrivals=300, seed=None):
        """
        patient_pool: optional pre-loaded dict with keys 'esi', 'equity',
        'hourly_weights' (see load_patient_pool below). Pass this to avoid
        re-reading the CSV for every episode during training/evaluation.
        """
        if patient_pool is not None:
            self.esi = patient_pool['esi']
            self.equity = patient_pool['equity']
            self.hourly_weights = patient_pool['hourly_weights']
        else:
            pool = load_patient_pool(data_path)
            self.esi = pool['esi']
            self.equity = pool['equity']
            self.hourly_weights = pool['hourly_weights']
        self.n_patients = len(self.esi)

        self.daily_arrivals = daily_arrivals
        self.rng = np.random.default_rng(seed)

        self.er_ratio_limit = 4
        self.icu_ratio_limit = 2
        self.ward_ratio_limit = 6
        self.total_float_nurses = 6
        self.er_nurses_base = 10
        self.icu_nurses_base = 5
        self.ward_nurses_base = 15

        self.reset()

    def reset(self):
        self.step_idx = 0
        self.departments = {'ER': [], 'ICU': [], 'Ward': []}
        self.wait_times = {g: [] for g in EQUITY_GROUPS}
        self.safety_violations = 0
        self.discharged_count = 0
        # capacity implied by the base-only staffing, used for the very
        # first observation before any action has been taken
        self.last_capacity = {
            'ER': self.er_nurses_base * self.er_ratio_limit,
            'ICU': self.icu_nurses_base * self.icu_ratio_limit,
            'Ward': self.ward_nurses_base * self.ward_ratio_limit,
        }
        return self._get_obs()

    def _sample_patients(self, n):
        idx = self.rng.integers(0, self.n_patients, size=n)
        return self.esi[idx], self.equity[idx]

    def _get_obs(self):
        # Utilization pressure (census / last-known capacity) is a far more
        # informative signal for the policy than raw census: "10 patients"
        # means nothing without knowing capacity, "0.9 utilized" does.
        er_util = len(self.departments['ER']) / max(self.last_capacity['ER'], 1)
        icu_util = len(self.departments['ICU']) / max(self.last_capacity['ICU'], 1)
        ward_util = len(self.departments['Ward']) / max(self.last_capacity['Ward'], 1)
        boarding = sum(1 for p in self.departments['ER'] if not p['admitted'])
        boarding_frac = min(boarding / 20.0, 1.0)  # normalized, capped at 1
        return np.array([er_util, icu_util, ward_util, boarding_frac,
                          self.step_idx / self.STEPS_PER_DAY], dtype=np.float32)

    def get_obs(self):
        return self._get_obs()

    def nurses_for(self, meta_action):
        meta_action = np.clip(np.array(meta_action, dtype=float), 0, self.total_float_nurses)
        total = meta_action.sum()
        if total > self.total_float_nurses:
            meta_action = meta_action / total * self.total_float_nurses
        er_f, icu_f, ward_f = np.floor(meta_action).astype(int)
        return {
            'ER': self.er_nurses_base + er_f,
            'ICU': self.icu_nurses_base + icu_f,
            'Ward': self.ward_nurses_base + ward_f,
        }

    def current_census(self):
        return {d: len(self.departments[d]) for d in self.departments}

    def local_state(self, dept):
        """
        3-dim local observation for department `dept`'s Local Agent:
        [own utilization, ER boarding-queue pressure, time of day]. This is
        a department-specific slice of the same global observation the
        Meta-Agent sees -- Local Agents get no privileged extra information,
        only their own view of it, consistent with decentralized execution.
        """
        obs = self._get_obs()
        util_idx = {'ER': 0, 'ICU': 1, 'Ward': 2}[dept]
        return np.array([obs[util_idx], obs[3], obs[4]], dtype=np.float32)

    def step(self, meta_action, local_actions=None):
        """
        meta_action: Meta-Agent's macro floating-nurse allocation [ER, ICU, Ward].
        local_actions: optional dict {'ER': [admission_rate, equity_priority],
        'ICU': [...], 'Ward': [...]} from the decentralized Local Agents
        (see agents.LocalAgent). When None -- the uncoordinated single-agent
        baselines (FCFS-Static, ESI-Proportional) -- admission reverts to the
        original flat rule with no learned prioritization or per-department
        throttling, matching the paper's "uncoordinated single-agent
        baseline" semantics for those methods.
        """
        nurses = self.nurses_for(meta_action)
        capacity = {
            'ER': nurses['ER'] * self.er_ratio_limit,
            'ICU': nurses['ICU'] * self.icu_ratio_limit,
            'Ward': nurses['Ward'] * self.ward_ratio_limit,
        }
        self.last_capacity = capacity

        # 1. new arrivals, scaled by the empirical hourly arrival profile
        hour = int((self.step_idx / self.STEPS_PER_DAY) * 24) % 24
        expected_arrivals = self.daily_arrivals * self.hourly_weights[hour] / (self.STEPS_PER_DAY / 24)
        n_arrivals = int(self.rng.poisson(max(expected_arrivals, 0.01)))
        if n_arrivals > 0:
            esi_batch, equity_batch = self._sample_patients(n_arrivals)
            for esi, equity in zip(esi_batch, equity_batch):
                target_dept = 'ICU' if esi <= 2 else ('Ward' if esi == 3 else 'ER')
                self.departments['ER'].append({'equity_group': str(equity), 'esi': int(esi),
                                                'wait': 0, 'target': target_dept, 'admitted': False})

        # 2. department admission/triage -- decentralized Local Agents if
        #    provided, else the original flat single-agent rule
        if local_actions is None:
            self.departments['ER'] = self._flat_admission_step(capacity)
        else:
            self.departments['ER'] = self._local_agent_admission_step(capacity, local_actions)

        # 3. stochastic discharge from each department
        safety_violation = 0
        for dept in ['ER', 'ICU', 'Ward']:
            census = len(self.departments[dept])
            if census > capacity[dept]:
                safety_violation = 1
            survivors = []
            for p in self.departments[dept]:
                if self.rng.random() < self.DISCHARGE_PROB[dept]:
                    self.discharged_count += 1
                else:
                    survivors.append(p)
            self.departments[dept] = survivors

        self.safety_violations += safety_violation
        self.step_idx += 1
        done = self.step_idx >= self.STEPS_PER_DAY

        utilization = {d: len(self.departments[d]) / max(capacity[d], 1) for d in capacity}
        all_boarding_waits = [p['wait'] for p in self.departments['ER'] if not p['admitted']]
        mean_wait = float(np.mean(all_boarding_waits)) if all_boarding_waits else 0.0

        info = {
            'safety_violation': safety_violation,
            'utilization': utilization,
            'mean_wait_steps': mean_wait,
            'nurses': nurses,
            'census': {d: len(self.departments[d]) for d in self.departments},
        }
        reward = self._step_reward(utilization, mean_wait, safety_violation)
        return self._get_obs(), reward, done, info

    def _flat_admission_step(self, capacity):
        """
        Original uncoordinated single-agent admission rule (no Local
        Agents): move boarding ER patients into their target dept if
        capacity allows (first-in-list order, no learned prioritization);
        ESI 4-5 patients are "treated" directly in the ER with a fixed
        per-step probability. Used by the flat baselines (FCFS-Static,
        ESI-Proportional) so their dynamics stay exactly as before.
        """
        new_er_list = []
        for p in self.departments['ER']:
            if p['target'] != 'ER' and len(self.departments[p['target']]) < capacity[p['target']]:
                p['admitted'] = True
                self.wait_times[p['equity_group']].append(p['wait'])
                self.departments[p['target']].append(p)
            elif p['target'] == 'ER' and not p['admitted'] and self.rng.random() < 0.5:
                p['admitted'] = True
                self.wait_times[p['equity_group']].append(p['wait'])
                new_er_list.append(p)
            else:
                p['wait'] += 1
                new_er_list.append(p)
        return new_er_list

    def _local_agent_admission_step(self, capacity, local_actions):
        """
        Decentralized Local Agent admission/triage (HC-MARL methods only).
        Each department always admits as many candidates as its *verified*
        capacity allows this step (structurally impossible to overcrowd via
        admission alone -- there is never a reward-side reason to admit
        slower than "as many as safely fit," so that volume is fixed rather
        than left to a noisy per-episode learning estimate). Candidates are
        ranked by clinical severity (ESI) first; the department's learned
        equity_priority breaks ties among patients of equal acuity.
        """
        admitted_ids = set()

        for target_dept in ['ICU', 'Ward']:
            candidates = [p for p in self.departments['ER']
                          if p['target'] == target_dept and not p['admitted']]
            if not candidates:
                continue
            equity_priority = local_actions[target_dept]
            if equity_priority > 0.5:
                candidates.sort(key=lambda p: (p['esi'], -p['wait']))  # longest-waiting first, within acuity
            else:
                candidates.sort(key=lambda p: (p['esi'], p['wait']))
            available_slots = max(capacity[target_dept] - len(self.departments[target_dept]), 0)
            n_admit = min(available_slots, len(candidates))
            for p in candidates[:n_admit]:
                p['admitted'] = True
                self.wait_times[p['equity_group']].append(p['wait'])
                self.departments[target_dept].append(p)
                admitted_ids.add(id(p))

        er_candidates = [p for p in self.departments['ER']
                          if p['target'] == 'ER' and not p['admitted']]
        if er_candidates:
            equity_priority = local_actions['ER']
            if equity_priority > 0.5:
                er_candidates.sort(key=lambda p: -p['wait'])  # longest-waiting first
            else:
                er_candidates.sort(key=lambda p: p['wait'])
            # ER "admission" here means starting treatment (occupied ER bed
            # count doesn't change either way), so it isn't gated by bed
            # capacity like ICU/Ward -- it's gated by *staff* throughput
            # instead: one new ESI 4-5 patient seen per ER nurse per step.
            # This is a structural, nurse-count-derived cap (not learned),
            # so the Meta-Agent's nurse allocation still matters for ER
            # throughput, while the Local Agent's equity_priority decides
            # who among the waiting queue gets seen first when demand
            # exceeds that throughput.
            er_treat_capacity = int(capacity['ER'] / self.er_ratio_limit)  # = ER nurse count
            n_treat = min(er_treat_capacity, len(er_candidates))
            for p in er_candidates[:n_treat]:
                p['admitted'] = True
                self.wait_times[p['equity_group']].append(p['wait'])
                admitted_ids.add(id(p))

        new_er_list = []
        for p in self.departments['ER']:
            if p['target'] != 'ER' and id(p) in admitted_ids:
                continue  # moved out to ICU/Ward above
            if id(p) not in admitted_ids and not p['admitted']:
                p['wait'] += 1
            new_er_list.append(p)
        return new_er_list

    @staticmethod
    def _step_reward(utilization, mean_wait, safety_violation, alpha=1.0, beta=0.05, violation_penalty=8.0):
        # R_step = alpha * sum(min(U_i, 1)) - overflow_penalty - beta * mean_wait
        #          - violation_penalty * safety_violation
        # matches the utilization/wait terms of the paper's global reward
        # (eq. 1); the equity term is added separately, once per episode.
        #
        # The flat violation_penalty term is deliberately large relative to
        # the per-department utilization reward (max +1.0/dept): the paper
        # treats clinical safety as a hard constraint and equity/throughput
        # as secondary, soft objectives, so a policy that trades a safety
        # violation for extra utilization elsewhere must never come out
        # ahead. Without this term we observed exactly that failure mode in
        # training: the graded overflow penalty on utilization alone (the
        # max(u-1,0)*2.0 term below) was not always enough to outweigh the
        # reward gained by packing another department to 100%, so ES
        # sometimes learned to sacrifice one department's safety for
        # another's throughput. This term makes any such trade strictly
        # reward-negative.
        utilization_term = sum(min(u, 1.0) - max(u - 1.0, 0.0) * 2.0 for u in utilization.values())
        return alpha * utilization_term - beta * mean_wait - violation_penalty * safety_violation

    def equity_gap(self):
        """Max |group mean wait - overall mean wait| in 15-min steps."""
        group_means = {g: float(np.mean(w)) for g, w in self.wait_times.items() if len(w) > 0}
        all_waits = [w for ws in self.wait_times.values() for w in ws]
        if len(group_means) < 2 or not all_waits:
            return 0.0, group_means
        overall = float(np.mean(all_waits))
        gap = max(abs(v - overall) for v in group_means.values())
        return gap, group_means


if __name__ == "__main__":
    env = HospitalEnv(seed=0)
    obs = env.reset()
    print(f"Initial State Observation: {obs}")

    total_reward = 0.0
    for _ in range(env.STEPS_PER_DAY):
        action = [3, 1, 2]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    gap, group_means = env.equity_gap()
    print(f"Episode total reward: {total_reward:.2f}")
    print(f"Safety violations: {env.safety_violations}/{env.STEPS_PER_DAY} steps")
    print(f"Equity gap (steps): {gap:.2f} | group means: {group_means}")
    print("Simulation engine test passed!")
