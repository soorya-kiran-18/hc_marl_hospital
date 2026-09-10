import os
import numpy as np
import pandas as pd

from src.hospital_env import HospitalEnv, load_patient_pool, EQUITY_GROUPS
from src.agents import MetaAgent
from src.controllers import FCFSController, ESIProportionalController
from src.constraints import ConstraintFilter, calculate_equity_penalty

MACRO_PERIOD = 16  # meta-agent / controller re-decides every 16 steps (~4h)
EPISODES_PER_SEED = 250
SEEDS = [0, 1, 2]


def run_episode(pool, controller_or_weights, action_fn, lambda_weight=0.0,
                 safety_filter=None, env_seed=0, daily_arrivals=300):
    """
    action_fn(controller_or_weights, state) -> raw 3-vector nurse allocation.
    Runs one simulated day and returns a dict of episode metrics.
    """
    env = HospitalEnv(patient_pool=pool, daily_arrivals=daily_arrivals, seed=env_seed)
    state = env.reset()

    action = action_fn(controller_or_weights, state)
    step_reward_sum = 0.0
    safety_checks = 0
    safety_violations_blocked = 0

    for t in range(env.STEPS_PER_DAY):
        if t % MACRO_PERIOD == 0:
            action = action_fn(controller_or_weights, state)
            if safety_filter is not None:
                proposed_nurses = env.nurses_for(action)
                census = env.current_census()
                safety_checks += 1
                if not safety_filter.verify_action_safety(proposed_nurses, census):
                    safety_violations_blocked += 1
                    action = np.zeros(3)  # fall back to base-only staffing

        state, reward, done, info = env.step(action)
        step_reward_sum += reward
        if done:
            break

    equity_gap, group_means = env.equity_gap()
    equity_penalty = calculate_equity_penalty(env.wait_times, lambda_weight)
    final_reward = step_reward_sum - equity_penalty

    return {
        'reward': final_reward,
        'throughput_reward': step_reward_sum,
        'equity_penalty': equity_penalty,
        'equity_gap_steps': equity_gap,
        'safety_violation_rate': env.safety_violations / env.STEPS_PER_DAY,
        'filter_block_rate': (safety_violations_blocked / safety_checks) if safety_checks else 0.0,
        'discharged': env.discharged_count,
    }


def rule_based_action_fn(controller, state):
    return controller.action(state)


def meta_agent_action_fn(agent_and_weights, state):
    agent, weights = agent_and_weights
    return agent.action(state, weights=weights)


def run_rule_based_method(name, controller_factory, pool, seeds, lambda_weight, daily_arrivals):
    print(f"\n--- Running {name} ---")
    filt = ConstraintFilter()
    rows = []
    for seed in seeds:
        controller = controller_factory()
        for ep in range(EPISODES_PER_SEED):
            m = run_episode(pool, controller, rule_based_action_fn, lambda_weight=lambda_weight,
                             safety_filter=filt, env_seed=seed * 1000 + ep, daily_arrivals=daily_arrivals)
            m.update({'method': name, 'seed': seed, 'episode': ep})
            rows.append(m)
    df = pd.DataFrame(rows)
    print(f"{name}: mean reward={df['reward'].mean():.2f} | "
          f"safety_violation_rate={df['safety_violation_rate'].mean():.3f} | "
          f"equity_gap={df['equity_gap_steps'].mean():.2f}")
    return df


def run_hc_marl(name, pool, seeds, lambda_weight, daily_arrivals, state_dim=5, total_nurses=6,
                 use_safety_filter=True):
    print(f"\n--- Running {name} (lambda={lambda_weight}, safety_filter={use_safety_filter}) ---")
    filt = ConstraintFilter() if use_safety_filter else None
    rows = []
    for seed in seeds:
        agent = MetaAgent(state_dim=state_dim, total_nurses=total_nurses, seed=seed)
        for ep in range(EPISODES_PER_SEED):
            # 1. evaluate current policy (this is what gets logged)
            m = run_episode(pool, (agent, agent.weights), meta_agent_action_fn,
                             lambda_weight=lambda_weight, safety_filter=filt,
                             env_seed=seed * 1000 + ep, daily_arrivals=daily_arrivals)
            m.update({'method': name, 'seed': seed, 'episode': ep})
            rows.append(m)

            # 2. antithetic ES update using a matched arrival stream so both
            # perturbations are compared on the same patient sequence
            def episode_return(weights, _seed=seed * 1000 + ep):
                res = run_episode(pool, (agent, weights), meta_agent_action_fn,
                                   lambda_weight=lambda_weight, safety_filter=filt,
                                   env_seed=_seed, daily_arrivals=daily_arrivals)
                return res['reward']

            agent.es_step(episode_return)
    df = pd.DataFrame(rows)
    print(f"{name}: mean reward (last 10 ep)={df.groupby('seed').tail(10)['reward'].mean():.2f} | "
          f"safety_violation_rate={df['safety_violation_rate'].mean():.3f} | "
          f"filter_block_rate={df['filter_block_rate'].mean():.3f} | "
          f"equity_gap={df['equity_gap_steps'].mean():.2f}")
    return df


def summarize(all_results):
    rows = []
    for name, df in all_results.items():
        tail = df.groupby('seed').tail(10)  # converged performance
        rows.append({
            'method': name,
            'mean_reward': tail['reward'].mean(),
            'std_reward': tail['reward'].std(),
            'mean_safety_violation_rate': df['safety_violation_rate'].mean(),
            'mean_filter_block_rate': df['filter_block_rate'].mean(),
            'mean_equity_gap_steps': tail['equity_gap_steps'].mean(),
        })
    return pd.DataFrame(rows).sort_values('mean_reward', ascending=False)


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)

    print("Loading training patient pool (data/processed/train.csv)...")
    pool = load_patient_pool("data/processed/train.csv")
    # Calibrated so the system is meaningfully resource-constrained: at low
    # arrival volumes nurse allocation barely matters (capacity is always
    # slack), which would make every method converge to the same outcome.
    daily_arrivals = 900

    esi_for_mix = pool['esi']

    all_results = {}

    all_results['FCFS-Static'] = run_rule_based_method(
        'FCFS-Static',
        lambda: FCFSController({'ER': 10, 'ICU': 5, 'Ward': 15}, total_float_nurses=6),
        pool, SEEDS, lambda_weight=0.0, daily_arrivals=daily_arrivals)

    all_results['ESI-Proportional'] = run_rule_based_method(
        'ESI-Proportional',
        lambda: ESIProportionalController(esi_for_mix, total_float_nurses=6),
        pool, SEEDS, lambda_weight=0.0, daily_arrivals=daily_arrivals)

    all_results['HC-MARL-NoFilter'] = run_hc_marl(
        'HC-MARL-NoFilter', pool, SEEDS, lambda_weight=0.0, daily_arrivals=daily_arrivals,
        use_safety_filter=False)

    all_results['HC-MARL-Phase1(lambda=0.0)'] = run_hc_marl(
        'HC-MARL-Phase1(lambda=0.0)', pool, SEEDS, lambda_weight=0.0, daily_arrivals=daily_arrivals)

    all_results['HC-MARL-Phase2(lambda=2.5)'] = run_hc_marl(
        'HC-MARL-Phase2(lambda=2.5)', pool, SEEDS, lambda_weight=2.5, daily_arrivals=daily_arrivals)

    for name, df in all_results.items():
        fname = "docs/" + name.replace('(', '_').replace(')', '').replace('=', '').replace('.', '') + "_results.csv"
        df.to_csv(fname, index=False)

    summary = summarize(all_results)
    summary.to_csv("docs/comparison_summary.csv", index=False)
    print("\n================ COMPARATIVE SUMMARY (converged, last 10 ep/seed) ================")
    print(summary.to_string(index=False))
    print("\nSaved per-method logs and docs/comparison_summary.csv")
