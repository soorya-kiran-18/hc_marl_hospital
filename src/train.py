import os
import numpy as np
import pandas as pd

from src.hospital_env import HospitalEnv, load_patient_pool, EQUITY_GROUPS
from src.agents import MetaAgent, LocalAgent
from src.controllers import FCFSController, ESIProportionalController
from src.constraints import ConstraintFilter, calculate_equity_penalty

MACRO_PERIOD = 16  # meta-agent / controller re-decides every 16 steps (~4h)
EPISODES_PER_SEED = 250
SEEDS = [0, 1, 2]
LOCAL_DEPTS = ['ER', 'ICU', 'Ward']


def run_episode(pool, controller_or_weights, action_fn, lambda_weight=0.0,
                 safety_filter=None, env_seed=0, daily_arrivals=300,
                 local_agents_and_weights=None):
    """
    action_fn(controller_or_weights, state) -> raw 3-vector nurse allocation
    from the Meta-Agent (or a rule-based controller for the flat baselines).

    local_agents_and_weights: optional dict {'ER': (LocalAgent, weights),
    'ICU': (...), 'Ward': (...)} -- when provided, each Local Agent re-acts
    every 15-minute step from its own department-local observation, giving
    the full hierarchical (Meta-Agent + Local Agents) architecture. When
    None, the environment falls back to the flat single-agent admission
    rule (used by the uncoordinated baselines).

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

        local_actions = None
        if local_agents_and_weights is not None:
            local_actions = {
                dept: agent.action(env.local_state(dept), weights=weights)
                for dept, (agent, weights) in local_agents_and_weights.items()
            }

        state, reward, done, info = env.step(action, local_actions)
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
                 use_safety_filter=True, use_local_agents=True):
    """
    Trains the full hierarchical stack: Global Meta-Agent (macro nurse
    allocation, every ~4h) + decentralized Local Agents (ER/ICU/Ward,
    micro admission/triage every 15 min) + optional CMDP Safety Filter.

    Training is coordinate-ascent over agents within each episode: the
    Meta-Agent takes one antithetic-ES step (all Local Agents held fixed),
    then each Local Agent takes its own ES step in turn (Meta-Agent and the
    other Local Agents held fixed). Every ES rollout for a given episode
    reuses that episode's env_seed, so perturbations are compared on a
    matched arrival stream. This is "hierarchical decentralized execution":
    each agent only ever sees its own local/global state to act, and the
    agents are coupled only indirectly, through the shared environment and
    the single shared episode reward used as every agent's fitness signal.
    """
    print(f"\n--- Running {name} (lambda={lambda_weight}, safety_filter={use_safety_filter}, "
          f"local_agents={use_local_agents}) ---")
    filt = ConstraintFilter() if use_safety_filter else None
    rows = []
    for seed in seeds:
        meta_agent = MetaAgent(state_dim=state_dim, total_nurses=total_nurses, seed=seed)
        local_agents = None
        if use_local_agents:
            local_agents = {
                dept: LocalAgent(dept, seed=seed * 10 + i)
                for i, dept in enumerate(LOCAL_DEPTS)
            }

        for ep in range(EPISODES_PER_SEED):
            ep_seed = seed * 1000 + ep
            local_aw = ({d: (a, a.weights) for d, a in local_agents.items()}
                        if local_agents is not None else None)

            # 1. evaluate current joint policy (this is what gets logged)
            m = run_episode(pool, (meta_agent, meta_agent.weights), meta_agent_action_fn,
                             lambda_weight=lambda_weight, safety_filter=filt,
                             env_seed=ep_seed, daily_arrivals=daily_arrivals,
                             local_agents_and_weights=local_aw)
            m.update({'method': name, 'seed': seed, 'episode': ep})
            rows.append(m)

            # 2. Meta-Agent ES step (Local Agents held fixed at current weights)
            def meta_episode_return(weights, _local_aw=local_aw, _seed=ep_seed):
                res = run_episode(pool, (meta_agent, weights), meta_agent_action_fn,
                                   lambda_weight=lambda_weight, safety_filter=filt,
                                   env_seed=_seed, daily_arrivals=daily_arrivals,
                                   local_agents_and_weights=_local_aw)
                return res['reward']

            meta_agent.es_step(meta_episode_return)

            # 3. each Local Agent's ES step, decentralized: Meta-Agent and
            #    every other Local Agent held fixed, all sharing the same
            #    global episode reward as their fitness signal
            if local_agents is not None:
                for dept, agent in local_agents.items():
                    def local_episode_return(weights, _dept=dept, _seed=ep_seed):
                        aw = {d: (a, weights if d == _dept else a.weights)
                              for d, a in local_agents.items()}
                        res = run_episode(pool, (meta_agent, meta_agent.weights), meta_agent_action_fn,
                                           lambda_weight=lambda_weight, safety_filter=filt,
                                           env_seed=_seed, daily_arrivals=daily_arrivals,
                                           local_agents_and_weights=aw)
                        return res['reward']

                    agent.es_step(local_episode_return)

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

    # All three HC-MARL variants use the full hierarchy (Meta-Agent + Local
    # Agents); they differ only in the safety filter / equity ablation.
    all_results['HC-MARL-NoFilter'] = run_hc_marl(
        'HC-MARL-NoFilter', pool, SEEDS, lambda_weight=0.0, daily_arrivals=daily_arrivals,
        use_safety_filter=False, use_local_agents=True)

    all_results['HC-MARL-Phase1(lambda=0.0)'] = run_hc_marl(
        'HC-MARL-Phase1(lambda=0.0)', pool, SEEDS, lambda_weight=0.0, daily_arrivals=daily_arrivals,
        use_safety_filter=True, use_local_agents=True)

    all_results['HC-MARL-Phase2(lambda=2.5)'] = run_hc_marl(
        'HC-MARL-Phase2(lambda=2.5)', pool, SEEDS, lambda_weight=2.5, daily_arrivals=daily_arrivals,
        use_safety_filter=True, use_local_agents=True)

    for name, df in all_results.items():
        fname = "docs/" + name.replace('(', '_').replace(')', '').replace('=', '').replace('.', '') + "_results.csv"
        df.to_csv(fname, index=False)

    summary = summarize(all_results)
    summary.to_csv("docs/comparison_summary.csv", index=False)
    print("\n================ COMPARATIVE SUMMARY (converged, last 10 ep/seed) ================")
    print(summary.to_string(index=False))
    print("\nSaved per-method logs and docs/comparison_summary.csv")
