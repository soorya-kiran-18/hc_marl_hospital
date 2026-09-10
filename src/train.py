import numpy as np
import pandas as pd
from agents import MetaAgent, LocalAgent
from constraints import ConstraintFilter, calculate_equity_penalty
import os

def run_training_loop(episodes=100, lambda_weight=0.0):
    print(f"Starting HC-MARL Training | Phase Lambda: {lambda_weight}")
    
    meta_agent = MetaAgent(state_dim=5, total_nurses=15)
    er_agent = LocalAgent('ER')
    safety_filter = ConstraintFilter()
    
    metrics = {'episode': [], 'reward': [], 'safety_violations': []}
    
    for ep in range(episodes):
        # Simulated environment reset (To be connected to Soorya's hospital_env.py)
        global_state = np.random.rand(5) 
        current_patients = {'ER': 12, 'ICU': 3, 'Ward': 20}
        
        # 1. Meta-Agent proposes macro-budget
        proposed_budget = meta_agent.get_macro_action(global_state)
        
        # 2. CMDP Safety Filter intercepts
        is_safe = safety_filter.verify_action_safety(proposed_budget, current_patients)
        
        violations = 0
        if not is_safe:
            violations += 1
            # Fallback to safe baseline if RL explores dangerously
            proposed_budget = {'ER': 4, 'ICU': 2, 'Ward': 9} 
            
        # 3. Simulate wait times for equity penalty calculation
        mock_wait_times = {
            'Private': np.random.normal(30, 5, 10),
            'Medicaid': np.random.normal(35, 10, 10)
        }
        
        # 4. Global Reward Calculation
        base_reward = 100 - (violations * 50) 
        equity_penalty = calculate_equity_penalty(mock_wait_times, lambda_weight)
        final_reward = base_reward - equity_penalty
        
        metrics['episode'].append(ep)
        metrics['reward'].append(final_reward)
        metrics['safety_violations'].append(violations)
        
    print(f"Training Complete. Average Reward: {np.mean(metrics['reward']):.2f}")
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    
    # Phase 1: Train with Safety Constraints Only (Lambda = 0)
    print("\n--- Running Phase 1 (Hard Constraints Only) ---")
    phase1_metrics = run_training_loop(episodes=500, lambda_weight=0.0)
    phase1_metrics.to_csv("docs/phase1_results.csv", index=False)
    
    # Phase 2: Train with Safety + Equity Penalty (Lambda > 0)
    print("\n--- Running Phase 2 (Hard Constraints + Soft Equity Penalty) ---")
    phase2_metrics = run_training_loop(episodes=500, lambda_weight=2.5)
    phase2_metrics.to_csv("docs/phase2_results.csv", index=False)