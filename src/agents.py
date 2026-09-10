import numpy as np

class MetaAgent:
    """
    Global Meta-Agent: Operates on a macro-time horizon to distribute floating nurses.
    """
    def __init__(self, state_dim, total_nurses):
        self.state_dim = state_dim
        self.total_nurses = total_nurses
        # Simplified policy weights for NumPy-based execution
        self.weights = np.random.randn(state_dim, 3) * 0.1 

    def get_macro_action(self, global_state):
        # Outputs a distribution of nurses to [ER, ICU, Ward]
        logits = np.dot(global_state, self.weights)
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
        
        # Allocate finite nurses based on probabilities
        allocation = np.floor(probs * self.total_nurses).astype(int)
        
        # Handle rounding remainders
        remainder = self.total_nurses - np.sum(allocation)
        if remainder > 0:
            allocation[np.argmax(probs)] += remainder
            
        return {'ER': allocation[0], 'ICU': allocation[1], 'Ward': allocation[2]}
        
    def update_policy(self, gradients):
        # Basic gradient ascent step for the Meta-Agent
        self.weights += 0.01 * gradients

class LocalAgent:
    """
    Local Department Agent: Manages micro-actions (triage queues) every 15 minutes.
    """
    def __init__(self, dept_name):
        self.dept_name = dept_name
        self.q_table = {} # Simplified Q-learning for local discrete state-actions

    def get_micro_action(self, local_state, valid_actions):
        state_key = str(local_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in valid_actions}
            
        # Epsilon-greedy action selection
        if np.random.rand() < 0.1:
            return np.random.choice(valid_actions)
        else:
            return max(self.q_table[state_key], key=self.q_table[state_key].get)