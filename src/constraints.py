import numpy as np

class ConstraintFilter:
    def __init__(self):
        # Maximum allowed patient-to-nurse ratios based on standard clinical guidelines
        self.max_ratios = {'ER': 4.0, 'ICU': 2.0, 'Ward': 6.0}

    def verify_action_safety(self, action_budget, current_patients):
        """
        Hard Constraint: Evaluates if a proposed nurse allocation violates safety ratios.
        action_budget: dict of proposed nurse counts per department
        current_patients: dict of current patient census per department
        """
        for dept in ['ER', 'ICU', 'Ward']:
            nurses = action_budget.get(dept, 1)
            patients = current_patients.get(dept, 0)
            
            if nurses <= 0:
                return False
                
            ratio = patients / nurses
            if ratio > self.max_ratios[dept]:
                return False  # Safety violation triggered
                
        return True  # Action is safe

def calculate_equity_penalty(wait_times_by_group, lambda_weight):
    """
    Soft Penalty: Calculates the variance in wait times across demographic groups.
    R_global equation: penalizes deviations from the global average wait time.
    """
    if not wait_times_by_group:
        return 0.0
        
    all_wait_times = []
    for times in wait_times_by_group.values():
        all_wait_times.extend(times)
        
    if len(all_wait_times) == 0:
        return 0.0
        
    t_all_mean = np.mean(all_wait_times)
    penalty = 0.0
    
    for group, times in wait_times_by_group.items():
        if len(times) > 0:
            t_d_mean = np.mean(times)
            penalty += abs(t_d_mean - t_all_mean)
            
    return lambda_weight * penalty