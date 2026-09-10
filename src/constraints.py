import numpy as np

EQUITY_GROUPS = ['Private', 'Medicaid', 'Medicare', 'SelfPay', 'Other/Unknown']


class ConstraintFilter:
    """
    Hard CMDP safety constraint: a proposed nurse allocation is rejected if
    it would push any department's current patient-to-nurse ratio beyond the
    clinical limit. `nurses` and `census` are the *total* (base + floating)
    nurse counts and current patient counts per department, as returned by
    HospitalEnv.nurses_for(...) and HospitalEnv.current_census().
    """

    def __init__(self):
        self.max_ratios = {'ER': 4.0, 'ICU': 2.0, 'Ward': 6.0}

    def verify_action_safety(self, nurses, census):
        for dept in ['ER', 'ICU', 'Ward']:
            n = nurses.get(dept, 1)
            p = census.get(dept, 0)
            if n <= 0:
                return False
            if p / n > self.max_ratios[dept]:
                return False
        return True


def calculate_equity_penalty(wait_times_by_group, lambda_weight):
    """
    Soft penalty term from the global reward:
        lambda * sum_d |E[T_d] - E[T_all]|
    wait_times_by_group: dict of equity_group -> list of observed wait times
    (in simulation steps or minutes; caller must be consistent).
    """
    all_waits = [w for times in wait_times_by_group.values() for w in times]
    if not all_waits:
        return 0.0

    t_all_mean = np.mean(all_waits)
    penalty = 0.0
    for times in wait_times_by_group.values():
        if len(times) > 0:
            penalty += abs(np.mean(times) - t_all_mean)

    return lambda_weight * penalty
