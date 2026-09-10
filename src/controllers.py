import numpy as np


def estimate_department_mix(esi_array):
    """Fraction of patients routed to ICU / Ward / ER given ESI codes."""
    esi_array = np.asarray(esi_array)
    n = len(esi_array)
    icu_share = np.mean(esi_array <= 2) if n else 1 / 3
    ward_share = np.mean(esi_array == 3) if n else 1 / 3
    er_share = np.mean(esi_array >= 4) if n else 1 / 3
    return np.array([er_share, icu_share, ward_share])


class FCFSController:
    """
    First-Come-First-Served / static staffing baseline: floating nurses are
    split in fixed proportion to each department's *base* permanent staff,
    with no responsiveness to real-time acuity or congestion.
    """

    def __init__(self, base_nurses, total_float_nurses):
        base = np.array([base_nurses['ER'], base_nurses['ICU'], base_nurses['Ward']], dtype=float)
        self.allocation = base / base.sum() * total_float_nurses

    def action(self, state):
        return self.allocation


class ESIProportionalController:
    """
    Acuity-first / ESI-proportional baseline: floating nurses are split in
    proportion to the historical mix of patient acuity (how many patients
    need ICU vs Ward vs ER-only care), regardless of current queue state.
    """

    def __init__(self, esi_array, total_float_nurses):
        er_share, icu_share, ward_share = estimate_department_mix(esi_array)
        self.allocation = np.array([er_share, icu_share, ward_share]) * total_float_nurses

    def action(self, state):
        return self.allocation
