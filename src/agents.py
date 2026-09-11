import numpy as np


def _antithetic_es_step(weights, rng, episode_return_fn, sigma, lr, n_directions, weight_clip):
    """
    Shared antithetic-ES update (Salimans et al., 2017) used by both
    MetaAgent and LocalAgent.

    Returns are rank/z-score normalized across the full batch of 2*n
    perturbations before computing the weighted-epsilon gradient estimate.
    This fitness-shaping step is standard in ES and is not optional here:
    without it, a raw (r_plus - r_minus) gradient's scale tracks the raw
    reward's variance, and once multiple agents are being co-trained the
    reward variance is high enough that un-normalized updates can blow the
    weights up to extreme magnitudes in a handful of iterations, collapsing
    the softmax into a degenerate corner policy (observed empirically here
    as a Meta-Agent that starved the ER of nurses). Weights are also
    hard-clipped after the update as a defense-in-depth bound, since this
    policy controls a safety-relevant resource allocation.
    """
    epsilons = [rng.normal(0, 1, size=weights.shape) for _ in range(n_directions)]
    r_plus = np.array([episode_return_fn(weights + sigma * eps) for eps in epsilons])
    r_minus = np.array([episode_return_fn(weights - sigma * eps) for eps in epsilons])

    all_r = np.concatenate([r_plus, r_minus])
    std = all_r.std()
    if std < 1e-8:
        std = 1.0
    mean = all_r.mean()
    r_plus_n = (r_plus - mean) / std
    r_minus_n = (r_minus - mean) / std

    grad = np.zeros_like(weights)
    for rp, rm, eps in zip(r_plus_n, r_minus_n, epsilons):
        grad += (rp - rm) * eps

    new_weights = weights + (lr / (2 * sigma * n_directions)) * grad
    np.clip(new_weights, -weight_clip, weight_clip, out=new_weights)
    return new_weights, r_plus.mean(), r_minus.mean()


class LocalAgent:
    """
    Decentralized Local Agent for one hospital department (ER, ICU, or Ward).

    Per the HC-MARL architecture, Local Agents manage micro-level triage and
    admission decisions every 15-minute simulation step, operating strictly
    within the nurse-derived capacity ("Verified Allocations") the Meta-Agent
    and Safety Filter hand down. They are not coupled to each other directly
    (no peer-to-peer communication) -- coordination happens only indirectly,
    through the shared simulation environment and the Meta-Agent's finite
    nurse budget, matching the paper's "hierarchical decentralized execution"
    paradigm.

    Action = equity_priority in (0, 1) via a sigmoid: a tie-break rule among
    admission candidates of equal clinical urgency (ESI). > 0.5 admits the
    longest-waiting patient first (closes the wait-time equity gap); <= 0.5
    keeps shorter-wait patients first. Clinical severity (ESI) always takes
    priority over this term -- equity only breaks ties within the same
    acuity level, so learned fairness behavior can never override immediate
    clinical need.

    Admission *volume* is deliberately not a free/learned parameter: the
    environment always admits as many candidates as the verified capacity
    allows (never more -- that hard cap is what makes ICU/Ward overcrowding
    via admission structurally impossible). Since admitting slower than
    "as many as safely fit" has no offsetting benefit under this reward --
    it only lets patients board longer, worsening both throughput and ER
    congestion -- that volume decision is a strictly dominated strategy
    with nothing to learn. Making it structural (rather than leaving it to
    a noisy per-episode Evolution Strategies estimate) removes a failure
    mode we hit empirically: an under-converged admission-rate parameter
    stuck near 50% choked ER outflow and drove violations *up*, not down.
    The one genuine trade-off left for the Local Agent to learn is fairness
    -- exactly where the paper's soft equity objective lives.

    Learns via the same antithetic Evolution Strategies method as the
    Meta-Agent, with a smaller perturbation budget matching its
    lower-dimensional policy.
    """

    def __init__(self, dept_name, state_dim=3, seed=None):
        self.dept_name = dept_name
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.1, size=(state_dim, 1))

    def action(self, local_state, weights=None):
        w = self.weights if weights is None else weights
        z = np.asarray(local_state) @ w
        return float(1.0 / (1.0 + np.exp(-z[0])))  # sigmoid -> equity_priority in (0,1)

    def es_step(self, episode_return_fn, sigma=0.2, lr=0.5, n_directions=4, weight_clip=8.0):
        """
        Same normalized antithetic-ES update as MetaAgent.es_step (see
        _antithetic_es_step), sized down for this agent's 3x1 weight matrix.
        """
        self.weights, r_plus_mean, r_minus_mean = _antithetic_es_step(
            self.weights, self.rng, episode_return_fn, sigma, lr, n_directions, weight_clip)
        return r_plus_mean, r_minus_mean


class MetaAgent:
    """
    Global Meta-Agent: a linear-softmax policy over the hospital's global
    state that outputs a floating-nurse allocation across [ER, ICU, Ward].

    Learning uses antithetic Evolution Strategies (Salimans et al., 2017):
    two perturbed copies of the weights are evaluated on full episodes and
    the weights are nudged toward whichever perturbation scored higher. This
    is a legitimate black-box policy-search method that needs no
    backpropagation, matching the project's plain-NumPy implementation.
    """

    def __init__(self, state_dim, total_nurses, seed=None):
        self.state_dim = state_dim
        self.total_nurses = total_nurses
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.1, size=(state_dim, 3))

    def action(self, state, weights=None):
        w = self.weights if weights is None else weights
        logits = np.asarray(state) @ w
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return probs * self.total_nurses

    def es_step(self, episode_return_fn, sigma=0.3, lr=1.0, n_directions=16, weight_clip=8.0):
        """
        episode_return_fn(weights) -> scalar return for running one full
        episode with the given weight matrix. Averages several antithetic
        perturbation directions per update, with returns normalized before
        the gradient estimate (see _antithetic_es_step) -- required for
        stability once this agent is co-trained alongside Local Agents,
        where reward variance is high enough to blow up an un-normalized
        update. n_directions=16 (rather than the single-agent system's
        original 8) empirically converges far more reliably across random
        seeds once Local Agents are in the loop -- with 8 directions the
        gradient estimate was noisy enough to occasionally collapse into a
        degenerate "give one department all the float nurses forever"
        policy regardless of state. Returns the mean (r_plus, r_minus)
        across directions for logging.
        """
        self.weights, r_plus_mean, r_minus_mean = _antithetic_es_step(
            self.weights, self.rng, episode_return_fn, sigma, lr, n_directions, weight_clip)
        return r_plus_mean, r_minus_mean
