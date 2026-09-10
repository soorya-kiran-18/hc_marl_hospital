import numpy as np


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

    def es_step(self, episode_return_fn, sigma=0.3, lr=1.0, n_directions=8):
        """
        episode_return_fn(weights) -> scalar return for running one full
        episode with the given weight matrix. Averages several antithetic
        perturbation directions per update to reduce the gradient-estimate
        variance inherent to a single noisy stochastic rollout.
        Returns the mean (r_plus, r_minus) across directions for logging.
        """
        grad = np.zeros_like(self.weights)
        r_plus_list, r_minus_list = [], []
        for _ in range(n_directions):
            eps = self.rng.normal(0, 1, size=self.weights.shape)
            r_plus = episode_return_fn(self.weights + sigma * eps)
            r_minus = episode_return_fn(self.weights - sigma * eps)
            grad += (r_plus - r_minus) * eps
            r_plus_list.append(r_plus)
            r_minus_list.append(r_minus)
        self.weights += (lr / (2 * sigma * n_directions)) * grad
        return np.mean(r_plus_list), np.mean(r_minus_list)
