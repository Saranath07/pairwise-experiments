"""Cyclic / rock-paper-scissors perturbation of BTL.

Per NEXT_PLAN §2b:
    P(i ≻ j) = sigma(theta_i - theta_j) + rho * c_{ij}
where `c` is a fixed skew-symmetric matrix with entries in [-1, 1] and
P is clipped to [0, 1]. theta is normalised so sigma(theta_i - theta_j)
maps to standard BTL (here we use the multiplicative convention used in
`common.Vote`, converted to the additive `delta = log(theta_i) - log(theta_j)`)."""
import math
import random
import numpy as np


def _sigma(x):
    if x > 50: return 1.0
    if x < -50: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def random_skew(n, seed=0):
    """Generate a fixed skew-symmetric perturbation matrix with entries
    drawn uniformly in [-1, 1] above the diagonal and mirrored with sign flip."""
    rng = np.random.default_rng(seed)
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.uniform(-1.0, 1.0)
            c[i, j] = v
            c[j, i] = -v
    return c


def cyclic_oracle(theta, rho, c):
    """Return vote(a, b). theta is the underlying (multiplicative) BTL score."""
    log_theta = np.log(np.maximum(theta, 1e-9))

    def vote(a, b):
        i, j = a.idx, b.idx
        p = _sigma(log_theta[i] - log_theta[j]) + rho * c[i, j]
        p = max(0.0, min(1.0, p))
        return random.random() < p

    return vote
