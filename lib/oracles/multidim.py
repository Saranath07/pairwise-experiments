"""Multidimensional preference oracle.

Per NEXT_PLAN §2b: two latent dimensions (e.g. coding skill, writing skill);
the oracle samples a dimension uniformly per query and returns BTL on that
dimension. Ground truth is defined by the caller (typically Borda winner over
both dims)."""
import math
import random
import numpy as np


def _sigma(x):
    if x > 50: return 1.0
    if x < -50: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def multidim_oracle(theta_dims, weights=None):
    """theta_dims: shape (n, D). Each query samples a dim with probability
    `weights[d]` (default uniform) and returns BTL on the log-scale of that dim."""
    theta_dims = np.asarray(theta_dims, dtype=float)
    n, D = theta_dims.shape
    if weights is None:
        weights = np.ones(D) / D
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    log_theta = np.log(np.maximum(theta_dims, 1e-9))

    def vote(a, b):
        i, j = a.idx, b.idx
        d = np.random.choice(D, p=weights)
        p = _sigma(log_theta[i, d] - log_theta[j, d])
        return random.random() < p

    return vote


def borda_winner(theta_dims):
    """Aggregate ground-truth winner under uniform Borda over dims:
    rank within each dim, sum the ranks (lower is better), pick argmin."""
    theta_dims = np.asarray(theta_dims, dtype=float)
    n, D = theta_dims.shape
    rank_sum = np.zeros(n)
    for d in range(D):
        order = np.argsort(-theta_dims[:, d])
        rank = np.zeros(n)
        for r, idx in enumerate(order):
            rank[idx] = r + 1
        rank_sum += rank
    return int(np.argmin(rank_sum))
