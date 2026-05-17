"""Thurstonian (probit-link) oracle.

P(i ≻ j) = Phi((theta_i - theta_j) / sigma)

with theta interpreted on the log-BTL scale so a pure-BTL run is well
matched in the limit sigma -> 1/sqrt(8/pi). For our purposes we expose
`sigma` as a free parameter and treat the sweep as a misspecification
stress test."""
import math
import random
import numpy as np


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def thurstone_oracle(theta, sigma=1.0):
    log_theta = np.log(np.maximum(theta, 1e-9))

    def vote(a, b):
        i, j = a.idx, b.idx
        p = _phi((log_theta[i] - log_theta[j]) / sigma)
        return random.random() < p

    return vote
