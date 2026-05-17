"""Reference BTL oracle.

Used as a sanity check: running with this oracle should reproduce the
default BTL behaviour of `common.Vote`."""
import random
import math


def _logit(theta_i, theta_j):
    # Standard BTL on theta values directly, where P(i > j) = theta_i / (theta_i + theta_j).
    s = theta_i + theta_j
    if s <= 0:
        return 0.5
    return theta_i / s


def btl_oracle(theta):
    def vote(a, b):
        return random.random() < _logit(a.theta, b.theta)
    return vote
