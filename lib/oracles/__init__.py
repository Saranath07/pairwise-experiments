"""Pairwise oracles for Phase-2b robustness sweeps.

Algorithms in this repo make every probabilistic comparison via
`Vote(scores[i-1], scores[j-1])`. To swap the oracle without touching the
algorithm code, we:

  1. wrap each item's score in an `Item` proxy that carries (index, theta);
  2. monkey-patch `lib.algorithms.common.Vote` (and the wrapping `Vote`s in
     `baselines.py` / `robust_fw.py` / `wisdom_variants.py`) for the duration
     of a trial so they consult the active oracle on those proxies.

Use:

    with use_oracle(cyclic_oracle(theta, rho, c)):
        run_robust_fw(n, scores_proxy, B, true_top, true_ranks)

`scores_proxy` is whatever `make_proxy_scores(theta)` returned.
"""
from contextlib import contextmanager
import random

from lib.algorithms import common as _common
from lib.algorithms import baselines as _baselines
from lib.algorithms import robust_fw as _robust_fw
from lib.algorithms import wisdom_variants as _variants


class Item:
    """Score proxy carrying (index, theta_value).

    Indexing matches the 0-based item index, exactly as the algorithm code
    passes `scores[i-1]` (i is 1-based item id, so idx = i-1)."""
    __slots__ = ("idx", "theta")

    def __init__(self, idx, theta):
        self.idx = idx
        self.theta = float(theta)

    def __repr__(self):
        return f"Item(idx={self.idx}, theta={self.theta:.3f})"


def make_proxy_scores(theta):
    return [Item(i, theta[i]) for i in range(len(theta))]


# ---------------------------------------------------------------------------
# Vote dispatcher: routes each Vote(a, b) call through the active oracle.
# ---------------------------------------------------------------------------

_active_oracle = None


def _vote_proxy(a, b):
    """Drop-in replacement for `common.Vote(a, b)` that handles either
    Item proxies (preferred, allows non-BTL oracles) or plain floats
    (default BTL fallback)."""
    if _active_oracle is None or not isinstance(a, Item):
        ai = a.theta if isinstance(a, Item) else a
        bj = b.theta if isinstance(b, Item) else b
        if ai + bj <= 0:
            return random.random() < 0.5
        return random.uniform(0, ai + bj) < ai
    return _active_oracle(a, b)


@contextmanager
def use_oracle(oracle_fn):
    """Activate `oracle_fn(item_a, item_b) -> bool` (True iff a beats b)
    for the duration of the with-block.

    Patches `Vote` in every module that defines a wrapper, so all algorithm
    code paths route through the oracle."""
    global _active_oracle
    saved = _active_oracle
    _active_oracle = oracle_fn

    saved_common = _common.Vote
    saved_baselines = _baselines.Vote
    saved_robust = _robust_fw.Vote
    saved_variants = _variants.Vote

    _common.Vote = _vote_proxy
    _baselines.Vote = _vote_proxy
    _robust_fw.Vote = _vote_proxy
    _variants.Vote = _vote_proxy

    try:
        yield
    finally:
        _active_oracle = saved
        _common.Vote = saved_common
        _baselines.Vote = saved_baselines
        _robust_fw.Vote = saved_robust
        _variants.Vote = saved_variants


@contextmanager
def use_proxy_dispatch():
    """Activate proxy-aware Vote without an oracle (BTL fallback for proxies).
    Useful when running default-BTL trials with proxy scores so the same
    algorithm code path is exercised in both BTL and non-BTL conditions."""
    with use_oracle(None):  # type: ignore
        yield
