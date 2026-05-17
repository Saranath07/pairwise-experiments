"""Cached pairwise-probability oracle.

Backs the §2c LLM-judge experiment. The expensive step (running the Qwen
judge over a full round-robin) happens once and produces:

  P[i, j] : float in [0, 1]  — P(item i beats item j) under the judge.
  C[i, j] : int              — token cost of the (i, j) judge call (debiased).

After that, every algorithm × budget × trial samples Bernoulli(P[i, j]) at
zero compute cost. Token cost per query is C[i, j], summed by the runner.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np


class CachedPairwiseOracle:
    """`Vote(Item, Item)` callable backed by a precomputed P matrix.

    Also tracks total token cost across all draws since `reset()`."""

    def __init__(self, p_matrix: np.ndarray, cost_matrix: Optional[np.ndarray] = None):
        p = np.asarray(p_matrix, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError("p_matrix must be square")
        if not np.all((0.0 <= p) & (p <= 1.0)):
            # Clamp tiny numerical drifts.
            p = np.clip(p, 0.0, 1.0)
        self.P = p
        self.N = p.shape[0]
        if cost_matrix is None:
            cost_matrix = np.zeros_like(p, dtype=np.int64)
        self.C = np.asarray(cost_matrix, dtype=np.int64)
        self._tokens = 0
        self._calls = 0

    def reset(self):
        self._tokens = 0
        self._calls = 0

    @property
    def tokens(self) -> int:
        return int(self._tokens)

    @property
    def calls(self) -> int:
        return int(self._calls)

    def __call__(self, a, b) -> bool:
        i, j = a.idx, b.idx
        p_ij = float(self.P[i, j])
        self._tokens += int(self.C[i, j])
        self._calls += 1
        return random.random() < p_ij


def cached_oracle_from_npz(path: str) -> CachedPairwiseOracle:
    """Load an oracle from `np.savez(path, P=P, C=C)` output."""
    npz = np.load(path)
    return CachedPairwiseOracle(npz["P"], npz.get("C"))
