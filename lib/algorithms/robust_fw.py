"""
Proposed algorithm: Robust Tournament + Winner-Focused Frank-Wolfe.

Phase 1:  Best-of-3 single-elimination bracket on all N items (~3N queries).
          Each match plays 3 games; winner is the side with >=2 wins.
Phase 2:  Winner-focused Frank-Wolfe on all N items; the FW objective picks
          the pair (i,j) that maximizes uncertainty/(gap^2). Argmax is
          restricted to pairs inside the current top-M by Elo (M = ceil(N^(1/4))).

Instrumented for three metrics per query step:
  - acc: 1 if predicted winner == true top-1
  - pf : reported rank of the true winner
  - ct : true rank of the reported winner

For the rank-based metrics we convert Elo scores to a full ranking via argsort.
"""
import math
import random
import numpy as np

from lib.algorithms import common as _common
from lib.algorithms.common import get_ranks


def Vote(a, b):
    """Delegating wrapper so monkey-patching `lib.algorithms.common.Vote`
    (e.g., by `lib.oracles`) reaches every call inside this module."""
    return _common.Vote(a, b)

ELO_SCALE = 400.0
LN10 = math.log(10)


def _elo_update(elo, winner, loser, k=32.0):
    exp = 1.0 / (1.0 + 10.0 ** ((elo[loser] - elo[winner]) / ELO_SCALE))
    elo[winner] += k * (1.0 - exp)
    elo[loser] -= k * (1.0 - exp)


def _mu_dot(gap):
    x = gap * LN10 / ELO_SCALE
    if abs(x) > 500: return 0.0
    ex = math.exp(-x)
    return ex / (1.0 + ex) ** 2


def _btl_gap(gap):
    return abs(gap * LN10 / ELO_SCALE)


def _play(scores, i, j):
    if Vote(scores[i - 1], scores[j - 1]): return i, j
    return j, i


# -- Frank-Wolfe (vectorized) ------------------------------------------------

def _precompute(pool, elo):
    pool_sorted = sorted(pool)
    m = len(pool_sorted)
    idx = {item: k for k, item in enumerate(pool_sorted)}
    pairs = [(pool_sorted[a], pool_sorted[b]) for a in range(m) for b in range(a + 1, m)]
    E = len(pairs)
    ii = np.empty(E, dtype=np.int64); jj = np.empty(E, dtype=np.int64)
    mus = np.empty(E); gap2 = np.empty(E)
    for k, (i, j) in enumerate(pairs):
        ii[k] = idx[i]; jj[k] = idx[j]
        mus[k] = _mu_dot(elo[i] - elo[j])
        g = max(_btl_gap(elo[i] - elo[j]), 1e-6)
        gap2[k] = g * g
    return pool_sorted, pairs, ii, jj, mus, gap2, m


def _phi(lam, ii, jj, mus, gap2, m):
    w = lam * mus
    H = np.zeros((m, m))
    np.add.at(H, (ii, ii),  w); np.add.at(H, (jj, jj),  w)
    np.add.at(H, (ii, jj), -w); np.add.at(H, (jj, ii), -w)
    H += 1e-6 * np.eye(m)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.inv(H + 1e-4 * np.eye(m))
    unc2 = H_inv[ii, ii] + H_inv[jj, jj] - 2.0 * H_inv[ii, jj]
    np.maximum(unc2, 0.0, out=unc2)
    return unc2 / gap2


def _fw_winner_focused(pool, elo, n_iters=40, M=3):
    pool_sorted, pairs, ii, jj, mus, gap2, m = _precompute(pool, elo)
    E = len(pairs)
    if E == 0: return {}
    lam = np.full(E, 1.0 / E)
    S = set(sorted(pool_sorted, key=lambda x: elo[x], reverse=True)[:M])
    mask = np.array([(i in S and j in S) for (i, j) in pairs], dtype=bool)
    if not mask.any(): mask = np.ones(E, dtype=bool)
    s_idx = np.nonzero(mask)[0]
    for t in range(n_iters):
        step = 2.0 / (t + 2)
        phi = _phi(lam, ii, jj, mus, gap2, m)
        best = int(s_idx[np.argmax(phi[s_idx])])
        lam *= (1.0 - step); lam[best] += step
    return {pairs[k]: float(lam[k]) for k in range(E)}


def _sample(lam):
    pairs = list(lam.keys())
    w = np.array([lam[p] for p in pairs], dtype=float)
    w /= w.sum()
    return pairs[np.random.choice(len(pairs), p=w)]


# -- Phase 1: robust tournament (best-of-3) ----------------------------------

def _robust_bracket(n, scores, elo, m_games=3, majority=2,
                    on_step=None, budget_cap=None):
    active = list(range(1, n + 1))
    random.shuffle(active)
    used = 0
    next_pow2 = 1 << (n - 1).bit_length()
    n_byes = next_pow2 - n
    byes = active[:n_byes] if n_byes > 0 else []
    remaining = active[n_byes:] if n_byes > 0 else active

    def match(p, q):
        nonlocal used
        wp = wq = 0
        for _ in range(m_games):
            if budget_cap is not None and used >= budget_cap: break
            w, l = _play(scores, p, q)
            _elo_update(elo, w, l)
            used += 1
            if w == p: wp += 1
            else: wq += 1
            if on_step: on_step(used)
        return p if wp >= majority else q

    nxt = list(byes)
    for i in range(0, len(remaining), 2):
        if budget_cap is not None and used >= budget_cap: break
        if i + 1 >= len(remaining):
            nxt.append(remaining[i]); continue
        nxt.append(match(remaining[i], remaining[i + 1]))
    active = nxt

    while len(active) > 1:
        if budget_cap is not None and used >= budget_cap: break
        random.shuffle(active)
        nxt = []
        for i in range(0, len(active), 2):
            if budget_cap is not None and used >= budget_cap: break
            if i + 1 >= len(active):
                nxt.append(active[i]); continue
            nxt.append(match(active[i], active[i + 1]))
        active = nxt

    return used


# -- Runner ------------------------------------------------------------------

def run_robust_fw(n, scores, total_budget, true_top_idx, true_ranks,
                  M=None, fw_iters=40, batch=5, m_games=3):
    if M is None:
        M = max(3, int(math.ceil(n ** 0.25)))

    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    true_top_1 = true_top_idx + 1
    elo = np.ones(n + 1) * 1500.0
    all_items = list(range(1, n + 1))

    def record(step_idx):
        if step_idx <= 0 or step_idx > total_budget: return
        # ranking from elo (descending)
        elo_vec = np.array([elo[i] for i in range(1, n + 1)])
        ranks = get_ranks(elo_vec)
        winner = int(np.argmax(elo_vec)) + 1
        idx = step_idx - 1
        acc[idx] = 1 if winner == true_top_1 else 0
        pf[idx] = ranks[true_top_idx]
        ct[idx] = true_ranks[winner - 1]

    used = _robust_bracket(n, scores, elo, m_games=m_games,
                           on_step=record, budget_cap=total_budget)

    lam = {}; t_since_rebuild = 0
    while used < total_budget:
        if t_since_rebuild % batch == 0 or not lam:
            lam = _fw_winner_focused(all_items, elo, fw_iters, M=M)
        if not lam: break
        p1, p2 = _sample(lam)
        w, l = _play(scores, p1, p2)
        _elo_update(elo, w, l)
        used += 1; t_since_rebuild += 1
        record(used)

    # pad any tail (e.g., if budget ran out mid-phase-2)
    for t in range(1, total_budget):
        if acc[t] == 0 and pf[t] == n and ct[t] == n:
            acc[t], pf[t], ct[t] = acc[t - 1], pf[t - 1], ct[t - 1]

    return acc, pf, ct
