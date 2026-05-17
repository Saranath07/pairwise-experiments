"""WiSDoM ablation variants for Phase 2a.

Five lines on the headline ablation plot:
  1. phase1_only       — bracket then output bracket winner; no Phase-2 spend.
  2. m2_verify         — Phase 1 + Bernoulli majority on (champion, runner-up).
  3. wisdom_lite       — Phase 1 + uniform random pairs on top-M.
  4. greedy_on_C       — Phase 1 + PARWiS-style King-of-the-Hill restricted to top-M.
  5. (full WiSDoM is `run_robust_fw` from robust_fw.py — kept as the 5th line.)

Each runner has signature
    run_<name>(n, scores, total_budget, true_top_idx, true_ranks) -> (acc, pf, ct)
matching the convention in baselines.py.

All routes go through `lib.algorithms.common.Vote`, so monkey-patching that
symbol (see `lib.oracles`) automatically swaps the oracle for every variant.
"""
import math
import random
import numpy as np

from lib.algorithms.common import Rank_Centrality, get_ranking, power_iter, get_ranks
from lib.algorithms.robust_fw import _robust_bracket, _elo_update, Vote


# ---------------------------------------------------------------------------
# Recording helper: ACC/PF/CT from an Elo vector at a given step index.
# ---------------------------------------------------------------------------

def _record_from_elo(elo, n, true_top_idx, true_ranks, step_idx, acc, pf, ct):
    if step_idx <= 0 or step_idx > len(acc):
        return
    elo_vec = np.array([elo[i] for i in range(1, n + 1)])
    ranks = get_ranks(elo_vec)
    winner = int(np.argmax(elo_vec)) + 1
    idx = step_idx - 1
    acc[idx] = 1 if winner == (true_top_idx + 1) else 0
    pf[idx] = ranks[true_top_idx]
    ct[idx] = true_ranks[winner - 1]


def _pad_tail(acc, pf, ct, n):
    """Forward-fill any all-zero / sentinel rows after the last recorded step."""
    for t in range(1, len(acc)):
        if acc[t] == 0 and pf[t] == n and ct[t] == n:
            acc[t], pf[t], ct[t] = acc[t - 1], pf[t - 1], ct[t - 1]


def _bracket_runner_up(elo, champion):
    """Pick the strongest non-champion item by current Elo (stable, deterministic)."""
    elo_vec = np.array(elo)
    elo_vec[champion] = -np.inf
    elo_vec[0] = -np.inf  # dummy slot
    return int(np.argmax(elo_vec))


# ---------------------------------------------------------------------------
# Variant 1: Phase 1 only (bracket, no Phase-2 budget).
# ---------------------------------------------------------------------------

def run_phase1_only(n, scores, total_budget, true_top_idx, true_ranks, m_games=3):
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    elo = np.ones(n + 1) * 1500.0

    def record(step_idx):
        _record_from_elo(elo, n, true_top_idx, true_ranks, step_idx, acc, pf, ct)

    used = _robust_bracket(n, scores, elo, m_games=m_games,
                           on_step=record, budget_cap=total_budget)
    # No Phase 2: forward-fill remaining steps with the bracket prediction.
    if used < total_budget:
        record(used or 1)
        last = max(used - 1, 0)
        for t in range(used, total_budget):
            acc[t] = acc[last]
            pf[t] = pf[last]
            ct[t] = ct[last]

    _pad_tail(acc, pf, ct, n)
    return acc, pf, ct


# ---------------------------------------------------------------------------
# Variant 2: M=2 verification — Bernoulli majority on (champion, runner-up).
# ---------------------------------------------------------------------------

def run_m2_verify(n, scores, total_budget, true_top_idx, true_ranks, m_games=3):
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    elo = np.ones(n + 1) * 1500.0

    def record(step_idx):
        _record_from_elo(elo, n, true_top_idx, true_ranks, step_idx, acc, pf, ct)

    used = _robust_bracket(n, scores, elo, m_games=m_games,
                           on_step=record, budget_cap=total_budget)

    # Phase 2 on (champion, runner-up): keep voting, update Elo each step.
    if used < total_budget:
        elo_vec_now = np.array([elo[i] for i in range(1, n + 1)])
        champion = int(np.argmax(elo_vec_now)) + 1  # 1-indexed
        runner = _bracket_runner_up(elo, champion)
        if runner == champion or runner == 0:
            others = [i for i in range(1, n + 1) if i != champion]
            runner = random.choice(others) if others else champion
        while used < total_budget:
            if Vote(scores[champion - 1], scores[runner - 1]):
                _elo_update(elo, champion, runner)
            else:
                _elo_update(elo, runner, champion)
            used += 1
            record(used)

    _pad_tail(acc, pf, ct, n)
    return acc, pf, ct


# ---------------------------------------------------------------------------
# Variant 3: WiSDoM-Lite — Phase 1 + uniform random pairs over top-M.
# ---------------------------------------------------------------------------

def _top_m_set(elo, n, M):
    elo_vec = np.array([elo[i] for i in range(1, n + 1)])
    order = np.argsort(-elo_vec)
    return [int(order[k]) + 1 for k in range(min(M, n))]


def run_wisdom_lite(n, scores, total_budget, true_top_idx, true_ranks,
                    M=None, m_games=3, refresh_every=5):
    if M is None:
        M = max(3, int(math.ceil(n ** 0.25)))

    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    elo = np.ones(n + 1) * 1500.0

    def record(step_idx):
        _record_from_elo(elo, n, true_top_idx, true_ranks, step_idx, acc, pf, ct)

    used = _robust_bracket(n, scores, elo, m_games=m_games,
                           on_step=record, budget_cap=total_budget)

    cand = _top_m_set(elo, n, M)
    steps_since = 0
    while used < total_budget:
        if steps_since % refresh_every == 0:
            cand = _top_m_set(elo, n, M)
        if len(cand) < 2:
            others = [i for i in range(1, n + 1) if i not in cand]
            cand = cand + others[: max(2 - len(cand), 0)]
        i, j = random.sample(cand, 2)
        if Vote(scores[i - 1], scores[j - 1]):
            _elo_update(elo, i, j)
        else:
            _elo_update(elo, j, i)
        used += 1
        steps_since += 1
        record(used)

    _pad_tail(acc, pf, ct, n)
    return acc, pf, ct


# ---------------------------------------------------------------------------
# Variant 4: greedy-on-C — PARWiS-style King-of-the-Hill restricted to top-M.
# ---------------------------------------------------------------------------

def _greedy_pick_pair_on_C(n, data, estimates, cand_set):
    """PARWiS power_iter score, but the picked partner must be inside cand_set
    (1-indexed). Champion = current top by Rank Centrality."""
    _, _, top = get_ranking(n, estimates)
    top = top or 1
    comp = np.zeros((n + 1, n + 1))
    for w, l in data:
        if w > 0 and l > 0:
            comp[w, l] += 1
    cand_pairs = []
    sc = []
    for p in cand_set:
        if p == top:
            continue
        m = power_iter(n, comp, (top, p), estimates)
        cand_pairs.append((top, p))
        sc.append(m)
    if not sc or max(sc) <= 1e-9:
        others = [i for i in cand_set if i != top]
        if not others:
            others = [i for i in range(1, n + 1) if i != top]
        return (top, random.choice(others)) if others else (1, 2)
    mx = max(sc)
    picks = [i for i, v in enumerate(sc) if v == mx]
    return cand_pairs[random.choice(picks)]


def run_greedy_on_C(n, scores, total_budget, true_top_idx, true_ranks,
                    M=None, m_games=3, refresh_every=5):
    if M is None:
        M = max(3, int(math.ceil(n ** 0.25)))

    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    elo = np.ones(n + 1) * 1500.0

    def record(step_idx):
        _record_from_elo(elo, n, true_top_idx, true_ranks, step_idx, acc, pf, ct)

    used = _robust_bracket(n, scores, elo, m_games=m_games,
                           on_step=record, budget_cap=total_budget)

    # Convert bracket outcomes into a Rank-Centrality data list.
    # Bracket's per-match votes were *Elo* updates. We need a (winner, loser)
    # data stream for choix's RC; reconstruct conservatively from elo
    # ordering by adding one prior match per top-M item against the champion.
    elo_vec_now = np.array([elo[i] for i in range(1, n + 1)])
    champion = int(np.argmax(elo_vec_now)) + 1
    cand = _top_m_set(elo, n, M)
    if champion not in cand:
        cand = [champion] + cand[: M - 1]

    data = [(champion, p) for p in cand if p != champion]
    if not data:
        data = [(champion, (champion % n) + 1)]
    estimates = Rank_Centrality(n, data)

    steps_since = 0
    while used < total_budget:
        if steps_since % refresh_every == 0:
            cand = _top_m_set(elo, n, M)
        p, q = _greedy_pick_pair_on_C(n, data, estimates, cand)
        if Vote(scores[p - 1], scores[q - 1]):
            data.append((p, q))
            _elo_update(elo, p, q)
        else:
            data.append((q, p))
            _elo_update(elo, q, p)
        estimates = Rank_Centrality(n, data)
        # Use the RC ranking for the recorded prediction (closer to PARWiS).
        _, ranks_rc, top_rc = get_ranking(n, estimates)
        used += 1
        idx = used - 1
        if idx < total_budget:
            acc[idx] = 1 if top_rc == (true_top_idx + 1) else 0
            pf[idx] = ranks_rc[true_top_idx]
            ct[idx] = true_ranks[top_rc - 1]
        steps_since += 1

    _pad_tail(acc, pf, ct, n)
    return acc, pf, ct
