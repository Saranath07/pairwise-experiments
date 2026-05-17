"""
Published baselines for top-1 winner identification from pairwise comparisons.

Each runner has the signature:
    run_<name>(n, scores, total_budget, true_top_idx, true_ranks) -> (acc, pf, ct)
  where each returned array has length `total_budget` and reports the cumulative
  metric at every query step.

Metrics:
  - acc[t]: 1 if predicted winner at step t equals true top-1, else 0
  - pf[t]:  reported rank of the true winner at step t
  - ct[t]:  true rank of the reported winner at step t

Baselines implemented:
  * PARWiS        (PAC Top-1 Identification under BTL)
  * SELECT        (Active Learning for Top-K Rank Aggregation)
  * RUCB          (Relative Upper Confidence Bound)
  * MultiSort     (Just Sort It!)
  * Knockout      (Maxing and Ranking with Few Assumptions)

All probabilistic queries go through the `Vote(a, b)` routine from common.py so
every algorithm draws from the same BTL simulator under the same RNG stream.
"""
import itertools
import math
import random
import numpy as np

from lib.algorithms import common as _common
from lib.algorithms.common import Rank_Centrality, get_ranking, power_iter, get_ranks


def Vote(a, b):
    """Delegating wrapper so monkey-patching `lib.algorithms.common.Vote`
    (e.g., by `lib.oracles`) reaches every call inside this module."""
    return _common.Vote(a, b)


# ---------------------------------------------------------------------------
# PARWiS (Standard: King-of-the-Hill init + exact-weighted pair selection)
# ---------------------------------------------------------------------------

def _reset_king_of_the_hill(n, scores):
    data = []
    shuffled = list(range(1, n + 1)); random.shuffle(shuffled)
    current_winner = shuffled[0]
    for i in range(1, n):
        challenger = shuffled[i]
        if Vote(scores[current_winner - 1], scores[challenger - 1]):
            data.append((current_winner, challenger))
        else:
            data.append((challenger, current_winner))
            current_winner = challenger
    estimates = Rank_Centrality(n, data)
    return data, len(data), estimates


def _parwis_pick_pair(n, data, estimates):
    _, _, top = get_ranking(n, estimates)
    top = top or 1
    comp = np.zeros((n + 1, n + 1))
    for w, l in data:
        if w > 0 and l > 0:
            comp[w, l] += 1
    cand, scores_vec = [], []
    for p in range(1, n + 1):
        if p != top:
            m = power_iter(n, comp, (top, p), estimates)
            cand.append((top, p)); scores_vec.append(m)
    if not scores_vec or max(scores_vec) <= 1e-9:
        others = [i for i in range(1, n + 1) if i != top]
        return (top, random.choice(others)) if others else (1, 2)
    mx = max(scores_vec)
    picks = [i for i, v in enumerate(scores_vec) if v == mx]
    return cand[random.choice(picks)]


def run_parwis(n, scores, total_budget, true_top_idx, true_ranks):
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    true_top_1 = true_top_idx + 1

    data, init_votes, estimates = _reset_king_of_the_hill(n, scores)

    if 0 < init_votes <= total_budget:
        _, ranks0, w0 = get_ranking(n, estimates)
        acc[:init_votes] = 1 if w0 == true_top_1 else 0
        pf[:init_votes] = ranks0[true_top_idx]
        ct[:init_votes] = true_ranks[w0 - 1]

    for t in range(init_votes, total_budget):
        if t > 0:
            acc[t], pf[t], ct[t] = acc[t - 1], pf[t - 1], ct[t - 1]
        p, q = _parwis_pick_pair(n, data, estimates)
        w, l = (p, q) if Vote(scores[p - 1], scores[q - 1]) else (q, p)
        data.append((w, l))
        estimates = Rank_Centrality(n, data)
        _, final_ranks, final_w = get_ranking(n, estimates)
        acc[t] = 1 if final_w == true_top_1 else 0
        pf[t] = final_ranks[true_top_idx]
        ct[t] = true_ranks[final_w - 1]

    return acc, pf, ct


# ---------------------------------------------------------------------------
# SELECT (deterministic single-elimination; each match = 1 vote)
# ---------------------------------------------------------------------------

def run_select(n, scores, total_budget, true_top_idx, true_ranks):
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)

    def _halve(items):
        votes = 0; new_items = []
        random.shuffle(items)
        for i in range(len(items) // 2):
            p, q = items[2 * i], items[2 * i + 1]
            if Vote(scores[p], scores[q]): new_items.append(p)
            else: new_items.append(q)
            votes += 1
        if len(items) % 2 == 1:
            new_items.append(items[-1])
        return new_items, votes

    items = list(range(n)); total = 0
    while len(items) > 1:
        items, v = _halve(items)
        total += v

    if 0 < total <= total_budget:
        predicted = items[0]
        is_correct = 1 if predicted == true_top_idx else 0
        acc[total - 1:] = is_correct
        pf[total - 1:] = 1.0 if is_correct else n
        ct[total - 1:] = true_ranks[predicted]

    return acc, pf, ct


# ---------------------------------------------------------------------------
# RUCB (Relative UCB, Zoghi et al. 2014)
# ---------------------------------------------------------------------------

def run_rucb(n, scores, total_budget, true_top_idx, true_ranks):
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    a = 0.51
    W = np.zeros((n, n)); T = np.zeros((n, n))

    for t in range(1, total_budget + 1):
        with np.errstate(divide='ignore', invalid='ignore'):
            U = (W / T) + np.sqrt(a * math.log(t) / T)
        np.nan_to_num(U, copy=False, nan=np.inf)
        np.fill_diagonal(U, 0.5)
        with np.errstate(divide='ignore', invalid='ignore'):
            J = W / T
        np.nan_to_num(J, nan=0.5, copy=False)
        copeland = (J > 0.5).sum(axis=1)
        c_star = int(np.argmax(copeland))
        C = {i for i in range(n) if U[i, c_star] >= 0.5}
        c = random.choice(list(C or range(n)))
        Uc = np.copy(U[:, c]); Uc[c] = -np.inf
        d = int(np.argmax(Uc))

        if Vote(scores[c], scores[d]): W[c, d] += 1
        else: W[d, c] += 1
        T[c, d] += 1; T[d, c] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            Jf = W / T
        np.nan_to_num(Jf, nan=0.5, copy=False)
        final_cope = (Jf > 0.5).sum(axis=1)
        ranks = get_ranks(final_cope)
        winner = int(np.argmax(final_cope))
        acc[t - 1] = 1 if winner == true_top_idx else 0
        pf[t - 1] = ranks[true_top_idx]
        ct[t - 1] = true_ranks[winner]

    return acc, pf, ct


# ---------------------------------------------------------------------------
# MultiSort / Just-Sort-It (Maystre & Grossglauser 2017)
# ---------------------------------------------------------------------------

def run_multisort(n, scores, total_budget, true_top_idx, true_ranks):
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    budget_spent = 0
    full_rankings = []

    def _copeland(data):
        if not data:
            return np.zeros(n)
        cope = np.zeros((n + 1, n + 1))
        for r in data:
            for j in range(len(r)):
                for kk in range(j + 1, len(r)):
                    cope[r[j], r[kk]] += 1
        cs = np.zeros(n)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if cope[i, j] > cope[j, i]: cs[i - 1] += 1
                elif cope[j, i] > cope[i, j]: cs[j - 1] += 1
                else: cs[i - 1] += 0.5; cs[j - 1] += 0.5
        return cs

    while budget_spent < total_budget:
        items = list(range(1, n + 1)); random.shuffle(items)
        current_sort = []
        for item in items:
            if not current_sort:
                current_sort.append(item); continue
            inserted = False
            for i in range(len(current_sort)):
                if budget_spent >= total_budget: break
                is_better = Vote(scores[item - 1], scores[current_sort[i] - 1])
                budget_spent += 1

                if full_rankings:
                    cs = _copeland(full_rankings)
                    cranks = get_ranks(cs)
                    winner0 = int(np.argmax(cs))
                    start_idx = budget_spent - 2 if budget_spent > 1 else 0
                    acc[start_idx:budget_spent - 1] = acc[start_idx]
                    pf[start_idx:budget_spent - 1] = pf[start_idx]
                    ct[start_idx:budget_spent - 1] = ct[start_idx]
                    acc[budget_spent - 1] = 1 if winner0 == true_top_idx else 0
                    pf[budget_spent - 1] = cranks[true_top_idx]
                    ct[budget_spent - 1] = true_ranks[winner0]

                if is_better:
                    current_sort.insert(i, item); inserted = True; break
            if not inserted:
                current_sort.append(item)
            if budget_spent >= total_budget: break
        if budget_spent < total_budget:
            full_rankings.append(current_sort)

    return acc, pf, ct


# ---------------------------------------------------------------------------
# Knockout (Falahatgar et al., "Maxing and Ranking with Few Assumptions")
# ---------------------------------------------------------------------------
#
# Single-elimination tournament where each match is repeated until one side
# has a confidently higher win rate. Simplified fixed-budget version:
# run a plain knockout bracket with m_games per match (constant), refining
# the running-best item. If budget remains after the bracket, keep replaying
# matches between the current top-2 (head-to-head refinement).

def _match_budgeted(a, b, scores, m_games, budget_left, pred_winner_update):
    """Play up to m_games between a and b, but stop early if budget runs out.
    Returns (winner, losses_played)."""
    wa = 0; wb = 0; played = 0
    while played < m_games and budget_left > 0:
        if Vote(scores[a - 1], scores[b - 1]): wa += 1
        else: wb += 1
        played += 1; budget_left -= 1
    winner = a if wa >= wb else b
    return winner, played, budget_left


def run_knockout(n, scores, total_budget, true_top_idx, true_ranks, m_games=3):
    """Knockout with best-of-m_games matches; pads remaining budget by refining
    head-to-head between the current bracket leader and nearest challenger."""
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    true_top_1 = true_top_idx + 1

    bracket = list(range(1, n + 1))
    random.shuffle(bracket)
    budget_spent = 0
    current_pred = bracket[0]

    def _record(step):
        # record current prediction at position `step` (0-indexed)
        if step >= total_budget: return
        acc[step] = 1 if current_pred == true_top_1 else 0
        # rank-based metrics require a full ranking; for Knockout we only know
        # the bracket leader, so approximate: rank of true winner in bracket order
        # and true rank of current prediction.
        pf[step] = 1.0 if current_pred == true_top_1 else n
        ct[step] = true_ranks[current_pred - 1]

    # Play bracket rounds
    round_items = list(bracket)
    while len(round_items) > 1 and budget_spent < total_budget:
        next_round = []
        for i in range(0, len(round_items), 2):
            if i + 1 >= len(round_items):
                next_round.append(round_items[i]); continue
            if budget_spent >= total_budget: break
            a, b = round_items[i], round_items[i + 1]
            w, played, _ = _match_budgeted(a, b, scores, m_games, total_budget - budget_spent, None)
            for k_ in range(played):
                budget_spent += 1
                # leader update after every game is just 'a' or 'b' tentatively
                # we don't change current_pred mid-match; keep prior best
                _record(budget_spent - 1)
            current_pred = w
            _record(budget_spent - 1) if budget_spent > 0 else None
            next_round.append(w)
        round_items = next_round

    # Refinement phase: keep replaying top-2 if we have budget left
    if len(round_items) >= 1 and budget_spent < total_budget:
        # find a plausible runner-up: the last loser, if any, else random other
        runner = None
        for cand in bracket:
            if cand != current_pred: runner = cand; break
        if runner is None:
            runner = (current_pred % n) + 1
        wa = m_games  # prior pseudo-win for current_pred
        wb = 0
        while budget_spent < total_budget:
            if Vote(scores[current_pred - 1], scores[runner - 1]): wa += 1
            else: wb += 1
            budget_spent += 1
            if wb > wa:
                current_pred, runner = runner, current_pred
                wa, wb = wb, wa
            _record(budget_spent - 1)

    # Fill any leading zeros with the initial prediction
    for t in range(total_budget):
        if acc[t] == 0 and pf[t] == n and ct[t] == n:
            if t > 0:
                acc[t], pf[t], ct[t] = acc[t - 1], pf[t - 1], ct[t - 1]

    return acc, pf, ct
