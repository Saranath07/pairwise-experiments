"""Phase 2c — LLM-as-judge experiment (GPU).

Two subcommands:

  precompute
    Builds (or loads) the Nectar candidate pool of size N, runs Qwen2.5-7B
    over the full round-robin (C(N,2) pairs × 2 orderings for position-bias
    debiasing), and saves:
      results/2c/<run_id>/candidates.jsonl   the candidate pool
      results/2c/<run_id>/roundrobin.npz     P[i,j] and C[i,j] matrices
      results/2c/<run_id>/manifest.json      metadata

  run
    Loads a roundrobin.npz, builds the cached oracle, and runs WiSDoM,
    PARWiS, and a uniform round-robin baseline over a budget grid in
    {N, 2N, 5N, 10N}. 30 trials. Token cost is the headline x-axis.
    Outputs:
      results/2c/<run_id>/<algo>.csv         per-step ACC/PF/CT + token cost
      results/2c/<run_id>/summary.csv

Ground truth is the Qwen-judge BT-implied top-1 (Bradley–Terry MLE on P).
A stronger judge (e.g. Opus 4.7) can overwrite this later by editing
`true_top` in `manifest.json` and rerunning the `run` step — algorithm
trajectories don't depend on the GT label.

Smoke mode (--smoke):
  uses StubPairwiseJudge and the synthetic candidate pool (no network,
  no GPU). Useful for shaking out the orchestration on a laptop.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.algorithms.common import get_ranks
from lib.algorithms.baselines import run_parwis
from lib.algorithms.robust_fw import run_robust_fw
from lib.oracles import use_oracle, make_proxy_scores
from lib.oracles.cached import CachedPairwiseOracle, cached_oracle_from_npz

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bradley–Terry MLE (used to pick the implied top-1 from the P matrix).
# ---------------------------------------------------------------------------

def bt_mle_from_p(P: np.ndarray, n_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Iterative BT-MLE on a soft-preference matrix using the standard
    Zermelo / minorization-maximization update.

    Treats P[i, j] as the empirical (or expected) win-rate of i over j.
    Returns positive BTL strengths, normalized so the max is 1.0."""
    N = P.shape[0]
    pi = np.ones(N)
    # Symmetric "matches played" weight = 1; effective wins matrix = P.
    for _ in range(n_iter):
        new = np.zeros(N)
        for i in range(N):
            num = 0.0
            den = 0.0
            for j in range(N):
                if i == j:
                    continue
                num += P[i, j]
                den += 1.0 / (pi[i] + pi[j])
            if den <= 0:
                new[i] = pi[i]
            else:
                new[i] = num / den
        new = np.maximum(new, 1e-12)
        new /= new.sum()
        if np.max(np.abs(new - pi / pi.sum())) < tol:
            pi = new
            break
        pi = new
    pi = pi / pi.max()
    return pi


# ---------------------------------------------------------------------------
# Round-robin precompute.
# ---------------------------------------------------------------------------

def _all_pairs(n: int) -> Iterable[Tuple[int, int]]:
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def precompute_round_robin(
    candidates,
    judge,
    out_dir: Path,
    progress: bool = True,
):
    """Run `judge` on every (i, j) pair (i < j) and save P, C matrices.

    The judge call already performs (A,B)+(B,A) debiasing internally, so
    P[i, j] is the position-bias-corrected estimate; we set P[j, i] = 1 - P[i, j]."""
    N = len(candidates)
    P = np.full((N, N), 0.5, dtype=np.float64)
    C = np.zeros((N, N), dtype=np.int64)

    pairs = list(_all_pairs(N))
    iter_obj = tqdm(pairs, desc="round-robin", ncols=72) if progress else pairs
    t0 = time.time()
    for i, j in iter_obj:
        out = judge.judge(
            prompt=candidates[i].prompt_text,
            response_a=candidates[i].response_text,
            response_b=candidates[j].response_text,
        )
        P[i, j] = out.p_a_beats_b
        P[j, i] = 1.0 - out.p_a_beats_b
        C[i, j] = out.tokens_in + out.tokens_out
        C[j, i] = C[i, j]
    elapsed = time.time() - t0
    log.info(f"round-robin: {len(pairs)} pairs in {elapsed:.1f}s "
             f"({len(pairs) / max(elapsed, 1e-6):.2f} pairs/s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "roundrobin.npz", P=P, C=C)
    return P, C


# ---------------------------------------------------------------------------
# Algorithm runners over the cached oracle.
# ---------------------------------------------------------------------------

def run_uniform_rr(n, scores, total_budget, true_top_idx, true_ranks):
    """Reference baseline: pick a random unordered pair every step."""
    from lib.algorithms.common import Vote, get_ranks as _gr
    acc = np.zeros(total_budget)
    pf = np.full(total_budget, n, dtype=float)
    ct = np.full(total_budget, n, dtype=float)
    wins = np.zeros((n, n), dtype=np.int64)
    plays = np.zeros((n, n), dtype=np.int64)
    for t in range(total_budget):
        i, j = random.sample(range(1, n + 1), 2)
        if Vote(scores[i - 1], scores[j - 1]):
            wins[i - 1, j - 1] += 1
        else:
            wins[j - 1, i - 1] += 1
        plays[i - 1, j - 1] += 1
        plays[j - 1, i - 1] += 1
        # Cheap aggregate: Copeland-on-empirical-rate.
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(plays > 0, wins / np.maximum(plays, 1), 0.5)
        cope = (rate > 0.5).sum(axis=1)
        winner = int(np.argmax(cope))
        ranks = _gr(cope)
        acc[t] = 1 if winner == true_top_idx else 0
        pf[t] = ranks[true_top_idx]
        ct[t] = true_ranks[winner]
    return acc, pf, ct


ALGOS = [
    ("WiSDoM",       run_robust_fw),
    ("PARWiS",       run_parwis),
    ("Uniform-RR",   run_uniform_rr),
]


def _candidates_to_proxy(candidates, btl_strengths):
    """Build the proxy `scores` array passed to algorithms.

    The numerical "score" is the BT-implied strength so any unmonkey-patched
    BTL fallback path is well-defined; the actual oracle lookups go through
    `Item.idx` because the cached oracle reads `a.idx` / `b.idx`."""
    return make_proxy_scores(np.asarray(btl_strengths, dtype=np.float64))


def run_algorithms_over_oracle(
    candidates,
    P: np.ndarray,
    C: np.ndarray,
    budgets: List[int],
    n_trials: int,
    base_seed: int,
    out_dir: Path,
    true_top_idx: int,
):
    """For each algorithm × budget × trial, sample the cached oracle and
    record per-step ACC + total token cost. Writes one CSV per algo with
    rows indexed by (budget, step)."""
    N = len(candidates)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `true_ranks` based on BT MLE on the cached P (caller passes top idx
    # from the same MLE so they're consistent).
    bt = bt_mle_from_p(P)
    true_ranks = get_ranks(bt)
    proxy = _candidates_to_proxy(candidates, bt)

    for name, fn in ALGOS:
        rows = []
        for B in budgets:
            for t in range(n_trials):
                seed = base_seed + t
                random.seed(seed); np.random.seed(seed)
                oracle = CachedPairwiseOracle(P, C)
                with use_oracle(oracle):
                    acc, pf, ct = fn(N, proxy, B, true_top_idx, true_ranks)
                # Tokens spent at each step are not tracked per-step (we only
                # know the *cumulative* token cost via oracle.tokens at the
                # end). For per-step token costs we re-derive: each step
                # consumed exactly the cost of the pair that the algorithm
                # queried, but the algorithm doesn't expose its query log.
                # Workable proxy: mean cost per query × step index. For
                # algorithms that bias toward short pairs (the WiSDoM
                # cost-aware extension would), this proxy under-counts the
                # gain — we leave that refinement to the post-run analysis.
                mean_cost = oracle.tokens / max(oracle.calls, 1)
                rows.append({
                    "budget": B, "trial": t,
                    "final_acc": float(acc[-1]),
                    "final_pf": float(pf[-1]),
                    "final_ct": float(ct[-1]),
                    "tokens_total": int(oracle.tokens),
                    "calls_total": int(oracle.calls),
                    "tokens_per_call_mean": float(mean_cost),
                })
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6f")
        # Aggregate summary across trials, per budget.
        agg = df.groupby("budget").agg(
            acc_mean=("final_acc", "mean"),
            acc_std=("final_acc", "std"),
            tokens_mean=("tokens_total", "mean"),
        ).reset_index()
        agg.to_csv(out_dir / f"{name}__summary.csv", index=False, float_format="%.6f")


# ---------------------------------------------------------------------------
# Top-level orchestrators.
# ---------------------------------------------------------------------------

def cmd_precompute(
    out_dir: Path,
    n: int,
    model_name: str,
    smoke: bool,
    seed: int,
    cache_dir: Optional[Path],
):
    from lib.judge.qwen import make_judge
    from lib.judge.nectar import build_candidate_pool, synthesize_candidate_pool

    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "candidates.jsonl"

    if smoke:
        log.info("[smoke] using synthetic candidate pool")
        candidates = synthesize_candidate_pool(n=n, cache_path=cand_path, seed=seed)
        judge = make_judge("stub")
    else:
        candidates = build_candidate_pool(
            n=n, cache_path=cand_path, seed=seed,
        )
        judge = make_judge(model_name)

    P, C = precompute_round_robin(candidates, judge, out_dir)

    bt = bt_mle_from_p(P)
    true_top = int(np.argmax(bt))
    manifest = {
        "n": len(candidates),
        "model_name": getattr(judge, "model_name", model_name),
        "device": getattr(judge, "device", "unknown"),
        "seed": seed,
        "smoke": smoke,
        "true_top_idx": true_top,
        "bt_strengths_top5": [
            (int(i), float(bt[i]))
            for i in np.argsort(-bt)[:5].tolist()
        ],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"manifest: top-1={true_top}, top-5={manifest['bt_strengths_top5']}")
    return P, C, candidates, true_top


def cmd_run(
    out_dir: Path,
    n_trials: int,
    base_seed: int,
    budgets: Optional[List[int]] = None,
):
    from lib.judge.nectar import _load as _load_candidates

    P_path = out_dir / "roundrobin.npz"
    if not P_path.exists():
        raise FileNotFoundError(f"{P_path} missing — run precompute first.")
    npz = np.load(P_path)
    P, C = npz["P"], npz["C"]
    N = P.shape[0]

    candidates = _load_candidates(out_dir / "candidates.jsonl")
    with open(out_dir / "manifest.json") as f:
        manifest = json.load(f)
    true_top = int(manifest["true_top_idx"])

    if budgets is None:
        budgets = [N, 2 * N, 5 * N, 10 * N]
    log.info(f"running algorithms: N={N} budgets={budgets} trials={n_trials}")

    run_algorithms_over_oracle(
        candidates=candidates,
        P=P, C=C,
        budgets=budgets,
        n_trials=n_trials,
        base_seed=base_seed,
        out_dir=out_dir,
        true_top_idx=true_top,
    )

    # Single combined summary across algos.
    summary = []
    for name, _ in ALGOS:
        df = pd.read_csv(out_dir / f"{name}__summary.csv")
        df["algo"] = name
        summary.append(df)
    pd.concat(summary, ignore_index=True).to_csv(out_dir / "summary.csv",
                                                  index=False, float_format="%.6f")


def run_2c(
    subcommand: str,
    n: int = 100,
    n_trials: int = 30,
    base_seed: int = 434,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    run_id: str = "default",
    smoke: bool = False,
    cache_dir: Optional[str] = None,
    budgets: Optional[List[int]] = None,
    results_dir: Optional[Path] = None,
):
    base = Path(results_dir) if results_dir else Path("results/2c")
    out_dir = base / run_id
    if subcommand == "precompute":
        cmd_precompute(out_dir, n=n, model_name=model_name,
                       smoke=smoke, seed=base_seed,
                       cache_dir=Path(cache_dir) if cache_dir else None)
    elif subcommand == "run":
        cmd_run(out_dir, n_trials=n_trials, base_seed=base_seed, budgets=budgets)
    elif subcommand == "all":
        cmd_precompute(out_dir, n=n, model_name=model_name,
                       smoke=smoke, seed=base_seed,
                       cache_dir=Path(cache_dir) if cache_dir else None)
        cmd_run(out_dir, n_trials=n_trials, base_seed=base_seed, budgets=budgets)
    else:
        raise ValueError(f"unknown subcommand: {subcommand}")
