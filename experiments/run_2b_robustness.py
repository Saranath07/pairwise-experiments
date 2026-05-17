"""Phase 2b — robustness to BTL violations.

Three perturbation families (per NEXT_PLAN.md §2b):

  cyclic    — P(i ≻ j) = sigma(theta_i − theta_j) + rho · c_ij,
              c skew-symmetric, rho ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.40}.
  thurstone — probit link with sigma ∈ {0.5, 1.0, 1.5, 2.0}.
              (sigma=1.0 is roughly BTL-matched; larger = more noise.)
  multidim  — two latent dimensions, oracle samples a dim per query;
              ground truth is uniform-Borda over both dims. Sweeps the
              correlation between the two dims to interpolate "perfect agreement"
              vs "RPS-like disagreement".

Algorithms compared per cell: WiSDoM (Robust-FW) vs PARWiS — these are the
two relevant heads for the BTL-reliance critique. Synthetic backbone:
N=100, B=500, k=75, 30 trials per (perturbation, cell, algo).

Outputs:
  results_dir/<perturbation>/<algo>__<cell>.csv
  results_dir/<perturbation>/config.json
  plots_dir/<perturbation>_ACC.png   (mean final ACC vs perturbation magnitude)
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.algorithms.common import init, get_ranks
from lib.algorithms.baselines import run_parwis
from lib.algorithms.robust_fw import run_robust_fw
from lib.oracles import use_oracle, make_proxy_scores
from lib.oracles.cyclic import cyclic_oracle, random_skew
from lib.oracles.thurstone import thurstone_oracle
from lib.oracles.multidim import multidim_oracle, borda_winner


ALGOS = [
    ("WiSDoM",  run_robust_fw),
    ("PARWiS",  run_parwis),
]


# ---------------------------------------------------------------------------
# Cell builders: each yields (cell_name, oracle_factory, scores_proxy_factory,
#                              true_top_factory, true_ranks_factory).
# ---------------------------------------------------------------------------

def _cyclic_cells(rho_grid):
    cells = []
    for rho in rho_grid:
        cells.append({"name": f"rho{rho:.2f}", "rho": rho})
    return cells


def _thurstone_cells(sigma_grid):
    return [{"name": f"sigma{s:.1f}", "sigma": s} for s in sigma_grid]


def _multidim_cells(corr_grid):
    return [{"name": f"corr{c:+.2f}", "corr": c} for c in corr_grid]


# ---------------------------------------------------------------------------
# Trial driver: for each perturbation we re-draw the synthetic instance.
# The skew-symmetric perturbation matrix is fixed across rho values within
# a trial (so rho=0 and rho=0.4 use the same c at the same trial seed).
# ---------------------------------------------------------------------------

def _run_perturbation(perturbation, cells, n_trials, base_seed,
                      results_dir, plots_dir, budget_cap=None):
    N = 100
    K_DIFF = 75
    B = budget_cap if budget_cap is not None else 5 * N
    out_dir = results_dir / perturbation
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump({"perturbation": perturbation, "N": N, "B": B,
                   "n_trials": n_trials, "base_seed": base_seed,
                   "cells": [c["name"] for c in cells],
                   "algos": [a for a, _ in ALGOS]}, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  2b / {perturbation}   N={N}  B={B}  trials={n_trials}  cells={len(cells)}")
    print("=" * 78)

    # Pre-allocate.
    acc_store = {(c["name"], a): np.zeros((n_trials, B)) for c in cells for a, _ in ALGOS}
    pf_store  = {(c["name"], a): np.zeros((n_trials, B)) for c in cells for a, _ in ALGOS}
    ct_store  = {(c["name"], a): np.zeros((n_trials, B)) for c in cells for a, _ in ALGOS}

    for t in tqdm(range(n_trials), desc=perturbation, ncols=72):
        seed_t = base_seed + t
        random.seed(seed_t); np.random.seed(seed_t)
        scores, true_top = init(n=N, precomputed=False, k=K_DIFF)
        true_ranks = get_ranks(scores)
        proxy = make_proxy_scores(scores)

        if perturbation == "cyclic":
            c_mat = random_skew(N, seed=seed_t)
        elif perturbation == "multidim":
            # build a second latent dim correlated with the first via target corr.
            base = scores
            theta_dims_factory = lambda corr: _make_correlated_dim(base, corr, rng_seed=seed_t)
        # else thurstone: uses scores directly.

        for cell in cells:
            cn = cell["name"]
            if perturbation == "cyclic":
                oracle = cyclic_oracle(scores, cell["rho"], c_mat)
                tt, tr, sc_proxy = true_top, true_ranks, proxy
            elif perturbation == "thurstone":
                oracle = thurstone_oracle(scores, sigma=cell["sigma"])
                tt, tr, sc_proxy = true_top, true_ranks, proxy
            elif perturbation == "multidim":
                theta_dims = theta_dims_factory(cell["corr"])
                tt = borda_winner(theta_dims)
                # ranks under Borda aggregate (-mean rank)
                tr = _borda_ranks(theta_dims)
                sc_proxy = make_proxy_scores(theta_dims.mean(axis=1))
                oracle = multidim_oracle(theta_dims)
            else:
                raise ValueError(f"unknown perturbation: {perturbation}")

            for name, fn in ALGOS:
                random.seed(seed_t); np.random.seed(seed_t)
                with use_oracle(oracle):
                    acc, pf, ct = fn(N, sc_proxy, B, tt, tr)
                acc_store[(cn, name)][t, :] = acc
                pf_store[(cn, name)][t, :]  = pf
                ct_store[(cn, name)][t, :]  = ct

    # Write per-cell CSVs and a wide summary across the cell axis.
    summary_rows = []
    for cell in cells:
        for name, _ in ALGOS:
            key = (cell["name"], name)
            df = pd.DataFrame({"Budget": np.arange(1, B + 1)})
            df["ACC_mean"] = acc_store[key].mean(axis=0)
            df["ACC_std"]  = acc_store[key].std(axis=0)
            df["PF_mean"]  = pf_store[key].mean(axis=0)
            df["PF_std"]   = pf_store[key].std(axis=0)
            df["CT_mean"]  = ct_store[key].mean(axis=0)
            df["CT_std"]   = ct_store[key].std(axis=0)
            df.to_csv(out_dir / f"{name}__{cell['name']}.csv",
                      index=False, float_format="%.6f")
            summary_rows.append({
                "cell": cell["name"],
                "algo": name,
                "ACC_final_mean": float(np.mean(acc_store[key][:, -1])),
                "ACC_final_std":  float(np.std(acc_store[key][:, -1])),
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_final_acc.csv",
                                       index=False, float_format="%.6f")

    _plot_perturbation(perturbation, cells, ALGOS, acc_store, B, plots_dir)
    print(f"  -> wrote {len(summary_rows)} curves under {out_dir}")


def _make_correlated_dim(base, corr, rng_seed):
    """Construct a second BTL-positive dim with target rank correlation `corr`
    in [-1, 1] versus `base`. Implemented by mixing the base ranks with a
    random permutation according to the target correlation magnitude."""
    rng = np.random.default_rng(rng_seed + 7919)
    n = len(base)
    base_ranks = np.argsort(np.argsort(-base))  # 0 = best
    rand_ranks = rng.permutation(n)
    if corr >= 0:
        alpha = corr
        target = alpha * base_ranks + (1 - alpha) * rand_ranks
    else:
        alpha = -corr
        flipped = (n - 1) - base_ranks
        target = alpha * flipped + (1 - alpha) * rand_ranks
    order = np.argsort(target)
    second = np.zeros(n)
    # Re-use the magnitudes of `base` but in the new rank order.
    sorted_mag = np.sort(base)[::-1]
    for r, i in enumerate(order):
        second[i] = sorted_mag[r]
    return np.column_stack([base, second])


def _borda_ranks(theta_dims):
    """Borda rank vector: lower = better."""
    n, D = theta_dims.shape
    rank_sum = np.zeros(n)
    for d in range(D):
        order = np.argsort(-theta_dims[:, d])
        for r, idx in enumerate(order):
            rank_sum[idx] += r + 1
    return get_ranks(-rank_sum)


def _plot_perturbation(perturbation, cells, algos, acc_store, B, plots_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    cell_x = [c["name"] for c in cells]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, _ in algos:
        means = [acc_store[(c["name"], name)][:, -1].mean() for c in cells]
        stds  = [acc_store[(c["name"], name)][:, -1].std()  for c in cells]
        ax.errorbar(cell_x, means, yerr=stds, label=name, marker="o", linewidth=1.6)
    ax.set_xlabel("Perturbation cell")
    ax.set_ylabel(f"Final top-1 accuracy at B={B}")
    ax.set_title(f"2b — robustness ({perturbation})")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{perturbation}_ACC.png", dpi=140)
    plt.close(fig)


DEFAULT_CYCLIC_RHO = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
DEFAULT_THURSTONE_SIGMA = [0.5, 1.0, 1.5, 2.0]
DEFAULT_MULTIDIM_CORR = [-0.5, 0.0, 0.5, 1.0]

DEFAULT_PERTURBATIONS = ["cyclic", "thurstone", "multidim"]


def run_2b(perturbations=None, n_trials=30, base_seed=434,
           results_dir=None, plots_dir=None, budget_cap=None):
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)

    for pert in perturbations:
        if pert == "cyclic":
            cells = _cyclic_cells(DEFAULT_CYCLIC_RHO)
        elif pert == "thurstone":
            cells = _thurstone_cells(DEFAULT_THURSTONE_SIGMA)
        elif pert == "multidim":
            cells = _multidim_cells(DEFAULT_MULTIDIM_CORR)
        else:
            raise ValueError(f"unknown perturbation: {pert}")

        _run_perturbation(pert, cells, n_trials, base_seed,
                          results_dir, plots_dir, budget_cap=budget_cap)

    print("\n2b done. Per-perturbation CSVs in results/2b/<pert>/.")
