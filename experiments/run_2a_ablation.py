"""Phase 2a — 5-line ablation attribution.

Lines compared (per NEXT_PLAN.md §2a):
  1. phase1_only       — bracket then output bracket winner.
  2. m2_verify         — Phase 1 + Bernoulli majority on (champion, runner-up).
  3. wisdom_lite       — Phase 1 + uniform random pairs on top-M (M=ceil(N^{1/4})).
  4. greedy_on_C       — Phase 1 + PARWiS-style greedy on top-M.
  5. Robust-FW         — full WiSDoM (Phase 1 + winner-focused FW on top-M).

Reference: PARWiS, kept as a sanity baseline.

Conditions:
  synth_k25, synth_k50, synth_k75, synth_k95   (N=100, B=500)
  netflix                                       (N=100)
  arena                                         (N=20, B=100 by default)

For each (condition, algorithm) tuple we record per-step ACC/PF/CT means and
stds across trials. Outputs:
  results_dir/<condition>/<algo>.csv      (Budget, ACC_mean, ACC_std, PF_*, CT_*)
  results_dir/<condition>/config.json
  plots_dir/<condition>_ACC.png           (one plot per condition)
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
from lib.algorithms.wisdom_variants import (
    run_phase1_only, run_m2_verify, run_wisdom_lite, run_greedy_on_C,
)


ALGOS = [
    ("Phase1-Only",  run_phase1_only),
    ("M2-Verify",    run_m2_verify),
    ("WiSDoM-Lite",  run_wisdom_lite),
    ("Greedy-on-C",  run_greedy_on_C),
    ("WiSDoM",       run_robust_fw),
    ("PARWiS",       run_parwis),
]


def _make_condition(spec):
    """Return (n, total_budget, scores, true_top_idx, true_ranks_factory).

    `scores` is the canonical BTL score array; for synthetic conditions the
    scores are re-drawn each trial (the factory handles that)."""
    name = spec["name"]
    if name.startswith("synth_k"):
        k = int(name.split("k")[-1])
        N = 100
        B = 5 * N
        def make(seed):
            random.seed(seed); np.random.seed(seed)
            scores, top = init(n=N, precomputed=False, k=k)
            return scores, top
        return N, B, make
    if name == "netflix":
        scores, top = init(n=100, dataset="netflix")
        N = len(scores); B = 5 * N
        def make(seed):
            random.seed(seed); np.random.seed(seed)
            return scores.copy(), int(top)
        return N, B, make
    if name == "arena":
        scores, top = init(n=20, dataset="arena")
        N = len(scores); B = 5 * N
        def make(seed):
            random.seed(seed); np.random.seed(seed)
            return scores.copy(), int(top)
        return N, B, make
    raise ValueError(f"unknown condition: {name}")


DEFAULT_CONDITIONS = [
    "synth_k25", "synth_k50", "synth_k75", "synth_k95",
    "netflix", "arena",
]


def run_2a(conditions=None, n_trials=30, base_seed=434,
           results_dir=None, plots_dir=None, budget_cap=None):
    if conditions is None:
        conditions = DEFAULT_CONDITIONS
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)

    for cond_name in conditions:
        out_dir = results_dir / cond_name
        out_dir.mkdir(parents=True, exist_ok=True)

        N, B, make = _make_condition({"name": cond_name})
        if budget_cap is not None:
            B = min(B, budget_cap)

        with open(out_dir / "config.json", "w") as f:
            json.dump({"condition": cond_name, "N": N, "B": B,
                       "n_trials": n_trials, "base_seed": base_seed,
                       "algos": [a for a, _ in ALGOS]}, f, indent=2)

        print("\n" + "=" * 78)
        print(f"  2a / {cond_name}   N={N}  B={B}  trials={n_trials}")
        print("=" * 78)

        acc_all = {a: np.zeros((n_trials, B)) for a, _ in ALGOS}
        pf_all  = {a: np.zeros((n_trials, B)) for a, _ in ALGOS}
        ct_all  = {a: np.zeros((n_trials, B)) for a, _ in ALGOS}

        for t in tqdm(range(n_trials), desc=cond_name, ncols=72):
            seed_t = base_seed + t
            scores, true_top = make(seed_t)
            true_ranks = get_ranks(scores)

            for name, fn in ALGOS:
                random.seed(seed_t); np.random.seed(seed_t)
                acc, pf, ct = fn(N, scores, B, true_top, true_ranks)
                acc_all[name][t, :] = acc
                pf_all[name][t, :] = pf
                ct_all[name][t, :] = ct

        for name, _ in ALGOS:
            df = pd.DataFrame({"Budget": np.arange(1, B + 1)})
            df[f"ACC_mean"] = acc_all[name].mean(axis=0)
            df[f"ACC_std"]  = acc_all[name].std(axis=0)
            df[f"PF_mean"]  = pf_all[name].mean(axis=0)
            df[f"PF_std"]   = pf_all[name].std(axis=0)
            df[f"CT_mean"]  = ct_all[name].mean(axis=0)
            df[f"CT_std"]   = ct_all[name].std(axis=0)
            df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6f")

        _plot_accuracy(cond_name, ALGOS, acc_all, B, plots_dir)

        print(f"\nFinal ACC at B={B}  (mean ± std):")
        for name, _ in ALGOS:
            m = float(np.mean(acc_all[name][:, -1]))
            s = float(np.std(acc_all[name][:, -1]))
            print(f"  {name:<14s} {m:.3f} ± {s:.3f}")

    print("\n2a done. Per-condition CSVs in results/2a/<cond>/.")


def _plot_accuracy(cond_name, algos, acc_all, B, plots_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    budget = np.arange(1, B + 1)
    for name, _ in algos:
        m = acc_all[name].mean(axis=0)
        s = acc_all[name].std(axis=0)
        ax.plot(budget, m, label=name, linewidth=1.6)
        ax.fill_between(budget, m - s, m + s, alpha=0.10)
    ax.set_xlabel("Budget (# queries)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(f"2a ablation — {cond_name}")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{cond_name}_ACC.png", dpi=140)
    plt.close(fig)
