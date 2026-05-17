"""CLI driver for Phase-2 experiments.

Usage:
    python run.py 2a   [--trials N] [--seed S] [--conditions ...]
    python run.py 2b   [--trials N] [--seed S] [--perturbations ...]
    python run.py 2c   precompute|run|all
                       [--n N] [--trials N] [--model NAME] [--run-id ID]
                       [--smoke]
    python run.py all
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TRIALS_DEFAULT = 30
SEED_DEFAULT = 434
DEFAULT_2C_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_dirs(tag):
    base = Path(__file__).parent
    results_dir = base / "results" / tag
    plots_dir = base / "plots" / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def cmd_2a(args):
    from experiments.run_2a_ablation import run_2a
    results_dir, plots_dir = get_dirs("2a")
    run_2a(
        conditions=args.conditions,
        n_trials=args.trials,
        base_seed=args.seed,
        results_dir=results_dir,
        plots_dir=plots_dir,
        budget_cap=args.budget_cap,
    )


def cmd_2b(args):
    from experiments.run_2b_robustness import run_2b
    results_dir, plots_dir = get_dirs("2b")
    run_2b(
        perturbations=args.perturbations,
        n_trials=args.trials,
        base_seed=args.seed,
        results_dir=results_dir,
        plots_dir=plots_dir,
        budget_cap=args.budget_cap,
    )


def cmd_2c(args):
    from experiments.run_2c_llm_judge import run_2c
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results_dir, _ = get_dirs("2c")
    budgets = args.budgets
    run_2c(
        subcommand=args.subcommand,
        n=args.n,
        n_trials=args.trials,
        base_seed=args.seed,
        model_name=args.model,
        run_id=args.run_id,
        smoke=args.smoke,
        cache_dir=args.cache_dir,
        budgets=budgets,
        results_dir=results_dir,
    )


def cmd_all(args):
    cmd_2a(args)
    cmd_2b(args)


def build_parser():
    p = argparse.ArgumentParser(description="Phase-2 CPU experiments.")
    sub = p.add_subparsers(dest="command", required=True)

    p_2a = sub.add_parser("2a", help="5-line ablation (Phase 2a).")
    p_2a.add_argument("--conditions", nargs="+", default=None,
                      help="Subset of conditions: synth_k25 synth_k50 synth_k75 synth_k95 netflix arena")
    p_2a.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    p_2a.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p_2a.add_argument("--budget-cap", type=int, default=None,
                      help="Override total budget (smoke tests).")
    p_2a.set_defaults(func=cmd_2a)

    p_2b = sub.add_parser("2b", help="BTL-violation robustness (Phase 2b).")
    p_2b.add_argument("--perturbations", nargs="+", default=None,
                      help="Subset of: cyclic thurstone multidim")
    p_2b.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    p_2b.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p_2b.add_argument("--budget-cap", type=int, default=None)
    p_2b.set_defaults(func=cmd_2b)

    p_2c = sub.add_parser("2c", help="LLM-as-judge experiment (GPU).")
    p_2c.add_argument("subcommand", choices=["precompute", "run", "all"],
                      help="precompute = round-robin Qwen judging; "
                           "run = algorithm sweep over cached oracle.")
    p_2c.add_argument("--n", type=int, default=100,
                      help="Candidate pool size (default 100).")
    p_2c.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    p_2c.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p_2c.add_argument("--model", type=str, default=DEFAULT_2C_MODEL,
                      help="HF model id for the judge (use 'stub' for offline).")
    p_2c.add_argument("--run-id", type=str, default="default",
                      help="Subdirectory under results/2c/ to write to / read from.")
    p_2c.add_argument("--smoke", action="store_true",
                      help="Use the stub judge + synthetic candidate pool. "
                           "No GPU and no network required.")
    p_2c.add_argument("--cache-dir", type=str, default=None,
                      help="HuggingFace cache dir (defaults to env-default).")
    p_2c.add_argument("--budgets", type=int, nargs="+", default=None,
                      help="Budget grid (default = [N, 2N, 5N, 10N]).")
    p_2c.set_defaults(func=cmd_2c)

    p_all = sub.add_parser("all", help="Run 2a then 2b (CPU only; 2c stays separate).")
    p_all.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    p_all.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p_all.add_argument("--budget-cap", type=int, default=None)
    p_all.add_argument("--conditions", nargs="+", default=None)
    p_all.add_argument("--perturbations", nargs="+", default=None)
    p_all.set_defaults(func=cmd_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
