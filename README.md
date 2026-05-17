# pairwise-experiments — Phase 2 (CPU + GPU)

Companion repo to `wisdom-improved/`. Implements the experimental tasks from
`NEXT_PLAN.md` Phase 2. CPU-bound experiments (2a, 2b) are ready to run; the
GPU-bound LLM-as-judge experiment (2c) is added in a separate pass.

## Layout

```
pairwise-experiments/
├── run.py                          # CLI driver
├── lib/
│   ├── algorithms/                 # WiSDoM + 5 baselines (verbatim from /wisdom/)
│   │   ├── common.py               # init(), Vote(), Rank Centrality wrapper
│   │   ├── baselines.py            # PARWiS, SELECT, RUCB, MultiSort, Knockout
│   │   ├── robust_fw.py            # full WiSDoM (Robust-FW)
│   │   └── wisdom_variants.py      # ablation lines: phase1_only, m2_verify,
│   │                               #                 wisdom_lite, greedy_on_C
│   └── oracles/                    # swappable oracles for §2b
│       ├── __init__.py             # use_oracle() context manager + Item proxy
│       ├── btl.py                  # reference BTL
│       ├── cyclic.py               # P(i>j) = sigma(theta_i-theta_j) + rho * c_ij
│       ├── thurstone.py            # probit link
│       └── multidim.py             # 2 latent dims, oracle samples a dim per query
├── experiments/
│   ├── run_2a_ablation.py
│   └── run_2b_robustness.py
├── results/                        # gitignored; per-run subdirs
└── plots/                          # gitignored
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, numpy 1.26, pandas 2.1, choix 0.3.4.

## Running CPU experiments

### 2a — 5-line ablation attribution

```bash
python run.py 2a                                  # all defaults
python run.py 2a --conditions synth_k75 netflix
python run.py 2a --trials 30
python run.py 2a --budget-cap 200                  # smoke test
```

Default conditions: `synth_k25 synth_k50 synth_k75 synth_k95 netflix arena`.
Default trials: 30. Default seed: 434.

Outputs:
- `results/2a/<condition>/<algo>.csv` — per-step ACC/PF/CT mean+std across trials.
- `results/2a/<condition>/config.json`
- `plots/2a/<condition>_ACC.png`

The five ablation lines plus PARWiS as a reference are run on every condition.

### 2b — robustness to BTL violations

```bash
python run.py 2b
python run.py 2b --perturbations cyclic
python run.py 2b --trials 30 --budget-cap 500
```

Perturbations: `cyclic thurstone multidim`.

- **cyclic** sweeps ρ ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.40}.
- **thurstone** sweeps σ ∈ {0.5, 1.0, 1.5, 2.0}.
- **multidim** sweeps the rank correlation between two latent dims ∈ {-0.5, 0, 0.5, 1.0}.

Algorithms compared: WiSDoM (full Robust-FW) vs PARWiS.

Outputs:
- `results/2b/<perturbation>/<algo>__<cell>.csv`
- `results/2b/<perturbation>/summary_final_acc.csv`
- `plots/2b/<perturbation>_ACC.png`

### Run both at once

```bash
python run.py all --trials 30
```

Estimated wall-clock at the defaults (single CPU thread, MacBook-class machine):

| stage   | trials | conditions/cells | rough time |
| ------- | -----: | ---------------: | ---------: |
| 2a      |     30 |                6 |     ~30 m  |
| 2b      |     30 |        6 + 4 + 4 |     ~45 m  |

Trivially parallelizable across trials and conditions if you split the CLI
arguments over machines.

## How the oracle swap works (§2b)

Algorithms make every probabilistic comparison via
`Vote(scores[i-1], scores[j-1])`. To swap the oracle without touching the
algorithm code, `lib/oracles/__init__.py` exposes:

- `Item(idx, theta)` — score proxy carrying a 0-based index.
- `make_proxy_scores(theta)` — wraps a numeric score vector into proxies.
- `use_oracle(oracle_fn)` — context manager that monkey-patches `Vote` in
  every algorithm module so calls route through `oracle_fn(Item, Item) -> bool`.

A pure-BTL run with proxies still works (the proxy wrapper exposes `.theta`
and `Vote` falls through to the standard BTL random draw).

## GPU experiments (§2c — LLM-as-judge)

The Nectar + Qwen2.5-7B-Instruct LLM-as-judge experiment lives under
`experiments/run_2c_llm_judge.py` and is exposed via `python run.py 2c …`.

### Architecture

Two-step design:

1. **`precompute`** — load Nectar, sample $N=100$ candidates across diverse
   model families, run **Qwen2.5-7B-Instruct** over the full $\binom{N}{2}=
   4950$ pairs, with each pair evaluated in both `(A=i, B=j)` and `(A=j, B=i)`
   orderings so the resulting probability is position-bias debiased.
   Saves:
     - `candidates.jsonl`     — the candidate pool (so the run is reproducible).
     - `roundrobin.npz`       — `P[i,j]` (debiased win probability) and
                                 `C[i,j]` (token cost per pair).
     - `manifest.json`        — model, device, top-1 (BT-MLE on `P`).

2. **`run`** — load the cached `P`, build a `CachedPairwiseOracle`, and run
   WiSDoM, PARWiS, and a uniform round-robin baseline across budgets
   $B \in \{N, 2N, 5N, 10N\}$ at 30 trials each. Token cost (sum of `C[i,j]`
   over queried pairs) is the headline x-axis.

The split makes the GPU pass a **one-shot cost**: 30 trials cost the same as
1 trial because the algorithms only sample from the cached matrix.

### Judge mechanism (no autoregressive generation)

The pairwise judge formats the prompt so the next token is forced to be `A`
or `B`, then reads only those two logits in a single forward pass. This is
~5× faster than autoregressive decoding, naturally produces a soft
probability, and avoids text-parsing brittleness.

### Ground truth (interim)

This commit deliberately does **not** include a strong (e.g. Opus 4.7)
judge step — that pass will run in a separate environment after the GPU
results land. Until then, ground-truth top-1 is the BT-MLE of the cached
Qwen `P` matrix. Algorithm trajectories don't depend on the GT label, so
overwriting `manifest.json["true_top_idx"]` later and rerunning `python
run.py 2c run` will refresh the headline metrics without recomputing the
expensive matrix.

### Commands

```bash
# Round-robin precompute (GPU; takes the bulk of the time).
python run.py 2c precompute --n 100 --run-id qwen7b

# Algorithm sweep over the cached oracle (CPU-only, fast).
python run.py 2c run --run-id qwen7b --trials 30

# Or end-to-end:
python run.py 2c all --run-id qwen7b
```

Useful flags:

- `--model`     HF model id; default `Qwen/Qwen2.5-7B-Instruct`. Pass `stub`
                for a length-based offline judge (no GPU, no network).
- `--smoke`     synthetic candidate pool + `stub` judge. Use to shake out the
                pipeline without weights.
- `--n`         candidate-pool size (default 100).
- `--budgets`   override the budget grid; default `[N, 2N, 5N, 10N]`.
- `--run-id`    subdirectory under `results/2c/`; lets you keep multiple runs.

### CPU/MPS smoke

```bash
python run.py 2c all --smoke --n 8 --trials 3 --run-id smoke
```

Runs end-to-end without `transformers` even being imported (the stub judge
is pure Python). Use this to sanity-check the orchestration before invoking
the real model.

### GPU sizing

| step                        | $N$ | pairs | calls (× 2 ordering) | rough wall-clock on a single A100 |
| --------------------------- | --: | ----: | -------------------: | --------------------------------: |
| precompute (Qwen2.5-7B bf16)| 100 | 4 950 |                9 900 | ~25–40 min                        |
| run (CPU only)              | 100 |   —   |                    — | ~5 min                            |

VRAM: Qwen2.5-7B in bf16 ≈ 15 GB. Fits a single A100 / H100 / 4090. On a
sub-12 GB GPU, switch to 4-bit via `transformers`’ `bitsandbytes` integration
(not enabled by default in this commit; add `load_in_4bit=True` to
`AutoModelForCausalLM.from_pretrained` if needed).

## Reproducibility

- Per-trial seed = `base_seed + trial_index` (default base seed = 434).
- Each algorithm sees the same per-trial seed → identical RNG stream where
  applicable.
- Skew-symmetric perturbation matrices in 2b are seeded by trial index, so
  ρ=0 and ρ=0.4 share the same `c` within a trial.
