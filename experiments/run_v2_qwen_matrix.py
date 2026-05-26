"""GPU step for the v2 per-prompt 2c experiment.

For each prompt directory under <root>/prompts/<pp>/ that contains a
candidates.jsonl built by build_v2_dataset.py, this script runs Qwen-7B
over all C(N,2) pairs (with (A,B)+(B,A) debiasing) and saves a frozen
roundrobin.npz to that directory.

Run on a single A100. Wall-clock per prompt at N=100: ~11 minutes.

Usage:
  python -m experiments.run_v2_qwen_matrix --root results/v2/nectar_v2
  python -m experiments.run_v2_qwen_matrix --root results/v2/nectar_v2 --only 0,1,2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def _load_candidates(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["idx"])
    return rows


def _all_pairs(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def run_one_prompt(
    sub_dir: Path,
    judge,
    overwrite: bool = False,
):
    cand_path = sub_dir / "candidates.jsonl"
    out_path = sub_dir / "roundrobin.npz"
    if not cand_path.exists():
        log.warning(f"skip {sub_dir} (no candidates.jsonl)")
        return None
    if out_path.exists() and not overwrite:
        log.info(f"skip {sub_dir} (roundrobin.npz exists; pass --overwrite to redo)")
        return None

    candidates = _load_candidates(cand_path)
    N = len(candidates)
    log.info(f"  N = {N}; running C(N,2) = {N*(N-1)//2} pairs")

    P = np.full((N, N), 0.5, dtype=np.float64)
    C = np.zeros((N, N), dtype=np.int64)

    pairs = list(_all_pairs(N))
    t0 = time.time()
    for i, j in tqdm(pairs, ncols=80, desc=f"qwen RR {sub_dir.name}"):
        out = judge.judge(
            prompt=candidates[i]["prompt_text"],
            response_a=candidates[i]["response_text"],
            response_b=candidates[j]["response_text"],
        )
        P[i, j] = out.p_a_beats_b
        P[j, i] = 1.0 - out.p_a_beats_b
        C[i, j] = out.tokens_in + out.tokens_out
        C[j, i] = C[i, j]
    elapsed = time.time() - t0
    log.info(f"  done in {elapsed/60:.1f} min "
             f"({len(pairs)/max(elapsed, 1e-6):.2f} pairs/s)")

    np.savez(out_path, P=P, C=C)
    log.info(f"  saved {out_path}")

    # quick sanity: BT-MLE top-5 + the Sonnet true_top_idx if we have it.
    try:
        from experiments.run_2c_llm_judge import bt_mle_from_p
        bt = bt_mle_from_p(P)
        top5 = np.argsort(-bt)[:5]
        manifest_path = sub_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.load(open(manifest_path))
        manifest["qwen_bt_top5"] = [
            (int(i), float(bt[i]), candidates[i]["model_name"], candidates[i]["kind"])
            for i in top5
        ]
        manifest["qwen_runtime_s"] = float(elapsed)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        log.warning(f"BT-MLE diagnostic failed (not fatal): {e}")

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/v2/nectar_v2")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--only", default=None,
                   help="comma-separated list of prompt subdir indices to run; default all")
    p.add_argument("--overwrite", action="store_true",
                   help="redo prompts that already have roundrobin.npz")
    p.add_argument("--smoke", action="store_true",
                   help="use the StubPairwiseJudge (no GPU); for orchestration tests only")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.root)
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        raise SystemExit(f"no {prompts_dir}; run build_v2_dataset.py first.")

    sub_dirs = sorted([d for d in prompts_dir.iterdir() if d.is_dir()])
    if args.only:
        keep = set(args.only.split(","))
        sub_dirs = [d for d in sub_dirs if d.name in keep or d.name.lstrip("0") in keep]
    log.info(f"will process {len(sub_dirs)} prompt subdirectories")

    if args.smoke:
        from lib.judge.qwen import StubPairwiseJudge
        judge = StubPairwiseJudge()
    else:
        from lib.judge.qwen import QwenPairwiseJudge
        judge = QwenPairwiseJudge(model_name=args.model, device=args.device)

    for sub in sub_dirs:
        log.info(f"=== {sub} ===")
        run_one_prompt(sub, judge, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
