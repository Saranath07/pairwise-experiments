"""Build the v2 per-prompt 2c dataset.

For each of K prompts taken from Nectar:
  * 7 REAL responses to that prompt (from Nectar).
  * ~80 RANDOM responses sampled from OTHER prompts in Nectar (off-topic
    distractors -- they are real LLM text but for a different question).
  * fill the rest (default 13) with GIBBERISH responses (deliberately bad).
Total = N (default 100) candidates per prompt.

Then call Sonnet listwise rank on the 7 real responses, and store the
result as `true_top_idx` (the global candidate index of the best real
response according to Sonnet).

CPU + AWS Bedrock only. No GPU needed for this step.

Outputs (under results/v2/<run_id>/prompts/<prompt_idx>/):
  candidates.jsonl       N rows: idx, kind in {real,random,gibberish},
                         prompt_text, response_text, model_name, source_prompt_id,
                         real_local_idx (0..6 for real rows, else -1)
  manifest.json          true_top_idx, sonnet ranking over the 7 reals,
                         counts per kind, etc.
  sonnet_rank.json       raw ranker output for audit
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class V2Candidate:
    idx: int
    kind: str                 # "real" | "random" | "gibberish"
    prompt_id: int            # the Nectar row this prompt came from (target prompt)
    prompt_text: str          # always the target prompt
    response_text: str
    model_name: str
    source_prompt_id: int     # for "real" == prompt_id; for "random" == the row the response came from; for "gibberish" == -1
    real_local_idx: int       # 0..6 for real rows (Nectar's own ordering); -1 otherwise
    nectar_rank: int          # Nectar's GPT-4 rank for real; 0 for distractors


# --------- gibberish templates ----------------------------------------------

GIBBERISH_TEMPLATES = [
    "i dont know lol",
    "{noise}",
    "yes.",
    "no comment.",
    "the the the the the the the the.",
    "asdf qwerty zxcv jkl asdf qwerty.",
    "{noise} {noise} {noise}",
    "404 not found",
    "as an AI language model i cannot help with that. {noise}",
    "[REDACTED]",
    "Please rephrase your question.",
    "Sorry I dont understand the question.",
    "{noise}\n{noise}\n{noise}",
    "Yes No Maybe So.",
]


def _make_gibberish(rng: random.Random) -> str:
    template = rng.choice(GIBBERISH_TEMPLATES)
    if "{noise}" in template:
        n = rng.randint(8, 60)
        noise = "".join(rng.choices(string.ascii_lowercase + " ", k=n)).strip()
        # token-style mash: split into 'words' of len 3-7
        words = []
        i = 0
        while i < len(noise):
            j = i + rng.randint(3, 7)
            words.append(noise[i:j])
            i = j
        template = template.replace("{noise}", " ".join(words))
    return template


# ---------- Nectar helpers --------------------------------------------------

def _load_nectar_rows(seed: int, dataset_split: str = "train"):
    from datasets import load_dataset
    log.info("Loading berkeley-nest/Nectar...")
    ds = load_dataset("berkeley-nest/Nectar", split=dataset_split)
    ds = ds.shuffle(seed=seed)
    return ds


def _select_target_prompts(ds, k: int, min_real: int = 7, max_chars: int = 4000):
    """Pick k Nectar rows that have >= min_real responses each."""
    chosen = []
    for row_idx, row in enumerate(ds):
        if len(chosen) >= k:
            break
        prompt = row.get("prompt") or row.get("instruction")
        if not prompt or len(prompt) > max_chars:
            continue
        answers = row.get("answers", [])
        good = [a for a in answers if (a.get("answer") or "").strip()]
        if len(good) < min_real:
            continue
        chosen.append((row_idx, prompt, good[:min_real]))
    if len(chosen) < k:
        raise RuntimeError(f"Only found {len(chosen)} prompts with >= {min_real} responses; needed {k}.")
    return chosen


def _collect_distractor_pool(ds, target_row_ids: set, n_needed: int, max_chars: int = 6000):
    """Pull responses from OTHER Nectar rows to use as off-topic distractors."""
    pool = []
    for row_idx, row in enumerate(ds):
        if row_idx in target_row_ids:
            continue
        for ans in row.get("answers", []):
            txt = (ans.get("answer") or "").strip()
            if not txt or len(txt) > max_chars:
                continue
            pool.append({
                "response_text": txt,
                "model_name": ans.get("model", "unknown"),
                "source_prompt_id": row_idx,
                "nectar_rank": int(ans.get("rank", 0)),
            })
            if len(pool) >= n_needed:
                return pool
    return pool


# ---------- Main builder ----------------------------------------------------

def build_per_prompt(
    target_row_idx: int,
    target_prompt: str,
    real_answers: List[dict],
    distractor_pool: List[dict],
    n_random: int,
    n_gibberish: int,
    rng: random.Random,
) -> List[V2Candidate]:
    cands: List[V2Candidate] = []

    # 1) real responses
    for li, ans in enumerate(real_answers):
        cands.append(V2Candidate(
            idx=-1,  # set below
            kind="real",
            prompt_id=target_row_idx,
            prompt_text=target_prompt,
            response_text=(ans.get("answer") or "").strip(),
            model_name=ans.get("model", "unknown"),
            source_prompt_id=target_row_idx,
            real_local_idx=li,
            nectar_rank=int(ans.get("rank", 0)),
        ))

    # 2) random distractors
    if n_random > len(distractor_pool):
        raise RuntimeError(f"distractor pool too small: have {len(distractor_pool)}, need {n_random}")
    picks = rng.sample(distractor_pool, n_random)
    for p in picks:
        cands.append(V2Candidate(
            idx=-1,
            kind="random",
            prompt_id=target_row_idx,
            prompt_text=target_prompt,
            response_text=p["response_text"],
            model_name=p["model_name"],
            source_prompt_id=p["source_prompt_id"],
            real_local_idx=-1,
            nectar_rank=p["nectar_rank"],
        ))

    # 3) gibberish
    for _ in range(n_gibberish):
        cands.append(V2Candidate(
            idx=-1,
            kind="gibberish",
            prompt_id=target_row_idx,
            prompt_text=target_prompt,
            response_text=_make_gibberish(rng),
            model_name="gibberish",
            source_prompt_id=-1,
            real_local_idx=-1,
            nectar_rank=0,
        ))

    # Shuffle in place so reals are not always at the bottom 0..6.
    rng.shuffle(cands)
    for new_idx, c in enumerate(cands):
        c.idx = new_idx

    return cands


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def _rank_reals_with_sonnet(
    prompt: str,
    cands: List[V2Candidate],
    model: str,
):
    from lib.judge.sonnet_ranker import SonnetListwiseRanker
    real_cands = [c for c in cands if c.kind == "real"]
    real_cands.sort(key=lambda c: c.real_local_idx)
    responses = [c.response_text for c in real_cands]
    ranker = SonnetListwiseRanker(model=model)
    out = ranker.rank(prompt=prompt, responses=responses)
    # ranking is in 0..len(reals)-1 over the local index passed in
    best_local = out.ranking[0]
    best_global = real_cands[best_local].idx
    return {
        "ranking_local": out.ranking,
        "ranking_global": [real_cands[r].idx for r in out.ranking],
        "best_real_global_idx": best_global,
        "tokens_in": out.tokens_in,
        "tokens_out": out.tokens_out,
        "responses_count": len(responses),
        "model": model,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/v2/nectar_v2")
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--n", type=int, default=100, help="candidates per prompt")
    p.add_argument("--n-real", type=int, default=7)
    p.add_argument("--n-random", type=int, default=80)
    p.add_argument("--n-gibberish", type=int, default=13)
    p.add_argument("--seed", type=int, default=434)
    p.add_argument("--ranker-model", default="sonnet")
    p.add_argument("--skip-sonnet", action="store_true",
                   help="Build pools only; do not call Sonnet (useful for offline prep).")
    args = p.parse_args()

    if args.n_real + args.n_random + args.n_gibberish != args.n:
        raise SystemExit(f"n_real+n_random+n_gibberish ({args.n_real}+{args.n_random}+{args.n_gibberish}) "
                         f"must equal --n ({args.n})")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    ds = _load_nectar_rows(seed=args.seed)

    log.info(f"Selecting {args.num_prompts} target prompts (>= {args.n_real} responses each)...")
    targets = _select_target_prompts(ds, k=args.num_prompts, min_real=args.n_real)
    target_row_ids = {row_idx for (row_idx, _, _) in targets}

    # We need num_prompts * n_random distractor responses, give a small safety margin.
    distractor_needed = args.num_prompts * args.n_random + 200
    log.info(f"Collecting distractor pool of size {distractor_needed}...")
    distractor_pool = _collect_distractor_pool(ds, target_row_ids, distractor_needed)
    log.info(f"Got {len(distractor_pool)} distractor responses.")

    summary = {
        "num_prompts": args.num_prompts,
        "n": args.n,
        "n_real": args.n_real,
        "n_random": args.n_random,
        "n_gibberish": args.n_gibberish,
        "seed": args.seed,
        "ranker_model": args.ranker_model,
        "prompts": [],
    }

    for pi, (row_idx, prompt, real_ans) in enumerate(targets):
        sub = out_root / "prompts" / f"{pi:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        log.info(f"[{pi+1}/{args.num_prompts}] target_row={row_idx}  reals={len(real_ans)}")

        cands = build_per_prompt(
            target_row_idx=row_idx,
            target_prompt=prompt,
            real_answers=real_ans,
            distractor_pool=distractor_pool,
            n_random=args.n_random,
            n_gibberish=args.n_gibberish,
            rng=rng,
        )
        _write_jsonl(sub / "candidates.jsonl", cands)

        manifest = {
            "prompt_idx": pi,
            "target_nectar_row_idx": row_idx,
            "prompt_text": prompt,
            "n": len(cands),
            "n_real": sum(c.kind == "real" for c in cands),
            "n_random": sum(c.kind == "random" for c in cands),
            "n_gibberish": sum(c.kind == "gibberish" for c in cands),
            "real_global_indices": sorted(c.idx for c in cands if c.kind == "real"),
        }

        if not args.skip_sonnet:
            log.info(f"  ranking 7 real responses with Sonnet ({args.ranker_model})...")
            rank_info = _rank_reals_with_sonnet(prompt, cands, model=args.ranker_model)
            with open(sub / "sonnet_rank.json", "w") as f:
                json.dump(rank_info, f, indent=2)
            manifest["sonnet_ranking_global"] = rank_info["ranking_global"]
            manifest["true_top_idx"] = rank_info["best_real_global_idx"]
            manifest["ranker_tokens_in"] = rank_info["tokens_in"]
            manifest["ranker_tokens_out"] = rank_info["tokens_out"]
            log.info(f"  Sonnet picked global idx {manifest['true_top_idx']}")
        else:
            manifest["true_top_idx"] = None
            manifest["sonnet_ranking_global"] = None

        with open(sub / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        summary["prompts"].append({
            "prompt_idx": pi,
            "target_nectar_row_idx": row_idx,
            "true_top_idx": manifest["true_top_idx"],
            "real_global_indices": manifest["real_global_indices"],
        })

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Done. Summary -> {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
