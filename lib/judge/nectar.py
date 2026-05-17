"""Nectar dataset loader for the §2c LLM-as-judge experiment.

Per NEXT_PLAN §2c (Phase 2 setup recommendations):
  - Nectar has ~7 candidate completions per prompt across diverse models.
  - We pool ~15 prompts to reach N=100 candidates.

Each candidate carries:
  - idx              (0-based index into the candidate pool)
  - prompt_id        (which Nectar row it came from)
  - prompt_text
  - response_text
  - model_name       (originator)
  - nectar_rank      (Nectar's GPT-4 ranking inside its own row, 1=best)

The loader caches its sample to JSONL so a precompute and an algorithm run
share the same N=100 pool deterministically.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    idx: int
    prompt_id: int
    prompt_text: str
    response_text: str
    model_name: str
    nectar_rank: int


def _save(path: Path, candidates: List[Candidate]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c)) + "\n")


def _load(path: Path) -> List[Candidate]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(Candidate(**d))
    return out


def build_candidate_pool(
    n: int = 100,
    cache_path: Optional[Path] = None,
    seed: int = 434,
    dataset_split: str = "train",
    min_models_per_prompt: int = 4,
) -> List[Candidate]:
    """Sample N candidates from Nectar.

    Strategy: shuffle Nectar rows under `seed`, walk through them, take all
    `min_models_per_prompt`+ responses from each, until we hit N candidates.
    """
    if cache_path is not None and Path(cache_path).exists():
        log.info(f"Loading cached Nectar candidates from {cache_path}")
        return _load(Path(cache_path))

    from datasets import load_dataset

    log.info("Loading Nectar (berkeley-nest/Nectar)…")
    ds = load_dataset("berkeley-nest/Nectar", split=dataset_split)

    # Stable shuffle under our seed.
    ds = ds.shuffle(seed=seed)

    candidates: List[Candidate] = []
    used_prompt_ids = []

    for row_idx, row in enumerate(ds):
        if len(candidates) >= n:
            break
        prompt_text = row.get("prompt") or row.get("instruction")
        answers = row.get("answers", [])
        if not prompt_text or not answers:
            continue
        if len(answers) < min_models_per_prompt:
            continue

        for ans in answers:
            if len(candidates) >= n:
                break
            response = (ans.get("answer") or "").strip()
            model = ans.get("model", "unknown")
            rank = int(ans.get("rank", 0))
            if not response:
                continue
            candidates.append(Candidate(
                idx=len(candidates),
                prompt_id=row_idx,
                prompt_text=prompt_text,
                response_text=response,
                model_name=model,
                nectar_rank=rank,
            ))
        used_prompt_ids.append(row_idx)

    log.info(f"Built pool of {len(candidates)} candidates "
             f"from {len(used_prompt_ids)} prompts.")

    if cache_path is not None:
        _save(Path(cache_path), candidates)
        log.info(f"Cached pool to {cache_path}")

    return candidates


def synthesize_candidate_pool(n: int, cache_path: Optional[Path] = None,
                              seed: int = 434) -> List[Candidate]:
    """Synthetic pool used by smoke tests (no `datasets` dependency).

    Builds N candidates: a single short prompt and N responses of varying
    length (longer = "more thorough" by stub-judge convention). The
    StubPairwiseJudge will rank longer responses higher, so the synthetic
    "true winner" is candidate index N-1."""
    import random
    rng = random.Random(seed)
    prompt = "Explain why the sky is blue in plain English."
    cands: List[Candidate] = []
    for i in range(n):
        # Length grows with i, plus light noise so all-equal tie cases don't dominate.
        base_len = 30 + 6 * i
        jitter = rng.randint(-3, 3)
        body = ("Light from the sun is scattered by the atmosphere. " * 2)
        text = (body * max(1, (base_len + jitter) // 50))[: base_len + jitter]
        cands.append(Candidate(
            idx=i,
            prompt_id=0,
            prompt_text=prompt,
            response_text=text,
            model_name=f"stub-model-{i}",
            nectar_rank=i + 1,
        ))
    if cache_path is not None:
        _save(Path(cache_path), cands)
    return cands
