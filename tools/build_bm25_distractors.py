"""Replace the 93 distractors in a single prompt subdir with BM25-retrieved
off-topic Nectar responses.

Concept (RAG framing):
  - For prompt P, BM25-search the rest of Nectar for the top-K most
    lexically similar *different* prompts.
  - From each retrieved Nectar row, take the rank-1 response (best).
  - Those K responses are real, well-formed, and lexically similar to P
    (high word overlap) -- but answer a different question. Exactly the
    "close-but-wrong" hard negatives a retriever would surface in a real
    pipeline.

What stays the same:
  - The 7 real Nectar responses to prompt P (kind="real").
  - Sonnet's listwise ranking on those 7 (already cached as
    manifest["sonnet_ranking_global"]).
  - Sonnet's pick (manifest["true_top_idx"]) -- remapped to the new
    global idx via real_local_idx.

What changes:
  - The 90 Gemma + 3 gibberish rows are replaced by 93 BM25 off-topic
    rows (kind="bm25_offtopic").
  - candidates.jsonl is shuffled + reindexed; manifest is updated.

Usage (run from repo root):
  python -m tools.build_bm25_distractors \
      --root results/v2/nectar_v2 --prompt 00 --n-distractors 93

This script does NOT need a GPU. It loads Nectar via HuggingFace
datasets and uses rank_bm25 if installed (falls back to sklearn TF-IDF
cosine otherwise -- both are CPU-only).

A backup of the previous candidates.jsonl is written to
candidates_pre_bm25.jsonl before overwriting.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class V2Candidate:
    idx: int
    kind: str
    prompt_id: int
    prompt_text: str
    response_text: str
    model_name: str
    source_prompt_id: int
    real_local_idx: int
    nectar_rank: int


# ---------------------------------------------------------------------------
# Tokenisation + BM25 / TF-IDF backends
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def _normalize_for_match(text: str) -> str:
    """Used to detect 'this is the same prompt as our target' so we can skip it."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _bm25_rank(query: str, corpus: List[str]) -> List[float]:
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        log.info("rank_bm25 not installed; falling back to sklearn TF-IDF cosine")
        return _tfidf_rank(query, corpus)
    tokenized_corpus = [_tokenize(d) for d in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return list(bm25.get_scores(_tokenize(query)))


def _tfidf_rank(query: str, corpus: List[str]) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9']+",
                          ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(corpus + [query])
    sims = linear_kernel(mat[-1:], mat[:-1]).ravel()
    return list(sims)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _load_candidates(path: Path) -> List[V2Candidate]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append(V2Candidate(**d))
    rows.sort(key=lambda r: r.idx)
    return rows


def _write_jsonl(path: Path, rows: List[V2Candidate]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def _load_nectar_corpus(split: str = "train", limit: Optional[int] = None
                        ) -> List[Tuple[str, str, str, int]]:
    """Returns a flat list of (prompt_text, best_response, model_name, rank)
    tuples, one per Nectar row, picking each row's rank-1 response.

    If `limit` is given, only the first `limit` rows are returned (after
    HuggingFace's deterministic shuffle is NOT applied -- we keep dataset
    order so the BM25 index is reproducible)."""
    from datasets import load_dataset
    log.info(f"Loading Nectar (berkeley-nest/Nectar, split={split}) ...")
    ds = load_dataset("berkeley-nest/Nectar", split=split)
    out: List[Tuple[str, str, str, int]] = []
    for row in ds:
        prompt = row.get("prompt") or row.get("instruction")
        answers = row.get("answers", [])
        if not prompt or not answers:
            continue
        # Pick the rank-1 answer (lowest 'rank' value). Nectar uses 1=best.
        best = None
        for a in answers:
            r = int(a.get("rank", 999))
            ans = (a.get("answer") or "").strip()
            if not ans:
                continue
            if best is None or r < best[2]:
                best = (ans, a.get("model", "unknown"), r)
        if best is None:
            continue
        out.append((prompt, best[0], best[1], best[2]))
        if limit is not None and len(out) >= limit:
            break
    log.info(f"  loaded {len(out)} Nectar rows")
    return out


def build_one(
    sub: Path,
    n_distractors: int,
    nectar_limit: Optional[int],
    seed: int,
):
    cand_path = sub / "candidates.jsonl"
    backup_path = sub / "candidates_pre_bm25.jsonl"
    manifest_path = sub / "manifest.json"

    rows = _load_candidates(cand_path)
    if not backup_path.exists():
        _write_jsonl(backup_path, rows)
        log.info(f"  backed up current pool -> {backup_path}")

    real_rows = sorted(
        [r for r in rows if r.kind == "real"],
        key=lambda r: r.real_local_idx,
    )
    if len(real_rows) != 7:
        raise RuntimeError(f"{sub}: expected 7 real rows, got {len(real_rows)}")

    manifest = json.load(open(manifest_path))
    old_true_top = int(manifest.get("true_top_idx", -1))
    real_idx_to_row = {r.idx: r for r in real_rows}
    if old_true_top not in real_idx_to_row:
        raise RuntimeError(
            f"{sub}: true_top_idx={old_true_top} not in real rows "
            f"{[r.idx for r in real_rows]}"
        )
    true_real_local = real_idx_to_row[old_true_top].real_local_idx
    log.info(f"  Sonnet's winner is real_local_idx={true_real_local} "
             f"(was global idx {old_true_top}); will remap.")

    target_prompt_text = real_rows[0].prompt_text
    target_norm = _normalize_for_match(target_prompt_text)

    # 1. Load Nectar corpus.
    corpus = _load_nectar_corpus(limit=nectar_limit)

    # 2. Drop any row whose prompt matches our target (so we never retrieve
    #    a response to the same question and accidentally call it off-topic).
    filtered = [(p, ans, m, r) for (p, ans, m, r) in corpus
                if _normalize_for_match(p) != target_norm]
    log.info(f"  filtered Nectar to {len(filtered)} rows after removing target-prompt matches")

    # 3. BM25 the prompts (we score by prompt similarity, not response similarity --
    #    we want lexically similar QUESTIONS that get DIFFERENT answers).
    prompt_corpus = [p for (p, _, _, _) in filtered]
    log.info(f"  BM25-ranking {len(prompt_corpus)} prompts ...")
    scores = _bm25_rank(target_prompt_text, prompt_corpus)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    if len(order) < n_distractors:
        raise RuntimeError(f"only {len(order)} Nectar prompts available, need {n_distractors}")

    # 4. Take top-N. Log a few so we can eyeball what's being retrieved.
    chosen = order[:n_distractors]
    log.info(f"  top BM25 hits (sample of 5):")
    for i in chosen[:5]:
        p_preview = filtered[i][0].replace("\n", " ").strip()[:120]
        log.info(f"    score={scores[i]:.3f}  prompt={p_preview!r}")

    # 5. Build new rows.
    new_rows: List[V2Candidate] = []
    for r in real_rows:
        new_rows.append(V2Candidate(
            idx=-1, kind="real", prompt_id=r.prompt_id,
            prompt_text=r.prompt_text, response_text=r.response_text,
            model_name=r.model_name, source_prompt_id=r.source_prompt_id,
            real_local_idx=r.real_local_idx, nectar_rank=r.nectar_rank,
        ))
    for rank_pos, ci in enumerate(chosen):
        (p, ans, model, nrank) = filtered[ci]
        new_rows.append(V2Candidate(
            idx=-1, kind="bm25_offtopic",
            prompt_id=real_rows[0].prompt_id,
            prompt_text=target_prompt_text,   # the candidate is "for" prompt P
            response_text=ans,                 # but the answer was written for a different prompt
            model_name=f"nectar_bm25@rank={rank_pos}|src_model={model}",
            source_prompt_id=-1,               # we don't track Nectar row idx; keep -1 (off-topic flag)
            real_local_idx=-1,
            nectar_rank=int(nrank),
        ))

    # 6. Shuffle + reindex.
    rng = random.Random(seed)
    rng.shuffle(new_rows)
    for new_idx, c in enumerate(new_rows):
        c.idx = new_idx

    # 7. Find Sonnet's pick at the new global idx.
    new_true_top = next(c.idx for c in new_rows
                        if c.kind == "real" and c.real_local_idx == true_real_local)
    new_real_global = sorted(c.idx for c in new_rows if c.kind == "real")

    _write_jsonl(cand_path, new_rows)

    # 8. Update manifest.
    manifest["n"] = len(new_rows)
    manifest["n_real"] = sum(c.kind == "real" for c in new_rows)
    manifest["n_random"] = 0
    manifest["n_gibberish"] = 0
    manifest["n_gemma_filler"] = 0
    manifest["n_bm25_offtopic"] = sum(c.kind == "bm25_offtopic" for c in new_rows)
    manifest["real_global_indices"] = new_real_global
    manifest["previous_true_top_idx"] = old_true_top
    manifest["true_top_idx"] = int(new_true_top)
    manifest["filler_version"] = "v4_bm25_offtopic"
    manifest["filler_model"] = "nectar_bm25"
    # If Sonnet's listwise ranking was stored, remap to new global ids.
    rank_path = sub / "sonnet_rank.json"
    if rank_path.exists():
        rank_doc = json.load(open(rank_path))
        loc_ranking = rank_doc.get("ranking_local")
        if loc_ranking is not None:
            local_to_new_global = {c.real_local_idx: c.idx
                                   for c in new_rows if c.kind == "real"}
            new_global_ranking = [local_to_new_global[loc] for loc in loc_ranking]
            manifest["sonnet_ranking_global"] = new_global_ranking
    json.dump(manifest, open(manifest_path, "w"), indent=2)

    log.info(f"  wrote {cand_path}: {len(new_rows)} rows "
             f"(7 real + {n_distractors} bm25_offtopic). "
             f"new true_top_idx={new_true_top} (was {old_true_top}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/v2/nectar_v2")
    ap.add_argument("--prompt", default="00",
                    help="Single prompt subdir name (e.g. '00').")
    ap.add_argument("--n-distractors", type=int, default=93)
    ap.add_argument("--nectar-limit", type=int, default=None,
                    help="Optional cap on Nectar rows to load (for fast dev).")
    ap.add_argument("--seed", type=int, default=434)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    sub = Path(args.root) / "prompts" / args.prompt
    if not sub.exists():
        raise FileNotFoundError(sub)
    log.info(f"[{args.prompt}] building BM25 distractors in {sub}")
    build_one(sub, n_distractors=args.n_distractors,
              nectar_limit=args.nectar_limit, seed=args.seed)


if __name__ == "__main__":
    main()
