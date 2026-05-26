"""Replace the 93 random/gibberish distractors per prompt with on-topic
Gemma-3-4B-it samples at varied temperatures.

Why: a reviewer can dismiss random Nectar responses + gibberish as a
strawman because BM25 / a tiny cross-encoder would filter them in ms
before any LLM is invoked. We replace those 93 fillers with on-topic
hard negatives produced by a small instruct model at varied T.

For each prompt:
  - Keep the 7 real Nectar responses verbatim, preserving their kind="real"
    rows.  We re-shuffle global indices so reals are not always at the bottom
    or top, but we DO update manifest.json so true_top_idx points to the same
    response (Sonnet's pick is unchanged; only its global index moves).
  - Generate 90 Gemma responses with the temperature mix:
        30 @ T=0.3, 30 @ T=0.7, 23 @ T=1.0, 7 @ T=1.3
  - Add 3 gibberish responses (small but present RAG noise floor).
  Total = 100.

Output: overwrites prompts/<pp>/candidates.jsonl in-place. Updates
prompts/<pp>/manifest.json: kind counts, real_global_indices, true_top_idx.
A backup of the previous candidates.jsonl is written to candidates_v1.jsonl.

Run on the A100 box. Wall-clock estimate at N=100, 10 prompts:
  ~93 generations/prompt x 10 prompts = 930 generations.
  Gemma-3-4B at 512 max_new_tokens does ~50-80 tok/s on A100, so each
  generation is ~6-10 s; whole run ~2-3 hours. Batch sampling brings
  this down to ~30-45 min if you set --batch-size 8 (default).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

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


# ---------- gibberish (kept tiny: 3 per prompt) -----------------------------

GIBBERISH_TEMPLATES = [
    "i dont know lol",
    "{noise}",
    "yes.",
    "404 not found",
    "[REDACTED]",
    "Sorry I dont understand the question.",
    "Yes No Maybe So.",
]


def _make_gibberish(rng: random.Random) -> str:
    template = rng.choice(GIBBERISH_TEMPLATES)
    if "{noise}" in template:
        n = rng.randint(8, 60)
        noise = "".join(rng.choices(string.ascii_lowercase + " ", k=n)).strip()
        words, i = [], 0
        while i < len(noise):
            j = i + rng.randint(3, 7)
            words.append(noise[i:j])
            i = j
        template = template.replace("{noise}", " ".join(words))
    return template


# ---------- Gemma generation ------------------------------------------------

# (temperature, count) pairs. Sums to 90.
DEFAULT_TEMP_MIX = [
    (0.3, 30),
    (0.7, 30),
    (1.0, 23),
    (1.3, 7),
]


def _gemma_chat_template(prompt: str) -> str:
    """Format the Nectar 'Human:/Assistant:' prompt as a Gemma chat turn."""
    user_text = prompt.replace("\n\nHuman:", "").replace("\n\nAssistant:", "").strip()
    if not user_text:
        user_text = prompt.strip()
    return user_text


class GemmaSampler:
    """Tiny wrapper around Gemma-3-4B-it for batched, varied-T sampling."""

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: Optional[str] = None,
        dtype=None,
        max_new_tokens: int = 512,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else ("mps" if torch.backends.mps.is_available() else "cpu"))
        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        log.info(f"Loading {model_name} on {self.device} ({dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

    def _format(self, user_text: str) -> str:
        msgs = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    def sample_batch(
        self,
        user_text: str,
        temperatures: List[float],
        top_p: float = 0.95,
        seed_base: int = 0,
    ) -> List[str]:
        """Generate one response per temperature in the input list, batched.

        Returns a list of response strings in the same order as `temperatures`.
        """
        import torch

        text = self._format(user_text)
        # Replicate the same prompt B times so the batch is uniform.
        B = len(temperatures)
        enc = self.tokenizer(
            [text] * B, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        # Group same-temperature items into a single .generate() call.
        outputs = [None] * B
        # Build groups: temperature -> list of indices in the batch
        groups = {}
        for i, T in enumerate(temperatures):
            groups.setdefault(float(T), []).append(i)

        for T, idxs in groups.items():
            torch.manual_seed(seed_base + int(T * 1000) + idxs[0])
            sub_input_ids = enc["input_ids"][idxs]
            sub_attn = enc["attention_mask"][idxs]
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=sub_input_ids,
                    attention_mask=sub_attn,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=T,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = out[:, sub_input_ids.shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for k, i in enumerate(idxs):
                outputs[i] = decoded[k].strip()
        return outputs


# ---------- main loop -------------------------------------------------------

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


def _temperatures_list(mix) -> List[float]:
    out = []
    for T, n in mix:
        out.extend([T] * n)
    return out


def _dedup_by_prefix(texts: List[str], min_len: int = 60) -> List[str]:
    """Replace exact duplicates / strict prefixes of another response with
    None so the caller knows to regenerate them. Cheap normalisation only.
    """
    norm = [(t or "").strip() for t in texts]
    keep = [True] * len(texts)
    for i in range(len(norm)):
        if not norm[i] or len(norm[i]) < min_len:
            keep[i] = False
            continue
        for j in range(len(norm)):
            if i == j or not keep[j]:
                continue
            if norm[i] == norm[j] and i > j:
                keep[i] = False
                break
            if norm[i].startswith(norm[j]) and len(norm[j]) >= min_len * 0.6:
                keep[i] = False
                break
    return [t if k else None for t, k in zip(texts, keep)]


def regenerate_one_prompt(
    sub: Path,
    sampler: "GemmaSampler",
    temp_mix,
    n_gibberish: int,
    rng: random.Random,
    batch_size: int,
    max_regen_attempts: int = 3,
):
    cand_path = sub / "candidates.jsonl"
    backup_path = sub / "candidates_v1.jsonl"
    manifest_path = sub / "manifest.json"

    rows = _load_candidates(cand_path)
    if not backup_path.exists():
        # Preserve the v1 pool for audit before we overwrite.
        _write_jsonl(backup_path, rows)
        log.info(f"  backed up v1 -> {backup_path}")

    real_rows = [r for r in rows if r.kind == "real"]
    real_rows.sort(key=lambda r: r.real_local_idx)
    if len(real_rows) != 7:
        raise RuntimeError(f"{sub}: expected 7 real rows, got {len(real_rows)}")

    # Sanity: verify true_top_idx in manifest matches one of the real rows.
    manifest = json.load(open(manifest_path))
    old_true_top = int(manifest.get("true_top_idx", -1))
    real_idx_to_row = {r.idx: r for r in real_rows}
    if old_true_top not in real_idx_to_row:
        raise RuntimeError(f"{sub}: true_top_idx={old_true_top} not in real rows {[r.idx for r in real_rows]}")
    true_real_local = real_idx_to_row[old_true_top].real_local_idx
    log.info(f"  Sonnet's winner is real_local_idx={true_real_local} "
             f"(was global idx {old_true_top}); will remap.")

    prompt_text = real_rows[0].prompt_text
    user_text = _gemma_chat_template(prompt_text)

    # Build the temperature list for the 90 (or whatever) Gemma rows.
    temps_full = _temperatures_list(temp_mix)
    n_gemma = len(temps_full)
    log.info(f"  generating {n_gemma} Gemma rows (T mix={temp_mix})")

    # Generate, with a few regeneration passes if we hit duplicates / empties.
    gen_texts: List[str] = [None] * n_gemma
    needed_indices = list(range(n_gemma))
    attempt = 0
    while needed_indices and attempt < max_regen_attempts:
        log.info(f"  attempt {attempt+1}: need {len(needed_indices)} samples")
        # Process in batches of `batch_size`.
        for start in range(0, len(needed_indices), batch_size):
            chunk = needed_indices[start:start + batch_size]
            T_chunk = [temps_full[i] for i in chunk]
            outs = sampler.sample_batch(
                user_text=user_text,
                temperatures=T_chunk,
                seed_base=rng.randint(0, 10**9) + attempt * 7919,
            )
            for slot, txt in zip(chunk, outs):
                gen_texts[slot] = txt
        # Light dedup pass: empties and obvious duplicates trigger a regen.
        deduped = _dedup_by_prefix(gen_texts)
        gen_texts = [d for d in deduped]
        needed_indices = [i for i, t in enumerate(gen_texts) if not t]
        attempt += 1

    # Anything still missing after max_regen_attempts: keep whatever we got
    # (may be a duplicate); it's already on-topic which is the main goal.
    for i in range(n_gemma):
        if not gen_texts[i]:
            gen_texts[i] = sampler.sample_batch(
                user_text=user_text, temperatures=[temps_full[i]],
                seed_base=rng.randint(0, 10**9),
            )[0] or "I am not sure how to answer this."

    # Build new candidate rows: 7 real (kept verbatim) + n_gemma gemma + n_gibberish
    new_rows: List[V2Candidate] = []
    for r in real_rows:
        new_rows.append(V2Candidate(
            idx=-1, kind="real", prompt_id=r.prompt_id,
            prompt_text=r.prompt_text, response_text=r.response_text,
            model_name=r.model_name, source_prompt_id=r.source_prompt_id,
            real_local_idx=r.real_local_idx, nectar_rank=r.nectar_rank,
        ))
    for i, txt in enumerate(gen_texts):
        new_rows.append(V2Candidate(
            idx=-1, kind="gemma_filler", prompt_id=real_rows[0].prompt_id,
            prompt_text=prompt_text, response_text=txt,
            model_name=f"{sampler.model_name}@T={temps_full[i]:.1f}",
            source_prompt_id=real_rows[0].prompt_id,
            real_local_idx=-1, nectar_rank=0,
        ))
    for _ in range(n_gibberish):
        new_rows.append(V2Candidate(
            idx=-1, kind="gibberish", prompt_id=real_rows[0].prompt_id,
            prompt_text=prompt_text, response_text=_make_gibberish(rng),
            model_name="gibberish", source_prompt_id=-1,
            real_local_idx=-1, nectar_rank=0,
        ))

    # Shuffle + reindex.
    rng.shuffle(new_rows)
    for new_idx, c in enumerate(new_rows):
        c.idx = new_idx

    # Find the new global idx of Sonnet's winner via real_local_idx.
    new_true_top = next(c.idx for c in new_rows
                        if c.kind == "real" and c.real_local_idx == true_real_local)
    new_real_global = sorted(c.idx for c in new_rows if c.kind == "real")

    _write_jsonl(cand_path, new_rows)

    # Update manifest.
    manifest["n"] = len(new_rows)
    manifest["n_real"] = sum(c.kind == "real" for c in new_rows)
    manifest["n_random"] = 0
    manifest["n_gibberish"] = sum(c.kind == "gibberish" for c in new_rows)
    manifest["n_gemma_filler"] = sum(c.kind == "gemma_filler" for c in new_rows)
    manifest["real_global_indices"] = new_real_global
    manifest["previous_true_top_idx"] = old_true_top
    manifest["true_top_idx"] = int(new_true_top)
    manifest["filler_model"] = sampler.model_name
    manifest["filler_temperature_mix"] = [list(p) for p in temp_mix]
    manifest["filler_version"] = "v2_gemma"
    # If sonnet_ranking_global was stored, remap to new global ids using
    # real_local_idx -> new global idx.
    rank_path = sub / "sonnet_rank.json"
    if rank_path.exists():
        rank = json.load(open(rank_path))
        loc_ranking = rank.get("ranking_local")
        if loc_ranking is not None:
            local_to_new_global = {c.real_local_idx: c.idx for c in new_rows if c.kind == "real"}
            new_global_ranking = [local_to_new_global[loc] for loc in loc_ranking]
            manifest["sonnet_ranking_global"] = new_global_ranking
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    log.info(f"  wrote {cand_path}; new true_top_idx={new_true_top}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/v2/nectar_v2")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-gemma", type=int, default=90,
                   help="total Gemma fillers per prompt (must equal sum(temp_mix))")
    p.add_argument("--n-gibberish", type=int, default=3)
    p.add_argument("--seed", type=int, default=434)
    p.add_argument("--only", default=None,
                   help="comma-separated prompt subdir indices; default all")
    p.add_argument("--no-regen-if-already-v2", action="store_true",
                   help="skip prompts whose manifest reports filler_version=v2_gemma")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if sum(n for _, n in DEFAULT_TEMP_MIX) != args.n_gemma:
        raise SystemExit(f"DEFAULT_TEMP_MIX sums to {sum(n for _,n in DEFAULT_TEMP_MIX)} "
                         f"but --n-gemma={args.n_gemma}. Edit DEFAULT_TEMP_MIX or pass matching --n-gemma.")

    root = Path(args.root)
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        raise SystemExit(f"no {prompts_dir}; run build_v2_dataset.py first.")

    sub_dirs = sorted([d for d in prompts_dir.iterdir() if d.is_dir()])
    if args.only:
        keep = set(args.only.split(","))
        sub_dirs = [d for d in sub_dirs if d.name in keep or d.name.lstrip("0") in keep]
    log.info(f"will process {len(sub_dirs)} prompt subdirectories")

    sampler = GemmaSampler(
        model_name=args.model, device=args.device, max_new_tokens=args.max_new_tokens,
    )
    rng = random.Random(args.seed)

    t0 = time.time()
    for i, sub in enumerate(sub_dirs):
        log.info(f"=== [{i+1}/{len(sub_dirs)}] {sub} ===")
        if args.no_regen_if_already_v2:
            mf = json.load(open(sub / "manifest.json"))
            if mf.get("filler_version") == "v2_gemma":
                log.info(f"  skip (already v2_gemma)")
                continue
        regenerate_one_prompt(
            sub=sub, sampler=sampler, temp_mix=DEFAULT_TEMP_MIX,
            n_gibberish=args.n_gibberish, rng=rng, batch_size=args.batch_size,
        )
    log.info(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
