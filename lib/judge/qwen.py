"""Qwen2.5-7B-Instruct as a pairwise judge.

Mechanism (standard pairwise-judge trick): format the prompt so the next
generated token is "A" or "B", then read just those two logits in a single
forward pass. We do not autoregressively generate — that's slower, gives
brittle text outputs that need parsing, and discards the soft probability.

Position-bias debiasing: every (i, j) pair is evaluated twice, once with
i=A,j=B and once with i=B,j=A. We return P(i beats j) = 0.5 * (P_AB(i is A
and chosen) + P_BA(i is B and chosen)).

Token cost: total prompt tokens consumed across the two orderings, plus the
2 "answer" tokens. This is the figure NEXT_PLAN §2c (token cost section)
asks for.

Device autodetect: cuda > mps > cpu. CUDA uses bf16 by default; MPS uses
fp16; CPU stays fp32.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# Standard pairwise-judge prompt template. Concise, explicit, and ends with
# "Answer: " so the model's next token is forced to be A or B.
JUDGE_TEMPLATE = (
    "You are an impartial judge comparing two AI responses to the same prompt. "
    "Choose the response that is more helpful, accurate, and well-written. "
    "If they are equal, pick whichever is marginally better. Answer with only "
    "the single letter A or B.\n\n"
    "[User prompt]\n{prompt}\n\n"
    "[Response A]\n{response_a}\n\n"
    "[Response B]\n{response_b}\n\n"
    "Better response (A or B): "
)


@dataclass
class JudgeOutput:
    """Soft pairwise probability and token accounting for one pair."""
    p_a_beats_b: float        # P(item passed as A is the chosen response)
    tokens_in: int
    tokens_out: int           # always 1 per call (we read a single logit)


def _autodetect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autodetect_dtype(device: str):
    import torch
    if device == "cuda":
        # Most modern CUDA GPUs do bf16 cleanly; fallback to fp16 on older cards.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


class QwenPairwiseJudge:
    """Stateless pairwise judge wrapping a HF causal LM.

    Use:
        judge = QwenPairwiseJudge(model_name="Qwen/Qwen2.5-7B-Instruct")
        p_ab, tin, tout = judge.judge(prompt, resp_a, resp_b)
        # P(item_a is preferred to item_b), debiased over (A,B) and (B,A).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        dtype=None,
        max_input_tokens: int = 6144,
        cache_dir: Optional[str] = None,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = device or _autodetect_device()
        self.dtype = dtype or _autodetect_dtype(self.device)
        self.max_input_tokens = max_input_tokens

        log.info(f"Loading {model_name} on {self.device} ({self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        # Padding side matters: we read logits at the last non-pad position.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # device_map='auto' is convenient on multi-GPU but disables custom
        # batching. For the CPU/MPS smoke path we keep things explicit.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Resolve "A" and "B" token ids. Many tokenizers prepend a leading
        # space when the token is mid-sentence, so we resolve both forms and
        # keep whichever is a single token.
        self.tok_a = self._resolve_letter_token("A")
        self.tok_b = self._resolve_letter_token("B")
        log.info(f"answer-token ids: A={self.tok_a}, B={self.tok_b}")

    def _resolve_letter_token(self, letter: str) -> int:
        for cand in (letter, " " + letter):
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # Fallback: take the first token id and warn.
        ids = self.tokenizer.encode(letter, add_special_tokens=False)
        log.warning(f"letter {letter!r} did not encode to a single token "
                    f"(got {ids}); falling back to first id.")
        return ids[0]

    def _format(self, prompt: str, response_a: str, response_b: str) -> str:
        # Truncate each response defensively so we don't blow past context.
        # Qwen2.5-7B context is 32k; we cap inputs at max_input_tokens to
        # keep token cost bounded for long-form responses.
        text = JUDGE_TEMPLATE.format(
            prompt=prompt[:8000],
            response_a=response_a[:8000],
            response_b=response_b[:8000],
        )
        return text

    @staticmethod
    def _softmax2(x_a: float, x_b: float) -> float:
        # Numerically stable 2-class softmax → P(A).
        m = max(x_a, x_b)
        ea = float(np.exp(x_a - m))
        eb = float(np.exp(x_b - m))
        return ea / (ea + eb)

    def _forward_one(self, text: str) -> Tuple[float, int]:
        """Run a single forward pass on `text` (which already ends with the
        "A or B): " stub) and return P(next token == A) and prompt-token count."""
        import torch
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        n_tokens = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model(**enc)
        # Last-position logits → score for A vs B.
        logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
        p_a = self._softmax2(float(logits[self.tok_a]),
                             float(logits[self.tok_b]))
        return p_a, n_tokens

    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        """Return P(item passed as A is preferred) averaged over both orderings.

        We pass A=response_a, B=response_b in the first call, and swap in the
        second. The two probabilities of "A wins" are averaged after correcting
        for the swap, yielding the position-bias-debiased estimate of
        P(response_a > response_b)."""
        text_ab = self._format(prompt, response_a, response_b)
        text_ba = self._format(prompt, response_b, response_a)
        p_ab_pickA, n_ab = self._forward_one(text_ab)
        p_ba_pickA, n_ba = self._forward_one(text_ba)
        # In text_ab, "A" is response_a → P(A wins) = P(a beats b).
        # In text_ba, "A" is response_b → P(A wins) = P(b beats a) = 1 - P(a beats b).
        p_a_beats_b = 0.5 * (p_ab_pickA + (1.0 - p_ba_pickA))
        return JudgeOutput(
            p_a_beats_b=float(p_a_beats_b),
            tokens_in=int(n_ab + n_ba),
            tokens_out=2,
        )


class StubPairwiseJudge:
    """Deterministic, no-GPU judge that compares response lengths.

    Used for offline tests of the orchestrator without loading a real model."""

    def __init__(self, **_kwargs):
        self.model_name = "stub:length"
        self.device = "cpu"

    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        # P(A wins) is a soft sigmoid of the length difference.
        diff = (len(response_a) - len(response_b)) / 200.0
        p = 1.0 / (1.0 + float(np.exp(-diff)))
        return JudgeOutput(p_a_beats_b=p, tokens_in=len(prompt) // 4, tokens_out=2)


def make_judge(model_name: str, device: Optional[str] = None) -> "QwenPairwiseJudge | StubPairwiseJudge":
    if model_name == "stub":
        return StubPairwiseJudge()
    return QwenPairwiseJudge(model_name=model_name, device=device)
