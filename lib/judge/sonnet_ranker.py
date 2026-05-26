"""Sonnet listwise ranker — takes a prompt + N responses, returns a ranking.

Used to set `true_top_idx` directly from Sonnet over the 7 real Nectar
responses for each prompt, without running pairwise comparisons.

The model returns a permutation 1..N where 1 = best. We parse the JSON and
fall back to a regex if the model wraps text around it.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


RANK_TEMPLATE = (
    "You are an impartial judge. Rank the following AI responses to the same "
    "prompt from BEST (rank 1) to WORST (rank {n}). Use only helpfulness, "
    "accuracy, and writing quality. Output ONLY a JSON list of indices in "
    "ranked order, e.g. [3, 0, 5, 2, 4, 1, 6]. Do not output anything else.\n\n"
    "[User prompt]\n{prompt}\n\n"
    "{responses_block}\n"
    "Ranked indices (best first), JSON list only: "
)


DEFAULT_MODELS = {
    "sonnet":   "global.anthropic.claude-sonnet-4-6",
    "sonnet45": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "opus":     "global.anthropic.claude-opus-4-7",
    "haiku":    "anthropic.claude-haiku-4-5-20251001-v1:0",
}


@dataclass
class RankOutput:
    ranking: List[int]
    tokens_in: int
    tokens_out: int


class SonnetListwiseRanker:
    def __init__(
        self,
        model: str = "sonnet",
        max_retries: int = 6,
        retry_base: float = 1.0,
        max_chars_per_response: int = 6000,
    ):
        import boto3
        self.client = boto3.client("bedrock-runtime")
        self.model = model
        self.model_id = DEFAULT_MODELS.get(model, model)
        self.max_retries = max_retries
        self.retry_base = retry_base
        self.max_chars_per_response = max_chars_per_response

    def _format(self, prompt: str, responses: List[str]) -> str:
        block_lines = []
        for i, r in enumerate(responses):
            r_trim = (r or "").strip()[: self.max_chars_per_response]
            block_lines.append(f"[Response {i}]\n{r_trim}")
        return RANK_TEMPLATE.format(
            n=len(responses),
            prompt=prompt[:8000],
            responses_block="\n\n".join(block_lines),
        )

    def _inference_config(self):
        cfg = {"maxTokens": 256}
        if "opus-4-7" not in self.model_id and "opus-4-6" not in self.model_id:
            cfg["temperature"] = 0.0
        return cfg

    def _invoke(self, text: str):
        for attempt in range(self.max_retries):
            try:
                return self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": text}]}],
                    inferenceConfig=self._inference_config(),
                )
            except Exception as e:
                msg = str(e)
                retryable = any(s in msg for s in (
                    "Throttling", "ServiceUnavailable", "InternalServer",
                    "ThrottlingException", "TooManyRequests", "ServiceQuotaExceeded",
                ))
                if not retryable or attempt == self.max_retries - 1:
                    raise
                wait = self.retry_base * (2 ** attempt)
                log.warning(f"Bedrock retry {attempt+1}/{self.max_retries} in {wait:.1f}s: {msg[:120]}")
                time.sleep(wait)

    @staticmethod
    def _parse(text: str, n: int) -> Optional[List[int]]:
        m = re.search(r"\[[^\[\]]*\]", text)
        if not m:
            return None
        try:
            arr = json.loads(m.group(0))
        except Exception:
            return None
        if not isinstance(arr, list):
            return None
        out = []
        for x in arr:
            try:
                v = int(x)
            except Exception:
                return None
            if v < 0 or v >= n or v in out:
                return None
            out.append(v)
        if len(out) != n:
            return None
        return out

    def rank(self, prompt: str, responses: List[str]) -> RankOutput:
        text = self._format(prompt, responses)
        resp = self._invoke(text)
        usage = resp.get("usage") or {}
        tokens_in = int(usage.get("inputTokens", 0))
        tokens_out = int(usage.get("outputTokens", 0))
        content = resp["output"]["message"]["content"]
        raw = "".join((b.get("text") or "") for b in content)
        parsed = self._parse(raw, n=len(responses))
        if parsed is None:
            log.warning(f"Sonnet ranking failed to parse: {raw!r}")
            parsed = list(range(len(responses)))
        return RankOutput(ranking=parsed, tokens_in=tokens_in, tokens_out=tokens_out)
