"""
llm_clients.py — real LLM clients.

Why REST instead of the google-genai SDK:
  numpy is this project's only third-party dependency. `pip install` failing
  in an interview environment is a real event, and every extra dependency is
  another way to lose time. The code below talks to the API over urllib with
  zero dependencies, and - just as importantly - it can be fully tested
  against a local fake server, rather than being "written but never run".
  If you prefer the SDK, replace _post; nothing else changes.

Three guards are built in. Each one corresponds to a real trap that fails
silently rather than loudly:

  1. Embedding count check
     gemini-embedding-2 returns a SINGLE aggregated vector when given
     multiple inputs. The _raw_embed contract is "pass N, get N back", and a
     count mismatch would make the downstream zip() drop data without any
     error. This asserts on it and returns an actionable message.

  2. Embedding dimension consistency check
     Vectors from different embed_models mixed into one cache make numpy
     raise an inhomogeneous-shape error inside retrieve(), killing the whole
     run. The dimension is locked on first call.

  3. finishReason check
     A MAX_TOKENS truncation produces a string that looks like JSON but is
     cut in half; parsing fails and the entire batch degrades. This detects
     it explicitly and tells you to raise max_output_tokens.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from engine import LLMClient, SearchConfig

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient(LLMClient):
    """
    Chat  : {base}/models/{model}:generateContent
    Embed : {base}/models/{model}:batchEmbedContents
            Uses the batch endpoint rather than passing a list to
            embedContent, which sidesteps the aggregation trap by
            construction: each request item yields its own vector.
    """

    def __init__(
        self,
        cache,
        config: SearchConfig,
        api_key: str,
        embed_key: str = "",
        embed_dim: Optional[int] = 768,
        base_url: str = GEMINI_BASE,
        max_output_tokens: int = 8192,
        timeout: float = 90.0,
    ):
        super().__init__(cache, config)
        self.api_key = api_key
        self.embed_key = embed_key or api_key
        self.embed_dim = embed_dim
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self._locked_dim: Optional[int] = None

    # ------------------------------------------------------------------
    def _post(self, path: str, key: str, body: Dict) -> Dict:
        url = f"{self.base_url}/{path}?key={key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            if e.code == 429:
                raise RuntimeError(f"RATE_LIMIT (429): {detail}") from e
            if e.code in (401, 403):
                raise RuntimeError(
                    f"AUTH ({e.code}): invalid key, or the API is not enabled. "
                    f"Check "
                    f"GEMINI_API_KEY。{detail}"
                ) from e
            if e.code == 404:
                raise RuntimeError(
                    f"MODEL_NOT_FOUND (404): model name may be retired or misspelled - {detail}"
                ) from e
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    # ------------------------------------------------------------------
    async def _raw_chat(self, system: str, user: str) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                # Note: recent Gemini models deprecate temperature / top_p /
                # top_k - do not send them.
                "responseMimeType": "application/json",
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        data = await asyncio.to_thread(
            self._post, f"models/{self.config.llm_model}:generateContent", self.api_key, body
        )

        if pf := data.get("promptFeedback", {}).get("blockReason"):
            raise RuntimeError(
                f"PROMPT_BLOCKED: {pf} (a candidate profile tripped the safety filter)")

        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"NO_CANDIDATES: {json.dumps(data)[:300]}")

        reason = cands[0].get("finishReason", "")
        parts = cands[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        if reason == "MAX_TOKENS":
            raise RuntimeError(
                f"MAX_TOKENS truncation ({len(text)} chars emitted). "
                f"Raise max_output_tokens, or lower --batch-size"
            )
        if reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
            raise RuntimeError(f"BLOCKED: finishReason={reason}")
        if not text:
            raise RuntimeError(f"EMPTY_RESPONSE: finishReason={reason}")
        return text

    # ------------------------------------------------------------------
    async def _raw_embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self.config.embed_model
        one = {"model": f"models/{model}", "content": {"parts": [{"text": ""}]}}
        if self.embed_dim:
            one["outputDimensionality"] = self.embed_dim

        body = {
            "requests": [
                {**one, "content": {"parts": [{"text": t[:30000]}]}} for t in texts
            ]
        }
        data = await asyncio.to_thread(
            self._post, f"models/{model}:batchEmbedContents", self.embed_key, body
        )

        embs = data.get("embeddings") or []
        vecs = [e.get("values") or [] for e in embs]

        # --- Guard 1: counts must correspond one-to-one ---
        if len(vecs) != len(texts):
            raise RuntimeError(
                f"EMBED_COUNT_MISMATCH: sent {len(texts)}, received {len(vecs)}. "
                f"If {model} aggregated the inputs into a single vector, switch "
                f"to gemini-embedding-001, or embed one item per call."
            )

        # --- Guard 2: dimension must stay consistent throughout ---
        dims = {len(v) for v in vecs if v}
        if len(dims) > 1:
            raise RuntimeError(f"EMBED_DIM_INCONSISTENT: mixed dimensions in one batch: {dims}")
        if dims:
            d = dims.pop()
            if self._locked_dim is None:
                self._locked_dim = d
            elif d != self._locked_dim:
                raise RuntimeError(
                    f"EMBED_DIM_CHANGED: was {self._locked_dim}, now {d}. "
                    f"If you switched embed_model, clear the cache first: "
                    f"rm cache.sqlite3"
                )
        return vecs

    # ------------------------------------------------------------------
    async def selftest(self) -> Dict[str, Any]:
        """
        Fire two minimal requests before burning any quota, validating the
        key, the model names, the embedding dimension and the aggregation
        trap in one shot. This should be the first thing you run on the day.
        """
        out: Dict[str, Any] = {}
        try:
            txt = await self._raw_chat(
                "Return JSON only.", 'Return exactly {"ok": true}'
            )
            out["chat"] = {"model": self.config.llm_model, "ok": True,
                           "sample": txt[:80]}
        except Exception as e:  # noqa: BLE001
            out["chat"] = {"model": self.config.llm_model, "ok": False, "error": str(e)[:300]}

        try:
            vs = await self._raw_embed(["alpha", "beta", "gamma"])
            out["embed"] = {
                "model": self.config.embed_model, "ok": True,
                "returned": len(vs), "dim": len(vs[0]) if vs else 0,
            }
        except Exception as e:  # noqa: BLE001
            out["embed"] = {"model": self.config.embed_model, "ok": False, "error": str(e)[:300]}
        return out


# ======================================================================

def make_client(cache, config: SearchConfig, settings) -> LLMClient:
    """Construct a client based on LLM_PROVIDER in .env."""
    p = settings.provider
    if p == "gemini":
        return GeminiClient(
            cache, config,
            api_key=settings.chat_key,
            embed_key=settings.embed_key,
            embed_dim=settings.embed_dim,
            # Allows pointing at a local fake server for integration tests,
            # or at a proxy / gateway
            base_url=os.environ.get("GEMINI_BASE_URL", GEMINI_BASE),
        )
    raise SystemExit(
        f"No client implemented for provider='{p}'. Only gemini is available.\n"
        f"Implement _raw_chat / _raw_embed following GeminiClient (~40 lines), "
        f"or set LLM_PROVIDER=gemini in .env."
    )
