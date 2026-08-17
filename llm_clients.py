"""
llm_clients.py — 真实 LLM 客户端。

为什么走 REST 而不是 google-genai SDK：
  这个项目的第三方依赖只有 numpy。面试环境里 `pip install` 失败是真会发生的
  事，多一个依赖多一个风险点。下面用 urllib 直连，零依赖，且可以对着一个本地
  假服务器完整测试请求/响应处理 —— 而不是"写完了但没跑过"。
  想用 SDK 也行，替换 _post 即可，其余逻辑不变。

已内置的三个防御（每一个都对应一个会静默出错的真实陷阱）：

  1. embedding 条数校验
     gemini-embedding-2 收到多条输入时会返回**单个聚合向量**。我们的
     _raw_embed 契约是"传 N 条回 N 条"，数量对不上会让后面的 zip() 静默
     丢数据。这里直接 assert 并给出可操作的报错。

  2. embedding 维度一致性校验
     缓存里混入不同 embed_model 的向量会让 numpy 在 retrieve() 里抛
     inhomogeneous shape，整轮 run 挂掉。第一次调用时锁定维度。

  3. finishReason 检查
     MAX_TOKENS 截断会产生"看起来像 JSON 的半截字符串"，parse 失败后整个
     batch 降级。这里显式识别并给出加大 max_output_tokens 的提示。
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
            —— 用 batch 端点而非 embedContent 传 list，天然绕开聚合陷阱，
               每个 request 项各自产出一个向量。
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
                    f"AUTH ({e.code}): key 无效或未启用 API。检查 .env 里的 "
                    f"GEMINI_API_KEY。{detail}"
                ) from e
            if e.code == 404:
                raise RuntimeError(
                    f"MODEL_NOT_FOUND (404): 模型名可能过期或拼错 —— {detail}"
                ) from e
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e

    # ------------------------------------------------------------------
    async def _raw_chat(self, system: str, user: str) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                # 注意：新版 Gemini 已废弃 temperature / top_p / top_k，不要再传
                "responseMimeType": "application/json",
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        data = await asyncio.to_thread(
            self._post, f"models/{self.config.llm_model}:generateContent", self.api_key, body
        )

        if pf := data.get("promptFeedback", {}).get("blockReason"):
            raise RuntimeError(f"PROMPT_BLOCKED: {pf}（候选人简历触发了安全过滤）")

        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"NO_CANDIDATES: {json.dumps(data)[:300]}")

        reason = cands[0].get("finishReason", "")
        parts = cands[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        if reason == "MAX_TOKENS":
            raise RuntimeError(
                f"MAX_TOKENS 截断（已输出 {len(text)} 字符）。"
                f"调大 max_output_tokens，或减小 --batch-size"
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

        # --- 防御 1：条数必须一一对应 ---
        if len(vecs) != len(texts):
            raise RuntimeError(
                f"EMBED_COUNT_MISMATCH: 传入 {len(texts)} 条，返回 {len(vecs)} 条。"
                f"若 {model} 把多条输入聚合成了单个向量，改用 "
                f"gemini-embedding-001，或逐条调用。"
            )

        # --- 防御 2：维度必须自始至终一致 ---
        dims = {len(v) for v in vecs if v}
        if len(dims) > 1:
            raise RuntimeError(f"EMBED_DIM_INCONSISTENT: 同一批出现多种维度 {dims}")
        if dims:
            d = dims.pop()
            if self._locked_dim is None:
                self._locked_dim = d
            elif d != self._locked_dim:
                raise RuntimeError(
                    f"EMBED_DIM_CHANGED: 之前是 {self._locked_dim}，现在是 {d}。"
                    f"换过 embed_model 的话请先清缓存：rm cache.sqlite3"
                )
        return vecs

    # ------------------------------------------------------------------
    async def selftest(self) -> Dict[str, Any]:
        """
        烧 quota 之前先打两发最小请求，把 key / 模型名 / 维度 / 聚合陷阱
        一次性验完。现场第一件事就该跑这个。
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
    """按 .env 里的 LLM_PROVIDER 造客户端。"""
    p = settings.provider
    if p == "gemini":
        return GeminiClient(
            cache, config,
            api_key=settings.chat_key,
            embed_key=settings.embed_key,
            embed_dim=settings.embed_dim,
            # 允许指向本地假服务器做集成测试，或指向代理/网关
            base_url=os.environ.get("GEMINI_BASE_URL", GEMINI_BASE),
        )
    raise SystemExit(
        f"provider='{p}' 的客户端尚未实现。当前只有 gemini。\n"
        f"照着 GeminiClient 实现 _raw_chat / _raw_embed 即可（约 40 行），"
        f"或在 .env 里设 LLM_PROVIDER=gemini。"
    )
