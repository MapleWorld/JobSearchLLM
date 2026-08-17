"""
engine.py — 检索排序主引擎。

关键设计：**分级 pipeline（--level 0..3）**。
每一级都是可独立提交的完整系统，级别越高越准但越慢。
75 分钟里你永远有一个能交的版本，不会出现"设计很美但跑不出结果"。

  L0  BM25 baseline            无 LLM，~30s 出结果，用来攒第一个分数和 gold set
  L1  + LLM 编译 + 硬过滤       结构化 criteria，graded relaxation
  L2  + Dense 检索 + RRF 融合   真正的 hybrid
  L3  + LLM 逐条 criteria 判定  最终排序：先按硬条件命中数，再按软性 fit
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cache import DiskCache
from diagnostics import SearchTrace


# ======================================================================
# 数据模型
# ======================================================================

@dataclass
class JobDescription:
    job_id: str
    title: str
    description: str
    hard_criteria: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        hc = "\n".join(f"- {c}" for c in self.hard_criteria)
        return f"{self.title}\n\n{self.description}\n\nHard criteria:\n{hc}"


@dataclass
class Candidate:
    candidate_id: str
    name: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)   # 原始记录，别丢
    profile_text: str = ""                              # 用于 BM25 / embedding 的拼接文本
    years_experience: Optional[float] = None
    skills: List[str] = field(default_factory=list)
    location: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class Criteria:
    """LLM 从 JD 编译出来的结构化查询。"""
    min_years_experience: float = 0.0
    required_skills: List[str] = field(default_factory=list)
    skill_synonyms: Dict[str, List[str]] = field(default_factory=dict)
    required_locations: List[str] = field(default_factory=list)
    semantic_query: str = ""
    keyword_query: str = ""
    checkable_criteria: List[str] = field(default_factory=list)  # 逐条可判定的硬条件

    @staticmethod
    def fallback(jd: JobDescription) -> "Criteria":
        return Criteria(
            semantic_query=f"{jd.title}. {jd.description}",
            keyword_query=f"{jd.title} {jd.description}",
            checkable_criteria=list(jd.hard_criteria),
        )


@dataclass
class ScoredCandidate:
    candidate_id: str
    final_score: float
    hard_passed: int = 0
    hard_total: int = 0
    soft_score: float = 0.0
    retrieval_score: float = 0.0
    checks: List[Dict] = field(default_factory=list)
    reasoning: str = ""
    degraded: bool = False       # True 表示这个分数是 LLM 失败后的兜底分


@dataclass
class SearchConfig:
    level: int = 3
    retrieve_k: int = 30
    final_k: int = 10
    min_pool_after_filter: int = 25       # 低于这个数就触发 relaxation
    rrf_k: int = 60
    rerank_batch_size: int = 5            # 一次 prompt 塞几个候选人
    max_concurrency: int = 8
    prompt_version: str = "v1"
    llm_model: str = "claude-sonnet-4-6"
    embed_model: str = "text-embedding-3-small"
    hard_criteria_weight: float = 1000.0  # 字典序：先比硬条件数，再比 soft
    use_cache: bool = True

    def signature(self) -> Dict:
        return asdict(self)


# ======================================================================
# LLM 客户端（带缓存 + 重试 + 健壮 JSON 解析）
# ======================================================================

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_loose(raw: str) -> Any:
    """LLM 的 JSON 输出经常带 markdown fence / 前后废话。三层兜底。"""
    if not raw:
        raise ValueError("empty response")
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 兜底：抓第一个 {...} 或 [...] 平衡括号
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = raw.find(open_c)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == open_c:
                depth += 1
            elif raw[i] == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"unparseable JSON: {raw[:200]}")


class LLMClient:
    """
    包一层，把 cache / retry / 并发限流 / 用量统计都收在这里。
    换 provider 只需要改 _raw_chat 和 _raw_embed 两个方法。
    """

    def __init__(self, cache: DiskCache, config: SearchConfig):
        self.cache = cache
        self.config = config
        self.sem = asyncio.Semaphore(config.max_concurrency)
        self.calls = Counter()
        self.latency_ms: List[float] = []

    # ---- 需要替换成真实 SDK 的两个方法 ----
    async def _raw_chat(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def _raw_embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    # ---- 对外接口 ----
    async def chat_json(self, system: str, user: str, ns: str) -> Any:
        payload = {
            "system": system,
            "user": user,
            "model": self.config.llm_model,
            "pv": self.config.prompt_version,
        }
        cached = self.cache.get(ns, payload)
        if cached is not None:
            self.calls["chat_cached"] += 1
            return cached

        last_err = None
        for attempt in range(3):
            try:
                async with self.sem:
                    t0 = time.perf_counter()
                    raw = await self._raw_chat(system, user)
                    self.latency_ms.append((time.perf_counter() - t0) * 1000)
                self.calls["chat"] += 1
                parsed = parse_json_loose(raw)
                self.cache.set(ns, payload, parsed)
                return parsed
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.calls["chat_retry"] += 1
                await asyncio.sleep(0.6 * (2**attempt))
        self.calls["chat_failed"] += 1
        raise RuntimeError(f"chat failed after retries: {last_err}")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding + 逐条缓存。别在循环里 await 单条。"""
        out: List[Optional[List[float]]] = [None] * len(texts)
        todo: List[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get("embed", {"t": t, "m": self.config.embed_model})
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)

        BATCH = 64
        for s in range(0, len(todo), BATCH):
            idxs = todo[s : s + BATCH]
            async with self.sem:
                vecs = await self._raw_embed([texts[i] for i in idxs])
            self.calls["embed"] += len(idxs)
            for i, v in zip(idxs, vecs):
                out[i] = v
                self.cache.set("embed", {"t": texts[i], "m": self.config.embed_model}, v)
        return [v if v is not None else [] for v in out]

    def usage(self) -> Dict:
        p50 = float(np.percentile(self.latency_ms, 50)) if self.latency_ms else 0.0
        p95 = float(np.percentile(self.latency_ms, 95)) if self.latency_ms else 0.0
        return {
            **dict(self.calls),
            "latency_p50_ms": round(p50),
            "latency_p95_ms": round(p95),
            "cache": self.cache.stats.as_dict(),
        }


# ======================================================================
# BM25（纯 python，零依赖，L0 baseline 用）
# ======================================================================

_TOKEN = re.compile(r"[a-z0-9\+\#\.]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / self.N if self.N else 0.0
        self.tf: List[Counter] = [Counter(t) for t in self.toks]
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        self.idf = {
            w: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            out[i] = s
        return out


def rrf_fuse(rankings: Sequence[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion —— 融合不同量纲的检索通道，比加权求和稳。"""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ======================================================================
# Prompts
# ======================================================================

COMPILER_SYSTEM = """You are a query compiler for a candidate search engine.
Turn a job description into (a) machine-checkable hard constraints and
(b) retrieval queries. Be conservative: only mark something as a hard
constraint if the JD states it as a requirement, not a preference.
Return JSON only, no prose."""


def compiler_user(jd: JobDescription) -> str:
    return f"""Job title: {jd.title}

Description:
{jd.description}

Stated hard criteria:
{json.dumps(jd.hard_criteria, ensure_ascii=False, indent=2)}

Return exactly this JSON schema:
{{
  "min_years_experience": number,
  "required_skills": [string],
  "skill_synonyms": {{"<skill>": [alternate spellings / adjacent techs]}},
  "required_locations": [string],
  "semantic_query": "one dense paragraph describing the ideal candidate",
  "keyword_query": "space-separated high-signal keywords for BM25",
  "checkable_criteria": ["each hard requirement restated as a single yes/no question"]
}}

Rules:
- skill_synonyms must cover casing/spacing variants (e.g. "PyTorch" -> ["pytorch","torch"]).
- checkable_criteria must be atomic: one requirement per entry.
- If years of experience is not stated, use 0."""


JUDGE_SYSTEM = """You are a strict technical recruiter grading candidates.
For each candidate, evaluate EACH criterion independently and answer yes/no
with a short evidence quote copied from the candidate's profile.
Never mark a criterion as passed without concrete evidence in the profile.
Then give a soft fit score 0-100 for overall quality beyond the hard bar.
Return JSON only."""


def judge_user(jd: JobDescription, criteria: Criteria, batch: List[Candidate]) -> str:
    crit = criteria.checkable_criteria or jd.hard_criteria or ["Relevant to the role"]
    cands = []
    for c in batch:
        cands.append(
            {
                "candidate_id": c.candidate_id,
                "years_experience": c.years_experience,
                "skills": c.skills,
                "location": c.location,
                "profile": c.profile_text[:4000],
            }
        )
    return f"""[ROLE]
{jd.title}
{jd.description[:2000]}

[CRITERIA] (evaluate each independently)
{json.dumps(crit, ensure_ascii=False, indent=2)}

[CANDIDATES]
{json.dumps(cands, ensure_ascii=False, indent=2)}

Return:
{{
  "results": [
    {{
      "candidate_id": string,
      "checks": [{{"criterion": string, "pass": boolean, "evidence": string}}],
      "soft_score": number,
      "reasoning": string
    }}
  ]
}}
One entry per candidate, same order. Evidence must be verbatim from the profile."""


# ======================================================================
# 引擎
# ======================================================================

class CandidateSearchEngine:
    def __init__(self, llm: LLMClient, config: SearchConfig):
        self.llm = llm
        self.config = config
        self._bm25_cache: Dict[int, BM25] = {}

    # ---------- Phase 1: 编译 ----------
    async def compile_criteria(self, jd: JobDescription, trace: SearchTrace) -> Criteria:
        if self.config.level == 0:
            return Criteria.fallback(jd)
        t0 = time.perf_counter()
        try:
            data = await self.llm.chat_json(
                COMPILER_SYSTEM, compiler_user(jd), ns="compile"
            )
            crit = Criteria(
                min_years_experience=float(data.get("min_years_experience") or 0),
                required_skills=[s.lower() for s in data.get("required_skills", [])],
                skill_synonyms={
                    k.lower(): [v.lower() for v in vs]
                    for k, vs in (data.get("skill_synonyms") or {}).items()
                },
                required_locations=[s.lower() for s in data.get("required_locations", [])],
                semantic_query=data.get("semantic_query") or jd.as_text(),
                keyword_query=data.get("keyword_query") or f"{jd.title} {jd.description}",
                checkable_criteria=data.get("checkable_criteria") or list(jd.hard_criteria),
            )
        except Exception as e:  # noqa: BLE001
            trace.warn(f"compile failed, using fallback: {e}")
            crit = Criteria.fallback(jd)
        trace.meta["compiled_criteria"] = asdict(crit)
        trace.meta["compile_ms"] = round((time.perf_counter() - t0) * 1000)
        return crit

    # ---------- Phase 2a: 分级放松的硬过滤 ----------
    def _skill_hit(self, cand: Candidate, skill: str, crit: Criteria) -> bool:
        variants = {skill, *crit.skill_synonyms.get(skill, [])}
        blob = (" ".join(cand.skills) + " " + cand.profile_text).lower()
        return any(v and v in blob for v in variants)

    def hard_filter(
        self, cands: List[Candidate], crit: Criteria, trace: SearchTrace
    ) -> List[Candidate]:
        """
        分级放松：从最严开始往下退，直到池子够大。
        不要一步退到"完全不过滤" —— 那等于放弃了硬条件信号。
        """
        if self.config.level == 0 or not (crit.required_skills or crit.min_years_experience):
            trace.meta["relaxation_level"] = "off"
            return cands

        def passes(c: Candidate, yrs_slack: float, need_all_skills: bool, check_loc: bool) -> bool:
            if crit.min_years_experience and c.years_experience is not None:
                if c.years_experience < crit.min_years_experience - yrs_slack:
                    return False
            if crit.required_skills:
                hits = [self._skill_hit(c, s, crit) for s in crit.required_skills]
                if need_all_skills and not all(hits):
                    return False
                if not need_all_skills and not any(hits):
                    return False
            if check_loc and crit.required_locations:
                if not any(l in c.location.lower() for l in crit.required_locations):
                    return False
            return True

        ladder = [
            ("L0_strict",        dict(yrs_slack=0, need_all_skills=True,  check_loc=True)),
            ("L1_drop_location", dict(yrs_slack=0, need_all_skills=True,  check_loc=False)),
            ("L2_any_skill",     dict(yrs_slack=0, need_all_skills=False, check_loc=False)),
            ("L3_years_slack",   dict(yrs_slack=2, need_all_skills=False, check_loc=False)),
        ]
        for name, kw in ladder:
            kept = [c for c in cands if passes(c, **kw)]
            if len(kept) >= self.config.min_pool_after_filter:
                trace.meta["relaxation_level"] = name
                return kept
        trace.meta["relaxation_level"] = "L4_semantic_only"
        trace.warn("hard filter 无法产出足够候选人，退化为纯语义检索")
        return cands

    # ---------- Phase 2b: 混合检索 ----------
    async def retrieve(
        self, crit: Criteria, pool: List[Candidate], trace: SearchTrace
    ) -> List[Tuple[Candidate, float]]:
        by_id = {c.candidate_id: c for c in pool}
        rankings: List[List[str]] = []

        # 通道 1: BM25（永远开着 —— 专有名词、公司名、证书靠它）
        t0 = time.perf_counter()
        bm = BM25([c.profile_text for c in pool])
        s = bm.scores(crit.keyword_query or crit.semantic_query)
        order = np.argsort(-s)
        lex_rank = [pool[i].candidate_id for i in order]
        rankings.append(lex_rank[: self.config.retrieve_k * 3])
        trace.stage(
            "retrieve_bm25",
            [c.candidate_id for c in pool],
            rankings[-1],
            (time.perf_counter() - t0) * 1000,
            note="lexical channel",
        )

        # 通道 2: Dense（level >= 2）
        if self.config.level >= 2:
            t0 = time.perf_counter()
            try:
                need = [c for c in pool if not c.embedding]
                if need:
                    vecs = await self.llm.embed([c.profile_text for c in need])
                    for c, v in zip(need, vecs):
                        c.embedding = v
                qv = (await self.llm.embed([crit.semantic_query]))[0]
                q = np.array(qv, dtype=float)
                q /= np.linalg.norm(q) + 1e-9
                M = np.array(
                    [c.embedding if c.embedding else [0.0] * len(q) for c in pool],
                    dtype=float,
                )
                M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
                sims = M @ q
                dorder = np.argsort(-sims)
                dense_rank = [pool[i].candidate_id for i in dorder]
                rankings.append(dense_rank[: self.config.retrieve_k * 3])
                trace.stage(
                    "retrieve_dense",
                    [c.candidate_id for c in pool],
                    rankings[-1],
                    (time.perf_counter() - t0) * 1000,
                    note="dense channel",
                )
            except Exception as e:  # noqa: BLE001
                trace.warn(f"dense retrieval failed, lexical only: {e}")

        fused = rrf_fuse(rankings, k=self.config.rrf_k)[: self.config.retrieve_k]
        trace.stage(
            "fuse_rrf",
            [c.candidate_id for c in pool],
            [cid for cid, _ in fused],
            0.0,
            note=f"{len(rankings)} channels",
        )
        return [(by_id[cid], sc) for cid, sc in fused if cid in by_id]

    # ---------- Phase 3: LLM 逐条判定 ----------
    async def rerank(
        self,
        jd: JobDescription,
        crit: Criteria,
        retrieved: List[Tuple[Candidate, float]],
        trace: SearchTrace,
    ) -> List[ScoredCandidate]:
        n_crit = len(crit.checkable_criteria or jd.hard_criteria) or 1

        if self.config.level < 3:
            # 不做 LLM 重排，直接用检索分
            return [
                ScoredCandidate(
                    candidate_id=c.candidate_id,
                    final_score=sc,
                    retrieval_score=sc,
                    hard_total=n_crit,
                )
                for c, sc in retrieved
            ]

        t0 = time.perf_counter()
        bs = self.config.rerank_batch_size
        batches = [retrieved[i : i + bs] for i in range(0, len(retrieved), bs)]
        results = await asyncio.gather(
            *[self._score_batch(jd, crit, b, trace) for b in batches],
            return_exceptions=False,
        )
        scored = [s for group in results for s in group]

        # 最终排序：字典序 —— 先比通过的硬条件数，再比软性 fit，最后比检索分
        for s in scored:
            s.hard_total = n_crit
            s.final_score = (
                s.hard_passed * self.config.hard_criteria_weight
                + s.soft_score
                + s.retrieval_score * 0.01
            )
        scored.sort(key=lambda x: -x.final_score)
        trace.stage(
            "llm_rerank",
            [c.candidate_id for c, _ in retrieved],
            [s.candidate_id for s in scored],
            (time.perf_counter() - t0) * 1000,
            note=f"{len(batches)} batches, degraded={sum(s.degraded for s in scored)}",
        )
        return scored

    async def _score_batch(
        self,
        jd: JobDescription,
        crit: Criteria,
        batch: List[Tuple[Candidate, float]],
        trace: SearchTrace,
    ) -> List[ScoredCandidate]:
        cands = [c for c, _ in batch]
        retr = {c.candidate_id: sc for c, sc in batch}
        try:
            data = await self.llm.chat_json(
                JUDGE_SYSTEM, judge_user(jd, crit, cands), ns="judge"
            )
            by_id = {r["candidate_id"]: r for r in data.get("results", [])}
            out = []
            for c in cands:
                r = by_id.get(c.candidate_id)
                if r is None:
                    trace.warn(f"judge 漏返回 {c.candidate_id}，用检索分兜底")
                    out.append(
                        ScoredCandidate(
                            candidate_id=c.candidate_id,
                            final_score=0.0,
                            soft_score=50.0,
                            retrieval_score=retr[c.candidate_id],
                            degraded=True,
                        )
                    )
                    continue
                checks = r.get("checks", []) or []
                out.append(
                    ScoredCandidate(
                        candidate_id=c.candidate_id,
                        final_score=0.0,
                        hard_passed=sum(1 for ch in checks if ch.get("pass")),
                        soft_score=float(r.get("soft_score") or 0),
                        retrieval_score=retr[c.candidate_id],
                        checks=checks,
                        reasoning=str(r.get("reasoning", ""))[:400],
                    )
                )
            return out
        except Exception as e:  # noqa: BLE001
            # 关键：失败不给 0 分（那会静默丢掉好候选人），退回检索分
            trace.warn(f"judge batch failed ({e}), 退回检索分")
            return [
                ScoredCandidate(
                    candidate_id=c.candidate_id,
                    final_score=0.0,
                    soft_score=50.0,
                    retrieval_score=retr[c.candidate_id],
                    degraded=True,
                )
                for c in cands
            ]

    # ---------- E2E ----------
    async def search(
        self, jd: JobDescription, pool: List[Candidate], run_id: str = ""
    ) -> Tuple[List[ScoredCandidate], SearchTrace]:
        trace = SearchTrace(job_id=jd.job_id, run_id=run_id)
        all_ids = [c.candidate_id for c in pool]
        trace.stage("pool", all_ids, all_ids, 0.0, note=f"level={self.config.level}")

        crit = await self.compile_criteria(jd, trace)

        t0 = time.perf_counter()
        filtered = self.hard_filter(pool, crit, trace)
        trace.stage(
            "hard_filter",
            all_ids,
            [c.candidate_id for c in filtered],
            (time.perf_counter() - t0) * 1000,
            note=trace.meta.get("relaxation_level", ""),
        )

        retrieved = await self.retrieve(crit, filtered, trace)
        scored = await self.rerank(jd, crit, retrieved, trace)
        top = scored[: self.config.final_k]
        trace.stage(
            "final_top_k",
            [s.candidate_id for s in scored],
            [s.candidate_id for s in top],
            0.0,
        )
        return top, trace.finish()
