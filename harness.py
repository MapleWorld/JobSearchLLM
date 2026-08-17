"""
harness.py — Evaluation 闭环。

题目要求"根据测试结果不断迭代，最终提交所有 Evaluation Results"。
这个文件负责的就是那个闭环：

    跑 pipeline -> 提交 eval endpoint -> 解析分数 -> 落盘 -> 吸收进 gold set
        -> 打印漏斗诊断 -> 改配置 -> 再跑 -> 对比两次 run 的每题 delta

每次 run 都会在 runs/<run_id>/ 下留一份完整快照（config + results + traces +
summary），最后一步 `python harness.py export` 直接产出可提交的汇总文件。

⚠️ 开场 5 分钟必须做的一件事：
   把 EvalClient.ADAPTER 里的三个函数改成和真实 endpoint 的 schema 对齐。
   其余代码不需要动。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cache import DiskCache
from diagnostics import GoldSet, SearchTrace
from engine import (
    Candidate,
    CandidateSearchEngine,
    JobDescription,
    LLMClient,
    ScoredCandidate,
    SearchConfig,
)

RUNS_DIR = "runs"


# ======================================================================
# Eval Endpoint 客户端
# ======================================================================

class EvalClient:
    """
    ADAPTER 区是唯一需要按真实 endpoint 改的地方。
    先用 curl 打一发，把响应贴进来，再对着改这三个函数。
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ---------------- ADAPTER: 按真实 schema 修改 ----------------
    def _build_payload(self, job_id: str, candidate_ids: List[str]) -> Dict:
        return {"job_id": job_id, "candidate_ids": candidate_ids}

    SCORE_KEYS = ("score", "overall_score", "precision", "precision_at_10",
                  "accuracy", "grade", "mean_score", "p_at_10")
    ID_KEYS = ("candidate_id", "candidateId", "id", "cid")
    VAL_KEYS = ("score", "relevant", "pass", "is_match", "match", "correct", "met")

    @staticmethod
    def _walk(node, depth: int = 0):
        """深度优先遍历整棵 JSON 树 —— 不假设字段在第几层。"""
        if depth > 6:
            return
        yield node
        if isinstance(node, dict):
            for v in node.values():
                yield from EvalClient._walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                yield from EvalClient._walk(v, depth + 1)

    @staticmethod
    def extract_overall_score(resp: Dict) -> float:
        """
        抠出这道题的总分。递归查找，兼容 {"data":{"evaluation":{"precision":..}}}
        这类嵌套结构 —— 硬编码顶层 key 在真实 API 上很容易落空。
        """
        for node in EvalClient._walk(resp):
            if isinstance(node, dict):
                for k in EvalClient.SCORE_KEYS:
                    v = node.get(k)
                    # 排除逐候选人条目里的 score（它们同名但含 id 字段）
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if not any(i in node for i in EvalClient.ID_KEYS):
                            return float(v)
        grades = EvalClient.extract_per_candidate(resp)
        if grades:
            return sum(1 for v in grades.values() if v >= 0.5) / len(grades)
        return 0.0

    @staticmethod
    def looks_unparsed(resp: Dict) -> bool:
        """
        关键：区分「真的 0 分」和「schema 没对上」。
        两者都返回 0.0，但前者要去改检索，后者要去改 ADAPTER —— 
        搞混会浪费掉现场十几分钟。
        """
        return (not EvalClient.extract_per_candidate(resp)) and not any(
            isinstance(n, dict) and any(
                isinstance(n.get(k), (int, float)) and not isinstance(n.get(k), bool)
                for k in EvalClient.SCORE_KEYS)
            for n in EvalClient._walk(resp)
        )

    @staticmethod
    def extract_per_candidate(resp: Dict) -> Dict[str, float]:
        """
        抠出 {candidate_id: 分数}。这是 gold set 的来源，比总分重要得多——
        没有它就只能盲调，有了它就能算分阶段召回。
        """
        out: Dict[str, float] = {}
        # 递归找出所有「同时含 id 字段和分值字段」的 dict，不管它埋在第几层
        for node in EvalClient._walk(resp):
            if not isinstance(node, dict):
                continue
            cid = next((node[k] for k in EvalClient.ID_KEYS if k in node), None)
            if cid is None or isinstance(cid, (dict, list)):
                continue
            val = next((node[k] for k in EvalClient.VAL_KEYS if k in node), None)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            if isinstance(val, (int, float)):
                out[str(cid)] = float(val)
        # 也支持 {"grades": {"cand_1": 1, "cand_2": 0}} 这种映射形态
        if not out:
            for node in EvalClient._walk(resp):
                if isinstance(node, dict) and node and all(
                    isinstance(v, (int, float, bool)) for v in node.values()
                ) and all(isinstance(k, str) for k in node):
                    if any(k.lower().startswith(("cand", "c_")) for k in node):
                        out = {k: float(v) for k, v in node.items()}
                        break
        return out
    # -------------------------------------------------------------

    async def submit(self, job_id: str, candidate_ids: List[str]) -> Dict:
        return await asyncio.to_thread(self._submit_sync, job_id, candidate_ids)

    def _submit_sync(self, job_id: str, candidate_ids: List[str]) -> Dict:
        body = json.dumps(self._build_payload(job_id, candidate_ids)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/evaluate", data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code < 500:
                    break            # 4xx 重试没意义，八成是 payload schema 不对
            except Exception as e:   # noqa: BLE001
                last = str(e)
            time.sleep(0.8 * (2**attempt))
        return {"error": last, "job_id": job_id}


class MockEvalClient(EvalClient):
    """离线自测用。真实 endpoint 接上后删掉。"""

    def __init__(self, truth: Dict[str, set]):
        super().__init__("http://mock")
        self.truth = truth

    async def submit(self, job_id: str, candidate_ids: List[str]) -> Dict:
        gold = self.truth.get(job_id, set())
        return {
            "job_id": job_id,
            "results": [
                {"candidate_id": c, "score": 1.0 if c in gold else 0.0}
                for c in candidate_ids
            ],
        }


# ======================================================================
# 实验运行器
# ======================================================================

class ExperimentRunner:
    def __init__(
        self,
        engine: CandidateSearchEngine,
        eval_client: EvalClient,
        gold: GoldSet,
        config: SearchConfig,
    ):
        self.engine = engine
        self.eval = eval_client
        self.gold = gold
        self.config = config

    async def run(
        self,
        jobs: List[JobDescription],
        pool: List[Candidate],
        run_id: Optional[str] = None,
        submit: bool = True,
    ) -> Dict:
        run_id = run_id or f"L{self.config.level}_{self.config.prompt_version}_{int(time.time())}"
        outdir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(outdir, exist_ok=True)

        results: List[Dict] = []
        traces: List[SearchTrace] = []
        t_start = time.perf_counter()

        for jd in jobs:
            top, trace = await self.engine.search(jd, pool, run_id=run_id)
            traces.append(trace)
            top_ids = [s.candidate_id for s in top]

            score, per_cand, raw = 0.0, {}, None
            if submit:
                raw = await self.eval.submit(jd.job_id, top_ids)
                if "error" in raw:
                    trace.warn(f"eval endpoint error: {raw['error']}")
                else:
                    score = self.eval.extract_overall_score(raw)
                    per_cand = self.eval.extract_per_candidate(raw)
                    if per_cand:
                        self.gold.absorb(jd.job_id, per_cand)

            results.append(
                {
                    "job_id": jd.job_id,
                    "title": jd.title,
                    "top_10": top_ids,
                    "score": score,
                    "per_candidate": per_cand,
                    "raw_response": raw,
                    "scored_detail": [asdict(s) for s in top],
                    "total_ms": trace.total_ms,
                    "relaxation_level": trace.meta.get("relaxation_level"),
                    "warnings": trace.warnings,
                }
            )
            # 边跑边打印，别等全部跑完 —— 中途 Ctrl-C 也不至于什么都没有
            print(trace.render_funnel(gold=self.gold.get(jd.job_id)))
            print(f"  -> score={score:.3f}  top10={top_ids[:3]}...\n", flush=True)

        scores = [r["score"] for r in results]
        summary = {
            "run_id": run_id,
            "config": self.config.signature(),
            "n_jobs": len(jobs),
            "mean_score": round(statistics.fmean(scores), 4) if scores else 0.0,
            "median_score": round(statistics.median(scores), 4) if scores else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "wall_clock_s": round(time.perf_counter() - t_start, 1),
            "llm_usage": self.engine.llm.usage(),
            "gold_coverage": self.gold.coverage(),
            "per_job": {r["job_id"]: r["score"] for r in results},
        }

        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "results.jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(outdir, "traces.jsonl"), "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

        print("=" * 72)
        print(f"RUN {run_id}: mean={summary['mean_score']:.3f} "
              f"median={summary['median_score']:.3f} "
              f"wall={summary['wall_clock_s']}s "
              f"cache_hit={summary['llm_usage']['cache']['hit_rate']:.0%}")
        print(f"saved -> {outdir}/")
        print("=" * 72)
        return summary


# ======================================================================
# Run 对比（回归检测）
# ======================================================================

def load_summaries() -> List[Dict]:
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for d in sorted(os.listdir(RUNS_DIR)):
        p = os.path.join(RUNS_DIR, d, "summary.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def compare_runs(a: Optional[str] = None, b: Optional[str] = None) -> None:
    """
    不要只看均值。均值涨了但某几道题崩了是常态，
    per-job delta 才能告诉你改动到底动了什么。
    """
    summaries = load_summaries()
    if not summaries:
        print("no runs yet")
        return

    print(f"\n{'run_id':<34}{'lvl':>4}{'mean':>8}{'median':>8}{'min':>7}{'wall':>7}")
    print("-" * 68)
    for s in summaries:
        print(f"{s['run_id']:<34}{s['config']['level']:>4}{s['mean_score']:>8.3f}"
              f"{s['median_score']:>8.3f}{s['min_score']:>7.3f}{s['wall_clock_s']:>7.0f}")

    if len(summaries) < 2:
        return
    ra = next((s for s in summaries if s["run_id"] == a), summaries[-2])
    rb = next((s for s in summaries if s["run_id"] == b), summaries[-1])
    print(f"\nPER-JOB DELTA:  {ra['run_id']}  ->  {rb['run_id']}")
    print(f"{'job_id':<24}{'before':>9}{'after':>9}{'delta':>9}")
    print("-" * 51)
    jobs = sorted(set(ra["per_job"]) | set(rb["per_job"]))
    regressions = []
    for j in jobs:
        x, y = ra["per_job"].get(j, 0.0), rb["per_job"].get(j, 0.0)
        d = y - x
        flag = "  <-- REGRESSION" if d < -0.05 else ("  ++" if d > 0.05 else "")
        if d < -0.05:
            regressions.append(j)
        print(f"{j:<24}{x:>9.3f}{y:>9.3f}{d:>+9.3f}{flag}")
    print(f"\nmean {ra['mean_score']:.3f} -> {rb['mean_score']:.3f} "
          f"({rb['mean_score'] - ra['mean_score']:+.3f})")
    if regressions:
        print(f"⚠ {len(regressions)} 道题回退，先去看它们的 traces.jsonl 漏斗：{regressions[:5]}")


def export_submission(out_path: str = "evaluation_results.json") -> None:
    """把所有 run 汇总成一份可提交的文件。"""
    summaries = load_summaries()
    best = max(summaries, key=lambda s: s["mean_score"]) if summaries else None
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs": len(summaries),
        "best_run": best["run_id"] if best else None,
        "best_mean_score": best["mean_score"] if best else None,
        "iteration_history": [
            {
                "run_id": s["run_id"],
                "level": s["config"]["level"],
                "prompt_version": s["config"]["prompt_version"],
                "mean_score": s["mean_score"],
                "config_delta": {
                    k: v for k, v in s["config"].items()
                    if k in ("retrieve_k", "rerank_batch_size", "min_pool_after_filter")
                },
            }
            for s in summaries
        ],
        "final_results": [],
    }
    if best:
        p = os.path.join(RUNS_DIR, best["run_id"], "results.jsonl")
        with open(p, encoding="utf-8") as f:
            payload["final_results"] = [
                {k: v for k, v in json.loads(l).items() if k in ("job_id", "top_10", "score")}
                for l in f
            ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"exported {len(payload['final_results'])} job results -> {out_path}")
    print(f"best run: {payload['best_run']} (mean={payload['best_mean_score']})")


# ======================================================================
# Mock 模式：离线验证整条链路
# ======================================================================

class MockLLM(LLMClient):
    async def _raw_chat(self, system: str, user: str) -> str:
        await asyncio.sleep(0.02)
        if "query compiler" in system:
            return json.dumps({
                "min_years_experience": 5,
                "required_skills": ["python", "distributed systems"],
                "skill_synonyms": {"python": ["py", "python3"],
                                   "distributed systems": ["distributed", "microservices"]},
                "required_locations": [],
                "semantic_query": "Senior backend engineer, Python, distributed systems, search ranking",
                "keyword_query": "python distributed systems search ranking backend senior",
                "checkable_criteria": ["Has 5+ years experience",
                                       "Proficient in Python",
                                       "Built distributed systems"],
            })
        payload = json.loads(user[user.index("[CANDIDATES]") + 12 : user.index("Return:")].strip())
        return json.dumps({"results": [
            {"candidate_id": c["candidate_id"],
             "checks": [{"criterion": "x", "pass": "python" in " ".join(c["skills"]).lower(),
                         "evidence": "skills"},
                        {"criterion": "y", "pass": (c["years_experience"] or 0) >= 5,
                         "evidence": "yrs"},
                        {"criterion": "z", "pass": "distributed" in c["profile"].lower(),
                         "evidence": "profile"}],
             "soft_score": 40 + (hash(c["candidate_id"]) % 60),
             "reasoning": "mock"} for c in payload]})

    async def _raw_embed(self, texts: List[str]) -> List[List[float]]:
        await asyncio.sleep(0.01)
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(64)
            # 让含关键词的文本在向量空间里靠拢，模拟语义信号
            bonus = np.zeros(64)
            for i, kw in enumerate(["python", "distributed", "search", "ranking"]):
                if kw in t.lower():
                    bonus[i * 8 : (i + 1) * 8] += 2.0
            v = v * 0.3 + bonus
            out.append((v / (np.linalg.norm(v) + 1e-9)).tolist())
        return out


def make_mock_data(n: int = 300) -> Tuple[List[JobDescription], List[Candidate], Dict[str, set]]:
    rng = random.Random(42)
    jobs = [
        JobDescription(
            job_id="job_001",
            title="Staff Backend / Search Engineer",
            description="Build distributed search and AI-driven ranking pipelines at scale.",
            hard_criteria=["5+ years of backend experience",
                           "Proficient in Python",
                           "Experience building distributed systems"],
        ),
        JobDescription(
            job_id="job_002",
            title="Senior ML Platform Engineer",
            description="Own the model serving and feature store for ranking models.",
            hard_criteria=["5+ years experience", "Python", "Distributed systems"],
        ),
    ]
    pool, truth = [], {j.job_id: set() for j in jobs}
    for i in range(n):
        good = i % 4 == 0
        skills = (["Python", "Distributed Systems", "Kubernetes"] if good
                  else rng.choice([["Java", "Spring"], ["Python", "Django"], ["Go", "gRPC"]]))
        yrs = rng.randint(5, 14) if good else rng.randint(0, 7)
        txt = ("Engineer working on distributed search ranking pipelines and Python services."
               if good else "Engineer working on web applications and CRUD services.")
        cid = f"cand_{i:03d}"
        pool.append(Candidate(
            candidate_id=cid, name=f"Candidate {i}",
            profile_text=f"{txt} Skills: {', '.join(skills)}. {yrs} years experience.",
            years_experience=yrs, skills=skills, location="Remote",
        ))
        if good and yrs >= 5:
            for j in jobs:
                truth[j.job_id].add(cid)
    return jobs, pool, truth


# ======================================================================
# CLI
# ======================================================================

async def _amain(args) -> None:
    cfg = SearchConfig(
        level=args.level,
        retrieve_k=args.retrieve_k,
        rerank_batch_size=args.batch_size,
        prompt_version=args.prompt_version,
        use_cache=not args.no_cache,
    )
    cache = DiskCache("cache.sqlite3", enabled=cfg.use_cache)
    gold = GoldSet("goldset.json")

    # ---- 数据 ----
    if args.data_dir and os.path.isdir(args.data_dir):
        with open(os.path.join(args.data_dir, "jobs.json"), encoding="utf-8") as f:
            jobs = [JobDescription(**j) for j in json.load(f)]
        with open(os.path.join(args.data_dir, "candidates.json"), encoding="utf-8") as f:
            pool = [
                Candidate(
                    candidate_id=c["candidate_id"], name=c.get("name", ""), raw=c,
                    profile_text=c.get("profile_text", ""),
                    years_experience=c.get("years_experience"),
                    skills=c.get("skills", []), location=c.get("location", ""),
                )
                for c in json.load(f)
            ]
        print(f"loaded {len(jobs)} jobs / {len(pool)} candidates from {args.data_dir}")
    else:
        jobs, pool, _ = make_mock_data()

    # ---- LLM ----
    from settings import Settings
    st = Settings.load()
    if args.mock_llm or not st.chat_key:
        llm: LLMClient = MockLLM(cache, cfg)
        if not args.mock_llm:
            print("! 未读到 chat_key，回退 MockLLM（--level 0 不受影响）")
    else:
        from llm_clients import make_client
        cfg.llm_model = st.chat_model
        cfg.embed_model = st.embed_model
        llm = make_client(cache, cfg, st)
        st.require(cfg.level)
        if args.selftest:
            print(json.dumps(await llm.selftest(), ensure_ascii=False, indent=2))
            return

    # ---- Eval endpoint ----
    url = args.eval_url or st.eval_url
    if url:
        ev: EvalClient = EvalClient(url, st.eval_key)
        print(f"eval endpoint -> {url}")
    else:
        _, _, truth = make_mock_data()
        ev = MockEvalClient(truth)

    engine = CandidateSearchEngine(llm, cfg)
    runner = ExperimentRunner(engine, ev, gold, cfg)
    await runner.run(jobs, pool, run_id=args.run_id, submit=not args.no_submit)
    return

    engine = CandidateSearchEngine(llm, cfg)
    runner = ExperimentRunner(engine, ev, gold, cfg)
    await runner.run(jobs, pool, run_id=args.run_id, submit=not args.no_submit)


def main() -> None:
    p = argparse.ArgumentParser(description="Candidate search harness")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="跑一轮实验并提交")
    r.add_argument("--level", type=int, default=3, choices=[0, 1, 2, 3])
    r.add_argument("--retrieve-k", type=int, default=30)
    r.add_argument("--batch-size", type=int, default=5)
    r.add_argument("--prompt-version", default="v1")
    r.add_argument("--run-id", default=None)
    r.add_argument("--mock", action="store_true")
    r.add_argument("--mock-llm", action="store_true", help="即使有 key 也用 MockLLM")
    r.add_argument("--data-dir", default=None, help="含 jobs.json/candidates.json 的目录")
    r.add_argument("--eval-url", default=None, help="覆盖 .env 里的 EVAL_URL")
    r.add_argument("--selftest", action="store_true", help="只打两发最小请求验通 key/模型/维度")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--no-submit", action="store_true")

    c = sub.add_parser("compare", help="对比历次 run")
    c.add_argument("--a", default=None)
    c.add_argument("--b", default=None)

    e = sub.add_parser("export", help="导出最终提交文件")
    e.add_argument("--out", default="evaluation_results.json")

    args = p.parse_args()
    if args.cmd == "run":
        asyncio.run(_amain(args))
    elif args.cmd == "compare":
        compare_runs(args.a, args.b)
    elif args.cmd == "export":
        export_submission(args.out)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
