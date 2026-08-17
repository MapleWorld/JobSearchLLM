"""
harness.py — Evaluation 闭环。

题目要求"根据测试结果不断迭代,最终提交所有 Evaluation Results"。
这个文件负责的就是那个闭环：

    跑 pipeline -> 提交 eval endpoint -> 解析分数 -> 落盘 -> 吸收进 gold set
        -> 打印漏斗诊断 -> 改配置 -> 再跑 -> 对比两次 run 的每题 delta

每次 run 都会在 runs/<run_id>/ 下留一份完整快照（config + results + traces +
summary）,最后一步 `python harness.py export` 直接产出可提交的汇总文件。

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
    先用 curl 打一发,把响应贴进来,再对着改这三个函数。
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ---------------- ADAPTER: 按真实 schema 修改 ----------------
    def _build_payload(self, job_id: str, candidate_ids: List[str]) -> Dict:
        return {"job_id": job_id, "candidate_ids": candidate_ids}

    @staticmethod
    def extract_overall_score(resp: Dict) -> float:
        """从响应里抠出这道题的总分。"""
        for key in ("score", "overall_score", "precision", "accuracy", "grade"):
            if isinstance(resp.get(key), (int, float)):
                return float(resp[key])
        # 常见形态：返回每个候选人的 pass/fail,总分 = 通过率
        grades = EvalClient.extract_per_candidate(resp)
        if grades:
            return sum(1 for v in grades.values() if v >= 0.5) / len(grades)
        return 0.0

    @staticmethod
    def extract_per_candidate(resp: Dict) -> Dict[str, float]:
        """
        抠出 {candidate_id: 分数}。这是 gold set 的来源,比总分重要得多——
        没有它就只能盲调,有了它就能算分阶段召回。
        """
        out: Dict[str, float] = {}
        for key in ("results", "candidates", "graded", "details", "evaluations"):
            items = resp.get(key)
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    cid = it.get("candidate_id") or it.get("id") or it.get("candidateId")
                    if cid is None:
                        continue
                    val = it.get("score")
                    if val is None:
                        val = it.get("relevant", it.get("pass", it.get("is_match")))
                    if isinstance(val, bool):
                        val = 1.0 if val else 0.0
                    if isinstance(val, (int, float)):
                        out[str(cid)] = float(val)
            elif isinstance(items, dict):
                for cid, val in items.items():
                    if isinstance(val, bool):
                        val = 1.0 if val else 0.0
                    if isinstance(val, (int, float)):
                        out[str(cid)] = float(val)
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
                    break            # 4xx 重试没意义,八成是 payload schema 不对
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
            # 边跑边打印,别等全部跑完 —— 中途 Ctrl-C 也不至于什么都没有
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
    不要只看均值。均值涨了但某几道题崩了是常态,
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
        print(f"⚠ {len(regressions)} 道题回退,先去看它们的 traces.jsonl 漏斗：{regressions[:5]}")


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
            # 让含关键词的文本在向量空间里靠拢,模拟语义信号
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

    if args.mock:
        jobs, pool, truth = make_mock_data()
        llm: LLMClient = MockLLM(cache, cfg)
        ev: EvalClient = MockEvalClient(truth)
    else:
        raise SystemExit(
            "接真实数据：在这里加载 jobs / candidates,并把 llm 换成真实 SDK 子类,"
            "ev = EvalClient(os.environ['EVAL_URL'], os.environ.get('EVAL_KEY',''))"
        )

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
