"""
mock_eval_server.py — 本地 evaluation endpoint（stdlib http.server，零依赖）。

为什么要起一个真的 HTTP 服务，而不是继续用 MockEvalClient：

  原来的 MockEvalClient 直接 override 了 submit()，于是 _build_payload、
  _submit_sync、urllib、重试退避、HTTP 错误分支 —— 整条真实链路一行都没跑过。
  面试当天那些代码是第一次执行。

  起一个 localhost 服务，EvalClient 走完整的 HTTP 路径，
  ADAPTER 三函数也真正被调用。

── 它还能干一件更重要的事 ─────────────────────────────────
  面试当天最大的未知数是「真实 endpoint 的响应 schema 长什么样」。
  --schema 可以在四种常见形态之间切换，用来演练那个动作：
  curl 一发 -> 看响应 -> 两分钟内改完 ADAPTER。
  练过三遍，当天就不会慌。
───────────────────────────────────────────────────────

用法：
    python mock_data.py                       # 先生成数据和标注
    python mock_eval_server.py --port 8000    # 起服务

    # 换 schema 演练 ADAPTER
    python mock_eval_server.py --schema nested
    python mock_eval_server.py --schema minimal     # 不返回逐候选人分数！

    # 注入故障，测重试和降级
    python mock_eval_server.py --fail-rate 0.3 --latency-ms 400 --malformed-rate 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List

STATE: Dict = {}


# ======================================================================
# 四种响应 schema —— 对应真实世界里常见的几种 API 风格
# ======================================================================

def schema_flat(job_id: str, graded: List[tuple], score: float) -> Dict:
    """最常见：顶层 score + results 数组。"""
    return {
        "job_id": job_id,
        "score": round(score, 4),
        "results": [{"candidate_id": c, "score": float(g)} for c, g in graded],
    }


def schema_nested(job_id: str, graded: List[tuple], score: float) -> Dict:
    """企业风：层层包裹 + 布尔字段 + 字段名不一样（id / relevant）。"""
    return {
        "status": "ok",
        "data": {
            "evaluation": {
                "jobId": job_id,
                "precision": round(score, 4),
                "candidates": [
                    {"id": c, "relevant": bool(g)} for c, g in graded
                ],
            }
        },
    }


def schema_minimal(job_id: str, graded: List[tuple], score: float) -> Dict:
    """最坏情况：只给总分，不给逐候选人结果 -> gold set 攒不起来。"""
    return {"job_id": job_id, "precision_at_10": round(score, 4)}


def schema_verbose(job_id: str, graded: List[tuple], score: float) -> Dict:
    """逐条 criteria 拆解 —— 信息最多，但字段藏得深。"""
    return {
        "job_id": job_id,
        "overall_score": round(score, 4),
        "details": [
            {
                "candidate_id": c,
                "pass": bool(g),
                "criteria_breakdown": [
                    {"criterion": f"c{i}", "met": bool(g)} for i in range(3)
                ],
            }
            for c, g in graded
        ],
    }


SCHEMAS = {
    "flat": schema_flat,
    "nested": schema_nested,
    "minimal": schema_minimal,
    "verbose": schema_verbose,
}


# ======================================================================
# Handler
# ======================================================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静音默认访问日志
        pass

    def _send(self, code: int, body) -> None:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "schema": STATE["schema"],
                             "calls": STATE["calls"], "jobs": list(STATE["labels"])})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        cfg = STATE
        cfg["calls"] += 1
        rng = cfg["rng"]

        if cfg["latency_ms"]:
            time.sleep(cfg["latency_ms"] / 1000.0)

        if self.path != "/evaluate":
            return self._send(404, {"error": f"unknown path {self.path}"})

        # --- 故障注入 ---
        if rng.random() < cfg["fail_rate"]:
            cfg["injected_5xx"] += 1
            return self._send(503, {"error": "service unavailable (injected)"})
        if rng.random() < cfg["malformed_rate"]:
            cfg["injected_malformed"] += 1
            return self._send(200, b'{"job_id": "x", "results": [{"candidate_i')

        # --- 解析请求（故意严格：schema 不对就 400，逼你把 ADAPTER 写对）---
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode())
        except Exception:
            return self._send(400, {"error": "body is not valid JSON"})

        job_id = payload.get("job_id") or payload.get("jobId")
        cands = payload.get("candidate_ids") or payload.get("candidates")
        if not job_id or not isinstance(cands, list):
            return self._send(400, {
                "error": "expected {'job_id': str, 'candidate_ids': [str]}",
                "received_keys": sorted(payload.keys()),
            })
        if job_id not in cfg["labels"]:
            return self._send(404, {"error": f"unknown job_id {job_id}"})

        if cfg["require_auth"] and not self.headers.get("Authorization"):
            return self._send(401, {"error": "missing Authorization header"})

        # --- 评分：只查标注文件，完全不碰检索代码 ---
        lab = cfg["labels"][job_id]
        cands = cands[: cfg["top_k"]]
        graded = [(c, lab.get(c, 0)) for c in cands]
        score = (sum(g for _, g in graded) / len(graded)) if graded else 0.0

        cfg["history"].append({"job_id": job_id, "n": len(cands), "score": round(score, 4)})
        self._send(200, SCHEMAS[cfg["schema"]](job_id, graded, score))


# ======================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Mock evaluation endpoint")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--labels", default="mock_data/labels.json")
    p.add_argument("--schema", default="flat", choices=sorted(SCHEMAS))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--fail-rate", type=float, default=0.0, help="返回 503 的概率")
    p.add_argument("--malformed-rate", type=float, default=0.0, help="返回截断 JSON 的概率")
    p.add_argument("--latency-ms", type=int, default=0)
    p.add_argument("--require-auth", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    with open(a.labels, encoding="utf-8") as f:
        labels = json.load(f)

    STATE.update(
        labels=labels, schema=a.schema, top_k=a.top_k,
        fail_rate=a.fail_rate, malformed_rate=a.malformed_rate,
        latency_ms=a.latency_ms, require_auth=a.require_auth,
        rng=random.Random(a.seed), calls=0, injected_5xx=0,
        injected_malformed=0, history=[],
    )

    srv = HTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock eval endpoint  ->  http://127.0.0.1:{a.port}/evaluate")
    print(f"  schema={a.schema}  jobs={list(labels)}  top_k={a.top_k}")
    if a.fail_rate or a.malformed_rate:
        print(f"  故障注入: 503={a.fail_rate:.0%}  malformed={a.malformed_rate:.0%}")
    print("  GET /health 查看状态   Ctrl-C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\ncalls={STATE['calls']} 503={STATE['injected_5xx']} "
              f"malformed={STATE['injected_malformed']}")


if __name__ == "__main__":
    main()
