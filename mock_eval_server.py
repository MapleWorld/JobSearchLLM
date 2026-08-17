"""
mock_eval_server.py — local evaluation endpoint (stdlib http.server, zero deps).

Why run a real HTTP service instead of keeping MockEvalClient:

  The original MockEvalClient overrode submit() outright, which meant
  _build_payload, _submit_sync, urllib, retry/backoff and every HTTP error
  branch had never executed even once. On interview day that code would be
  running for the first time.

  With a localhost service, EvalClient takes the full HTTP path and the three
  ADAPTER functions are genuinely exercised.

── It also does something more important ───────────────────────────
  The biggest unknown on the day is what the real endpoint's response schema
  looks like. --schema switches between four common shapes so you can
  rehearse the actual motion: curl it, read the response, fix the ADAPTER in
  two minutes. Do that three times and it stops being scary.
────────────────────────────────────────────────────────────────────

Usage:
    python mock_data.py                       # generate data + labels first
    python mock_eval_server.py --port 8000    # start the server

    # Rehearse the ADAPTER against different shapes
    python mock_eval_server.py --schema nested
    python mock_eval_server.py --schema minimal     # no per-candidate scores!

    # Inject failures to exercise retry and degradation
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
# Four response schemas, mirroring API styles you actually meet in the wild
# ======================================================================

def schema_flat(job_id: str, graded: List[tuple], score: float) -> Dict:
    """Most common: top-level score plus a results array."""
    return {
        "job_id": job_id,
        "score": round(score, 4),
        "results": [{"candidate_id": c, "score": float(g)} for c, g in graded],
    }


def schema_nested(job_id: str, graded: List[tuple], score: float) -> Dict:
    """Enterprise style: deeply nested, boolean fields, different key names."""
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
    """Worst case: aggregate score only, no per-candidate results -> no gold set."""
    return {"job_id": job_id, "precision_at_10": round(score, 4)}


def schema_verbose(job_id: str, graded: List[tuple], score: float) -> Dict:
    """Per-criterion breakdown - richest data, but buried deepest."""
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
    def log_message(self, fmt, *args):  # Silence the default access log
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

        # --- Failure injection ---
        if rng.random() < cfg["fail_rate"]:
            cfg["injected_5xx"] += 1
            return self._send(503, {"error": "service unavailable (injected)"})
        if rng.random() < cfg["malformed_rate"]:
            cfg["injected_malformed"] += 1
            return self._send(200, b'{"job_id": "x", "results": [{"candidate_i')

        # --- Parse request. Deliberately strict: a wrong shape returns 400,
        #     which forces the ADAPTER to actually be correct. ---
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

        # --- Grading: reads the label file only, never touches retrieval code ---
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
    p.add_argument("--fail-rate", type=float, default=0.0, help="probability of a 503")
    p.add_argument("--malformed-rate", type=float, default=0.0,
                   help="probability of truncated JSON")
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
        print(f"  failure injection: 503={a.fail_rate:.0%}  malformed={a.malformed_rate:.0%}")
    print("  GET /health for status   Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\ncalls={STATE['calls']} 503={STATE['injected_5xx']} "
              f"malformed={STATE['injected_malformed']}")


if __name__ == "__main__":
    main()
