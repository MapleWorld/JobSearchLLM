"""
diagnostics.py — stage-by-stage recall diagnostics.

This is the highest-value code in the project.

Central claim: **if a correct candidate is dropped during retrieval, no
reranker, however good, can bring them back.**
So every search records a funnel:
    pool → after_hard_filter → after_lexical → after_dense → after_fusion → final_top10
...along with exactly who was lost at each stage.

Once a gold set exists (see below), recall can be computed per stage, which
makes the bottleneck obvious at a glance. When an interviewer asks "how did
you find the problem?", this table is the answer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set


# ======================================================================
# 1. Funnel Trace
# ======================================================================

@dataclass
class StageRecord:
    name: str
    count_in: int
    count_out: int
    ids_out: List[str]           # Candidate ids surviving this stage (used for recall)
    dropped_sample: List[str]    # Up to 10 dropped ids, for eyeballing
    elapsed_ms: float
    note: str = ""

    @property
    def survival_rate(self) -> float:
        return self.count_out / self.count_in if self.count_in else 0.0


@dataclass
class SearchTrace:
    job_id: str
    run_id: str = ""
    stages: List[StageRecord] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)     # relaxation_level, compiled_criteria, etc.
    warnings: List[str] = field(default_factory=list)
    total_ms: float = 0.0

    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def stage(
        self,
        name: str,
        ids_in: List[str],
        ids_out: List[str],
        elapsed_ms: float,
        note: str = "",
    ) -> None:
        dropped = [i for i in ids_in if i not in set(ids_out)]
        self.stages.append(
            StageRecord(
                name=name,
                count_in=len(ids_in),
                count_out=len(ids_out),
                ids_out=list(ids_out),
                dropped_sample=dropped[:10],
                elapsed_ms=round(elapsed_ms, 1),
                note=note,
            )
        )

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def finish(self) -> "SearchTrace":
        self.total_ms = round((time.perf_counter() - self._t0) * 1000, 1)
        return self

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("_t0", None)
        for s in d["stages"]:
            # ids_out can be long; truncate non-final stages when persisting
            if len(s["ids_out"]) > 200:
                s["ids_out"] = s["ids_out"][:200]
        return d

    # ---------- Human-readable funnel table ----------
    def render_funnel(self, gold: Optional[Set[str]] = None) -> str:
        head = f"[FUNNEL] job={self.job_id}  total={self.total_ms}ms"
        if self.meta.get("relaxation_level"):
            head += f"  relax={self.meta['relaxation_level']}"
        lines = [head]
        cols = f"  {'stage':<22}{'in':>6}{'out':>6}{'surv':>8}{'ms':>8}"
        if gold:
            cols += f"{'recall':>9}{'lost':>7}"
        lines.append(cols)
        lines.append("  " + "-" * (len(cols) - 2))

        prev_recall = 1.0
        for s in self.stages:
            row = (
                f"  {s.name:<22}{s.count_in:>6}{s.count_out:>6}"
                f"{s.survival_rate:>7.1%}{s.elapsed_ms:>8.0f}"
            )
            if gold:
                kept = len(gold & set(s.ids_out))
                recall = kept / len(gold) if gold else 0.0
                lost = prev_recall - recall
                flag = "  <-- LEAK" if lost > 0.15 else ""
                row += f"{recall:>8.1%}{lost:>7.1%}{flag}"
                prev_recall = recall
            lines.append(row)

        if gold:
            for i, s in enumerate(self.stages):
                upstream = gold if i == 0 else gold & set(self.stages[i - 1].ids_out)
                leaked = upstream - set(s.ids_out)
                if leaked:
                    lines.append(
                        f"    ! {s.name} lost {len(leaked)} gold: {sorted(leaked)[:5]}"
                    )
        for w in self.warnings:
            lines.append(f"    ! WARN: {w}")
        return "\n".join(lines)


# ======================================================================
# 2. Gold Set - reconstruct "correct answers" from eval endpoint responses
# ======================================================================

class GoldSet:
    """
    Tasks like this rarely ship explicit labels, but the evaluation endpoint
    does tell you which of your submitted 10 were correct. Accumulating every
    candidate_id ever graded as correct yields a pseudo ground truth that
    thickens with each run.

    That is enough to answer the question that actually matters:
      "this run scored worse - did retrieval miss them, or did rerank
       order them wrong?"

    Usage:
        gold = GoldSet("goldset.json")
        gold.absorb(job_id, {"c_01": 1.0, "c_07": 0.0, ...})   # parsed from eval response
        gold.get("job_123")  -> {"c_01", ...}
    """

    def __init__(self, path: str = "goldset.json", threshold: float = 0.5):
        self.path = path
        self.threshold = threshold
        self._data: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def absorb(self, job_id: str, grades: Dict[str, float]) -> None:
        """grades: {candidate_id: score}. Keeps the historical max so a single
        noisy run cannot erase a known-good candidate."""
        bucket = self._data.setdefault(job_id, {})
        for cid, sc in grades.items():
            bucket[cid] = max(bucket.get(cid, float("-inf")), float(sc))
        self.save()

    def get(self, job_id: str) -> Set[str]:
        return {c for c, s in self._data.get(job_id, {}).items() if s >= self.threshold}

    def negatives(self, job_id: str) -> Set[str]:
        """Known-bad answers - useful as few-shot negatives, or to verify the filter."""
        return {c for c, s in self._data.get(job_id, {}).items() if s < self.threshold}

    def coverage(self) -> Dict[str, int]:
        return {j: len(self.get(j)) for j in self._data}


# ======================================================================
# 3. Fallback diagnostic when no labels exist yet: Oracle Ceiling
# ======================================================================

def oracle_ceiling_note(trace: SearchTrace, gold: Set[str]) -> str:
    """
    For the first couple of runs, before the gold set has accumulated, use
    this rough check: turn every filter off, push top_k to 200, and re-run.
    If the score jumps noticeably, the bottleneck is an over-aggressive
    filter (wrong precision/recall operating point), not the reranker.
    """
    if not trace.stages:
        return "no stages recorded"
    first, last = trace.stages[0], trace.stages[-1]
    if not gold:
        return (
            f"pool={first.count_in} -> final={last.count_out}; "
            "gold set is empty; run a --level 0 baseline first to seed it"
        )
    reached = len(gold & set(last.ids_out))
    return f"ceiling: gold={len(gold)}, gold reaching final={reached}"
