"""
diagnostics.py — 分阶段召回诊断。

这是整个项目里最值钱的 100 行代码。

核心命题：**如果正确候选人在 retrieval 阶段就被丢了,reranker 再强也救不回来。**
所以每次搜索都要记录漏斗（funnel）：
    pool → after_hard_filter → after_lexical → after_dense → after_fusion → final_top10
以及每一层"丢了谁"。

有了 gold set（见下）之后,就能算出每一层的 recall,一眼看出瓶颈在哪一层。
面试官问"你是怎么发现问题的",答案就是这张表。
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
    ids_out: List[str]           # 出口存活的 candidate id（用于算 recall）
    dropped_sample: List[str]    # 被丢掉的样本（最多 10 个,用于人工 eyeball）
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
    meta: Dict = field(default_factory=dict)     # 放 relaxation_level / compiled_criteria 等
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
            # ids_out 可能很长,落盘时截断非 final 阶段
            if len(s["ids_out"]) > 200:
                s["ids_out"] = s["ids_out"][:200]
        return d

    # ---------- 人类可读的漏斗表 ----------
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
                        f"    ! {s.name} 丢失 gold {len(leaked)} 个: {sorted(leaked)[:5]}"
                    )
        for w in self.warnings:
            lines.append(f"    ! WARN: {w}")
        return "\n".join(lines)


# ======================================================================
# 2. Gold Set —— 用 eval endpoint 的返回反推"正确答案"
# ======================================================================

class GoldSet:
    """
    这类题通常拿不到显式标注,但 evaluation endpoint 会告诉你
    "你提交的 10 个人里哪几个是对的"。把历次跑分中被判为『对』的
    candidate_id 累积起来,就得到一个逐轮变厚的伪 ground truth。

    有了它就能回答那个关键问题：
      「这一轮分数掉了,是 retrieval 漏召回,还是 rerank 排错了？」

    用法：
        gold = GoldSet("goldset.json")
        gold.absorb(job_id, {"c_01": 1.0, "c_07": 0.0, ...})   # 从 eval 响应解析
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
        """grades: {candidate_id: score}。取历史最大值,避免被单次波动抹掉。"""
        bucket = self._data.setdefault(job_id, {})
        for cid, sc in grades.items():
            bucket[cid] = max(bucket.get(cid, float("-inf")), float(sc))
        self.save()

    def get(self, job_id: str) -> Set[str]:
        return {c for c, s in self._data.get(job_id, {}).items() if s >= self.threshold}

    def negatives(self, job_id: str) -> Set[str]:
        """已知的错误答案 —— 可以拿来做 few-shot 负例,或验证 filter 是否有效。"""
        return {c for c, s in self._data.get(job_id, {}).items() if s < self.threshold}

    def coverage(self) -> Dict[str, int]:
        return {j: len(self.get(j)) for j in self._data}


# ======================================================================
# 3. 无标注时的替代诊断：Oracle Ceiling
# ======================================================================

def oracle_ceiling_note(trace: SearchTrace, gold: Set[str]) -> str:
    """
    gold 还没攒起来的时候（前两轮）,用这个粗判：
    把 filter 全关、top_k 拉到 200 跑一次,如果分数明显变高,
    说明瓶颈在 filter 太严（precision-recall 折中点选错了）,而不是 rerank。
    """
    if not trace.stages:
        return "no stages recorded"
    first, last = trace.stages[0], trace.stages[-1]
    if not gold:
        return (
            f"pool={first.count_in} -> final={last.count_out}; "
            "gold set 为空,建议先跑一次 --level 0 baseline 攒 gold"
        )
    reached = len(gold & set(last.ids_out))
    return f"ceiling: gold={len(gold)}, 进入 final 的 gold={reached}"
