"""
mock_data.py — 生成"有难度"的合成数据集 + 独立标注。

为什么要重做数据集：
  原来的 make_mock_data 里，"好候选人"关键词密度高、坏候选人完全不沾边，
  BM25 一把就能抓完，所有 run 都是 1.000 —— 这种数据集测不出任何东西。

这里刻意埋了四类候选人，对应四种真实失败模式：

  QUALIFIED   真合格，关键词也明显            -> 基线，都该找到
  DISTRACTOR  关键词齐全但硬条件不达标         -> 测 precision：光靠语义会被骗
  HIDDEN_GEM  真合格但用同义表述，不含原词      -> 测 recall：精确匹配会漏
  IRRELEVANT  无关                            -> 噪声

标注（labels.json）由本文件独立生成，检索代码看不到，
mock server 只读标注文件 —— 这样 eval 才是外部裁判，不是自己给自己打分。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

OUT_DIR = "mock_data"

JOBS = [
    {
        "job_id": "job_001",
        "title": "Staff Backend / Search Engineer",
        "description": "Build distributed search and ranking pipelines at scale.",
        "hard_criteria": [
            "At least 5 years of backend engineering experience",
            "Proficient in Python",
            "Has built distributed systems",
        ],
    },
    {
        "job_id": "job_002",
        "title": "Senior ML Platform Engineer",
        "description": "Own model serving and the feature store for ranking models.",
        "hard_criteria": [
            "At least 5 years of engineering experience",
            "Production machine learning experience",
            "Experience with large-scale data infrastructure",
        ],
    },
]

# HIDDEN_GEM 用这些说法，绝不出现 "Python" / "distributed systems" 原词
SYNONYM_PHRASES = [
    "large-scale service mesh and sharded storage layers",
    "horizontally scaled multi-region infrastructure",
    "high-throughput streaming pipelines across clusters",
]
SYNONYM_SKILLS = [["Py3", "Django", "Celery"], ["CPython", "FastAPI"], ["Python3", "Ray"]]


def generate(n: int = 400, seed: int = 7, scarce: bool = False) -> Tuple[List[Dict], List[Dict], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    candidates: List[Dict] = []
    labels: Dict[str, Dict[str, int]] = {j["job_id"]: {} for j in JOBS}

    for i in range(n):
        cid = f"cand_{i:04d}"
        r = i % 10
        if scarce:
            # 稀缺模式：合格者只有 ~6%，top-10 装不满 -> precision@10 才有区分度
            r = {0: 0, 1: 4}.get(i % 33, 2 if i % 3 == 0 else 9)

        if r in (0, 1):  # QUALIFIED
            kind = "QUALIFIED"
            yrs = rng.randint(5, 15)
            skills = ["Python", "Distributed Systems", "Kubernetes", "Spark"]
            text = ("Senior engineer building distributed search ranking pipelines "
                    "and production machine learning services in Python at scale.")
            qualified = True

        elif r in (2, 3):  # DISTRACTOR 20% —— 关键词全有，但年限不够
            kind = "DISTRACTOR"
            # 关键：年限也达标、技能关键词也齐全 —— 结构化字段无法区分。
            # 只有读懂"他是在写文档/做QA，不是在建系统"才能判掉。
            # 若 distractor 能被 years>=5 一刀切掉，那 mock 就退化成了 oracle，
            # 任何配置都拿满分，测不出东西。
            yrs = rng.randint(5, 14)
            skills = ["Python", "Distributed Systems", "Kubernetes", "Spark"]
            text = ("Technical writer documenting distributed search ranking pipelines "
                    "and production machine learning systems. Wrote Python tutorials "
                    "and QA test plans for the platform team at scale.")
            qualified = False

        elif r == 4:  # HIDDEN_GEM 10% —— 真合格但不含原词
            kind = "HIDDEN_GEM"
            yrs = rng.randint(6, 16)
            skills = rng.choice(SYNONYM_SKILLS)
            text = (f"Principal engineer. Designed {rng.choice(SYNONYM_PHRASES)} "
                    "serving ranked results, plus model deployment and feature storage "
                    "for recommendation workloads.")
            qualified = True

        else:  # IRRELEVANT 50%
            kind = "IRRELEVANT"
            yrs = rng.randint(0, 12)
            skills = rng.choice([["Java", "Spring"], ["PHP", "Laravel"], ["Excel", "SQL"]])
            text = rng.choice([
                "Frontend developer focused on responsive web design and CSS.",
                "IT support specialist handling desktop provisioning and ticketing.",
                "Marketing analyst producing campaign dashboards.",
            ])
            qualified = False

        candidates.append({
            "candidate_id": cid,
            "name": f"Candidate {i}",
            "kind": kind,                      # 只用于事后分析，检索侧不读
            "years_experience": yrs,
            "skills": skills,
            "location": rng.choice(["Remote", "Seattle, WA", "New York, NY"]),
            "profile_text": f"{text} Skills: {', '.join(skills)}. {yrs} years of experience.",
        })

        for j in JOBS:
            labels[j["job_id"]][cid] = 1 if qualified else 0

    return JOBS, candidates, labels


def dump(out_dir: str = OUT_DIR, n: int = 400, seed: int = 7, scarce: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jobs, cands, labels = generate(n, seed, scarce)

    with open(os.path.join(out_dir, "jobs.json"), "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    # 候选人文件里剥掉 kind 和 labels —— 检索侧不该看见答案
    with open(os.path.join(out_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in c.items() if k != "kind"} for c in cands],
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "_kinds.json"), "w", encoding="utf-8") as f:
        json.dump({c["candidate_id"]: c["kind"] for c in cands}, f, indent=2)

    from collections import Counter
    dist = Counter(c["kind"] for c in cands)
    pos = sum(labels[JOBS[0]["job_id"]].values())
    print(f"wrote {out_dir}/  jobs={len(jobs)} candidates={len(cands)}")
    print(f"  分布: {dict(dist)}")
    print(f"  每题合格者 {pos} 人 -> top-10 理论满分可达 1.000，但需同时避开 "
          f"{dist['DISTRACTOR']} 个 distractor")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--scarce", action="store_true", help="合格者稀缺，precision@10 才有区分度")
    a = p.parse_args()
    dump(a.out, a.n, a.seed, a.scarce)
