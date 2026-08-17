"""
mock_data.py — generate a genuinely hard synthetic dataset + independent labels.

Why the dataset was rebuilt:
  In the original make_mock_data, "good" candidates were keyword-dense and
  "bad" ones shared no vocabulary at all, so BM25 alone swept every one of
  them up and every run scored 1.000. A dataset like that measures nothing.

Four candidate classes are planted here, each targeting a real failure mode:

  QUALIFIED   Genuinely qualified, obvious keywords -> baseline; must be found
  DISTRACTOR  All the keywords, fails a hard bar    -> tests precision: pure
                                                       semantics gets fooled
  HIDDEN_GEM  Qualified but phrased in synonyms     -> tests recall: exact
                                                       matching misses these
  IRRELEVANT  Unrelated                             -> noise

The labels (labels.json) are produced here, independently. The retrieval code
never sees them, and the mock server reads nothing but the label file - which
is what makes the evaluation an external judge rather than the system grading
its own homework.
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

# HIDDEN_GEM candidates use these phrasings and never the literal
# "Python" / "distributed systems" tokens
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
            # Scarce mode: only ~6% qualify, so top-10 cannot be filled with
            # easy positives - which is what gives precision@10 any resolution
            r = {0: 0, 1: 4}.get(i % 33, 2 if i % 3 == 0 else 9)

        if r in (0, 1):  # QUALIFIED
            kind = "QUALIFIED"
            yrs = rng.randint(5, 15)
            skills = ["Python", "Distributed Systems", "Kubernetes", "Spark"]
            text = ("Senior engineer building distributed search ranking pipelines "
                    "and production machine learning services in Python at scale.")
            qualified = True

        elif r in (2, 3):  # DISTRACTOR - every keyword present
            kind = "DISTRACTOR"
            # Critical: years of experience ALSO clears the bar and the skill
            # keywords are ALSO all present, so no structured field separates
            # these from real hires. Only reading the text - "this person
            # documents and QAs the system, they do not build it" - rejects them.
            # If a distractor could be removed by a years>=5 predicate, the mock
            # degenerates into an oracle: every config scores a perfect 1.000
            # and the harness measures nothing.
            yrs = rng.randint(5, 14)
            skills = ["Python", "Distributed Systems", "Kubernetes", "Spark"]
            text = ("Technical writer documenting distributed search ranking pipelines "
                    "and production machine learning systems. Wrote Python tutorials "
                    "and QA test plans for the platform team at scale.")
            qualified = False

        elif r == 4:  # HIDDEN_GEM - qualified, but never uses the literal keywords
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
            "kind": kind,                      # Post-hoc analysis only; retrieval never reads it
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
    # Strip kind and labels from the candidate file - retrieval must not see answers
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
    print(f"  distribution: {dict(dist)}")
    print(f"  {pos} qualified per job -> a perfect 1.000 is reachable, but only "
          f"by avoiding all {dist['DISTRACTOR']} distractors")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--scarce", action="store_true",
                   help="make positives scarce so precision@10 has resolution")
    a = p.parse_args()
    dump(a.out, a.n, a.seed, a.scarce)
