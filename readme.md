# Candidate Search — a 75-minute end-to-end battle plan

Given a job description (title / description / hard criteria), retrieve and rank
candidates from a database and return the top 10.

> This is not a polished "final architecture" document. It is a **battle plan**.
> Guiding principle: **get a score first, architecture second.**
> You should have a submittable baseline by minute 10.

---

## Quick start (fully offline, no API key required)

```bash
pip install numpy                                  # the only third-party dependency

python mock_data.py --scarce                       # generate data + independent labels
python mock_eval_server.py --port 8000 &           # start a local eval endpoint
python harness.py run --level 3 --mock-llm \
       --data-dir mock_data --eval-url http://127.0.0.1:8000

python harness.py compare                          # diff previous runs
python harness.py export                           # produce evaluation_results.json
```

When wiring up a real LLM, **run selftest first**. Two minimal requests validate
the key, model names and embedding dimension before you burn any quota:

```bash
cp .env.example .env && vi .env          # fill in GEMINI_API_KEY
python harness.py run --level 3 --selftest
# {"chat": {"model": "gemini-3.7-flash", "ok": true, ...},
#  "embed": {"model": "gemini-embedding-001", "ok": true, "returned": 3, "dim": 768}}
```

---

## File map

| File | Lines | Responsibility |
|---|---:|---|
| `engine.py` | ~670 | Data models, LLM wrapper, BM25, RRF, leveled pipeline |
| `harness.py` | ~565 | Eval client + ADAPTER, experiment runner, run comparison, CLI |
| `llm_clients.py` | ~200 | Real Gemini client (REST, zero deps) + selftest |
| `diagnostics.py` | ~200 | Funnel trace, gold set accumulation, stage-wise recall |
| `mock_eval_server.py` | ~216 | Local eval endpoint (real HTTP, switchable schema, fault injection) |
| `mock_data.py` | ~161 | Deliberately hard synthetic dataset + independent labels |
| `settings.py` | ~176 | `.env` loading (zero deps) + config validation |
| `cache.py` | ~110 | Persistent sqlite cache + hit-rate statistics |

Dependencies: `numpy` plus the standard library. BM25 is pure Python (no
`rank_bm25`); the `.env` parser is built in (no `python-dotenv`); the mock server
uses `http.server` (no Flask).

**`pip install` failing in an interview environment is a real event** — every
dependency removed is one less way to lose time.

---

## Architecture

```
Input: Job Description (title, description, hard_criteria[])
   |
   +- Phase 1  Query Compilation (LLM-as-Compiler, JSON mode)
   |     -> min_years / required_skills / skill_synonyms /
   |        semantic_query / keyword_query / checkable_criteria[]
   |
   +- Phase 2  Retrieval
   |     +- 2a  Graded hard filter (stepwise relaxation)
   |     +- 2b  BM25 channel    <- proper nouns, companies, certs, framework versions
   |     +- 2c  Dense channel   <- semantic generalization, synonyms
   |     +- 2d  RRF fusion -> top 30
   |
   +- Phase 3  Rerank (LLM-as-Judge, 5 candidates per call)
   |     -> per-criterion binary verdict + evidence quote + soft_score
   |
   +- Phase 4  Ordering and submission
         final = hard_passed * 1000 + soft_score + retrieval_score * 0.01
         Lexicographic: hard criteria met, then soft fit, then retrieval score
```

### Three decisions worth defending

**1. Why not a holistic 0-100 score?**
A single holistic score clusters candidates in the 82-88 band, so **rank 10 and
rank 15 become indistinguishable** — and the entire scoring difference in a
top-10 task lives exactly at that boundary. Per-criterion binary verdicts give
natural coarse strata; the `evidence` field lets you tell "the LLM judged wrong"
apart from "the data never contained it"; and the objective aligns directly with
what the evaluation measures.

**2. Why keep a BM25 channel?**
The high-signal tokens in hard criteria — company names, schools, certifications,
`PyTorch 2.x` — are precisely what embeddings handle worst and BM25 handles best.
A dense-only "hybrid" is hybrid in name only.

**3. Why RRF instead of weighted scores?**
BM25 scores are unbounded, cosine lives in [-1, 1]. The scales are incomparable,
so weighted summation needs per-query tuning. RRF uses ranks only: zero tuning.

---

## Configuration: `.env`

```bash
cp .env.example .env      # fill in your key
python settings.py        # confirm it loaded (keys are masked)
```

> **If this repository is public**: once `.env` is committed the key is public,
> and deleting the file does not help because git history retains it.
> `.gitignore` must exist *before* `.env` — verified to exclude `.env`,
> `cache.sqlite3`, `goldset.json` and `runs/`.
> Consider switching the repo back to private after the interview.

Precedence is **`.env` < real environment variables**, not the other way around —
an already-exported value is never silently overwritten by `.env`.

`settings.py` reports which level is currently reachable and fails fast **before
burning quota**:

```
  chat_key    : AIzaSy...cdef (36 chars)
  embed_key   : AIzaSy...cdef (36 chars)
  -> can run up to --level 3
```

### Choosing a provider

| | Chat | Embedding | Keys | Notes |
|---|---|---|---:|---|
| **Gemini** (recommended) | `gemini-3.7-flash` | `gemini-embedding-001` | **1** | One key covers both |
| Anthropic + Voyage | Claude | `voyage-3.5` | 2 | Anthropic ships no embedding model |
| OpenAI | GPT | `text-embedding-3-small` | 1 | — |

`llm_clients.py` uses REST (urllib) rather than the `google-genai` SDK: no new
dependencies, and it can be fully tested against a local fake server. Set
`GEMINI_BASE_URL` to point at a proxy or a test server.

Gemini pitfalls (surfaced as startup warnings in `settings.py` and as runtime
assertions in `llm_clients.py`):

- **`gemini-embedding-2` returns a single aggregated vector for a list input**,
  not one per item. `_raw_embed` expects a one-to-one mapping, so using it
  directly **misaligns data silently** with no error.
  Sidestepped by using the `batchEmbedContents` endpoint, with an
  `EMBED_COUNT_MISMATCH` assertion as a backstop.
- **Embeddings below 3072 dimensions are not pre-normalized** and must be
  normalized manually (`engine.retrieve()` already does this).
- `temperature` / `top_p` / `top_k` are deprecated on recent Gemini models — do
  not send them.

---

## The evaluation loop

```
run pipeline -> submit to eval endpoint -> parse per-candidate scores -> absorb into gold set
     ^                                                                        |
     |                                                                        v
  change config <- read per-job delta <- persist to runs/<run_id>/ <- print funnel diagnostics
```

Every run writes three files, so **you always have something submittable even if
interrupted**:

```
runs/<run_id>/
  |- summary.json    config + mean/median/min + LLM usage + cache hit rate
  |- results.jsonl   per-job top-10, score, per-candidate scoring detail
  +- traces.jsonl    the full funnel for each job
```

### The local mock endpoint

`mock_eval_server.py` is a **real HTTP service**, not a mocked class — so
`_build_payload`, urllib, retry/backoff and every HTTP error branch genuinely
execute.

It also rehearses the biggest unknown of the day: **what the real endpoint's
response schema looks like**.

```bash
python mock_eval_server.py --schema nested     # nested, different key names (id/relevant)
python mock_eval_server.py --schema minimal    # aggregate only, no per-candidate -> no gold set
python mock_eval_server.py --schema verbose    # per-criterion breakdown
python mock_eval_server.py --fail-rate 0.3 --malformed-rate 0.1 --require-auth
```

Practise "curl it, read the response, fix the ADAPTER in two minutes" three times
and it stops being scary.

### The ADAPTER

`EvalClient` in `harness.py` contains an ADAPTER block — the only part that must
change for the real endpoint:

```python
def _build_payload(job_id, candidate_ids) -> Dict     # request body
def extract_overall_score(resp) -> float              # aggregate score
def extract_per_candidate(resp) -> Dict[str, float]   # per-candidate scores
```

The third matters most — **it is the only source of the gold set**, and without
it you can only tune blind.

The current implementation walks the JSON tree recursively and assumes nothing
about depth; it parses flat / nested / minimal / verbose / map shapes in testing.
`looks_unparsed()` separates **a genuine zero** from **a schema mismatch**: both
return 0.0, but one sends you to fix retrieval and the other to fix the ADAPTER.

---

## Stage-wise recall diagnostics

> **If a correct candidate is dropped during retrieval, no reranker can recover them.**

```
[FUNNEL] job=job_002  total=56.0ms  relax=L0_strict
  stage                     in   out    surv      ms   recall   lost
  ------------------------------------------------------------------
  pool                     300   300 100.0%       0  100.0%   0.0%
  hard_filter              300    75  25.0%       1  100.0%   0.0%
  fuse_rrf                  75    30  40.0%       0   72.2%  27.8%  <-- LEAK
  llm_rerank                30    30 100.0%      28   72.2%   0.0%
  final_top_k               30    10  33.3%       0   55.6%  16.7%  <-- LEAK
```

**This table is the answer to "how did you find the problem?"** Above, the
bottleneck is clearly `fuse_rrf`, not the reranker — so tune `retrieve_k`, not
the prompt.

### Iteration decision tree

```
Read the recall column and find the first LEAK:
  hard_filter leaks  -> relaxation too shallow / skill_synonyms too thin /
                        compiler treated a preference as a requirement -> fix compiler prompt
  fuse_rrf leaks     -> retrieve_k too small (30 -> 50), or one channel is weak
  llm_rerank leaks   -> now it IS a prompt problem -> check whether checks[].evidence
                        is empty (data absent) or wrong (prompt unclear)
  final_top_k leaks  -> ordering formula -> are too many candidates tied on hard_passed?
```

### Where the gold set comes from

Explicit labels are rarely provided, but the eval endpoint tells you which of
your submitted 10 were correct. Accumulating every id ever graded correct into
`goldset.json` yields a **pseudo ground truth that thickens each run**.

It is biased — it only contains candidates you have already retrieved — so it
answers "why is this run worse than the last?" but **not "how far am I from the
ceiling?"** While the gold set is still thin, use an oracle ceiling check: turn
every filter off, push `retrieve_k` to 200, and re-run. A noticeable jump means
the filter is too aggressive.

---

## Graded relaxation of the hard filter

Dropping straight to "no filtering" throws away the hard-criteria signal. Use a
ladder, and record which rung was used in the trace:

| Level | Years | Skills | Location |
|---|---|---|---|
| `L0_strict` | strict | **all** must match | checked |
| `L1_drop_location` | strict | all must match | ignored |
| `L2_any_skill` | strict | **any** match | ignored |
| `L3_years_slack` | -2 years | any match | ignored |
| `L4_semantic_only` | off | off | off |

If `relax=L2_any_skill` appears too often in the logs, the compiler is
misclassifying preferences as requirements — go fix the "Be conservative" line in
its prompt.

---

## Caching

You will run a dozen-plus iterations in 75 minutes. sqlite on disk means it
**still hits after a restart**; the key carries `model` and `prompt_version`, so
**changing the prompt misses automatically and never serves a stale result**.
Measured: re-running the same config hits 100%.

Embeddings are stored as Python `list[float]` rather than `np.float32`. At 1536
dimensions:

| Candidates | RAM | sqlite file |
|---:|---:|---:|
| 1,000 | 0.05 GB | 0.02 GB |
| 10,000 | 0.49 GB | 0.20 GB |
| 50,000 | **2.46 GB** | 1.00 GB |

Above ~50k, switch to `float32` (roughly 1/8 the size) or use Gemini's
`output_dimensionality=768`.

---

## Time-degradation roadmap

**The easiest way to fail: a beautiful architecture that never produces results,
so you submit nothing.**

Every level is a **complete, independently submittable system** (`--level 0..3`).
Whenever you get stuck, the previous level's results are already on disk.

| Minutes | Goal | If you get stuck |
|---|---|---|
| **0-10** | Read the schema, curl the endpoint, fix the ADAPTER, **submit an L0 BM25 baseline** | Endpoint unreachable? Run the pipeline with `--no-submit` |
| **10-25** | Add LLM compilation + graded hard filter, submit L1 | Compiler returns invalid JSON? Three fallback layers already handle it |
| **25-40** | Add dense + RRF, submit L2 | Embedding too slow? Lower `retrieve_k`, or stop at L1 |
| **40-58** | Add per-criterion LLM rerank, submit L3 | Timing out? Raise `--batch-size` |
| **58-70** | **Iterate on the funnel**: fix whichever stage leaks | See the decision tree above |
| **70-75** | `export` and prepare your narrative | — |

**Hard rules**

- No submitted score by minute 15 → run `--level 0` immediately.
- Still debugging L3 at minute 50 → abandon L3, stabilize on L2, spend the rest
  on diagnostics and narrative.

A working L2 whose debugging story you can explain beats an unfinished L3.

---

## Robustness checklist

| Failure | Handling | Location |
|---|---|---|
| LLM output wrapped in fences or padded with prose | Three layers: direct parse -> strip fence -> balanced-bracket scan | `parse_json_loose` |
| LLM call fails | Exponential backoff, 3 attempts; **on final failure fall back to the retrieval score, never 0** | `chat_json` |
| Why that matters | Scoring 0 **silently discards good candidates**, and shows up in the funnel as a rerank-stage leak, sending you to fix the wrong thing | — |
| Judge omits a candidate from a batch | Align by `candidate_id` rather than position; fill the gap and log a warning | `_score_batch` |
| Hard filter empties the pool | Graded relaxation, not an on/off switch | `hard_filter` |
| Dense channel fails | Degrade to lexical only, log a warning, do not abort | `retrieve` |
| Eval endpoint returns 4xx | Do not retry — the payload schema is almost certainly wrong | `_submit_sync` |
| Ctrl-C mid-run | Print as you go, write to disk after each job | `ExperimentRunner.run` |

---

## Known issues (found by testing, not yet fixed)

Ordered by priority. These are measured, not hypothetical.

| # | Issue | Impact | Cost to fix |
|---|---|---|---|
| 1 | **The funnel treats parallel retrieval channels as sequential** | A candidate dropped by BM25 but recovered by dense is reported as a **false LEAK** (measured at 50%), plus a meaningless negative loss. The diagnostic tool actively misleads you | Medium |
| 2 | `parse_json_loose` does not handle truncation / single quotes / trailing commas | When batch scoring hits `max_tokens` the batch degrades (no crash, but wasted) | Low |
| 3 | No client for providers other than Gemini | Setting anthropic / openai in `.env` fails fast with a clear message | Low, ~40 lines |

Fences, surrounding prose and braces inside strings are already handled.

### Fixed (verified by testing)

| Issue | Fix | Measured effect |
|---|---|---|
| No real LLM client | `llm_clients.py` REST implementation + `--selftest` | Seven failure modes (429 / 404 / MAX_TOKENS / SAFETY / aggregation / dimension mismatch / success) all handled correctly |
| Skill substring false positives | Word-boundary regex, right boundary tolerant of `c++` / `c#` | `"go"` no longer matches `google`; `Go` / `C++` / `R` match correctly |
| BM25 index rebuilt per job | Cached on a hash of the candidate id sequence | 10k docs: 2.06s -> 0.01s (**200x**) |
| Crash on inconsistent embedding dimensions | Lock dimension on first call + per-batch consistency assertion | Raises `EMBED_DIM_CHANGED` with a clear fix, instead of a numpy exception |
| ADAPTER parsed only 2 of 4 schemas | Recursive JSON tree walk + `looks_unparsed()` | 5/5 parsed; unknown schemas flagged as mismatches rather than reported as zero |

---

## Lessons from designing the mock (worth reading)

Three iterations of the dataset, and **the first two both scored a perfect 1.000**:

**Iteration 1** — 80 qualified candidates, only 10 slots. Breaking the funnel down
by candidate class revealed the real story:

```
hard_filter   80  {'QUALIFIED': 80}     <- all 40 HIDDEN_GEMs were filtered out
final_top_k   10  {'QUALIFIED': 10}     <- yet the score was 1.000
```

**The system had 0% recall on an entire class of qualified candidates while
precision@10 reported a perfect score.** When positives are abundant, that metric
has no resolution.

**Iteration 2** — positives cut to 6%, still 1.000. Distractors had 0-4 years of
experience while the compiler emitted `min_years=5`, so **the years filter became
a perfect oracle**.

**Iteration 3** — give distractors 5-14 years and every keyword (a technical
writer who documents distributed search ranking rather than building it). No
structured field separates them; only reading the text does. That finally
measured something:

```
A_L1  mean=1.000
B_L3  mean=0.300     job_001  1.000 -> 0.300  -0.700  <-- REGRESSION
```

L3 scored 0.7 below L1 because the dense channel pulled in every distractor
(`QUALIFIED` fell from 13 to 5 after `retrieve_dense`). Only then did the
regression-detection path execute for the first time.

> **Conclusion**: if the mock's ground truth is a deterministic function of fields
> your filter already checks, it will always hand you a perfect score.
> **A mock must encode judgment that lives outside the structured fields** — which
> in practice means either hand labels or an independent LLM grader.
> If you choose the latter, note that using one model as both judge and grader
> produces correlated errors: you will optimize toward the grader's biases.

---

## Interview preparation

**Q: How did you use the LLM?**

Three places, with clear division of labour:

1. **Query compiler** (JSON mode) — turns a natural-language JD into
   machine-executable predicates. The key is a conservative prompt: only mark
   something as a hard constraint if the JD states it as a requirement. Otherwise
   nice-to-haves become filters and recall collapses.
2. **Skill synonym generation** — produced in the same compile call, solving the
   exact-match recall problem at zero extra cost.
3. **Judge** (5 candidates per call) — per-criterion binary verdicts with evidence
   quotes, not a holistic score.

What the LLM is **not** used for: the final ordering decision (a deterministic
formula) and anything BM25 or a rule can do. LLM calls are the slowest, most
expensive and least reliable step — avoid them where possible.

**Q: How did you find problems?**

The funnel table. Two real examples: (1) recall dropped 27.8% at `fuse_rrf` while
`llm_rerank` lost nothing — the problem was retrieval width, not ranking quality,
and editing the judge prompt would have been entirely the wrong direction.
(2) Breaking the funnel down by candidate class revealed that the system was
missing an entire class of qualified candidates *while scoring a perfect 1.000* —
the aggregate score had completely hidden it.

**Q: What would you do with more time?**

(1) Fix the three known issues above. (2) Use the accumulated gold set as few-shot
negatives to attack judge false positives. (3) Use the LLM at index time to
normalize profiles into a standard skill taxonomy, moving normalization cost off
the query path. (4) Replace pointwise with listwise reranking to optimize top-10
order directly. (5) Infer `years_experience` from resume text rather than trusting
the field — internships and gaps make it subtle.
