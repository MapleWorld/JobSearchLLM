# Candidate Search — 75 分钟 E2E 作战计划

> 这不是一份"完美终态设计文档",而是一份**作战计划**。
> 核心原则：**先有分数,再有架构。** 第 10 分钟就要有一个能提交的 baseline。

---

## 0. TL;DR — 开场三条命令

```bash
# 1. 离线自测,确认整条链路能跑（不消耗任何 API quota）
python harness.py run --mock --level 0

# 2. 接上真实 endpoint,先交 baseline 拿第一个分数 + 攒 gold set
python harness.py run --level 0 --run-id L0_baseline

# 3. 逐级加码,每一级都提交
python harness.py run --level 3 --run-id L3_v1
python harness.py compare          # 看 per-job delta,抓回退
python harness.py export           # 产出 evaluation_results.json
```

---

## 1. 系统架构

```
Input: Job Description (title, description, hard_criteria[])
   │
   ├─ Phase 1  Query Compilation (LLM-as-Compiler, temp=0, JSON mode)
   │     └─ 输出：min_years / required_skills / skill_synonyms /
   │              semantic_query / keyword_query / checkable_criteria[]
   │
   ├─ Phase 2  Retrieval
   │     ├─ 2a  Graded Hard Filter（分级放松,见 §3）
   │     ├─ 2b  BM25 通道   ← 专有名词、公司名、证书、框架版本
   │     ├─ 2c  Dense 通道  ← 语义泛化、同义表述
   │     └─ 2d  RRF 融合 → Top-30
   │
   ├─ Phase 3  Rerank (LLM-as-Judge, 批量 5 人/次)
   │     └─ 逐条 criteria 二元判定 + evidence 引用 + soft_score
   │
   └─ Phase 4  排序 & 提交
         final_score = hard_passed×1000 + soft_score + retrieval_score×0.01
         → 字典序：先比硬条件命中数,再比软性 fit,最后用检索分打破平局
```

### 三个关键设计决策（面试官一定会问）

**① 为什么打分不用 holistic 0-100？**

单次 holistic 打分会把大量候选人挤在 82-88 区间,**第 10 名和第 15 名区分不开**——而 Top-10 任务的得分差异恰恰全在这个边界上。改成"逐条硬条件二元判定 + evidence"之后：

- 排序有了天然的粗粒度分层（通过 3 条 > 通过 2 条）,边界清晰
- `evidence` 字段让 debug 时能立刻分辨：**是 LLM 判错了,还是数据里本来就没有这个信息**
- 如果 eval 是按"top10 里几个满足硬条件"算分,这个目标函数和 eval 直接对齐

**② 为什么坚持保留 BM25 通道？**

硬性条件里的高信号词——公司名、学校名、certification、`PyTorch 2.x`——恰恰是 embedding 最不擅长、BM25 最擅长的。纯 dense 的"hybrid"名不副实。RRF 融合不需要调量纲权重,比加权求和稳。

**③ 为什么用 RRF 而不是分数加权？**

BM25 分数无上界,cosine 在 [-1,1],两者量纲不可比,加权求和的权重需要 per-query 调。RRF 只用排名,零调参。

---

## 2. Evaluation 闭环（`harness.py`）

题目要求"根据测试结果不断迭代,最终提交所有 Evaluation Results"。闭环长这样：

```
跑 pipeline → 提交 eval endpoint → 解析每候选人分数 → 吸收进 gold set
     ↑                                                      │
     │                                                      ↓
  改配置 ← 看 per-job delta ← 落盘 runs/<run_id>/ ← 打印漏斗诊断
```

每次 run 落盘四个文件,**保证任何时刻被打断都有可提交的东西**：

```
runs/<run_id>/
  ├── summary.json    配置 + 均值/中位数/最低分 + LLM 用量 + 缓存命中率
  ├── results.jsonl   每题的 top10 / 分数 / 逐候选人打分明细
  └── traces.jsonl    每题的完整漏斗 trace
```

### ⚠️ 开场 5 分钟唯一必须手改的地方

`harness.py` 的 `EvalClient` 里有一段 **ADAPTER 区**（三个函数）,需要按真实 endpoint 的 schema 对齐：

```python
def _build_payload(self, job_id, candidate_ids) -> Dict     # 请求体长什么样
def extract_overall_score(resp) -> float                    # 总分在哪个字段
def extract_per_candidate(resp) -> Dict[str, float]         # ★ 逐候选人分数在哪
```

第三个最重要。**`extract_per_candidate` 是 gold set 的唯一来源**——没有它就只能盲调超参,有了它才能做分阶段召回诊断。拿到 endpoint 后第一件事是 `curl` 打一发,把响应贴出来对着改这三个函数,其余代码一行不用动。

默认实现已经兼容了常见的几种 schema（`results` / `candidates` / `graded` 数组,`score` / `pass` / `relevant` 字段,bool 自动转 float）,大概率能直接用。

---

## 3. 分阶段召回诊断（`diagnostics.py`）

> **如果正确候选人在 retrieval 阶段就被丢了,reranker 再强也救不回来。**

这是整个项目最值钱的部分。每次搜索记录完整漏斗,配合 gold set 算出**每一层的 recall**：

```
[FUNNEL] job=job_002  total=56.0ms  relax=L0_strict
  stage                     in   out    surv      ms   recall   lost
  ------------------------------------------------------------------
  pool                     300   300 100.0%       0  100.0%   0.0%
  hard_filter              300    75  25.0%       1  100.0%   0.0%
  retrieve_bm25             75    75 100.0%       1  100.0%   0.0%
  retrieve_dense            75    75 100.0%       1  100.0%   0.0%
  fuse_rrf                  75    30  40.0%       0   72.2%  27.8%  <-- LEAK
  llm_rerank                30    30 100.0%      28   72.2%   0.0%
  final_top_k               30    10  33.3%       0   55.6%  16.7%  <-- LEAK
    ! fuse_rrf 丢失 gold 5 个: ['cand_048', 'cand_052', 'cand_056', ...]
    ! final_top_k 丢失 gold 3 个: ['cand_036', 'cand_040', 'cand_060']
```

**这张表直接回答"你是怎么发现问题的"。** 上面这个例子一眼看出：瓶颈在 `fuse_rrf`（漏了 27.8%）,不在 reranker——所以该调的是 `retrieve_k`,不是 prompt。

### Gold Set 从哪来？

这类题通常没有显式标注,但 eval endpoint 会告诉你"你提交的 10 个里哪几个对"。把历次跑分中被判为对的 `candidate_id` **累积**起来（`goldset.json`）,就得到一个**逐轮变厚的伪 ground truth**。

它天然有偏（只包含你曾经召回过的人）,但足以回答那个最关键的问题：**上一轮找到的好候选人,这一轮为什么没了？** 这正是 `<-- LEAK` 标记在做的事。

前两轮 gold 还很薄的时候,用 **oracle ceiling** 粗判：把 filter 关掉、`retrieve_k` 拉到 200 跑一次,如果分数明显变高,说明瓶颈在 filter 太严,不在 rerank。

---

## 4. 分级放松的硬过滤

一步退到"完全不过滤"等于放弃硬条件信号。用阶梯,并把用到第几级记进 trace：

| 级别 | 年限 | 技能 | 地点 |
|---|---|---|---|
| `L0_strict` | 严格 | **全部**命中 | 检查 |
| `L1_drop_location` | 严格 | 全部命中 | 忽略 |
| `L2_any_skill` | 严格 | **任一**命中 | 忽略 |
| `L3_years_slack` | −2 年 | 任一命中 | 忽略 |
| `L4_semantic_only` | 关 | 关 | 关 |

从上往下退,直到池子 ≥ `min_pool_after_filter`(25)。技能匹配走 `skill_synonyms` 归一化（`"PyTorch"` → `["pytorch","torch"]`）,因为**小写精确串匹配是这类题的第一大 recall 杀手**。

日志里 `relax=L2_any_skill` 出现得太频繁,就说明 compiler 把 preference 误判成了 requirement——回去改 compiler prompt 里那句 "Be conservative"。

---

## 5. 缓存（`cache.py`）

75 分钟里你会跑十几轮。没有缓存,时间和 quota 全烧在重复调用上。

- sqlite 落盘,**进程重启后仍然命中**
- key = `sha256(namespace + payload)`,payload 里带 `model` 和 `prompt_version`——**改了 prompt 自动 miss,不会拿到脏结果**
- embedding 逐条缓存 + 批量请求（`BATCH=64`）,绝不在循环里 `await` 单条
- 统计 hit rate,面试时可以直接报数

实测：同配置复跑一轮,缓存命中 100%,wall clock 从 0.3s → 0.0s。真实 API 下这个差距是分钟级的。

```bash
python harness.py run --level 3 --run-id L3_rerun
# RUN L3_rerun: mean=1.000 wall=0.0s cache_hit=100%
```

---

## 6. ⏱ 时间降级路线图

**最容易的死法：架构设计得很漂亮,但跑不出结果,交不上 evaluation results。**

每一级都是**可独立提交的完整系统**（`--level 0..3`）。任何时刻卡住,上一级的结果都已经落盘了。

| 时间 | 目标 | 命令 | 卡住了怎么办 |
|---|---|---|---|
| **0–10** | 读数据 schema,`curl` 打通 eval endpoint,改 ADAPTER 三函数,**提交 L0 BM25 baseline** | `run --level 0` | endpoint 打不通就先 `--no-submit` 把 pipeline 跑通 |
| **10–25** | 加 LLM 编译 + 分级硬过滤,提交 L1 | `run --level 1` | compiler 返回不合法 JSON → 已有三层兜底解析 + fallback,不会崩 |
| **25–40** | 加 dense + RRF,提交 L2 | `run --level 2` | embedding 太慢 → 降 `retrieve_k`,或直接停在 L1 |
| **40–58** | 加 LLM 逐条判定重排,提交 L3 | `run --level 3` | 超时 → 调大 `--batch-size`（一次 prompt 塞更多人） |
| **58–70** | **看漏斗迭代**：哪一层 LEAK 就调哪一层 | `compare` | 见下方决策树 |
| **70–75** | 导出所有结果,准备讲述 | `export` | — |

### 迭代决策树（58–70 分钟这段最关键）

```
看 render_funnel 的 recall 列,找第一个 LEAK：

  hard_filter 漏  → relax 级别太低 / skill_synonyms 不够 / compiler 把
                    preference 当成了 requirement → 改 compiler prompt
  fuse_rrf 漏     → retrieve_k 太小 → 30 → 50；或某个通道质量差,
                    单独跑 --level 1（纯 BM25）对比
  llm_rerank 漏   → 这才是 prompt 问题 → 看 checks[].evidence 是空的
                    （数据没有）还是判错的（prompt 没说清）
  final_top_k 漏  → 排序公式问题 → 检查是不是大量候选人 hard_passed 相同
```

### 硬性时间纪律

- **第 15 分钟还没有任何一个提交过的分数 → 立刻停下来跑 `--level 0`。**
- **第 50 分钟还在调 L3 → 放弃 L3,回到 L2 稳住,把剩余时间全部用在诊断和讲述上。**

一个能跑通并且你能讲清楚 debug 过程的 L2,远好过一个跑不完的 L3。

---

## 7. 健壮性清单（每一条都对应一个真实故障）

| 故障 | 处理 | 位置 |
|---|---|---|
| LLM 输出带 markdown fence / 前后废话 | 三层解析：直接 parse → 剥 fence → 平衡括号扫描 | `engine.parse_json_loose` |
| LLM 调用失败 | 指数退避重试 3 次；**最终失败退回检索分,不给 0 分** | `LLMClient.chat_json` |
| ← 为什么这条重要 | 给 0 分会**静默丢掉好候选人**,而且在漏斗里表现为"rerank 层泄漏",误导你去改 prompt | — |
| batch 里 LLM 漏返回某个候选人 | 按 `candidate_id` 对齐而非位置,缺失的单独兜底并记 warning | `_score_batch` |
| 硬过滤清空候选池 | 分级放松,非一刀切 | `hard_filter` |
| dense 通道挂了 | 降级为纯 lexical,记 warning,不中断 | `retrieve` |
| eval endpoint 4xx | 不重试（八成是 payload schema 不对）,直接报错让你去改 ADAPTER | `_submit_sync` |
| 中途 Ctrl-C | 边跑边打印 + 每题跑完即写盘 | `ExperimentRunner.run` |

---

## 8. 面试问答准备

**Q: 你是怎么使用 LLM 的？**

三个位置,各有明确分工：

1. **Query Compiler**（temp=0, JSON mode）——把自然语言 JD 编译成机器可执行的谓词。关键是 prompt 里明确要求 "conservative：只有 JD 明说是 requirement 才标为硬条件",否则会把 nice-to-have 也变成过滤条件,导致召回崩塌。
2. **Skill Synonym 生成**——同一次 compile 调用里顺带产出,解决字符串精确匹配的召回问题。这是零额外成本的收益。
3. **Judge**（批量 5 人/次）——逐条 criteria 二元判定 + evidence 引用,而非 holistic 打分。

**没有**用 LLM 做的：候选人排序的最终决策（用确定性公式）、以及任何可以用 BM25/规则解决的事。LLM 调用是最慢最贵最不稳定的环节,能不用就不用。

**Q: 遇到问题是怎么发现的？**

`render_funnel` 那张表。举例：我发现 `fuse_rrf` 那一层 recall 掉了 27.8%,而 `llm_rerank` 一点没掉——说明问题在检索宽度不够,不在排序质量。当时如果直接去改 judge prompt,方向就完全错了。

**Q: 如果给你更多时间会做什么？**

按优先级：① 用累积的 gold set 做 few-shot 负例,专治 judge 的假阳性；② 在 index 阶段用 LLM 把候选人 profile 结构化成标准 skill taxonomy,把归一化成本从 query time 挪到 index time；③ listwise rerank 替代 pointwise,直接优化 top-10 的相对顺序；④ `years_experience` 从简历文本推断而非信任字段（实习/gap 怎么算是个坑）。

---

## 9. 文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| `cache.py` | ~110 | sqlite 持久化缓存 + 命中率统计 |
| `diagnostics.py` | ~180 | 漏斗 trace、gold set 累积、recall 诊断 |
| `engine.py` | ~470 | 数据模型、LLM 封装、BM25、RRF、分级 pipeline |
| `harness.py` | ~400 | eval 客户端、实验运行器、run 对比、CLI、mock |

依赖：`numpy` + 标准库。BM25 是纯 Python 实现,无需 `rank_bm25`。

**接真实数据只需改两处**：
1. `harness.py` 的 `EvalClient` ADAPTER 三函数
2. 继承 `LLMClient`,实现 `_raw_chat` 和 `_raw_embed`（照抄 `MockLLM` 的结构）