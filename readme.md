# Candidate Search — 75 分钟 E2E 作战计划

给定职位描述（title / description / hard criteria），从候选人库里检索并排序，返回 Top-10。

> 这不是"完美终态设计文档"，是**作战计划**。核心原则：**先有分数，再有架构。**
> 第 10 分钟就要有一个能提交的 baseline。

---

## 快速开始（完全离线，不需要任何 API key）

```bash
pip install numpy                                  # 唯一的第三方依赖

python mock_data.py --scarce                       # 生成数据 + 独立标注
python mock_eval_server.py --port 8000 &           # 起本地 eval endpoint
python harness.py run --level 3 --mock-llm \
       --data-dir mock_data --eval-url http://127.0.0.1:8000

python harness.py compare                          # 对比历次 run
python harness.py export                           # 产出 evaluation_results.json
```

接真实 LLM 时，**第一件事**是 selftest —— 两发最小请求验通 key / 模型名 / 维度，
在烧 quota 之前把配置问题挡掉：

```bash
cp .env.example .env && vi .env          # 填 GEMINI_API_KEY
python harness.py run --level 3 --selftest
# {"chat": {"model": "gemini-3.7-flash", "ok": true, ...},
#  "embed": {"model": "gemini-embedding-001", "ok": true, "returned": 3, "dim": 768}}
```

---

## 文件地图

| 文件 | 行数 | 职责 |
|---|---:|---|
| `engine.py` | ~670 | 数据模型、LLM 封装、BM25、RRF、分级 pipeline |
| `llm_clients.py` | ~200 | Gemini 真实客户端（REST，零依赖）+ selftest |
| `harness.py` | ~500 | eval 客户端 + ADAPTER、实验运行器、run 对比、CLI |
| `diagnostics.py` | ~200 | 漏斗 trace、gold set 累积、分阶段 recall |
| `mock_eval_server.py` | ~200 | 本地 eval endpoint（真 HTTP，可切 schema / 注故障） |
| `mock_data.py` | ~150 | 有难度的合成数据集 + 独立标注 |
| `cache.py` | ~110 | sqlite 持久化缓存 + 命中率统计 |
| `settings.py` | ~150 | `.env` 加载（零依赖）+ 配置校验 |

依赖：`numpy` + 标准库。BM25 是纯 Python，无需 `rank_bm25`；`.env` 解析器自带，无需 `python-dotenv`；mock server 用 `http.server`，无需 Flask。

**面试环境里 `pip install` 失败是真会发生的事**，少一个依赖少一个风险点。

---

## 架构

```
Input: Job Description (title, description, hard_criteria[])
   │
   ├─ Phase 1  Query Compilation (LLM-as-Compiler, JSON mode)
   │     └─ min_years / required_skills / skill_synonyms /
   │        semantic_query / keyword_query / checkable_criteria[]
   │
   ├─ Phase 2  Retrieval
   │     ├─ 2a  Graded Hard Filter（分级放松）
   │     ├─ 2b  BM25 通道   ← 专有名词、公司名、证书、框架版本
   │     ├─ 2c  Dense 通道  ← 语义泛化、同义表述
   │     └─ 2d  RRF 融合 → Top-30
   │
   ├─ Phase 3  Rerank (LLM-as-Judge, 批量 5 人/次)
   │     └─ 逐条 criteria 二元判定 + evidence 引用 + soft_score
   │
   └─ Phase 4  排序 & 提交
         final = hard_passed×1000 + soft_score + retrieval_score×0.01
         字典序：先比硬条件命中数，再比软性 fit，最后用检索分打破平局
```

### 三个关键决策

**① 为什么不用 holistic 0–100 打分？**
单次 holistic 打分会把候选人挤在 82–88 区间，**第 10 名和第 15 名区分不开**——而 Top-10 的得分差异全在这个边界。改成逐条二元判定后：排序有天然分层；`evidence` 字段让你能分辨"LLM 判错了"还是"数据里本来没有"；目标函数和 eval 直接对齐。

**② 为什么保留 BM25 通道？**
硬条件里的高信号词——公司名、学校名、certification、`PyTorch 2.x`——恰恰是 embedding 最不擅长、BM25 最擅长的。纯 dense 的"hybrid"名不副实。

**③ 为什么用 RRF 而不是分数加权？**
BM25 分数无上界，cosine 在 [-1,1]，量纲不可比，加权需要 per-query 调参。RRF 只用排名，零调参。

---

## 配置：`.env`

```bash
cp .env.example .env      # 填 key
python settings.py        # 确认读到了（key 脱敏显示）
```

> ⚠️ **如果这个仓库是 public**：`.env` 一旦 commit，key 就等于公开了，删文件也没用（git 历史还在）。
> `.gitignore` 必须先于 `.env` 存在——已验证它会挡住 `.env` / `cache.sqlite3` / `goldset.json` / `runs/`。
> 面试结束后建议把仓库改回 private。

优先级是 **`.env` < 真实环境变量**，不是反过来——已经 `export` 过的值不会被 `.env` 悄悄改掉。

`settings.py` 会告诉你当前能跑到第几级，并在**烧 quota 之前** fail fast：

```
  chat_key    : AIzaSy…cdef (36 chars)
  embed_key   : AIzaSy…cdef (36 chars)
  -> 可运行到 --level 3
```

### Provider 选择

| | Chat | Embedding | Key 数 | 备注 |
|---|---|---|---:|---|
| **Gemini**（推荐） | `gemini-3.7-flash` | `gemini-embedding-001` | **1** | 一个 key 覆盖两者 |
| Anthropic + Voyage | Claude | `voyage-3.5` | 2 | Anthropic 不提供 embedding 模型 |
| OpenAI | GPT | `text-embedding-3-small` | 1 | — |

`llm_clients.py` 走 REST（urllib）而非 `google-genai` SDK —— 零新增依赖，
且可以对着本地假服务器完整测试。设 `GEMINI_BASE_URL` 可指向代理或测试服务器。

Gemini 的坑（`settings.py` 做成启动告警，`llm_clients.py` 做成运行时断言）：

- **`gemini-embedding-2` 对 list 输入返回单个聚合向量**，不是每条一个。`_raw_embed` 期望等长返回，直接用会**静默错位**且不报错。
  → 已用 `batchEmbedContents` 端点天然规避，并加了 `EMBED_COUNT_MISMATCH` 断言兜底。
- **非 3072 维的 embedding 不是预归一化的**，须自行 normalize（`engine.retrieve()` 已处理）。
- 新版 Gemini 的 `temperature` / `top_p` / `top_k` 已废弃，别再传。

---

## Evaluation 闭环

```
跑 pipeline → 提交 eval endpoint → 解析逐候选人分数 → 吸收进 gold set
     ↑                                                      │
     │                                                      ↓
  改配置 ← 看 per-job delta ← 落盘 runs/<run_id>/ ← 打印漏斗诊断
```

每次 run 落盘三个文件，**任何时刻被打断都有可提交的东西**：

```
runs/<run_id>/
  ├── summary.json    配置 + 均值/中位数/最低分 + LLM 用量 + 缓存命中率
  ├── results.jsonl   每题的 top10 / 分数 / 逐候选人打分明细
  └── traces.jsonl    每题的完整漏斗
```

### 本地 mock endpoint

`mock_eval_server.py` 是**真的 HTTP 服务**，不是 mock 一个类——所以 `_build_payload`、urllib、重试退避、HTTP 错误分支全都会被真正执行。

它还能演练当天最大的未知数：**真实 endpoint 的响应 schema 长什么样**。

```bash
python mock_eval_server.py --schema nested     # 嵌套 + 字段名不同（id/relevant）
python mock_eval_server.py --schema minimal    # 只给总分，不给逐候选人 → gold 攒不起来
python mock_eval_server.py --schema verbose    # 逐条 criteria 拆解
python mock_eval_server.py --fail-rate 0.3 --malformed-rate 0.1 --require-auth
```

练三遍"curl → 看响应 → 两分钟改完 ADAPTER"，当天就不会慌。

### ADAPTER

`harness.py` 的 `EvalClient` 里有一段 ADAPTER 区，是唯一需要按真实 endpoint 改的地方：

```python
def _build_payload(job_id, candidate_ids) -> Dict     # 请求体
def extract_overall_score(resp) -> float              # 总分
def extract_per_candidate(resp) -> Dict[str, float]   # ★ 逐候选人分数
```

第三个最重要——**它是 gold set 的唯一来源**，没有它就只能盲调超参。

当前实现用递归遍历 JSON 树，不假设字段在第几层，实测能解析 flat / nested / minimal / verbose / map 五种形态。另外 `looks_unparsed()` 区分**真的 0 分**和**schema 没对上**：两者都返回 0.0，但一个要改检索、一个要改 ADAPTER，搞混会浪费现场十几分钟。

---

## 分阶段召回诊断

> **正确候选人在 retrieval 阶段被丢了，reranker 再强也救不回来。**

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

**这张表直接回答"你是怎么发现问题的"。** 上例一眼看出瓶颈在 `fuse_rrf`，不在 reranker——该调 `retrieve_k`，不是 prompt。

### 迭代决策树

```
看 recall 列，找第一个 LEAK：
  hard_filter 漏  → relax 级别太低 / skill_synonyms 不够 /
                    compiler 把 preference 当成了 requirement → 改 compiler prompt
  fuse_rrf 漏     → retrieve_k 太小（30 → 50），或某通道质量差
  llm_rerank 漏   → 这才是 prompt 问题 → 看 checks[].evidence 是空的
                    （数据没有）还是判错的（prompt 没说清）
  final_top_k 漏  → 排序公式问题 → 是否大量候选人 hard_passed 相同
```

### Gold Set 从哪来

拿不到显式标注，但 eval endpoint 会告诉你"你提交的 10 个里哪几个对"。把历次被判为对的 id 累积到 `goldset.json`，得到**逐轮变厚的伪 ground truth**。

它有偏（只含你曾召回过的人），能回答"这轮为什么比上轮差"，**回答不了"我离天花板还有多远"**。前两轮 gold 还薄时，用 oracle ceiling 粗判：把 filter 关掉、`retrieve_k` 拉到 200 跑一次，分数明显变高说明瓶颈在 filter 太严。

---

## 分级放松的硬过滤

一步退到"完全不过滤"等于放弃硬条件信号。用阶梯，并把用到第几级记进 trace：

| 级别 | 年限 | 技能 | 地点 |
|---|---|---|---|
| `L0_strict` | 严格 | **全部**命中 | 检查 |
| `L1_drop_location` | 严格 | 全部命中 | 忽略 |
| `L2_any_skill` | 严格 | **任一**命中 | 忽略 |
| `L3_years_slack` | −2 年 | 任一命中 | 忽略 |
| `L4_semantic_only` | 关 | 关 | 关 |

日志里 `relax=L2_any_skill` 出现太频繁，说明 compiler 把 preference 误判成了 requirement——回去改 prompt 里那句 "Be conservative"。

---

## 缓存

75 分钟里你会跑十几轮。sqlite 落盘，**进程重启后仍然命中**；key 里带 `model` 和 `prompt_version`，**改了 prompt 自动 miss，不会拿到脏结果**。实测同配置复跑命中率 100%。

⚠️ embedding 存成 Python `list[float]` 而非 `np.float32`，1536 维下：

| 候选人数 | 内存 | sqlite 文件 |
|---:|---:|---:|
| 1,000 | 0.05 GB | 0.02 GB |
| 10,000 | 0.49 GB | 0.20 GB |
| 50,000 | **2.46 GB** | 1.00 GB |

5 万人以上建议改 `float32`（降至约 1/8），或用 Gemini 的 `output_dimensionality=768`。

---

## ⏱ 时间降级路线图

**最容易的死法：架构很漂亮，但跑不出结果，交不上 evaluation results。**

每一级都是**可独立提交的完整系统**（`--level 0..3`）。任何时刻卡住，上一级的结果都已落盘。

| 时间 | 目标 | 卡住了怎么办 |
|---|---|---|
| **0–10** | 读 schema，curl 打通 endpoint，改 ADAPTER，**提交 L0 BM25 baseline** | endpoint 不通就先 `--no-submit` 跑通 pipeline |
| **10–25** | 加 LLM 编译 + 分级硬过滤，提交 L1 | compiler 返回非法 JSON → 已有三层兜底 + fallback |
| **25–40** | 加 dense + RRF，提交 L2 | embedding 太慢 → 降 `retrieve_k`，或停在 L1 |
| **40–58** | 加 LLM 逐条判定重排，提交 L3 | 超时 → 调大 `--batch-size` |
| **58–70** | **看漏斗迭代**：哪层 LEAK 调哪层 | 见上方决策树 |
| **70–75** | `export` 导出，准备讲述 | — |

**硬性纪律**

- 第 15 分钟还没有任何提交过的分数 → 立刻跑 `--level 0`。
- 第 50 分钟还在调 L3 → 放弃 L3，回到 L2 稳住，剩余时间全用在诊断和讲述上。

一个能跑通并且你讲得清 debug 过程的 L2，远好过一个跑不完的 L3。

---

## 健壮性清单

| 故障 | 处理 | 位置 |
|---|---|---|
| LLM 输出带 fence / 前后废话 | 三层解析：直接 parse → 剥 fence → 平衡括号扫描 | `parse_json_loose` |
| LLM 调用失败 | 指数退避重试 3 次；**最终失败退回检索分，不给 0 分** | `chat_json` |
| ↑ 为什么重要 | 给 0 分会**静默丢掉好候选人**，且在漏斗里表现为"rerank 层泄漏"，误导你去改 prompt | — |
| batch 里漏返回某候选人 | 按 `candidate_id` 对齐而非位置，缺失单独兜底并记 warning | `_score_batch` |
| 硬过滤清空候选池 | 分级放松，非一刀切 | `hard_filter` |
| dense 通道挂了 | 降级为纯 lexical，记 warning，不中断 | `retrieve` |
| eval endpoint 4xx | 不重试（八成是 payload schema 不对） | `_submit_sync` |
| 中途 Ctrl-C | 边跑边打印 + 每题跑完即写盘 | `ExperimentRunner.run` |

---

## ⚠️ 已知问题（实测发现，尚未修复）

按优先级排列。这些是真实测出来的，不是理论风险。

| # | 问题 | 影响 | 修复成本 |
|---|---|---|---|
| 1 | **漏斗把并行检索通道当串行处理** | BM25 漏掉但被 dense 捞回的候选人，会报**假 LEAK**（实测 50%）和无意义的负泄漏。诊断工具主动误导你 | 中 |
| 2 | `parse_json_loose` 接不住截断 / 单引号 / 尾逗号 | 批量打分撞 `max_tokens` 时整个 batch 降级（不崩，但那批白跑） | 低 |
| 3 | 非 Gemini provider 的客户端未实现 | `.env` 设 anthropic / openai 会 fail fast 并提示 | 低，照抄 40 行 |

fence、前后废话、字符串内含花括号这三种已经接住了。

### 已修复（实测验证）

| 问题 | 修法 | 实测效果 |
|---|---|---|
| 真实 LLM 客户端缺失 | `llm_clients.py` REST 实现 + `--selftest` | 七种故障模式（429/404/MAX_TOKENS/SAFETY/聚合/维度不一致/正常）全部正确处理 |
| 技能匹配子串假阳性 | 词边界正则，右边界兼容 `c++` / `c#` | `"go"` 不再命中 `google`；`Go` / `C++` / `R` 正确命中 |
| BM25 每题重建索引 | 按候选人 id 序列 hash 缓存 | 10k 文档：2.06s → 0.01s（**200×**） |
| embedding 维度不一致崩溃 | 首次调用锁定维度 + 批内一致性断言 | 报 `EMBED_DIM_CHANGED` 并提示清缓存，不再抛 numpy 异常 |
| ADAPTER 只认 2/4 种 schema | 递归遍历 JSON 树 + `looks_unparsed()` | 5/5 解析；陌生 schema 标为"没对上"而非 0 分 |

---

## 设计 mock 时踩过的坑（值得读）

我迭代了三版数据集，**前两版都是满分 1.000**：

**第一版**——合格者 80 人，top-10 只要 10 个。按类别拆解漏斗才发现：

```
hard_filter   80  {'QUALIFIED': 80}     ← 40 个 HIDDEN_GEM 全被滤掉
final_top_k   10  {'QUALIFIED': 10}     ← 但分数 1.000
```

**系统在整整一类合格候选人上召回率为 0，precision@10 却给满分。** 正样本充足时这个指标没有区分度。

**第二版**——正样本压到 6%，还是 1.000。因为 distractor 年限是 0–4 年，而 compiler 输出 `min_years=5`，**年限过滤成了完美 oracle**。

**第三版**——让 distractor 也有 5–14 年、关键词也齐全（技术文档工程师：写的是 distributed search ranking 的*文档*）。结构化字段无法区分，只能靠读懂语义。终于测出东西：

```
A_L1  mean=1.000
B_L3  mean=0.300     job_001  1.000 → 0.300  -0.700  <-- REGRESSION
```

L3 比 L1 差 0.7——dense 通道把 distractor 全捞上来了（`retrieve_dense` 后 QUALIFIED 从 13 掉到 5）。回归检测路径这才第一次真正跑通。

> **结论**：如果 mock 的 ground truth 是"你的过滤器已经在检查的字段"的确定性函数，它永远给你满分。
> **mock 必须编码结构化字段里没有的判断**——要么手工标注，要么用独立的 LLM 当裁判。
> 后者要注意：同一个模型既当裁判又当 judge 会产生相关误差，你会优化到裁判的偏见上。

---

## 面试问答准备

**Q: 你是怎么使用 LLM 的？**

三个位置，分工明确：

1. **Query Compiler**（JSON mode）——把自然语言 JD 编译成机器可执行的谓词。关键是 prompt 里要求 conservative："只有 JD 明说是 requirement 才标为硬条件"，否则 nice-to-have 也变成过滤条件，召回崩塌。
2. **Skill Synonym 生成**——同一次 compile 调用里顺带产出，解决精确匹配的召回问题，零额外成本。
3. **Judge**（批量 5 人/次）——逐条 criteria 二元判定 + evidence 引用，而非 holistic 打分。

**没有**用 LLM 做的：最终排序决策（用确定性公式）、任何 BM25 或规则能解决的事。LLM 是最慢最贵最不稳定的环节，能不用就不用。

**Q: 遇到问题是怎么发现的？**

漏斗表。举两个真实例子：① `fuse_rrf` 那层 recall 掉 27.8% 而 `llm_rerank` 一点没掉——问题在检索宽度，不在排序质量，当时若去改 judge prompt 方向就全错了。② 按候选人类别拆解漏斗，发现系统在满分情况下漏掉了整整一类合格者——总分完全掩盖了这个问题。

**Q: 更多时间会做什么？**

① 修上面那 6 个已知问题；② 用累积的 gold set 做 few-shot 负例，专治 judge 假阳性；③ index 阶段用 LLM 把 profile 结构化成标准 skill taxonomy，把归一化成本从 query time 挪到 index time；④ listwise rerank 替代 pointwise，直接优化 top-10 相对顺序；⑤ `years_experience` 从简历文本推断而非信任字段（实习/gap 怎么算是个坑）。
