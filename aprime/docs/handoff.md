# 交接说明：E4 候选选择实验

> **用途**：新开对话时把本文件贴给 AI，即可无缝接上当前进度。
> **最后更新**：2026-11-22（跨数据集任务刚开头）

---

## 0. 一句话现状

E4 章节（"LLM 能否替代覆盖贪心做候选选择"）已有 **4 组独立证据、6 个 commit**，全部指向
同一结论：**LLM 既无优势也无额外脆弱性，退化成不够精确的 `topk_cov`**。

已完成**跨数据集外部效度**（GeoLife 真实轨迹，sim + V4-Flash + E4.3 污染），与 T-Drive 同向；文档已补 §4.5。

---

## 1. 仓库与环境

| 项 | 值 |
|---|---|
| 仓库根 | `D:\composition` |
| 分支 | `cursor/course-paper-aprime-1d60` |
| 子项目 | `aprime/`（主实验）、`mapus/`（路线 / 城市属性）、`agentsense/`（实例与 `SETTINGS`） |
| 工作目录 | `D:\composition\aprime` |
| 解释器 | `py -3.14` |
| Shell | **Windows PowerShell**：分隔用 `;`，**不能用 `&&`** |

**跑 LLM 前必须先清环境变量**（shell 里有个 53 字符旧 key 会覆盖 `.env` 里有效的 51 字符 key）：

```powershell
Remove-Item Env:SILICONFLOW_API_KEY -ErrorAction SilentlyContinue
```

- API key：`D:\composition\agentsense\.env`
- provider 统一 `siliconflow`
- 三个模型：`deepseek-ai/DeepSeek-V4-Flash`、`Qwen/Qwen3.8-27B`、`deepseek-ai/DeepSeek-V3.2`
- 缓存 `results/llm_cache`（**不要提交**）；`results/_smoke/`、根目录 `_*.ps1` 均为未跟踪临时物

---

## 2. 核心代码

| 文件 | 作用 |
|---|---|
| `aprime/aprime/dense.py` | `make_dense_instance(dataset, scale, n_workers, seed, index, source=...)`，确定性 `random.Random(seed*7919+index)` |
| `aprime/aprime/candidates.py` | `build_candidates(inst, k, seed, lams, n_random)`；按 `frozenset(cells)` 去重后由 `_diverse` 封顶 k 条 |
| `aprime/aprime/llm_gen.py` | `MenuSelector`；菜单行 `[i] 覆盖 N 格 \| 满意度 x \| 步数 s`；`PROMPT_VERSION="menu-select-v2"` |
| `aprime/experiments/run_llm_candidates.py` | E4.2 / E4.3 主实验，`--mode sim\|llm` |
| `aprime/experiments/compare_llm_theta.py` | 多档 θ 结果按 模型 × 规模 × θ 聚合 |
| `aprime/experiments/run_llm_pollution.py` | E4.4 菜单污染（真实 LLM vs 算法），含"被迫 / 净吸收"分解 |
| `aprime/docs/e4_llm_candidate_experiment.md` | 实验文档：§4.1–§4.4 结果、§5 定位、§8 局限、§10 文件表 |

---

## 3. 标准实验参数

`tdrive / small`、`n_workers=20`、`k_full=12`、`k=3`、`budget_steps=120`、
`theta=0.3`、`seeds=10`、`prompt-style=coverage`。

菜单实测去重后 **4–12 条（均值 10.36）**。

常用命令（sim，免费）：

```powershell
py -3.14 experiments/run_llm_candidates.py --dataset tdrive --scale small --n-workers 20 `
  --k-full 12 --k 3 --budget-steps 120 --theta 0.3 --seeds 10 --out-dir results
```

污染实验：

```powershell
py -3.14 experiments/run_llm_pollution.py --provider siliconflow `
  --model deepseek-ai/DeepSeek-V4-Flash --env-file D:\composition\agentsense\.env `
  --cache-dir results/llm_cache --eta-grid 0,0.5,0.75 --seeds 10 --n-workers 20 `
  --k-full 12 --k 3 --budget-steps 120 --theta 0.3 --prompt-style coverage --out-dir results
```

---

## 4. 已完成结果（均为 10 种子）

### §4.2 命中率 → 近似比（sim，small，θ=0.3）

| 生成器 | 命中率 | A′/OPT |
|---|---|---|
| `full` | 1.000 | 0.989 |
| `topk_cov` | 0.443 | **0.939** |
| `topk_sat` | 0.262 | 0.885 |
| `random` | 0.275 | 0.919 |
| `oracle_eps` 0.0 → 1.0 | 0.416 → 1.000 | 0.900 → 0.989 |

结论：`topk_cov` 近乎最优；命中率是近似比的单调驱动（定理 3 的界结构成立）。

### §4.3 θ 敏感性（A′/OPT）

| 模型 | θ=0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| sim small | 0.963 | 0.939 | 0.946 | 0.940 |
| sim medium | 0.967 | 0.974 | 0.976 | 0.973 |
| V4-Flash | 0.947 | 0.936 | 0.946 | 0.926 |
| Qwen3.8-27B | 0.936 | 0.931 | 0.940 | 0.938 |
| V3.2 | 0.931 | 0.932 | 0.935 | 0.938 |

结论：θ 升高不伤 LLM（候选池与 θ 无关）；退化**跨规模稳健**（medium 仅 −2.3pp vs small −5.6pp）。
V4-Flash θ=0.3 的 cov 59 / OPT 63，用量 183,104 token。

### §4.4 污染鲁棒性（A′/OPT，三模型）

| 污染 | η | V4-Flash | Qwen | V3.2 | `topk_cov` | `random` |
|---|---|---|---|---|---|---|
| random | 0.00 | 0.925 | 0.919 | 0.923 | 0.937 | 0.926 |
| random | 0.50 | 0.933 | 0.919 | 0.922 | 0.931 | 0.847 |
| random | 0.75 | 0.915 | 0.914 | 0.914 | 0.917 | 0.809 |
| adversarial | 0.50 | 0.900 | 0.895 | 0.898 | 0.901 | 0.814 |
| adversarial | 0.75 | 0.837 | 0.833 | 0.833 | 0.840 | 0.726 |

**净吸收**（吸收率 − 被迫率）：三模型 **0.000–0.025**；`random` 0.373–0.555、`topk_sat` 0.12–0.19。
污染主要靠"削减好候选"（η=0.75 adversarial 被迫率 0.472），而非欺骗模型。

用量：V4-Flash 405 次 / 178k token、Qwen 787 / 376k、V3.2 787 / 345k；失败 0、解析失败 0。

### §4.5 跨数据集（GeoLife，sim + V4-Flash，θ=0.3）

| 设置 | 生成器 | 命中率 | A′/OPT |
|---|---|---|---|
| sim | full | 1.000 | 0.973 |
| sim | topk_cov | 0.449 | 0.890 |
| sim | random | 0.356 | 0.868 |
| sim | topk_sat | 0.261 | 0.845 |
| V4-Flash llm | llm | 0.379 | 0.878 |

设置：8×8/8 时槽（与 T-Drive small 口径一致），--source geolife --min-cand 4 --n-workers 60 --seeds 10；
每种子丢弃候选 <4 条者，有效 314 人·种子、丢弃 286 人·种子。结论：与 T-Drive 同向，绝对值
略低（候选池更稀）。

**E4.3 污染（GeoLife，V4-Flash，A′/OPT）**：

| 污染 | η | `llm` | `topk_cov` | `topk_sat` | `random` | `llm` 净吸收 |
|---|---|---|---|---|---|---|
| random | 0.50 | 0.885 | 0.902 | 0.850 | 0.795 | 0.018 |
| random | 0.75 | 0.844 | 0.844 | 0.815 | 0.721 | 0.000 |
| adversarial | 0.50 | 0.850 | 0.850 | 0.800 | 0.754 | 0.016 |
| adversarial | 0.75 | 0.725 | 0.725 | 0.690 | 0.628 | 0.000 |

`llm` 与 `topk_cov` 全 η 下几乎重合，净吸收 0.000–0.018，与 T-Drive（0.000–0.025）一致；
`random` 净吸收 0.253–0.455、`topk_sat` 0.092–0.196，与 T-Drive 同向。
用量：1195 实发调用（689 缓存命中）、740,682 token（实发 474,513）、0 失败、38 补足。

---

## 5. 两个必须知道的方法学坑

1. **菜单小 ⇒ 必须做"被迫 vs 净吸收"分解。**
   small 菜单最少 4 条、`k=3`，η=0.75 会逼到"只剩 1 条好候选"，此时选诱饵是**被迫**。
   只看吸收率（0.465）会误判成"LLM 被骗近一半"。

2. **控制台"每人候选 12–12"是 medium 的，不是 small。**
   small 一直是 4–12。当前代码**完全可复现**已提交的 sim small 结果
   （`cov/opt/ratio/hit_rate/n_selected` 逐位吻合），不要因为这个再怀疑复现性。

---

## 6. 提交历史

```
7f5b123  E4.3 污染鲁棒性扩到 Qwen + V3.2，三模型一致（净吸收≈0）
730e441  E4.3 污染鲁棒性真实 LLM 版（V4-Flash）
6b3e353  medium 规模 θ 敏感性复跑（跨规模）
221a459  跨模型 θ 扫描（Qwen、V3.2）
b2bc196  V4-Flash θ 扫描
38344d9  θ 敏感性能力 + sim small 四档
```

---

## 7. 当前任务：跨数据集外部效度（已完成，含污染）

GeoLife 外部效度已跑完并写进文档：

- sim：8×8 栅格（与 T-Drive small 口径完全一致），`--source geolife --min-cand 4 --n-workers 60 --seeds 10`；
  有效 314 人·种子，菜单 4–12 条，topk_cov 0.890 / random 0.868 / topk_sat 0.845，oracle_eps 单调。
- V4-Flash `--mode llm`：314 次调用、123,482 token、0 失败、命中率 0.379 / 近似比 0.878；
  介于 random 与 topk_cov 之间，与 T-Drive 同向。
- V4-Flash E4.3 污染：`run_llm_pollution.py --source geolife --min-cand 4 --n-workers 60`；
  1195 实发调用 / 740,682 token / 0 失败；`llm` 与 `topk_cov` 全 η 重合、净吸收 0.000–0.018，
  与 T-Drive 同向。为此给 `run_llm_pollution.py` 加了 `--source` / `--min-cand`。
- 文档 `docs/e4_llm_candidate_experiment.md` §4.5 已含污染结果，并更新 §6 用量表、
  §8 限制（两个数据集都跑了污染）、§10 文件表、§7.8 过滤必须报告丢弃人数。

下一步：提交本次更改（`run_llm_pollution.py` 的 `--source`/`--min-cand`，污染结果文件，文档）。

## 8. 不要做的事

- 不要提交 `results/llm_cache/`、`results/_smoke/`、根目录 `_*.ps1`；
- 不要用 `&&` 连命令（PowerShell）；
- 不要在未清 `SILICONFLOW_API_KEY` 时跑 LLM（会 401 或串 key）；
- 不要凭控制台"每人候选 N–M"判断可复现性——真正口径看 JSON 的
  `cov / opt / ratio / hit_rate / n_selected`；
- 不要用 gMission 跑 E4（无轨迹）。

---

## 9. 下次开始时的第一条命令

仓库已干净，可提交。如需重跑 GeoLife sim（免费，约 20 秒）：

```powershell
cd D:\composition\aprime
py -3.14 experiments/run_llm_candidates.py --dataset tdrive --scale small --source geolife --min-cand 4 --n-workers 60 --k-full 12 --k 3 --budget-steps 120 --theta 0.3 --seeds 10 --out-dir results
```

如需重跑 V4-Flash llm（约 0.3 元、约 300 次调用）：先执行
`Remove-Item Env:SILICONFLOW_API_KEY -ErrorAction SilentlyContinue`，再加
`--mode llm --provider siliconflow --model deepseek-ai/DeepSeek-V4-Flash --env-file D:\composition\agentsense\.env`。
