# 交接说明：E4 候选选择实验

> **用途**：新开对话时把本文件贴给 AI，即可无缝接上当前进度。
> **最后更新**：2026-11-23（跨数据集 + 跨规模全部完成）

---

## 0. 一句话现状

E4 章节（"LLM 能否替代覆盖贪心做候选选择"）已有 **4 组独立证据**，全部指向
同一结论：**LLM 既无优势也无额外脆弱性，退化成不够精确的 `topk_cov`**。

已完成 **跨数据集 × 跨模型 × 跨规模**：
- E4.2 真实 LLM 候选生成：T-Drive small + medium，三模型均完成；medium 命中率 0.704–0.800，仍低于 `topk_cov` 0.876，但差距比 small 收窄。
- E4.3 污染鲁棒性：T-Drive small + medium，三模型均完成；GeoLife small 三模型均完成。
- E4.2 候选生成：T-Drive small / medium / GeoLife small **三个设置 × 三模型全部完成**。
- 污染下三模型始终与 `topk_cov` 同步退化，medium 上极差 ≤0.010；污染是实例级削减好候选，不是模型级欺骗。

**论文主张边界**：α(θ) 近似比论证已撤回（见 `paper_sec6_experiments.md` §6.8.2）；不要将它作为结论。

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

`tdrive / medium`：其余相同，`budget_steps=240`、16×16 栅格、**每人候选固定 12 条**。

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
| random | 0.75 | 0.922 | 0.920 | 0.919 | 0.923 | 0.806 |
| adversarial | 0.50 | 0.887 | 0.880 | 0.870 | 0.895 | 0.829 |
| adversarial | 0.75 | 0.829 | 0.828 | 0.817 | 0.829 | 0.739 |

**净吸收**（吸收率 − 被迫率）：三模型 **0.000–0.027**；`random` 0.385–0.565、`topk_sat` 0.117–0.163。
污染主要靠"削减好候选"（η=0.75 adversarial 被迫率 0.450），而非欺骗模型。

用量（修复后重跑）：V4-Flash 实发 220 / 524,664 token、Qwen 220 / 571,486、V3.2 220 / 524,314；失败 0。
补足率：V4 1.0%、Qwen 0.25%、**V3.2 6.75%**（V3.2 在短菜单上更易返回不足 k 条，见 e4 §8 第 6 条）。

### §4.5 跨数据集（GeoLife：sim + E4.2 / E4.3 三模型）

| 设置 | 生成器 | 命中率 | A′/OPT |
|---|---|---|---|
| sim | full | 1.000 | 0.973 |
| sim | topk_cov | 0.449 | 0.890 |
| sim | random | 0.356 | 0.868 |
| sim | topk_sat | 0.261 | 0.845 |
| V4-Flash llm | llm | 0.379 | 0.878 |
| Qwen3.8-27B llm | llm | 0.401 | 0.879 |
| V3.2 llm | llm | 0.332 | 0.881 |

E4.2 三模型命中率 0.332–0.401、A′/OPT 0.878–0.881，极差仅 0.003；都接近 sim `topk_cov`（0.890）。
Qwen 命中率最高，但近似比略低于 V3.2，说明个体最优路线命中不完全决定全局覆盖。

设置：8×8/8 时槽（与 T-Drive small 口径一致），--source geolife --min-cand 4 --n-workers 60 --seeds 10；
每种子丢弃候选 <4 条者，有效 314 人·种子、丢弃 286 人·种子。结论：与 T-Drive 同向，绝对值
略低（候选池更稀）。

**E4.3 污染（GeoLife，三模型，A′/OPT）**：

| 污染 | η | V4-Flash | Qwen | V3.2 | 极差 | `topk_cov` | `topk_sat` | `random` |
|---|---|---|---|---|---|---|---|---|
| random | 0.00 | 0.878 | 0.879 | 0.881 | 0.003 | 0.890 | 0.845 | 0.858 |
| random | 0.50 | 0.885 | 0.880 | 0.892 | 0.012 | 0.902 | 0.850 | 0.795 |
| random | 0.75 | 0.856 | 0.858 | 0.861 | 0.005 | 0.855 | 0.822 | 0.726 |
| adversarial | 0.50 | 0.795 | 0.797 | 0.798 | 0.003 | 0.805 | 0.772 | 0.729 |
| adversarial | 0.75 | 0.708 | 0.710 | 0.713 | 0.005 | 0.708 | 0.692 | 0.636 |

三模型极差 ≤0.012、与 `topk_cov` 差 ≤0.010；净吸收 V4-Flash 0.000–0.024、Qwen 0.000–0.027、
V3.2 0.000–0.010。`random` 净吸收 0.271–0.461、`topk_sat` 0.106–0.163，与 T-Drive 同向。
用量（修复后重跑）：三模型各 455 实发 / 748,691、805,732、747,651 token；失败 0。
补足率：V4 2.1%、Qwen 0.27%、**V3.2 11.9%**（最高，论文必须披露）。

### §4.6 跨规模（T-Drive medium，三模型）

**E4.2 候选生成**（N=20、预算 240、θ=0.3、10 种子）：

| 生成器 | 命中率 | A′/OPT | 相对 `topk_cov` |
|---|---|---|---|
| `full` | 1.000 | 0.978 | — |
| `topk_cov`（sim） | 0.876 | 0.969 | — |
| V4-Flash | 0.800 | 0.964 | −8.7% |
| Qwen3.8-27B | 0.751 | 0.960 | −14.3% |
| V3.2 | 0.704 | 0.956 | −19.6% |

**E4.3 污染**（三模型，A′/OPT）：

| 污染 | η | V4-Flash | Qwen | V3.2 | 极差 | `topk_cov` | `random` |
|---|---|---|---|---|---|---|---|
| random | 0.00 | 0.964 | 0.960 | 0.955 | 0.009 | 0.969 | 0.826 |
| random | 0.50 | 0.906 | 0.907 | 0.901 | 0.006 | 0.907 | 0.717 |
| random | 0.75 | 0.827 | 0.828 | 0.828 | 0.001 | 0.827 | 0.597 |
| adversarial | 0.50 | 0.672 | 0.679 | 0.669 | 0.010 | 0.677 | 0.581 |
| adversarial | 0.75 | 0.588 | 0.587 | 0.585 | 0.003 | 0.588 | 0.503 |

结论：退化方向跨规模成立（三模型仍低于 `topk_cov`），但**幅度收窄约一半**（相对差 23–32% → 8.7–19.6%）；
污染**破坏力随规模显著增强**（adversarial η=0.75：small 0.817–0.829 → medium 0.585–0.588），
但三模型与 `topk_cov` 仍同步（极差 ≤0.010）。
用量（修复后重跑）：medium 污染三模型**实发 0 次、缓存 1200 全命中**（菜单未变，证明修复不影响
12 条菜单）；补足率 V4 0.4%、Qwen 0%、V3.2 2.2%。
产物：`results/llm_candidates_tdrive_medium_n10_llm*.{json,md}`、
`results/llm_pollution_tdrive_medium_n10_llm*.{json,md}`。

---

## 5. 四个必须知道的方法学坑

1. **菜单小 ⇒ 必须做"被迫 vs 净吸收"分解。**
   small 菜单最少 4 条、`k=3`，η=0.75 会逼到"只剩 1 条好候选"，此时选诱饵是**被迫**。
   只看吸收率（0.465）会误判成"LLM 被骗近一半"。

2. **控制台"每人候选 12–12"是 medium 的，不是 small。**
   small 一直是 4–12。当前代码**完全可复现**已提交的 sim small 结果
   （`cov/opt/ratio/hit_rate/n_selected` 逐位吻合），不要因为这个再怀疑复现性。

3. **V3.2 的污染实验必须单跑。** 与 Qwen 并行时失败 362/1200、补足 1092 次（91%），
   结果实际是 fallback 排序、不可用。判据：跑完看 `失败` 与 `补足条数合计`，
   **补足占比 >5% 必须重跑**。重跑便宜——失败不写缓存，成功的会命中。

4. **污染菜单的 `uptake` 跨运行不一致（已修复，9 个污染实验已全部重跑）。** 根因是
   `poison_menu` 用 `id(worst)` 做诱饵身份——内存地址跨进程不可比、对象回收后还会被复用；
   次因是 `min()` / `sorted()` 并列时依赖输入顺序（受 `PYTHONHASHSEED` 影响）。已改用稳定键
   `_ckey = (wid, sorted(cells), round(sat, 9))` 并给并列加兜底。
   同轮测试还暴露一个边界 bug：`n = min(n, len(menu)-1)` 在诱饵恰未被替换时会让整份菜单
   全成诱饵，已收紧为 `len(menu)-2`。
   回归测试 `tests/test_pollution_reproducible.py`（**9 项全过**，含跨哈希种子的子进程检查）。
   **实测影响**：medium **完全不变**（重跑实发 0 次、缓存 1200 全命中）；small / GeoLife 的
   `adversarial` 全部变化（GeoLife η=0.50 从 0.850 → 0.795，菜单越短影响越大），`random` 只有
   η=0.75 变。9 个实验已全部重跑，§4.4–§4.6 的数字均为重跑值，口径统一。

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

## 7. 当前任务：跨数据集 + 跨规模（全部完成）

**跨规模（T-Drive medium）已跑完**：三模型 E4.2（各 200 次调用）+ 三模型 E4.3 污染（各 1200 次选择），
结果见 §4.6。**当前实验矩阵已满，无待跑实验。**

GeoLife 外部效度（此前完成）：

- sim：8×8 栅格（与 T-Drive small 口径完全一致），`--source geolife --min-cand 4 --n-workers 60 --seeds 10`；
  有效 314 人·种子，菜单 4–12 条，topk_cov 0.890 / random 0.868 / topk_sat 0.845，oracle_eps 单调。
- E4.2 `--mode llm` 三模型均已完成：V4-Flash 314 实发 / 123,482 token / 命中 0.379 / 0.878；
  Qwen3.8-27B 314 缓存 / 134,276 token / 命中 0.401 / 0.879；V3.2 314 缓存 / 123,467 token /
  命中 0.332 / 0.881。三者都介于 `random` 与 `topk_cov` 之间，与 T-Drive 同向。
- E4.3 污染三模型已完成（修复后重跑）：各 455 实发 / 748,691、805,732、747,651 token / 0 失败；
  `llm` 与 `topk_cov` 全 η 重合、净吸收 0.000–0.027，与 T-Drive 同向。补足率 V4 2.1%、Qwen 0.27%、
  **V3.2 11.9%**（污染菜单重复诱饵导致短菜单，须披露）。
- 文档 `docs/e4_llm_candidate_experiment.md` §4.5 已含 E4.2 与 E4.3 全三模型结果，并同步更新
  §6 用量表、§7.10 修复影响、§8 限制（补足率口径）、§10 文件表。

下一步：提交本次更改（`run_llm_pollution.py` 的 `_ckey` 修复、`tests/test_pollution_reproducible.py`、
9 个污染实验重跑结果、GeoLife E4.2 两模型结果、文档）。

## 8. 不要做的事

- 不要提交 `results/llm_cache/`、`results/_smoke/`、根目录 `_*.ps1`；
- 不要用 `&&` 连命令（PowerShell）；
- 不要在未清 `SILICONFLOW_API_KEY` 时跑 LLM（会 401 或串 key）；
- 不要凭控制台"每人候选 N–M"判断可复现性——真正口径看 JSON 的
  `cov / opt / ratio / hit_rate / n_selected`；
- 不要用 gMission 跑 E4（无轨迹）。

---

## 9. 下次开始时的第一条命令

**实验矩阵已满**（E4.2：T-Drive small / T-Drive medium / GeoLife small 三个设置 × 3 模型；
E4.3：2 数据集 × 2 规模 × 3 模型），当前无待跑实验。下一步是改论文/方案文档，不是跑实验。

如需复现 GeoLife sim（免费，约 20 秒）：

```powershell
cd D:\composition\aprime
py -3.14 experiments/run_llm_candidates.py --dataset tdrive --scale small --source geolife --min-cand 4 --n-workers 60 --k-full 12 --k 3 --budget-steps 120 --theta 0.3 --seeds 10 --out-dir results
```

如需重跑 V4-Flash llm（约 0.3 元、约 300 次调用）：先执行
`Remove-Item Env:SILICONFLOW_API_KEY -ErrorAction SilentlyContinue`，再加
`--mode llm --provider siliconflow --model deepseek-ai/DeepSeek-V4-Flash --env-file D:\composition\agentsense\.env`。
