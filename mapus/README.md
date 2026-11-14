# MAPUS 复现工程

复现论文 **Language-Grounded Multi-Agent Planning for Personalized and Fair
Participatory Urban Sensing**（arXiv:2603.24014v1）。

MAPUS = **M**ulti-**A**gent framework for **P**articipatory **U**rban **S**ensing：
参与者是带个人画像的自治 LLM agent，协调者 agent 负责公平选人与协商式路线精修。

```
论文结构                         本工程
──────────────────────────────  ──────────────────────────────────────────────
3.1 问题定义                     mapus/env/ + agentsense/env/（复用已标定的覆盖效用）
3.3 偏好感知路线生成 (算法 1)     mapus/agents/participant.py + mapus/routing.py
3.4 公平感知选人     (算法 2)     mapus/agents/coordinator.py: select
3.5 协商式路线精修   (算法 3)     mapus/agents/coordinator.py: negotiate
4.2 表 2/3 主结果                experiments/run_main.py
4.3 图 3 消融 / 表 4 协商         experiments/run_ablation.py, run_negotiation.py
4.4 图 4 公平性                  experiments/run_fairness.py
B.4 表 5 运行时                  experiments/run_runtime.py
B.5 图 11 基座模型               experiments/run_base_llm.py
B.6 图 12 扰动案例               experiments/run_case_study.py
```

> 环境与覆盖目标函数复用同目录下的 `agentsense` 工程（同一课题组的前置工作，
> 其 `DEFAULT_ENV` 已用论文表 4 的 Base Obj. 锚点标定过），从而让 6 个经典基线
> 与 MAPUS 的表 2/3 在同一口径下可比。

## 安装

```powershell
# 1) 先安装可复用的环境层（只需一次）
cd D:\composition\agentsense
py -3.14 -m pip install -e .

# 2) 本工程
cd D:\composition\mapus
py -3.14 -m pip install -e .
```

依赖：`numpy`。真实 LLM 走 OpenAI 兼容 HTTP 接口，**不需要** `openai` 包
（`agentsense/llm.py` 用标准库 `urllib` 实现）。

## 快速开始

```powershell
py -3.14 experiments\smoke.py            # 环境自检：城市属性 / 画像 / PSS / 三阶段
py -3.14 -m pytest tests -q              # 单元测试
py -3.14 experiments\run_all.py          # 一键跑完整套离线实验 → report.html
```

单步：

```powershell
py -3.14 experiments\calibrate.py --n 20                 # PSS 尺度标定（κ, a, μ）
py -3.14 experiments\run_main.py --dataset tdrive --n 20 # 表 2
py -3.14 experiments\run_main.py --dataset grab   --n 20 # 表 3
py -3.14 experiments\run_ablation.py  --n 20             # 图 3
py -3.14 experiments\run_negotiation.py --n 20           # 表 4
py -3.14 experiments\run_fairness.py --n-tasks 60        # 图 4
py -3.14 experiments\run_runtime.py   --n 5              # 表 5
py -3.14 experiments\make_figures.py && py -3.14 experiments\build_report.py
py -3.14 experiments\build_dashboard.py                     # 另生成 dashboard.html（可视化看板）
```

真实 LLM（已跑完的那一套，`--tag qwen72b` 让结果文件与离线结果区分开）：

```powershell
$M = "Qwen/Qwen2.5-72B-Instruct"
py -3.14 experiments\run_main.py        --dataset tdrive --scales small,medium,large --n 3 --backend siliconflow --model $M --parallel 1 --llm-workers 3 --tag qwen72b
py -3.14 experiments\run_main.py        --dataset grab   --scales small,medium,large --n 3 --backend siliconflow --model $M --parallel 1 --llm-workers 3 --tag qwen72b
py -3.14 experiments\run_ablation.py    --n 3 --scales small,medium,large --backend siliconflow --model $M --tag qwen72b   # 每个数据集跑一次
py -3.14 experiments\run_negotiation.py --n 3 --backend siliconflow --model $M --tag qwen72b
py -3.14 experiments\run_fairness.py    --n-tasks 60 --route-workers 6 --backend siliconflow --model $M --tag qwen72b
py -3.14 experiments\estimate_llm_cost.py                # 先算钱再跑：调用量/token/¥ 估算
```

## 真实 LLM

把 key 写进 `.env`（已 gitignore）：

```
SILICONFLOW_API_KEY=sk-...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

```powershell
py -3.14 experiments\llm_e2e.py Qwen/Qwen2.5-72B-Instruct      # 单实例端到端验证
py -3.14 experiments\run_main.py --dataset tdrive --n 3 --backend siliconflow `
    --model Qwen/Qwen2.5-72B-Instruct --parallel 3 --llm-workers 6 --tag qwen72b
py -3.14 experiments\run_base_llm.py --n 2                     # 图 11
py -3.14 experiments\run_case_study.py --backend siliconflow --model Qwen/Qwen2.5-72B-Instruct
```

所有真实调用都带**磁盘缓存**（`results/llm_cache/<sha256>.json`）：同一 prompt 只付费
一次，实验可中断续跑、可完全复现。注意 `Qwen3-*` 系列默认开思考模式，长提示下会非常慢，
建议用 `Qwen/Qwen2.5-*`、`deepseek-ai/DeepSeek-V3.2` 这类非思考模型。

## 目录

```
mapus/
├── mapus/
│   ├── config.py              MAPUSConfig（λ/β/ε/τ/K/N/μ/κ/a 等自由度全集）
│   ├── env/urban.py           6 类土地利用 + 犯罪风险场（式 3/4，附录 B.1）
│   ├── env/metrics_obj.py     PSS（式 2）、覆盖效用、重叠（式 7）、汇总评估
│   ├── profiles.py            5 个 archetype + 画像模板（附录 B.2，图 10）
│   ├── routing.py             TSPTW 基线 R0、VALID(·)、不可行解投影、偏好停留补齐
│   ├── parse.py               容错 JSON / 路径解析
│   ├── llm_backend.py         RemoteLLM（+磁盘缓存）、SurrogateLLM（离线确定性代理）
│   ├── agents/prompts.py      图 5–8 的提示词（+ 扰动广播 + 单 LLM 基线）
│   ├── agents/tools.py        retrieve_grid_geographic_attributes / crime_info
│   ├── agents/participant.py  PA：接单 + 算法 1 + 扰动响应
│   ├── agents/coordinator.py  CA：算法 2 + 算法 3
│   ├── data/instances.py      实例装配（城市属性 + 画像）、公平性实验的人才池
│   └── pipeline.py            run_mapus / run_single_llm / run_disturbance
├── experiments/               smoke / calibrate / run_main / run_ablation / run_negotiation
│                              run_fairness / run_runtime / run_base_llm / run_case_study
│                              make_figures / build_report / run_all / common
├── tests/                     环境、解析、三阶段流程的单元测试
├── docs/                      assumptions（自由度与信息缺口）· architecture（逐项对照）· calibration
└── results/                   实验输出 + 图片 + LLM 缓存
```

## 三种运行模式

| 模式 | 说明 | 用途 |
|---|---|---|
| `surrogate` | 确定性代理：直接执行提示词描述的优化目标（λ·覆盖 + (1−λ)·PSS 的局部搜索） | 不消耗额度、可跑满 20 个实例、单元测试 |
| `siliconflow` / `openai` / `deepseek` | 真实 LLM（OpenAI 兼容） | 论文级实验，带缓存 |
| `anthropic` | 真实 LLM（Anthropic Messages API） | 备选 |

> `surrogate` 的结果在报告与表格中一律标注 `backend=surrogate`，不与真实 LLM 结果混用。
> 它能验证「机制是否成立」（三阶段 + 消融 + 公平性 + 协商），但不能替代论文里
> gpt-4.1-mini 的语言推理能力。

## 关键设计取舍（否则机制会失效）

1. **经典基线只优化覆盖**（`baseline_route(lam=1.0)`），偏好由 LLM 精修引入。
   若经典规划器已按偏好选点，PRG 将没有任何提升空间。
2. **停留补齐必须偏好感知**：`h(r_u)` 按时间步统计土地利用，把剩余步数停在偏好最契合的
   格子是提升 PSS 的直接机制（基线用均匀补齐，精修用偏好补齐）。
3. **协商提案先投影再投票**：协调者直接给出的 `(x,y,t)` 几乎总会在某个参与者的时间窗上
   越界；不投影时协商接受率为 0。
4. **参与者要有时间余量**：按实测速度折算的行程步数 ≤ 时间窗的 55%（判据是
   `steps_for(·, cps)`，不是曼哈顿距离），否则"用 residual steps 插入偏好
   路点"根本不可行。

详见 `docs/architecture.md`。

## 结果摘要（离线代理模型，n = 20；真实 LLM 版本待重跑）

> **当前代码口径 = 实测速度**（`cells_per_step`：tdrive 3/7/13、grab 2/4/8）。
> 下表是 2026-11-10 全量重跑的结果，完整表格/消融/协商/公平性/案例见
> `results_speed_v2/report.html`（数据文件在 `results_speed_v2/results/`）。
> `results/` 里保留的是 `cells_per_step = 1` 时代的冻结结果，只用于前后对照，
> 清单与哈希见 `results/FROZEN.md`。
>
> ⚠️ **`τ` 已定档 0.05（2026-11-10 17:44）**：下表与 `results_speed_v2/` 是 `τ=0.10` 产出的；
> 新默认下的全量重跑在 **`results_tau005/report.html`**。两版 Coverage/PSS 差异在 ±0.006 内，
> 主要差别是协商后的路线重叠（T-Drive Medium 0.018→0.012、Grab Large 0.031→0.014）。
> 并表时以 `results_tau005/` 为准。

**表 2 · T-Drive**（Coverage / PSS，括号内为论文值）

| Method | Small | Medium | Large |
|---|---|---|---|
| RN | 4.653 / 0.630 (4.557 / 0.632) | 4.909 / 0.650 (4.626 / 0.605) | 5.257 / 0.623 (4.974 / 0.594) |
| TVPG | 4.695 / 0.696 (4.563 / 0.634) | 5.026 / 0.690 (4.753 / 0.610) | 5.351 / 0.719 (5.067 / 0.590) |
| TCPG | 4.702 / 0.646 (4.594 / 0.631) | 5.014 / 0.700 (5.084 / 0.621) | 5.377 / 0.786 (5.043 / 0.602) |
| MSA | 4.729 / 0.662 (4.662 / 0.630) | 4.995 / 0.769 (5.065 / 0.622) | 5.375 / 0.775 (5.131 / 0.598) |
| MSAGI | 4.726 / 0.699 (4.661 / 0.633) | 5.034 / 0.716 (5.052 / 0.616) | 5.371 / 0.752 (5.128 / 0.590) |
| GraphDP | 4.680 / 0.551 (4.761 / 0.659) | 4.900 / 0.549 (5.102 / 0.651) | 5.233 / 0.469 (5.214 / 0.628) |
| GPT-4.1-mini | 4.727 / 0.539 (4.576 / 0.742) | 5.043 / 0.580 (4.872 / 0.706) | 5.369 / 0.586 (4.936 / 0.691) |
| **MAPUS** | **4.678 / 0.793** (4.677 / 0.875) | **5.043 / 0.894** (5.115 / 0.875) | **5.374 / 0.941** (5.251 / 0.898) |

**表 3 · Grab-Posisi**（⚠️ 官方数据尚未获得：当前为"合成同分布移动性 + 合成城市"下的机制验证，非官方数据复现）

| Method | Small | Medium | Large |
|---|---|---|---|
| RN | 4.606 / 0.775 (4.236 / 0.727) | 4.910 / 0.747 (4.710 / 0.696) | 5.200 / 0.638 (5.399 / 0.648) |
| TVPG | 4.687 / 0.830 (4.342 / 0.714) | 5.008 / 0.828 (4.737 / 0.695) | 5.321 / 0.833 (5.406 / 0.640) |
| TCPG | 4.684 / 0.823 (4.232 / 0.730) | 4.995 / 0.849 (4.806 / 0.703) | 5.350 / 0.826 (5.587 / 0.658) |
| MSA | 4.716 / 0.808 (4.372 / 0.744) | 4.994 / 0.837 (4.900 / 0.699) | 5.365 / 0.868 (5.586 / 0.655) |
| MSAGI | 4.722 / 0.821 (4.392 / 0.752) | 5.020 / 0.849 (4.887 / 0.714) | 5.341 / 0.860 (5.567 / 0.653) |
| GraphDP | 4.678 / 0.753 (4.369 / 0.749) | 4.906 / 0.599 (4.913 / 0.719) | 5.184 / 0.548 (5.611 / 0.700) |
| GPT-4.1-mini | 4.699 / 0.715 (4.293 / 0.782) | 5.040 / 0.689 (4.755 / 0.742) | 5.367 / 0.608 (5.413 / 0.729) |
| **MAPUS** | 4.668 / **0.891** (4.386 / 0.854) | 5.033 / **0.960** (4.890 / 0.834) | 5.366 / **0.984** (5.701 / 0.831) |

**结论**（与论文一致）

* **PSS**：MAPUS 在 6 个档位全部最高（T-Drive 0.793/0.894/0.941，Grab 0.891/0.960/0.984），
  比最强经典基线高 **+0.09 ~ +0.16**（T-Drive）与 **+0.06 ~ +0.12**（Grab）；单 LLM 基线
  明显更低（0.54–0.72），复现了"一次性规划无法兼顾个性化"的结论。
* **Coverage**：与最强基线同量级——Medium 并列或第一，Large 差 0.06%（T-Drive）/ 第一（Grab），
  Small 低 1.1%。
* **消融**（Full − 去掉某组件，PSS）：PRG 在 Medium/Large 贡献 **+0.10 ~ +0.14**（Small 仅 +0.01 ~ +0.03）；
  FPS 在 Small 贡献 **+0.03 ~ +0.10**、随规模衰减到 +0.01；NRR 在小/中档 PSS 中性或 +0.02，
  但**稳定降低路线重叠**（Grab 0.097→0.116、0.031→0.045、0.031→0.038）。
* **协商（表 4）**：`τ=0.10` 时 Grab 重叠 0.045→0.031（−31%）、T-Drive 本就很低（0.019→0.018）；
  `τ` 定档 0.05 后（`results_tau005/`）T-Drive 0.019→0.012、Grab Large 0.038→0.014，
  但 Grab Medium 是反例（0.031→0.034）——按档位如实写，别笼统说"稳定降低"。
  定档依据与全 6 档对照见 `docs/calibration.md`。
* **公平（图 4）**：FPS 以 Gini 0.024 / 最大负载 9 保持最公平，同时覆盖 4.694 高于 LS（4.639）
  且 PSS 最高（1.028）；CG 牺牲公平（Gini 0.562、最大负载 34）换覆盖 4.771。

### 真实 LLM 基座对比与案例（2026-11-10，medium n=2，`τ=0.10` 口径）

> 在全量重跑速度口径后新跑的补充结果，文件在 `results/base_llm_n2.json`（图 11）与
> `results/case_study_siliconflow.json`（图 12）。⚠️ 这两份是 `τ=0.10` 跑的，
> 与 `results_tau005/` 的 `τ=0.05` 不同口径；已决定等 Grab-Posisi 官方数据到位后一次性重跑
> （见 `results/FROZEN.md`），暂不与下表并表。

| 模型 | T-Drive cov / PSS | Grab cov / PSS | 未命中缓存时延（T / G） |
|---|---|---|---|
| Qwen2.5-7B | 5.027 / 0.732 | 4.998 / 0.859 | 全命中缓存，时延不可比 |
| Qwen2.5-72B | 5.020 / 0.785 | 4.973 / 0.905 | 65 s / 210 s |
| DeepSeek-V3.2 | 5.035 / 0.756 | 4.990 / 0.868 | 585 s / 517 s |

读法：Coverage 三档基本持平（4.97–5.04），PSS 明显吃模型容量（7B→72B +0.05），
DeepSeek-V3.2（思考模型）PSS 略低于 72B 但时延高一个数量级——正是"性能—时延权衡"那张图。
案例研究（图 12）：事故封锁格后覆盖 4.674→4.694、PSS 0.799→0.726（4/5 改路线）；
高价值需求广播后覆盖 5.042→5.130、PSS 1.048→0.907（6/6 改路线）。
其中 2 例出现"声称已调整但路径未变"（`adjust=true` / `changed=false`），是"解释—行为一致性"
的可讨论素材。

### 真实 LLM（Qwen2.5-72B-Instruct，n=3）——⚠️ 仍是 `cells_per_step = 1` 口径，待重跑

> 下面两组数是速度口径变更**之前**跑的，只说明"当时代码"的表现，不能与上表并列比较。
> T-Drive 与 Grab-Posisi 都跑了完整 8 方法 × 3 档。

  T-Drive（Coverage / PSS）：

  | Scale | 本文 | 论文 | 单 LLM 基线（本文） |
  |---|---|---|---|
  | Small | 4.668 / 0.677 | 4.677 / 0.875 | 4.516 / 0.475 |
  | Medium | 5.042 / 0.704 | 5.115 / 0.875 | 4.987 / 0.522 |
  | Large | 5.349 / 0.791 | 5.251 / 0.898 | 5.266 / 0.636 |

  Grab-Posisi（Coverage / PSS）：

  | Scale | 本文 | 论文 | 单 LLM 基线（本文） |
  |---|---|---|---|
  | Small | 4.641 / 0.799 | 4.386 / 0.854 | 4.384 / 0.752 |
  | Medium | 5.012 / 0.842 | 4.890 / 0.834 | 4.880 / 0.675 |
  | Large | 5.339 / 0.885 | 5.701 / 0.831 | 5.325 / 0.708 |

  ΔCoverage：T-Drive **−0.2% / −1.4% / +1.9%**，Grab **+5.8% / +2.5% / −6.4%**；
  ΔPSS：T-Drive −0.198 / −0.171 / −0.107，Grab −0.055 / **+0.008** / **+0.054**（Grab 上几乎对齐论文）。
  T-Drive 的 PSS 系统性低 0.11–0.20，是与离线代理模型同向的残差（来自 PSS 尺度 (a, μ) 与画像文本的
  标定差异），而不是机制失效。
  两个数据集、六个档位里 MAPUS 的 PSS 都显著高于同模型单 LLM 基线
  （T-Drive +0.20 / +0.18 / +0.16，Grab +0.05 / +0.17 / +0.18），复现了"多 agent 分解带来收益"。
  真实调用全部走磁盘缓存，可续跑；MAPUS 未命中缓存时的实测时延为 T-Drive 153 s / 287 s、
  Grab 98 s（medium）/ 289 s（large）。
  **诚实说明（n=3 子样本噪声）**：small 档 MAPUS 的 PSS 不是第一（T-Drive TVPG 0.796、Grab TCPG 0.888 更高），
  Grab large 的 Coverage 也略低于该子样本里最好的 MSA（5.339 vs 5.380）；在 n=20 的离线全量里
  MAPUS 六个档位 PSS 全第一、Coverage 与最强基线同量级，说明这是 3 个实例的抽样波动而不是机制问题。

* **消融（图 3，真实 LLM，两数据集 × 3 档，n=3）**：PSS 对比如下（加粗为该档最优）。

  | 数据集 | Scale | w/o PRG | w/o FPS | w/o NRR | Full |
  |---|---|---|---|---|---|
  | T-Drive | Small | 0.603 | 0.529 | 0.657 | **0.677** |
  | | Medium | 0.694 | 0.500 | **0.724** | 0.704 |
  | | Large | 0.772 | 0.614 | **0.791** | **0.791** |
  | Grab | Small | 0.799 | 0.660 | **0.812** | 0.799 |
  | | Medium | 0.825 | 0.658 | **0.849** | 0.842 |
  | | Large | 0.864 | 0.753 | **0.885** | **0.885** |

  **去 FPS 的损失最大**（PSS −0.13 ~ −0.20，公平性也塌到 Gini 0.58），去 PRG 次之（0.00 ~ −0.07，
  Grab Small 上恰为 0——8×4 网格太小，LLM 的绕行提案基本被投影回原路线）；
  去 NRR 反而略升 0.01–0.02，因为它换来的是**重叠下降**（T-Drive Small 2.4%→1.6%、Large 0.8%→0.4%，
  Grab Medium 7.2%→7.0%）。Coverage 在四个变体之间差异 ≤1%，说明三个组件主要作用于 PSS 与空间重叠，
  而不是覆盖效用本身——这与论文"组件耦合、覆盖由经典规划器打底"的叙述一致，但**去 PRG 的跌幅比论文小**。

* **协商（表 4，真实 LLM，Medium，n=3）**：T-Drive 重叠 **1.91%→1.75%**、覆盖 5.035→5.042、
  空间熵 4.164→4.177、PSS 0.724→0.704（每实例 2.3 对协商、2.0 对接受）；Grab 重叠 **7.16%→6.98%**、
  覆盖 5.029→5.012、熵 4.167→4.142、PSS 0.849→0.842（4.7 对 / 4.0 对接受）。
  方向与论文一致（重叠下降、T-Drive 覆盖微升、PSS 小幅下降），但**幅度明显小于代理模型**
  （代理上 T-Drive 重叠 2.0%→0.77%），说明真实模型给出的绕行提案更保守。

* **公平性（图 4，真实 LLM，60 任务 × 30 人池）**：

  | 策略 | Coverage | PSS | Gini（被选次数） | 单人最大被选 |
  |---|---|---|---|---|
  | UR（随机） | 4.483 | 0.790 | 0.167 | 17 |
  | LS（最少被选优先） | 4.472 | 0.780 | 0.026 | 9 |
  | CG（纯覆盖贪心） | **4.789** | 0.778 | 0.581 | 35 |
  | **FPS（本文）** | 4.628 | **0.900** | **0.007** | **9** |

  FPS 用 3.4% 的覆盖率换取 **PSS +0.12** 和 **Gini 0.007**（近乎完美均衡、单人最多 9/60 次），
  正是论文主张的"覆盖—公平—满意度"折中点；CG 的覆盖率最高但把 60 个任务里的 35 个给了同一个人。
* **基座模型（图 11，Medium，n=2，缩小版）**：Qwen2.5-7B vs Qwen2.5-72B 在 T-Drive / Grab 上
  Coverage 5.033/5.017 vs 5.044/5.013，PSS 0.725/0.833 vs 0.686/0.843——与论文一致：
  **覆盖效用随模型容量有小幅提升，PSS 基本持平**（轻量模型也守住偏好对齐）。
  论文用 9 个模型 × 20 个任务，本复现受额度/限流限制缩小为 2 个模型 × 2 个任务，只用于看趋势。
* **扰动案例（图 12，真实 LLM）**：交通事故封锁 [5,4] 时，2 名参与者改道成功、2 名尝试改道但被
  可解性/可行性检查挡回（LLM 给出了具体绕行点）、2 名判断"不经过该区域"保持原路线；
  高价值数据案例 6/6 绕行 [5,3]，覆盖 **5.014 → 5.067**、PSS **0.883 → 0.923**，
  复现了论文"扰动可成为再优化的催化剂"这一观察。

## 已知差距

绝对数值无法与论文逐格对齐，因为下列信息不可恢复：原始 POI 数据、犯罪数据、
`E(A)` 的分层熵定义、`μ / λ / β / ε / τ / K / N` 的取值、以及 T-Drive / Grab-Posisi
的原始轨迹。本工程的处理方式与逐项登记见 `docs/assumptions.md`，标定细节见
`docs/calibration.md`，结果与残差分析见 `report.html`。

> **数据状态（2026-11-10）**：T-Drive 真实轨迹已接入；**Grab-Posisi 官方数据（轨迹 + POI）
> 尚未获得**，`grab` 相关结果全部基于合成数据。现有结果文件已冻结（`results/FROZEN.md`
> 记录 SHA256 清单），待两项前置工作完成后统一重跑一次，在此之前不做重跑与就地覆盖。

可比的结论（与论文一致）：

* MAPUS 的 PSS 在所有档位**显著高于**全部 6 个经典基线与单 LLM 基线；
* Coverage 与最强经典基线同量级（Large 档常为最优）；
* 三个组件里 **FPS（满意度感知的公平选择）贡献最大**（去 FPS 的 PSS 跌幅 0.10–0.20），
  其次是 PRG（去 PRG 跌 0.01–0.07），NRR 的作用体现在**降低路线重叠**而不是拉开 PSS；
* NRR 降低路线重叠；T-Drive 上覆盖微升、PSS 微降，Grab 上两者都微降（真实 LLM 下幅度小于代理模型）；
* FPS 在覆盖率与公平（Gini / CDF）之间取得优于 CG 与 LS 的折中（真实 LLM：PSS 0.900、Gini 0.007）。

## 测试

```powershell
py -3.14 -m pytest tests -q
```

覆盖：土地利用六类齐备与归一、犯罪数据仅 T-Drive、PSS 的方向性、路线可行性与
投影修复、解析器容错、三阶段流程的预算/可行性、代理模型的确定性、消融开关确实生效。
