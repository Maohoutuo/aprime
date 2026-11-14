# 假设清单与不可恢复信息

> 原则：论文未给出的自由度，一律在代码里留一个显式开关（`MAPUSConfig` / `EnvConfig`），
> 并在本文档登记，便于对照原文复核与二次标定。

## 1. 论文给出、直接落地的部分

| 项 | 取值 | 出处 |
|---|---|---|
| 三档规模配置（网格、N、B、H） | 见 `agentsense.config.SETTINGS`（与 MAPUS 表 1 完全一致） | 表 1 |
| 六个土地利用类别 | vegetation / industrial / institutional / medical / residential / commercial | 式 (4) |
| 覆盖效用 | `φ(A) = α·E(A) + (1−α)·log Q(A)`，α = 0.5 | 式 (12)、式 (13) |
| PSS | `S(r) = sim(p_u, h(r)) − μ·risk(r)`，sim 为余弦相似度 | 式 (2) |
| 经验土地利用分布 | `h(r)(k) = (1/L)Σ I[κ(g_i)=k]` | 式 (3) |
| 公平分 | `F(u) = 1/(1+h_u)` | 式 (5) 后 |
| 复合选择分 | `J(u) = β·f(Δφ) + (1−β)·e(F)`，f/e 为归一化 | 式 (6) |
| 路线重叠 | `Σ min(1[g∈r_u],1[g∈r_v]) / Σ max(·)` | 式 (7) |
| 算法 1–3 的控制流 | 原样实现（含可行性校验、平局裁决、协商准入） | 附录 A.2–A.4 |
| 五个 archetype | 生态爱好者 / 都市居民 / 产业工人 / 社区助手 / 探索者 | 附录 B.2 |
| 六个基线 | RN / TVPG / TCPG / MSA / MSAGI / GraphDP | 附录 B.3 |
| 每档 20 个任务实例 | `N_TRIALS = 20` | §4.1 |
| 感知间隔 | T-Drive 15 min，Grab-Posisi 5 min | §4.1 |

## 2. 论文未给出、必须自行确定的自由度

| # | 自由度 | 取值 | 理由 / 影响 |
|---|---|---|---|
| A1 | 参与者每步可移动格数（cells_per_step） | **实测值**：tdrive small=3 / medium=7 / large=13（8×8、16×16、32×32 @15 min）；grab small=2 / medium=4 / large=8（Grab 官方轨迹未获得，用 T-Drive 在"同一 bbox、同一网格、同一 5 min 时槽"实测的位移作代理） | 口径唯一：**相邻时槽、位移非零的曼哈顿位移中位数**（`agentsense/experiments/measure_speed.py` 可复现）。该值是成本口径的唯一真源：`Setting.cells_per_step` → `Worker.speed` + `meta["cells_per_step"]`，所有步数一律走 `steps_for()` = `ceil(d/cps)` |
| A2 | 招募成本 `c_u` | 路线时间步数（报酬 1.0/步） | §4.1"each sensing action yields a normalized reward of 1.0 per time step"；预算 B=40/60/100 与表 1 一致 |
| A3 | 任务激励成本（TCPG 用） | 插入路点的绕行步数 | 原文未定义，仅用于排序 |
| A4 | **参与者时间窗内的余量** | 起终点曼哈顿距离 ≤ 55% 窗口 | 图 1(b) 的参与者都有大段余量；无余量时"用 residual steps 插入偏好路点"不可行，PRG 无法发挥作用 |
| A5 | 路线局部效用 `λ` | 0.5（`MAPUSConfig.lam`） | 算法 1 未给 λ；0.5 表示覆盖与满意等权 |
| A6 | 式 (1) 的 `η` | 0.5（`MAPUSConfig.eta`） | 论文未给；用于选择阶段的收益项组合 |
| A7 | 式 (6) 的 `β` | 0.5 | 论文未给；0.5 表示覆盖效率与公平等权 |
| A8 | 算法 2 的平局阈值 `ε` | 0.02（归一化后） | 论文未给；控制 LLM 平局裁决的触发频率 |
| A9 | 算法 3 的 `τ / K / N` | 0.10 / 3 / 5 | 论文未给；τ 是"起点重叠下界"，K 单对轮数，N 最大尝试对数 |
| A10 | 接单决策 | LLM 判断（`llm_accept=True`） | 论文"each PA decides whether to accept"；解析失败时**默认接受**，避免格式噪声被当作拒单 |
| A11 | LLM 输出不可行时的处理 | 路点投影回可行空间（`infeasible_policy="repair"`） | 算法 1 第 6 行只丢弃不可行解；实测丢弃率过高会让 PRG 退化为基线，因此保留投影（只保留方向语义，不保留越界/跳步部分） |
| A12 | 协商中协调者提案不可行时 | 先投影再交给参与者评估 | 算法 3 要求"preserving route feasibility"；否则协商几乎 100% 被否决 |
| A13 | 土地利用的粒度 | 细粒度混合（1–2 格相关长度） | 参与者路线只有 5–15 步，若地块是 10 格宽的大片区，`h(r)` 必然退化成 one-hot，PSS 失去意义 |
| A14 | `h(r)` 的 Laplace 平滑 `κ` | 6（`MAPUSConfig.pss_prior`） | 短路线 + 6 个类别时经验分布方差极大，向全局用地占比收缩是标准做法 |

## 3. 不可恢复的信息（不在代码里造假，只记录）

1. **原始 POI 数据**：土地利用由 POI 聚合得到（附录 B.1），论文与仓库均未提供。
   本工程用"结构化合成城市 + 配额约束抽样"生成，目标是让 6 类占比可控且空间混合。
2. **犯罪数据**（Zhang et al. 2025）：未提供。本工程用与商业/工业密度相关的合成场，
   并按附录 B.1 剔除记录数 < 10 的格子。
3. **`E(A)` 的分层熵定义**：论文引用 Ji et al. (2016)，未复述。沿用 agentsense 工程
   已标定的相对归一化熵（一个尺度常数 `e_scale`）。
4. **T-Drive / Grab-Posisi 原始轨迹**：需要下载 / Kaggle 授权。默认用同分布的合成
   移动性；`make_mapus_instance(..., source="tdrive")` 预留了真实轨迹接口。
   **Grab-Posisi 官方数据（轨迹 + POI）截至目前尚未获得**，因此 `dataset="grab"`
   的表 3 结果全部建立在"合成同分布移动性 + `_GRAB_LAYOUT` 合成城市"之上，
   属于**机制层面的跨数据集验证**，不是官方数据复现；拿到官方数据后需统一重跑。
5. **PSS 的绝对尺度与 μ**：由表 2 的 6 个基线锚点拟合尺度 a（μ 固定 0.5），
   见 `docs/calibration.md`。

## 4. 与论文的对齐检查表

| 检查项 | 状态 |
|---|---|
| 预算是"时间步之和"，且 6 个配置的预算利用率接近 100% | ✅（实测 cost/budget ≈ 0.95–1.0） |
| 所有输出路线 100% 满足起点/终点/时序/速度/边界 | ✅（`validate_route`，测试覆盖） |
| 基线之间 Coverage 排序（GraphDP/MSAGI 强、RN 弱） | ✅ 与论文一致 |
| MAPUS 的 PSS 显著高于所有基线 | ✅（实测速度口径下 6 档全部第一，+0.06 ~ +0.16，见 `results_speed_v2/`） |
| MAPUS 的 Coverage 与最强基线同量级 | ✅（Medium 并列/第一，Large ±0.06%，Small −1.1%） |
| NRR 降低重叠 | ✅ Grab 三档 0.045→0.031 / 0.031→0.045 / 0.038→0.031；⚠️ T-Drive Medium/Large 重叠地板已降到 0.015–0.019，低于触发阈值 τ=0.10，几乎不触发（τ=0.05 时 T-Drive 0.019→0.012） |
| NRR 提升覆盖 / PSS 略降 | ⚠️ **口径变更后不成立**：PSS 中性到 +0.02（Grab Large 反而 −0.002），覆盖变化 ≤0.004；NRR 的主要作用是压重叠，不是抬 PSS |
| FPS 在覆盖率与公平（Gini/CDF）之间取得折中 | ✅（FPS 覆盖率 4.694 > LS 4.639，Gini 0.024 ≈ LS 0.029，PSS 1.028 最高） |
| 消融：去 PRG 掉 PSS、去 NRR 掉 Coverage | ⚠️ **部分成立**：去 PRG 掉 PSS 只在 Medium/Large 明显（−0.06 ~ −0.14），Small 仅 −0.006 ~ −0.026；去 FPS 是 Small 档最主要的损失（−0.029 ~ −0.098）；去 NRR 对覆盖几乎无影响 |

## 5. 数据状态与结果冻结（2026-11-10）

| 项 | 状态 | 说明 |
|---|---|---|
| Grab-Posisi 官方数据（轨迹 + POI） | ❌ 尚未获得 | 轨迹需 Kaggle 授权、POI 论文未公开。当前 `dataset="grab"` 全部是"合成同分布移动性 + `_GRAB_LAYOUT` 合成城市" |
| T-Drive 真实轨迹 | ✅ 已接入 | `data/tdrive/tdrive_clean.txt.gz`，`make_mapus_instance(..., source="tdrive")` |
| Grab 犯罪数据 | ❌ 不存在 | 附录 B.1：Grab-Posisi 区域无可靠犯罪数据，`crime ≡ 0`（PSS 的风险项恒为 0） |
| 历史结果文件 | 🔒 已冻结 | `main_tdrive_*`、`main_grab_*`、`ablation_*`、`negotiation_*`、`fairness_*`、`case_study_*`、`runtime_*`，SHA256 清单见 `results/FROZEN.md` |

**冻结规则**

1. 在 ①② 两项前置工作完成之前，不重跑、不改标定口径、不覆盖上述结果文件；
   对表 2/3 的任何数值改动都视为"解冻"，需要负责人显式确认。
2. `results/llm_cache/` **不冻结**：它是按 prompt 寻址的付费缓存，重跑时应尽量命中以省额度。
3. ①② 完成后**统一重跑一次**（`run_all.py` 全流程），新结果写入新的文件名/目录，
   旧冻结结果保留用于前后对照（而不是就地覆盖）。
4. 在官方 Grab 数据接入前，对外表述 Grab 结果时必须带"合成数据"限定语，不能写成
   "在 Grab-Posisi 数据集上复现"。

### 5.1 ①② 的落地情况（2026-11-10）

**① 成本口径接真实速度**：`manhattan(a,b)` 形式的"步数"已全部换成
`steps_for(a,b,cps) = ceil(manhattan/cps)`，逐处核对如下（cps 的唯一真源是
`Setting.cells_per_step`，见第 4 行）：

| 位置 | 现状 |
|---|---|
| `agentsense/env/model.py` `steps_for` | 定义 `ceil(manhattan/speed)`；`:207` 的 `d > w.speed` 是等价的单步校验（d > cps ⟺ ceil(d/cps) > 1），保持 |
| `agentsense/baselines/planners.py:56,67`（`_reachable` / `_reach_set`） | 已用 `steps_for(·, w.speed)` |
| `mapus/routing.py:145`（`validate_route` 的速度校验） | 已用 `steps_for(·, speed)`，`speed` 缺省取 `worker.speed` |
| `agentsense/repair.py:47`（可达性收尾） | 已用 `steps_for(cells[i], w.dest, w.speed)`；`:42/:78` 的 `manhattan > speed` 是单步越界校验，保持 |

真正的阻断**不在**这些成本函数，而在"速度从来没写进 `Worker.speed`"（生成器恒置 1）。

**② 生成器写真实速度**：

* 新增 `agentsense/experiments/measure_speed.py`（可复现口径：相邻时槽 + 位移非零的
  位移中位数），实测 tdrive 3/7/13、grab 代理 2/4/8，写入 `config.SETTINGS`；
* `synthetic_workers` / `tdrive_workers` / `mapus_workers` 全部把 `speed=cps` 写进
  `Worker`，`meta["cells_per_step"]` 同步为实测值并附 `cells_per_step_source`；
* 余量与可达性的判据由"曼哈顿距离"改为"按 cps 折算的步数"
  （`sample_endpoints` / `slack = 1 − steps_for/dspan`），否则速度 > 1 时会把
  "其实有余量"的参与者误判为没余量。

**副作用（重跑必须注意）**：速度变大后所有方法的 PSS 都上升（参与者有更多剩余步数
可以停在偏好地块上），PSS 的尺度 `a` 是在 `cps=1` 下标定的，因此
**重跑顺序必须先 `calibrate.py` 再跑主表**，否则 PSS 会越界（实测出现 > 1.0）。

**重跑已完成（2026-11-10 12:28）**：代理模型全量 n=20 跑完，产物在
`mapus/results_speed_v2/`（`results/*.json`、`results/figures/*.png`、`report.html`），
旧的 `results/` 一个字节未动。`agentsense` 侧的表 4 也已在同一口径下重跑
（`agentsense/results_speed_v2/`），`e_scale` 重拟合仍为 4.00。待办：真实 LLM 两项、
以及 `overlap_tau` 是否由 0.10 改为 0.05（见 `docs/calibration.md`）。
