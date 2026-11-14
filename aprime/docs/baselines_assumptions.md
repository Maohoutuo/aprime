# 基线假设声明 & 数据可得性（务必在论文里如实标注）

本文件记录两件事：**没能拿到的东西**，以及**在拿不到原文/数据时的替代实现及其假设**。
目的是防止把"按方案描述重写"写成"复现"——那会是审稿时的硬伤。

## 1. 数据可得性

| 数据 | 状态 | 影响 | 退路 |
|---|---|---|---|
| **Grab-Posisi 官方轨迹 + POI** | **未获得**（轨迹需 Kaggle 授权，POI 论文未公开） | 无法做"官方数据集"复现；`dataset="grab"` 的实例全部是合成同分布 | 不做 Grab 官方复现（`CROSS` 搁置）；`mapus.data.instances.data_source_label()` 会把 `synthetic(grab_layout)` 写进结果文件，不得当官方复现读 |
| **gMission** | **本地不存在**（`D:\composition\data` 下只有 `tdrive`） | 方案 R2 的检查点 D5（"确认 gMission 时间字段可用"）**无法执行**；E0.1 **阻塞在 D0** | 接入方案见 `docs/e0_1_data_access_plan.md`（获取途径 / D0–D5 管线 / 验收断言）。数据到位前，E0.1 的"真实时间戳"用 **T-Drive 真实轨迹**承担（通路已实测可用），并在论文写明来源与替代理由 |
| **东华 CAFCTA/LPFCTA 原文** | **未获得**（2026 年同期工作） | `MSCTA_Greedy` / `SA 贪心` 无法逐行对齐 | 见下面 §2 的替代实现与假设；拿到原文后必须重跑并用同一实例对比 |

> 结论：R2（gMission 时间字段）目前是**阻塞**状态，不是"已确认可用"。不要在任何内部汇报里写成"已验证"。
> 接入方案见 `docs/e0_1_data_access_plan.md`；真实数据实验的搁置清单见该文件 §8 与本文 §4。

## 2. 替代实现与假设（`aprime/solvers/baselines.py`）

### 2.1 `mscta_greedy(cap_per_worker=2)`

**假设**：东华方案描述的是"最大化传感覆盖 + 每人容量上限 `cnt_j < cap_j`"。
本实现取：

* 目标：纯覆盖（无个性化、无公平项）；
* 约束：每人最多 `cap_per_worker` 条路线（**这就是与 A′「每人至多一条」的结构差异**）；
* **不检查 θ**。

**不是复现**：`cap_per_worker` 是超参，原文的"能力容量"定义未知。

### 2.2 `sa_caftca(iters, lam=0.5)`

**假设**：东华"贪心 + 模拟退火"的目标可写成 `coverage − λ·N·cfi_gap`，
其中 `cfi_gap` 是"名额份额 − 能力份额"的最大偏差（即 CFI 型公平的自洽度）。

* 初值取 `capacity_greedy`（能力比例贪心），退火在 add / remove / swap 上做；
* 同样不检查 θ、允许每人多名额；
* `λ` 是超参，原文未给出。

### 2.3 `capacity_greedy`

**假设**：CFI 型"能力比例公平"= 名额 ∝ 能力。实现为按能力（候选路线数）从高到低取人，
再补次优路线。这是 §2.5 要打的靶子，**故意**忽略 θ，因此会选中 `S(r_u) < θ_u` 的路线
（结果里 `theta_violations` 就是它的违约次数，实测 0.2–0.4）。

### 2.4 与 A′ 的可比性说明（写进论文的"实验设置"）

| 维度 | A′ | capacity / mscta / sa_caftca |
|---|---|---|
| 满意度下界 θ | **硬约束** | 忽略（违约 0.2–1.0 次） |
| 每人路线数 | 至多 1 | 1 或 `cap_per_worker` |
| 公平机制 | 分组配额一致性（待实现，见 D5） | CFI 型能力比例 |
| 求解 | 贪心 + 局部搜索 | 贪心 / 贪心 + SA |

因此主对比表必须**同时报 `theta_violations`**，否则拿"高 PSS/高 CFI"当优势是不公平的比较。

## 3. 待办（拿到原文后）

1. 重跑 `mscta_greedy` / `sa_caftca`，把假设换成原文定义；
2. 用**同一批实例**（同一 seed、同一候选池）重跑，保证"只有算法不同"；
3. 若原文的公平指标与 `cfi_gap` 不同，按原文口径实现并在附录并列两套指标；
4. gMission 到位后，先做 R2 检查（时间字段完整性 + 时槽对齐），再重跑 E0.1/E1.1；
   检查项与阈值见 `docs/e0_1_data_access_plan.md` §6（D0–D5）与 §7（A1–A9 断言）。

## 4. 搁置状态（2026-11 起）

E0.1 阻塞期间，**所有依赖真实数据的实验一律搁置**，状态由 `aprime/availability.py` 单点判定
（`require_available(eid)` 会对搁置项直接抛错并给出解冻条件）：

| 状态 | 实验 |
|---|---|
| `blocked` | E0.1（gMission 缺失）、E1.2（批量管线未实现）、CROSS（Grab 官方未获得） |
| `deferred` | E1.1、E2.3、E3.1、SEC6（§6 表 6-1…6-9 的真实数据复现批） |
| `ready` | E2.1、E2.2（抽象实例）、回归测试、理论工作 |

必须记住：现有 `results/*.md` 的 `dataset=tdrive` 数字**全部来自合成参与者**
（`make_mapus_instance()` 默认 `source="synthetic"`、`make_dense_instance()` 用 `mapus_workers()`），
真实元素只有栅格/速度/时槽三个标定常量。详见 `docs/e0_1_data_access_plan.md` §2.3。
