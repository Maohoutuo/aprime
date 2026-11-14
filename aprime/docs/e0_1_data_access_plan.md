# E0.1 数据接入方案：真实时间戳

> **状态：阻塞（停在 D0，未过）。** 本文只定义"数据怎么进来、怎么验收"，不含任何实验结果。
> 依赖真实数据的实验已按 §8 全部搁置；仍可推进的工作见 §9。
> 机读状态以 `aprime/availability.py` 为唯一来源，本文与之冲突时以代码为准。

---

## 0. 一页结论

| 问题 | 结论 |
|---|---|
| E0.1 要做什么 | 把一份**带真实时间戳**的群智感知数据接进 `agentsense → mapus → aprime` 三段管线，并给出可执行验收；对应方案 R2 的检查点 **D5（确认时间字段可用）** |
| 现在有什么 | T-Drive 清洗文件（真实，`data/tdrive/tdrive_clean.txt.gz`，22.4 MB）＋可用的真实轨迹读取器与实例入口 |
| 缺什么 | **gMission**（`D:\composition\data` 下不存在）、Grab-Posisi 官方数据 |
| 最要紧的发现 | 现有 9 个实验脚本**没有一条**真正走过真实轨迹（§2.3）。"tdrive_small" 只是**栅格/速度标定取自 T-Drive**，参与者是合成的 |
| 处置 | 真实数据实验全部搁置（§8）；E0.1 按 D0→D5 推进，当前停在 D0 |

一句话：**E0.1 不是"再补一份数据集"，而是"第一次把真实轨迹接进实验"。** 在此之前所有
`results/*.md` 都是合成参与者结果，论文里必须如实标注。

---

## 1. E0.1 的定义与三条判据

E0.1 的产出不是一个数字，而是一条**可信的数据通路**。三条判据缺一不可：

| 判据 | 内容 | 为什么必须有 |
|---|---|---|
| **J1 时间可用** | 每个事件有可解析的绝对时间，且能对齐到 `slot_min` 网格 | 方案 R2/D5 的定义；没有它 `Instance.n_slots` 无从谈起 |
| **J2 轨迹可切段** | `_contiguous_runs` + `_trips` 能产出 `(origin, dest, duration_slots)` | 候选路线生成（`aprime/candidates.py`）的输入就是这三元组 |
| **J3 时间窗有余量** | `slack = 1 − 曼哈顿距离/(span−1)` 的分布足以支撑"插入偏好路点" | A′ 全部创新点的物理前提；余量为 0 时 PRG 与偏好感知绕行都不可行 |

J3 的口径已在 `data/tdrive/audit_slack.py` 里固化，gMission 必须复用同一套定义才可比。

---

## 2. 现状盘点

### 2.1 已有资产（实测可用）

| 资产 | 位置 | 实测/说明 |
|---|---|---|
| T-Drive 清洗文件 | `data/tdrive/tdrive_clean.txt.gz` | 22.4 MB；`taxi_id,ts,lon,lat`，已 bbox 过滤 + 按 (day,slot) 去重 + 排序 |
| T-Drive 原始数据 | `data/tdrive/raw/{06..014}/` + 同名 `.zip` | 9 个压缩包，已解包 |
| 清洗脚本 | `data/tdrive/prep_tdrive.py` | bbox → `(day*24+hour)*60+minute` 去重 → 排序 → 丢弃 <3 点的轨迹 |
| 余量审计 | `data/tdrive/audit_slack.py` | 真实 vs 合成 slack 分布对照（J3 的度量工具） |
| 真实轨迹读取器 | `agentsense/agentsense/data/instances.py::load_tdrive_traces` | `lru_cache` 缓存；首次解析 214 万点约 **14.3 s**（实测） |
| 真实实例入口 | `mapus/mapus/data/instances.py::make_mapus_instance(source="tdrive")` | **实测可用**：`meta.source="tdrive"`，`n_workers=20`，`data_source_label()→"tdrive(real)"` |

### 2.2 缺失数据

| 数据 | 状态 | 影响 |
|---|---|---|
| **gMission** | 本地不存在 | E0.1 的主数据源缺位；R2/D5 无法执行，不能写成"已验证" |
| **Grab-Posisi 官方** | 未获得（轨迹需 Kaggle 授权，POI 论文未公开） | 跨数据集验证只能停留在合成层面 |

### 2.3 必须正视的现状：现有实验全部是合成参与者

这是写本文时最重要的核对结果。逐一核查 `experiments/`：

| 脚本 | 实例构造 | 参与者来源 |
|---|---|---|
| `run_theta_curve.py` | `make_mapus_instance(args.setting, i)` | **默认 `source="synthetic"`** → `mapus_workers()` |
| `run_density_stress.py` | `make_dense_instance()` | `mapus_workers(slack=0.55)` |
| `run_fairness_tradeoff.py` | `make_dense_instance()` | 同上 |
| `run_cfi_target.py` | `make_dense_instance()` | 同上 |
| `run_pref_homogeneity.py` | `make_dense_instance()` | 同上 |
| `run_route_vs_task.py` | `make_dense_instance()` | 同上 |
| `run_llm_candidates.py` | `make_dense_instance()` | 同上 |
| `run_bound_tightness.py` | `random_instance()` + `SimpleNamespace` | **纯抽象，不涉及任何数据** |

全仓检索 `source="tdrive"` 在 `experiments/` 下 **0 命中**。真实元素只有三个标定常量：
栅格尺寸、`cells_per_step`（速度）、`slot_min`。结果文件的 `meta.source` 会写
`synthetic(slack)` / `dense(...)`，但**文件名的 `tdrive` 前缀容易被误读成"真实轨迹结果"**。

> 因此 §8 的搁置范围不是"少数依赖 gMission 的实验"，而是**整批 §6 实证结果在真实数据意义上都待重做**。

---

## 3. 数据需求规格

### 3.1 三类实体的最小字段集

| 实体 | 必需字段 | 用途 | 缺失后果 |
|---|---|---|---|
| **worker** | `worker_id`；可选画像（能力/兴趣） | 参与者标识、分组（`aprime/groups.py`） | 无法构造 `Worker.wid` |
| **event / check-in** | `worker_id, timestamp, lon, lat` | 轨迹 → 连续段 → 移动段 → `(origin,dest,duration)` | **E0.1 直接失败**（J1/J2 无从谈起） |
| **task** | `task_id, lon, lat, start_time, end_time`；可选 `required_workers / type` | 时域与空间范围、`Instance.budget` 定标 | 无法定 `horizon_min`；退化为"只有轨迹" |
| **acceptance**（可选） | `task_id, worker_id, timestamp` | 真实的任务—工作匹配、能力分层 | E3.1 只能用合成能力分组 |

gMission 的实际文件名以压缩包为准，**接入时必须在 `manifest.json` 里记录真实列名**，不要凭记忆写死。

### 3.2 时间字段的硬要求（D5 的判定内容）

| 编号 | 要求 | 判定方法 | 阈值 |
|---|---|---|---|
| **T1** | 绝对时间可解析 | 按 `%Y-%m-%d %H:%M:%S` 解析成功率 | ≥ 99.5% |
| **T2** | 无时区歧义 | 全表同一时区、无夏令时；`manifest.tz` 非空 | 必须 |
| **T3** | 单调可排序 | 同 worker 内按时间排序后，负间隔占比 | < 0.1% |
| **T4** | 可对齐时槽 | `slot=(h*60+m)//slot_min` 后，事件时间窗跨度 ≥ 2 slot 的占比 | ≥ 99% |
| **T5** | **跨天不塌陷** | 多天数据不得叠在同一组槽上 | 见下 |

**T5 是已经踩过的坑，必须复刻防护。** `load_tdrive_traces` 里的槽计算是
`slot = (hour*60+minute)//slot_min`——**丢掉了日期**。单天数据没问题，多天数据会把不同天的
同一钟点压进同一个槽，制造虚假的"连续轨迹"。`prep_tdrive.py` 用
`(day*24+hour)*60+minute` 做去重键来规避。gMission 若跨天，`prep_gmission.py` 必须：
1. 去重键带日期；
2. 在 `manifest.json` 与实例 `meta` 里记 `n_days`；
3. 若确实要跨天合并，必须显式写进 `meta["time_collapse"]="by_design"`，否则视为缺陷。

### 3.3 与 `SETTINGS` 的映射

`agentsense/config.py::Setting` 的字段需要逐项由数据实测确定，**禁止沿用 T-Drive 的常数**：

| 字段 | 来源 | 规则 |
|---|---|---|
| `grid_x / grid_y` | bbox + 目标单元格尺寸 | 保留 8×8 / 16×16 / 32×32 三档尺度以便与 T-Drive 可比 |
| `slot_min` | 事件时间粒度 | 建议对齐 15 min；若原始粒度过粗（如仅日期）→ **D5 直接判失败** |
| `horizon_min` | 数据覆盖窗口 | 取分位数内的稳态窗口，不用全量跨度（否则空槽过多） |
| `cells_per_step` | **由 gMission 轨迹实测** | `_cps_source()` 必须返回 `gmission_measured`，不得复用 `tdrive_measured` |
| `total_budget` | 沿用步数口径 40 / 60 / 100 | 与论文表 1 一致，便于可比 |
| `n_workers / n_instances` | 数据规模与实验设计 | `n_instances=20` 沿用 |

### 3.4 一个必须先修的坑：bbox 写死在北京

`_load_tdrive_traces` 与 `tdrive_workers` 的 `bbox` 默认值硬编码为北京
`(116.20, 39.80, 116.55, 40.05)`，且 `Setting` **没有 bbox 字段**。gMission 在广州，
照现状接入的后果是：**所有点被 bbox 静默过滤掉，得到空轨迹，然后回退合成数据而不报错**
（`make_mapus_instance` 的 `except Exception` 会打印一行 stderr 就继续）。

修法（D4 之前必须完成）：
1. 在 `mapus/data/instances.py` 增加 dataset 级 `BBOXES`；
2. gMission 的框由**数据分位数实测**（0.5% / 99.5%）得到，写进 `manifest.json`；
3. 加载后断言 `len(traces) > 0` 且 `n_points > 0`，**空结果必须是异常，不能是回退**。

---

## 4. 获取途径

三条腿并行，任一条通了即可开工：

| 途径 | 动作 | 产出 | 备注 |
|---|---|---|---|
| **P1 官方/课题组** | 查 gMission 论文（Chen et al., KDD'14）的通讯作者与课题组主页，找 dataset 下载页 | 原始压缩包 + 许可 | 最权威；优先 |
| **P2 公开镜像/学术索引** | 数据集索引站、论文补充材料、GitHub 上的复现仓库 | 可能是他人整理的子集 | **必须核对与原文口径是否一致**，否则只能当"同类数据"用 |
| **P3 邮件申请** | 向作者说明：学术用途、不二次分发、需要**时间字段**与**任务有效期** | 授权 + 数据 | 邮件里要明确索要字段清单（§3.1），否则常拿到脱敏版 |

申请信要点（照抄即可）：
1. 研究主题：群智感知中的路线级参与者选择与个性化满意度硬约束；
2. 需要的字段：worker_id / timestamp / 经纬度 / 任务起止时间（**明确强调时间戳是必需的**）；
3. 用途承诺：仅用于学术论文、不再分发、论文中注明来源；
4. 若能提供：请附字段说明文档（codebook）。

**落盘约定**：拿到后原样解压到 `data/gmission/raw/`，**不要手改原始文件**；
所有清洗只写新文件（§5）。

---

## 5. 目录与文件契约

```
data/gmission/
├── raw/                        # 原始压缩包解压结果，只读，永不修改
├── manifest.json               # 来源、许可、列名映射、bbox、时区、量级（见下表）
├── gmission_clean.txt.gz       # 统一格式：worker_id,YYYY-MM-DD HH:MM:SS,lon,lat
├── prep_gmission.py            # raw → clean（幂等、可重跑）
├── audit_gmission.py           # 结构与时间字段审计（D1）
└── audit_gmission_slack.py     # 余量审计（D3，口径同 audit_slack.py）
```

`manifest.json` 必需字段：

| 键 | 含义 | 示例 |
|---|---|---|
| `source_url` / `obtained_at` | 出处与获取日期 | `"https://..."` / `"2026-11-xx"` |
| `license` | 许可或授权方式 | `"academic-use-by-permission"` |
| `files` | 原始文件名 → 行数 → 列名列表 | 用于核对 codebook |
| `field_map` | 原始列名 → 标准字段（§3.1） | `{"checkin_time": "timestamp"}` |
| `tz` | 时区（T2） | `"Asia/Shanghai"` |
| `bbox` | 实测经纬度框（§3.4） | `[113.2, 23.0, 113.5, 23.2]` |
| `n_days` / `n_workers` / `n_events` / `n_tasks` | 量级（T5） | — |
| `time_field_verdict` | **D5 结论**：`pass` / `fail` / `partial` + 一句理由 | `"pass: 99.97% parseable"` |

> `time_field_verdict` 是 E0.1 的**签收字段**。没有它，E0.1 视为未完成。

---

## 6. 接入管线（D0 → D5）

| 检查点 | 动作 | 脚本 / 命令 | 产出 | 失败判据 | 当前状态 |
|---|---|---|---|---|---|
| **D0** 落地 | 原始数据进 `data/gmission/raw/`，写 manifest 骨架 | 人工 + 邮件授权 | `raw/` 非空、`manifest.json` 存在 | 目录为空 → 后续全部无法开始 | **✗ 未过（阻塞）** |
| **D1** 结构审计 | 列名、量级、时间分布、缺失率 | `py -3.14 data/gmission/audit_gmission.py` | 审计报告（stdout + `audit.json`） | T1 不达标或时间字段根本不存在 | 未做 |
| **D2** 清洗 | 统一为 `worker_id,ts,lon,lat`；按 (day,slot) 去重；排序；丢 <3 点轨迹 | `py -3.14 data/gmission/prep_gmission.py` | `gmission_clean.txt.gz` | 输出行数 ≈ 0 或 bbox 内占比 < 99% | 未做 |
| **D3** 余量审计 | 复用 `audit_slack.py` 口径，对比真实 vs 合成 | `py -3.14 data/gmission/audit_gmission_slack.py` | slack 分位表 | `slack ≥ 0.55` 的片段占比过低（J3 不成立） | 未做 |
| **D4** 标定 | 定 `grid / slot_min / horizon / cells_per_step`；**先修 bbox 配置化**（§3.4） | 改 `agentsense/config.py` + `mapus/data/instances.py` | 新增 `SETTINGS[("gmission", scale)]` | 若只能沿用 T-Drive 的 cps → 标为 `proxy(...)` 而非 `measured` | 未做 |
| **D5** 验收 | 跑 §7 断言 + 一次端到端冒烟 | 见下 | `manifest.time_field_verdict="pass"` | 任一硬门不过 | 未做 |

D5 冒烟命令（数据到位后应一次通过）：

```powershell
# 真实 gMission 实例（对标 T-Drive 那条已验证的通路）
py -3.14 -c "import sys; sys.path.insert(0,r'D:\composition\aprime'); import aprime; from mapus.data.instances import make_mapus_instance, data_source_label; i=make_mapus_instance('gmission_small', 0, source='gmission'); print(i.base.meta['source'], len(i.workers), data_source_label('gmission'))"
# 期望输出: gmission 20 gmission(real)
```

参照：T-Drive 的同一命令已实测通过（`meta.source=tdrive`、20 workers、14.3 s）。

---

## 7. 验收标准（可执行断言）

**硬门**（不过则 E0.1 判失败，不得进入 §8 的解冻）：

| 编号 | 断言 | 阈值 |
|---|---|---|
| **A1** | 时间字段解析成功率（T1） | ≥ 99.5% |
| **A2** | `manifest.tz` 已记录且全表一致（T2） | 必须 |
| **A3** | 同 worker 负时间间隔占比（T3） | < 0.1% |
| **A4** | 时间窗跨度 ≥ 2 slot 的事件占比（T4） | ≥ 99% |
| **A5** | 跨天不塌陷（T5）：去重键含日期，`meta.n_days` 已记录 | 必须 |
| **A6** | bbox 内点占比，且 bbox 来自数据分位数而非抄 T-Drive（§3.4） | ≥ 99% |
| **A7** | 能切出长度 ≥ 3 的连续 run；`_trips` 非空 | > 0 |
| **A8** | 端到端冒烟：`make_mapus_instance(source="gmission")` 返回非空 workers，且 `meta.source=="gmission"` | 必须 |
| **A9** | `manifest.time_field_verdict` 已填 `pass` | 必须 |

**软门**（可带条件通过，但必须在论文里写明）：

| 编号 | 断言 | 建议阈值 | 不达标的处置 |
|---|---|---|---|
| **B1** | `slack ≥ 0.55` 的片段占比（J3） | ≥ 60% | 放宽 `slack` 参数并报"余量受限" |
| **B2** | 可用参与者数 ≥ `n_instances × n_workers` | 400 | 降 `n_workers` 或减少 `n_instances`，并记录 |
| **B3** | 清洗幂等：`prep_gmission.py` 重跑逐字节一致 | 必须 | 修排序/浮点格式化 |
| **B4** | `cells_per_step` 为实测值 | 必须 | 用代理值时必须标 `proxy(...)`，且不写"实测" |

---

## 8. 搁置清单（真实数据实验全部暂停）

**搁置原因分三类**：`blocked` = 数据不在磁盘上，跑不了；`deferred` = 数据在，但按本次决定
暂不产出真实数据结果；`ready` = 不依赖真实数据，可继续。

| 实验 | 依赖数据 | 状态 | 说明 / 解冻条件 |
|---|---|---|---|
| **E0.1** 真实时间戳接入与验收 | gMission | **blocked** | 停在 D0；解冻 = D5 全硬门通过 |
| **E1.1** θ 权衡曲线（真实轨迹版） | T-Drive 真实 / gMission | **deferred** | 现版 `run_theta_curve.py` 走合成参与者（§2.3）；解冻 = 真实实例冒烟 + 同 seed 重跑 |
| **E1.2** 大规模密度（500→5000 workers 真实批量） | 真实批量管线 | **blocked** | `dense.py` 明确写着"不要当作规模复现" |
| **E2.1** 界的紧性 | —（抽象实例） | **ready** | `run_bound_tightness.py` 纯抽象，不受影响 |
| **E2.2** 界紧性的补充配置 | —（抽象实例） | **ready** | 同上 |
| **E2.3** α(θ) 真实实例分析 | T-Drive 真实 / gMission | **deferred** | 与 E1.1 同批 |
| **E3.1** 低能力子群（θ 更严） | 真实能力分层 | **deferred** | `config.py` 已留 `per_worker_theta` 接口；需真实能力信号 |
| **§6 表 6-1…6-9 的真实数据复现批** | T-Drive 真实 / gMission | **deferred** | 现有数字均为合成参与者，真实版待重跑 |
| **跨数据集验证（Grab-Posisi 官方）** | Grab 官方轨迹 + POI | **blocked** | 需 Kaggle 授权；POI 未公开 |

**解冻 gate（三条同时满足才可恢复）**：
- **G1** E0.1 的 D5 硬门全过，`manifest.time_field_verdict == "pass"`；
- **G2** `make_mapus_instance(source=<dataset>)` 冒烟返回真实来源标签；
- **G3** 重跑时**同一 seed、同一候选池**，保证"只有数据不同"（与 `baselines_assumptions.md` §3 的公平性要求一致）。

> 状态在代码里由 `aprime/availability.py` 统一判定，`require_available("E0.1")` 会直接抛错并
> 附带解冻条件——防止"某个脚本忘了改默认值就悄悄跑出合成数字"。

---

## 9. 搁置期间可以推进的工作

不被真实数据阻塞、且不会产出"看起来像真实数据结果"的事项：

1. **定理 2 措辞定稿**（纯理论，见 `docs/theory_alpha_min.md` §5.4）；
2. **E2.1/E2.2 界紧性**（抽象实例，`run_bound_tightness.py` 可直接跑）；
3. **回归测试**（`py -3.14 -m pytest tests -q`，合成数据，13 项）；
4. **E0.1 的代码侧准备**（不产出论文数字）：
   - bbox 配置化（§3.4）——**建议优先做**，它是 D2/D4 的前置；
   - `data_source_label()` 增加 `gmission` 分支；
   - `SETTINGS` 预留 `("gmission", scale)` 骨架（`cells_per_step` 标 `proxy`）；
   - `prep_gmission.py` / `audit_gmission.py` 骨架，用 1k 行构造数据自测（**自测数据不得进 `results/`**）；
5. **论文数据可得性声明的措辞定稿**（§2.3 的事实必须在文中出现，不能只写在内部文档里）。

---

## 10. 风险与工时

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| gMission 最终拿不到 | 中 | E0.1 主源缺位 | **退路：用 T-Drive 真实轨迹承担 E0.1**（通路已实测可用），论文写明来源与替代理由 |
| gMission 只有任务、无轨迹/签到 | 中 | 只能做任务分配侧，不能做路线侧 | 降级为 **E0.1b（任务级）**，路线级结论仍走 T-Drive |
| 时间粒度过粗（如仅到日期） | 中 | T4 失败，无法对齐 slot | D5 直接判 fail，不留模糊地带 |
| 多天数据塌陷到同一组槽（T5） | **高**（读取器丢 day） | 伪造"连续轨迹"，结论不可信 | 去重键带日期 + `n_days` 记录 + A5 断言 |
| bbox 写死北京导致空轨迹 | **必然**（若不改） | 静默回退合成数据，无人发现 | §3.4 配置化 + 空结果抛异常 + A6 |
| 许可/二次分发越界 | 低 | 合规问题 | manifest 记 license，不二次分发 |
| 标定沿用 T-Drive 常数 | 中 | "实测"变"代理"，可比性被高估 | B4：代理值必须标 `proxy(...)` |

工时估算（不含等待数据授权）：D1 0.5 人日、D2 1 人日、D3 0.5 人日、D4 1 人日、D5 0.5 人日，
合计 **≈3.5 人日**。其中 D4 的 bbox 配置化可提前做，不依赖数据。

---

## 11. 执行检查表

```
[ ] D0  data/gmission/raw/ 非空；manifest.json 骨架已建
[ ] D1  audit_gmission.py 跑通；T1 解析成功率 ≥ 99.5%
[ ] D1  T2 时区已记录；T3 负间隔 < 0.1%
[ ] D2  prep_gmission.py 产出 gmission_clean.txt.gz；幂等（B3）
[ ] D2  T5 去重键含日期；meta.n_days 已记录
[ ] D3  slack 分布与 T-Drive 对照已出表（B1）
[ ] D4  BBOXES 配置化完成；gMission bbox 由分位数实测（A6）
[ ] D4  SETTINGS[("gmission",*)] 就位；cells_per_step 实测或标 proxy（B4）
[ ] D5  make_mapus_instance(source="gmission") 冒烟通过（A8）
[ ] D5  manifest.time_field_verdict = "pass"（A9）
[ ] 论文数据可得性声明已更新（§2.3 的事实）
```
