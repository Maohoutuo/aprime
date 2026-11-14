# 替代数据调研：Grab-Posisi、稠密快照、CrowdSenSim 虚拟机

> 本文回答三个具体问题：
> 1. 方案里的 **Grab-Posisi** 拿不到，GitHub 上有没有可替代的同类数据？
> 2. 方案要的"**有更稠密快照的数据集**"能不能搞到？
> 3. 从 <https://crowdsensim.gforge.uni.lu/> 下载的**虚拟机**能提供什么数据，能不能用在 A′ 实验中？
>
> 性质：**调研与决策建议**，不含任何实验结果。机读状态仍以 `aprime/availability.py` 为准。
> 撰写日期：2026-11-13。所有"已验证"标记均为当天实测（见附录 A）。

---

## 0. 一页结论

| 问题 | 结论 |
|---|---|
| Grab-Posisi 能拿到吗 | **不能（50 天周期内）**：官方只走邮件授权（`grab.posisi@grabtaxi.com`），无公开直链 |
| 拿到就够用吗 | **不够**：论文里那 13 项道路/POI 属性**没有随数据发布**，而它正是 `S(·)` 需要的料 |
| 有替代吗 | **有，且是拆分替代**：轨迹 → GeoLife + Porto Taxi；属性层 → OSM/OSMnx + Overture Maps；两者都免费可直链下载 |
| "更稠密快照"存在吗 | **已经在盘里**：`data/sc_dense/raw/synthetic_onrta/3000_15000_1_10_0.5_6_10`（84.0% 覆盖 / 平均 1.88 候选 / 单任务最多 10）；要真实稠密 → Foursquare Global Check-in / Gowalla 自建 |
| CrowdSenSim VM 有用吗 | **有用但只限两处**：E1.2/E6.1 的参与者生成器、E0.1 的 J1/J2/J3 与 slack 对照。**不能**做匹配型实验与 `S(·)` 标定，也不能当"真实人群"写进论文 |
| 要启动那个 5 GB 的 VM 吗 | **不需要**：数据是纯文本，随源码包/镜像一起发布（见 §5.4） |

---

## 1. 先把"两个数据集"拆成三件不同的事

方案里"Grab-Posisi + 更稠密的快照"这句话盖住了三件互不相同的需求，它们对应不同实验、能被不同数据满足：

| # | 需求 | 谁在用 | 真正要的是什么 |
|---|---|---|---|
| **A** | 真实稠密 GPS 轨迹 | CROSS（跨数据集验证）、E0.1 的 J2/J3 | 1–15 s 级采样、真实城市、能切 `(origin, dest, duration)` |
| **B** | 道路 / POI 属性层 | 创新点 1、E0.3 的满意度 `S(·)` 物理含义 | 每段路 / 每个 POI 的语义标签（类别、密度、道路等级…） |
| **C** | worker–task 匹配快照的稠密性 | E1.2 / E5.14 / MRG 设计动机 | 每个任务有多个可行候选（`|F_u|` 中位数不是 1） |

现状是：A + B 都押在 Grab-Posisi 上，C 押在"更稠密快照"上。
实测结论是 **A/B 最好补、C 其实已经解决**（见 §3.3）。

---

## 2. Grab-Posisi 的真实状态（2026-11-13 核实）

来源：<https://engineering.grab.com/grab-posisi>（官网正文，已实测可访问）。

| 项 | 事实 |
|---|---|
| 获取方式 | **发邮件到 `grab.posisi@grabtaxi.com`** 申请（姓名/联系方式/单位/用途），非 Kaggle、非公开直链、禁商业用途 |
| 内容 | 84K 条轨迹、80M+ GPS ping、**1 s 采样**、新加坡 + 雅加达、Parquet 约 2 GB |
| 字段 | `trajectory_id` / `lat` / `lon` / `timestamp(UTC)` / `accuracy` / `bearing` / `speed`；按 Car\|Motorcycle、Android\|iOS 打标 |
| **未发布的部分** | **论文里的 13 项道路/POI 属性表不随数据发布**；起终点脱敏；无用户 ID、无任务、无接单记录 |

两个比"拿不到"更严重的后果：

1. **周期不可控。** 邮件授权 + 法务审批，等不起方案 §10 的 R2 检查点（D5）与 §13.5 的分级检查点。
2. **`CROSS` 的解冻条件实际不可满足。** 代码里写的是"拿到官方轨迹 + POI"，但 Grab 从未公开发布属性层——即使拿到轨迹，`S(·)` 的语义属性仍然缺位。

> 因此 `CROSS` 必须**改设计**，而不是继续等数据（改法见 §4）。

### 2.1 顺带纠正两处过期事实

| 位置 | 写的 | 实际 |
|---|---|---|
| `aprime/availability.py` → `DATASETS["grab"].reason` | "需 Kaggle 授权" | 应为"需**邮件**授权，且属性层未公开" |
| `docs/e0_1_data_access_plan.md` §2.2 / §8 → gMission | "本地不存在 → blocked" | `data/gmission/raw/{gMission_cap1, EverySender_cap1}` **已在位**（2026-11-12 落地），实际只差 `manifest.json`（`data/README.md` §6 已记为待办） |

第二条直接影响 D5：E0.1 的真实卡点已经不是"没数据"，而是"没走完 manifest 签收 + bbox 配置化 + 时隙参数化"。

---

## 3. 替代数据集清单

可达性列是**当天实测**的 HTTP 结果（附录 A）；标"未实测"的只做了页面级确认。

### 3.1 替代 A：真实稠密 GPS 轨迹

| # | 数据集 | 规模 / 采样 | 可达性 | 可替代 Grab-Posisi 的哪半边 | 许可 |
|---|---|---|---|---|---|
| **A1** | **GeoLife 1.3**（MSRA） | 182 用户 / 17,621 条轨迹 / **91% 为 1–5 s 或 5–10 m 稠密采样** / 1.2M km / 2007-04–2012-08 北京 / 298.7 MB | **HTTP 200（直链）** | **稠密快照这一半，最强**；自带 11 类出行方式标签（walk/bus/car/subway/…），可替代 mode detection 用途 | MS 学术使用 |
| **A2** | **Porto Taxi**（ECML/PKDD 2015，UCI #339） | 442 辆出租 / 1.7M 行程 / **15 s 采样** / POLYLINE 全轨迹 | **HTTP 200（UCI 静态 ZIP）** | **"叫车场景"这一半**：`CALL_TYPE`（A/B/C 三种叫车渠道）、`TIMESTAMP`、`ORIGIN/DEST`、`MISSING_DATA`，语义上最接近 worker–task | 公开研究用 |
| **A3** | **CrowdSenSim 卢森堡**（本次下载的 VM） | 2000 agents / 1 天 / **分钟级** / OSM 路网 | 已在本地 | 真实路网 + 稠密位置流；**人群是模拟的**（见 §5） | GPLv3（代码） |
| A4 | **T-Drive** | 8,560 车 / 214 万点 / 15 min | 已在库 | 已作为真实轨迹主源（通路实测可用） | 需引用 Yuan et al. 2010 |
| A5 | DiDi GAIA / 成都 DiDi | OD 级 | 申请制 | 只能补 OD，补不了快照 | 申请制 |

A1 / A2 直链（均实测 200）：

```powershell
# A1 GeoLife（298.7 MB）
curl.exe -L -o "D:\composition\data\geolife\Geolife Trajectories 1.3.zip" `
  "https://download.microsoft.com/download/f/4/8/f4894aa5-fdbc-481e-9285-d5f8c4c4f039/Geolife%20Trajectories%201.3.zip"

# A2 Porto Taxi（UCI #339）
curl.exe -L -o "D:\composition\data\porto\uci339.zip" `
  "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip"
```

> 注意：GitHub 上以 Grab-Posisi 命名的仓库（`Tayzhiyi/Grab-Posisi`、`rehohoho/grab_posisi_datathon`、`danyelys/Grab-Posisi-Dataset` 等）**只含分析与爬虫，没有数据本体**，不能当数据源。

### 3.2 替代 B：道路 / POI 属性层（Grab-Posisi 那 13 项属性的重建）

这一层是真正"不可替代但可以重建"的部分，而且**重建版本比私有表更可复现**：

| # | 来源 | 可达性 | 能重建的属性 |
|---|---|---|---|
| **B1** | **OSM + OSMnx** | 免费，`pip install osmnx` | 道路等级、单/双向、车道数、`maxspeed`、交叉口密度、人行道/自行车道、路段长度 → 覆盖 Grab-Posisi 属性表里的**全部物理项** |
| **B2** | **Overture Maps Places** | **HTTP 200**（S3 上的 GeoParquet，全球 1 亿+ POI，含类别/品牌/置信度） | POI 密度、类别占比（植被/工业/教育/居住/治安…），正好对上方案 §4.1 E0.3 与 §7.7 的"城市属性数据"科目 |
| B3 | Foursquare OS Places | HuggingFace 端点实测 **401**，托管位置需再确认 | 同 B2（备选） |
| B4 | Mapillary | 未实测 | 绿视率、隔离带、路面状态（需 CV，50 天内不做） |

**这条路线的三个好处**：
1. 属性是**可计算、可复现**的，能写进论文附录，比"从 Grab 拿一份不公开的表"更好答辩；
2. CrowdSenSim 本身就用 OSMnx 建图（A3），两套数据口径天然统一；
3. 完全免费、无需授权。

### 3.3 替代 C："更稠密的匹配快照"（已经解决）

`data/README.md` 已量化过，此处复核：

| # | 数据 | 状态 | 覆盖率 / 平均候选 / 单任务最多 |
|---|---|---|---|
| **C1** | **`synthetic_onrta/3000_15000_1_10_0.5_6_10`** | **已在库** | 84.0% / **1.88** / **10** ← 现成可用的"更稠密快照"，也是唯一撑得起 MRG 设计动机的实例 |
| C2 | ONRTA `data/scaleData/syn1/10000_50000_...` | 同仓库未下载（2.7 MB） | 直接补规模 |
| C3 | `genDataSynthetic.py`（GOMA / ONRTA 自带） | 已在库可改 | **密度是参数**：改 `rwList`/`ddlList` 即可造任意平均度数 |
| **C4** | **Foursquare Global Check-in**（Yang et al., TIST 2015） | 页面可访问，775 MB zip | 33,278,683 check-ins / 266,909 用户 / 3,680,126 venue，**含细粒度 venue 类别** → 按 E0.1 映射规则（worker=用户、task=venue、可行=半径+时间窗）造**真实稠密**实例，同时补 B 层语义 |
| **C5** | **Gowalla**（SNAP） | **HTTP 200（直链）** | 6.4M check-ins / 196,591 用户 / 1.28M places（含类别） |
| C6 | gMission / EverySender / Didi | 已在库 | 都**稀**：中位 `|F_u|` = 1，56–71% 的人只有 1 个选项 |

诚实表述：**"真实 + 稠密"的匹配快照在公开数据里基本不存在**（`data/README.md` §0 的实测结论）。
所以论文里应把密度当**自变量**扫（E1.2 的 density stress），而不是找一份稠密真实数据当主实验。

```bash
# C5 Gowalla（实测 200）
curl.exe -L -o gowalla_totalCheckins.txt.gz https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
```

---

## 4. 推荐的"B 计划"数据方案

| 实验 | 原定 | 改为 | 理由 |
|---|---|---|---|
| **CROSS** 跨数据集验证 | Grab-Posisi 官方轨迹 + POI | **GeoLife（A1）+ Porto（A2）** 两份；属性层按 Huang et al. 定义在 **OSM/OSMnx + Overture（B1/B2）** 上重建 | 明天就能下；属性可复现；不再依赖邮件 |
| **E0.1** 真实时间戳接入 | gMission | **T-Drive（通路已实测）+ CrowdSenSim 卢森堡（A3）** 作第二份真实路网轨迹；gMission 原始其实已在盘，只差 `manifest.json` | 见 §2.1 |
| **E1.2 / E6.1** 规模 500→5,000 | 真实批量管线 | **CrowdSenSim 作参与者生成器**（`Number of users` 可配）+ ONRTA `scaleData` | 见 §5.2，这是 VM 最实在的用处 |
| 稠密匹配主实验 | "更稠密快照" | **ONRTA（C1）主用**，**Foursquare / Gowalla（C4/C5）造的真实稠密实例做 cross-check** | 合成撑机制、真实撑外部效度 |

**论文里要加"数据可得性声明"一段**，写清三件事：

1. Grab-Posisi 需邮件授权，且其道路/POI 属性层未公开发布；
2. 本文按同构定义在 OSM + Overture 上重建属性层（给出可复现脚本）；
3. 稠密匹配快照以「合成族 + 真实签到族」双轨报告。

这一段是**加分项**：审稿人最反感"用了一份没人能复现的私有数据"。

---

## 5. CrowdSenSim 虚拟机：里面是什么，能不能用

### 5.1 它是什么

| 项 | 事实 |
|---|---|
| 站点 | <https://crowdsensim.gforge.uni.lu/>（卢森堡大学） |
| 定位 | **移动群智感知的离散事件仿真器**，GPLv3；支持数据采集、任务分配、监控、资源管理 |
| 下载物 | `CrowdSensim-VM.ova`（5,014,681,600 B，2018-01-30 构建） |
| OVA 内容 | `CrowdSensim-VM.ovf`（9.4 KB）+ `CrowdSensim-VM-disk001.vmdk`（**20 GiB，streamOptimized**） |
| 客户机 | **Ubuntu 16.04**、1 vCPU、4096 MB RAM、VirtualBox 5.1（Guest Additions 5.1.26）、NAT 网络 |
| 软件 | **CrowdSenSim 1.2.0**：C++ 引擎（`Simulation.cc` 110 KB + `Event/Statistics/HeatMap`）+ Python 菜单 + HTML 统计页 |
| 本质 | **是仿真器，不是数据集包** |

### 5.2 它到底提供哪些数据

来源：官方站点文档、站点上的 `Inputs/README.txt` 系列，以及 GitHub 上的一份源码镜像
[`prguin/CrowdSensim_simulator`](https://github.com/prguin/CrowdSensim_simulator)（其 `simulator/data/Inputs/`），
并与引擎里写死读路径的 `ReadValues.cc` 逐条对齐。

| 文件（VM 内 `simulator/data/Inputs/…`） | 内容 | 实测 / 文档量级 |
|---|---|---|
| `Default/Mobility/UserMovementsListEventsDefault.txt` | **每个 agent 的逐位置事件**：`ID-User Lat Long Alt Day Hour Minute` | 5.5 MB，约 13 万事件；**2000 用户 / 1 天 / 卢森堡市 / 06:23–20:56**，坐标 49.61 N / 6.10 E，ray 250 m，行程 20–40 min |
| `Default/Mobility/UserMovementsListEvents.txt` | 同格式的另一份 | 5.5 MB；坐标 42.14 N / 12.50 E → **这是罗马，不是卢森堡** |
| `MapPointsCenter.txt` | 城市内可行位置点（OSM 路网节点，`Lat Long Alt`） | 随机游走的图节点 |
| `CoordinatesAntennasLuxembourg.txt` / `CoordinatesAntennas.txt` | 3G/WiFi 基站坐标（`ID-Antenna Lat Long`） | 11 KB / 26 B |
| `hours.txt` / `perc_connections` / `durations` | 每小时连接比例、会话时长分布 | 24 行 / 24 行 / 1 行 |
| `Setup.txt`（及 `Setupddf/Setuppda/Setuppcs.txt`） | 仿真参数：天数、用户数、行程时长上下界、起止时刻、基站类型、ray、是否用真实电量轨迹、采集框架（DDF/PDA/PCS） | 每份约 1.3–1.6 KB |
| `energy-traces.zip`（站点单独发布） | **真实手机耗电曲线** | 独立压缩包 |
| 输出 | `Statistics.txt` / `Statistics.html`、`day_0.html`（覆盖热图）、`route_usr_1day_0.html`（单用户路径） | 覆盖点数、电量、邻居接触 |

默认场景（官方 `DefaultConfigurationDetails.txt`）：**Luxembourg，2000 users，3G，06:23–20:56，1 天，ray 250 m**。

实测行样例（取自默认文件，注意它确实是分钟级稠密快照）：

```
/ID-User/-/Lat/-/Long/-/Alt/-/Day/-/Hour/-/Minute/-
1 49.6128730128 6.10936355371 0 0 7 40
1 49.6126731 6.1073639 0 0 7 42
1 49.6128502 6.106991  0 0 7 42
```

### 5.3 能派上用场吗：按 A′ 的判据逐条对

| A′ 的需求 | 能否满足 | 说明 |
|---|---|---|
| **J1 绝对时间可解析** | ✅ | Day/Hour/Minute 齐全，可拼绝对时间（需自补基准日期） |
| **J2 轨迹可切段** | ✅ | 每人是一条连续位置序列，`_contiguous_runs` + `_trips` 直接可用 |
| **J3 时间窗余量（slack）** | ✅ 且比 T-Drive 更易测 | 分钟级 + 单城市内圈，可作 `data/tdrive/audit_slack.py` 口径的**第三组真实路网对照** |
| 候选路线生成（E0.2） | ✅ | 路网来自 OSM，`(origin, dest, duration)` 质量优于当前合成的 `tdrive_workers` |
| **规模实验 500→5,000（E1.2 / E6.1）** | ✅ **最有价值的一项** | `Setup.txt` 的 `|Number of users|` 是参数，可生成任意数量 agent + 真实时隙 → 正是 §8 中 `blocked` 的"批量管线"最省事的替代 |
| 能力分层（E3.1） | ⚠️ 部分 | 无 `rate`/能力字段；但 **`energy-traces.zip` 可作"电量预算"这一维的真实能力代理**（比纯合成能力强） |
| **worker–task 匹配（E1.3 / 主对比表）** | ❌ | **没有任务、没有报酬、没有接单/拒绝记录**，无法构造 `Worker–Task` 可行边 |
| **`S(·)` 的物理含义（E0.3）** | ❌ | **没有 POI 语义、没有道路属性**，只有坐标；属性仍须走 B1/B2 |
| 当"真实人群轨迹" | ❌ | mobility 是 **OSM 图上的随机游走**（ray 250 m，行程 20–40 min），并非真实人类移动 |

### 5.4 三个必须注意的坑

1. **不能写成"真实轨迹"。** 它和 T-Drive 是两类东西：T-Drive 是真人真车；CrowdSenSim 是"真实路网 + 模拟 agent"。混着写会被审稿人抓。
2. **`Day` 字段出现 `0` 与 `7` 混用**（同一用户同一天既有 day=0 又有 day=7）。这正是 `docs/e0_1_data_access_plan.md` §3.2 的 **T5 跨天塌陷**风险：接进来必须"去重键带日期 + 记 `n_days`"，否则会伪造"连续轨迹"。
3. **2018 年的老环境**：Ubuntu 16.04 + VirtualBox 5.1（本机 VBox 为新版），首次导入可能要重装 Guest Additions；1 vCPU / 4 GB 跑 5000 agents 会很慢。

### 5.5 实操：不需要启动这个 VM

数据是纯文本，且随源码包一起发布（源码 zip 的 `simulator/data/Inputs/` 就是全套）。三条路，按推荐度排序：

1. **源码镜像（推荐）**：`https://github.com/prguin/CrowdSensim_simulator` → `simulator/data/Inputs/`。GitHub 直连本机可用（实测 200）。
2. **官方源码包**：`https://crowdsensim.gforge.uni.lu/ftp/CrowdSenSim2.0.zip` 或 `CrowdSenSim_1.1.1.zip`。**但本机直连 uni.lu 的 80/443 均不通**（实测连接超时，只能通过外部代理抓网页），需换网络或让他人代下。
3. **真读 VM 里原件**：`VBoxManage clonemedium disk CrowdSensim-VM-disk001.vmdk out.raw --format RAW`（**本机实测可跑通**，产出 20 GiB raw），再到 Linux 上 `mount`。**但本机 WSL 未装好**（`wsl --status` 报 `WSL/CallMsi/Install/REGDB_E_CLASSNOTREG`），所以这条路目前走不通。

### 5.6 价值评级

| 用途 | 评级 | 理由 |
|---|---|---|
| E1.2 / E6.1 的参与者生成器（500→5000 agents，真实时隙 + 真实路网） | **值得做** | 直接解 `blocked`，且不牺牲时隙真实性 |
| J1/J2/J3 通路与 slack 审计的第三组对照 | **值得做**（约半天） | 分钟级、单城市、结构干净 |
| E3.1 的"电量能力"维度 | 可选 | 给公平性加一个真实异质性来源 |
| 主对比表 / 匹配型实验 / `S(·)` 标定 | **不能用** | 缺任务与语义 |
| 作为"真实人群"证据 | **不能用** | mobility 是模拟的 |

---

## 6. 落地改动清单（约 1.5–2 人日）

1. `aprime/availability.py`
   - 新增 `geolife` / `porto` / `crowdsensim` 三个 `Dataset`（`probe` 用文件路径，风格照 `_probe_tdrive`）；
   - `grab.reason` 改为"需邮件授权，且道路/POI 属性层未公开"；
   - `CROSS.unblock` 从"拿到官方轨迹 + POI"改为"OSM/Overture 属性重建管线可复现 + GeoLife/Porto 落盘"。
2. `data/README.md`
   - 新增一节"§8 替代数据与属性层重建"，收录 A/B/C 三层、实测可达性、许可与引用要点（延续现有文档风格：一页速查表 + 坑 + 怎么读）。
3. `data/` 新增目录
   - `data/geolife/`、`data/porto/`：各配 `load_*.py`，字段映射与 `load_sc.py` 同一套契约（`load_instance` / `feasible_pairs` / `density`）；
   - `data/crowdsensim/`：`UserMovementsListEventsDefault.txt` → `worker_id,ts,lon,lat`，**去重键带日期**，`meta.n_days` 必填。
4. `aprime/experiments/preflight_gmission.py`
   - 第 97、101 行硬编码的 `t.t_arrive // 300` 参数化（`data/README.md` §6 已记为待办）；否则 A3 的分钟级数据会被压成 1–2 个时隙，时间维度消失。
5. `data/gmission/manifest.json`
   - 补 `time_field_verdict`，解除 E0.1 的**假阻塞**（原始快照其实早已在位）。

---

## 7. 结论

1. **Grab-Posisi 应当放弃**，改用"**轨迹（GeoLife + Porto）+ 属性（OSM/Overture 重建）**"的双轨替代；`CROSS` 的解冻条件必须改写成"属性重建管线可复现"。
2. **"更稠密快照"不是要找的数据集，而是已经在库的 `synthetic_onrta`**；真实稠密实例用 Foursquare / Gowalla 自建；密度本身应作为 E1.2 的自变量扫描。
3. **CrowdSenSim VM 是 P1 级补充源**：只用在"参与者生成器"和"J1/J2/J3 + slack 对照"两处，且**无需启动 VM**（数据是纯文本，随源码包发布）。

---

## 8. 实测回填（2026-11-14）：三条纠正 + 一条新增坑

本节记录按 §6 清单动手后的实测结果，其中三条**修正了本文前面的判断**。

### 8.1 GeoLife 可以直接用，但必须先重采样（本文 §3.1 没提）

GeoLife 的"1–5 s 稠密采样"是卖点，但对**本工程的管线**恰恰是陷阱：
`load_tdrive_traces` 不做 slot 去重，而 `_contiguous_runs` 要求相邻观测的 slot 差**恰为 1**。
原样喂进去，同一 15 分钟槽内会产生成百上千个重复 slot——实测 **99.4% 重复**，
363 万点只切出 59 段、**15 条**出行段，连 20 个参与者都凑不齐。

**处置**：`data/geolife/normalize_geolife.py` 按 `(tid, slot)` 去重、每槽保留槽末一点。
产出 **98,709 行 / 10,055 条（用户×日）轨迹**，加载 **0.8 s**（原样 460 s），
三个尺度分别切出 11,317 / 13,880 / 15,245 条出行段。

**第二个坑（本文 §5.4 只对 CrowdSenSim 提过）**：`tid` 必须带日期。下游
`slot = (hour*60+minute)//slot_min` **不含日分量**，GeoLife 跨 2007-04…2012-08 五年，
按用户 ID 分组会把不同天的同一时刻压进同一个 slot。已用 `tid = <用户>_<yyyymmdd>` 规避。

### 8.2 Porto 本轮拿不到（本文 §3.1 标的是"HTTP 200 直链"）

直链确实返回 200，但**吞吐不可用**：实测 **19.4 KB/s**，下载到 15.29 MB 时
`curl: (18) transfer closed with outstanding read data remaining` 中断；
再试 `-C -` 续传返回 **exit 33（服务端不支持 Range）**，只能从头再来。
按此速率，几百 MB 的包需要数小时，本轮放弃。

**结论修正**：Porto 的"可达"是**HTTP 层可达**，不是**工程上可获得**。
若后续要用，需换网络环境或走代理；在此之前，"轨迹 + 叫车场景"双轨里只有 GeoLife 这一轨落地。

### 8.3 E0.1 的卡点不只是"手续没走完"，有几条是结构性不可满足

本文 §2.1 写"E0.1 的真实卡点已经不是'没数据'，而是'没走完 manifest 签收 + bbox 配置化 + 时隙参数化'"。
**时隙参数化已过时**（`preflight_gmission.py` 现有 `--slot-sec`，缺省按任务时间窗中位数自动推导，
gMission→300 s / EverySender→600 s 均正确）。但更重要的是：跑完 D1 审计后发现，
**gMission/EverySender 有三条硬门是数据本身决定不可过的**：

| 硬门 | 实测 | 为什么不可满足 |
|---|---|---|
| A2 `tz` | 无 | 数据未发布时区，也**没有纪元**；`begTime` 只是整数秒偏移（0…529147 s） |
| A5 跨天不塌陷 | 无 | **没有日期字段**，`n_days` 无法确定（只能报 `span_days=6.12`） |
| A6 地理 bbox | 无 | 坐标是 **0–5 归一化平面**（x∈[0.0005,4.9986]），不是经纬度 |

另有 A7/A8 **不适用**：gMission/EverySender 是 worker–task **匹配快照**，不是轨迹，
`_trips` 与 `make_mapus_instance(source='gmission')` 都不该在它上面成立。

**处置**：写了 `data/gmission/manifest.json`，`time_field_verdict` 如实标为
**`partial`**（A1 通过：100% 整数秒可解析；A4 分裂：gMission 0% / EverySender 100%），
并把 A2/A5/A6 的结构性不可满足写进 `open_items`。
同时把 `_probe_gmission()` 收紧为**校验 verdict 以 `pass` 开头**——
否则"写个 manifest 骨架"就会误触发 E0.1 解冻。E0.1 仍 blocked，但理由已精确。

### 8.4 属性层（§3.2 的 B1/B2）**已建成**——A′ 不再有任何"拿不到 Grab"的依赖

**没有用 osmnx，也没有用 Overture**，改走**纯 OSM + 纯 requests** 直连 Overpass。
理由是 B1 的几何运算在这里其实用不上：我们只需要"按经纬度网格分桶计数"，
而 osmnx 会拖进 geopandas / shapely / pyproj 一整串重依赖（Python 3.14 上
wheel 可用性存疑）；Overture 的 GeoParquet 又需要 pyarrow/duckdb + S3。

产出（`data/osm_attrs/attrs.csv.gz`，1344 行 / 35 列）：

| 成分 | 实测规模 |
|---|---|
| 道路 | 69,247 条 → 段数、总长、交叉口数、车道/限速均值、六档等级构成 |
| 用地多边形 | 15,561 个 → 六类**面积**占比 |
| 兴趣点 + 建筑 | 85,823 条 → 六类**个数**占比、建筑密度 |

**一个把结果从"错"救回"对"的关键实现细节**：用地面积必须**跨格切分**。
最初用"多边形质心归格"，实测让单格累加出**自身面积 8.2 倍**的多边形面积，
把市中心占比彻底扭曲。改为 128×128 细网格采样近似交面积后，溢出降到 2.6 倍以内
（残余来自 OSM 多边形嵌套，是数据性质而非 bug）。

**两条外部合理性检查**（都独立于本工程）：

1. 16×16 下主导类为住宅 158 格 / 商业 51 / 植被 28 / 机构 16 / **工业仅 3** 格，
   与 `data/context/` 北京年鉴"第二产业就业 16.4%"的去工业化格局一致；
2. 北海公园所在格植被面积占比 0.78，与该公园实测 67.9 万 m² / 格面积 80.8 万 m² 吻合。

**仍然诚实的保留项**：犯罪层是合成的（北京无公开细粒度犯罪数据），
故 `urban_note` 一律标为 `landuse=osm(实测), crime=synthetic`。
论文里不能因为用地变真就宣称"城市属性全真"。

**状态变化**：`osm_attrs=ok`，`CROSS` 由 `blocked` → **`deferred`**（数据齐，
只剩 `SHELVE_REAL_DATA` 这道全局闸门）。

### 8.5 状态板现状

```
数据集：tdrive=ok, gmission=missing(partial), onrta_didi=ok, grab=missing,
        geolife=ok, onrta_synthetic=ok, osm_attrs=ok
```

`grab=missing` 仍在表上，但**已不再是任何实验的依赖**——它只是"官方数据确实拿不到"的记录。
`CROSS` 的数据侧已全部就位，状态为 `deferred`（被全局闸门 `SHELVE_REAL_DATA` 搁置），
不再是 `blocked`。

### 8.6 θ 的尺度口径：一个会毁掉全部跨数据集对比的坑（已修）

**问题。** θ 的合法条件原本是 `S_u(r) ≥ θ`——θ 是一个**绝对**满意度下限。
换成实测数据后满意度整体下移（合成 → OSM 实测下移约 0.17–0.26）且离散变宽，
于是**同一个 θ 的严格程度完全不同**：

| θ=0.6 时合法人数 | 全合成 | 仅OSM用地 | 两者均实测 |
|---|---|---|---|
| small 尺度 | 59 | 36 | 30 |
| medium 尺度 | 60 | 39 | 30 附近 |

表面上"θ=0.6"是同一实验条件，实际约束强度差了一倍。**任何把不同数据源
放在同一张 θ 曲线上的对比都不成立**。

**修复。** `APrimeConfig.theta_mode` 新增 `"relative"` 口径：
`S_u(r) ≥ θ · max_r S_u(r)`，即"达到本人最佳满意度的 θ 倍"。θ 变成无量纲比例，
对满意度的整体平移/缩放不变。默认仍是 `"absolute"`，历史实验逐位不变。

**验证（同一批实例重跑）。** 相对口径下合法池对 θ 变成**平的**，
不再随数据源塌陷——这正是"尺度已被剔除"的直接证据：

| θ=0.6 时合法人数 | 全合成 | 仅OSM用地 | 两者均实测 |
|---|---|---|---|
| small | 60 | 59 | 57 |
| medium | 60 | 60 | 60 |

**修复后结论的变化（必须一起记）：**

* **核心结论变强**：用地压低 PSS 的效应从 -0.156/-0.150（z≈3.0–3.4）
  放大到 **-0.267/-0.186（z=7.2/3.7）**。原因是绝对口径下"人数流失"
  把一部分效应吸收掉了；剔除后效应更干净。
* **一条结论被打掉**：仅换用地对公平差距（`gqc_gap`）的影响，
  在 medium 上 z 从 3.3 掉到 **1.8**，不再显著。这条**不能写进论文**。
* **覆盖度依旧不可用**：方向在两个尺度间翻转。

**注意相对口径的行为变化**：由于 `θ·best_u ≤ best_u`（θ≤1），
**每个候选池非空的工作者在 θ≤1 时必然仍有可行解**——θ 不再造成"参与者掉队"。
这让它天然干净，但也意味着**命题 4 里关于 dropout 的讨论只在绝对口径下有意义**，
两套口径的结论不能混用。

**复现命令：**

```powershell
$env:APRIME_ALLOW_SHELVED = '1'
py -3.14 experiments/run_cross_trial.py --scale small  --theta-mode relative
py -3.14 experiments/run_cross_trial.py --scale medium --theta-mode relative
py -3.14 experiments/compare_cross_scales.py `
    --files results/cross_trial_small60_relative.json,results/cross_trial_medium60_relative.json
```

### 8.7 `gqc_gap` 在基线上是退化的——之前"实测数据抬高公平差距"的结论作废

**结论先说：这条结论是假的，不能进论文。** 拆解见 `experiments/diagnose_gqc_gap.py`。

**为什么。** `gqc_gap` 比较"选中份额"与"人数份额"。但能力分组按**每人合法路线条数**
切三档（1 条 / 2 条 / 3+ 条）。当 θ 较松时，**所有人都有 3 条以上合法路线**，
全部落进 `ample` 一档——只剩**一个组**。此时人数份额与选中份额双方都是 100%，
`gqc_gap ≡ 0`，**恒等于零且不携带任何信息**。

于是"全合成基线 gap≈0、换实测用地后 gap>0"并不是"公平性变差"，
而是"基线分组塌成一档、实测数据才让分组分化"。这是一个**基线退化伪影**。

**实测证据（60 人，5 种子 × θ∈{0,0.3,0.6}）：**

| 口径 / 分组 | small 全合成 | small 仅OSM | medium 全合成 | medium 仅OSM |
|---|---|---|---|---|
| absolute / value | 0（**1 组**） | 0.117（3 组） | 0（**1 组**） | 0.033（3 组） |
| relative / value | 0（**1 组**） | 0.116（3 组） | 0（**1 组**） | 0（**1 组**） |
| relative / quantile_tied | 0.250（2 组） | 0.148（3 组） | 0.133（2 组） | 0.200（2 组） |

前两行是退化格（基线恒为 0）。唯一两边都非退化的组合（最后一行）
**方向翻转**：small 上 OSM 让差距下降 0.10，medium 上让它上升 0.07。

**进一步拆解**（`floor` = 整数选取本身造成的不可消除下限，`excess = gap − floor`）：
absolute/value 下 Δgap 几乎全部由 **Δfloor** 贡献（+0.064 / +0.033，Δexcess 仅 +0.033 / 0.000）——
即差距上升来自"组规模更偏斜导致整数分配凑不齐"，而不是"算法更不公平"。

**该怎么做。**

* 比较两组数据源的 `gqc_gap` 前，**必须对两边都检查组数**；
  `aprime.groups.grouping_is_degenerate()` 就是为此加的护栏。
* 基线退化时不要报 `gqc_gap`，改报**组份额本身**（`group_shares`），
  或换用 `quantile_tied` 分组（该方案退化程度较低，但仍须检查方向稳定性）。
* 论文里凡是用 `gqc_gap` 论证"公平性随 X 变化"的地方，都要重新过一遍这个检查。

**复现：**

```powershell
$env:APRIME_ALLOW_SHELVED = '1'
py -3.14 experiments/diagnose_gqc_gap.py --theta-mode absolute --scheme value
py -3.14 experiments/diagnose_gqc_gap.py --theta-mode relative --scheme quantile_tied
```

### 8.8 全量审计：`gqc_gap` 的论证普遍建立在退化/不可判读的格子上

按 §8.7 的判据把**所有**用 `gqc_gap` 论证"公平性随某因素变化"的实验扫了一遍
（`experiments/audit_grouping_degeneracy.py`，只统计分组、不调用求解器）：

| 实验 | 方案 | 论证的因素 | 格数 | 退化 | 不可判读 | 可用 |
|---|---|---|---|---|---|---|
| E5.15 偏好混合敏感性 | value | 差距是否随偏好先验移动 | 3 | 0 | **3** | 0 |
| 公平-覆盖权衡（**Fig.3 / E3.1/E3.2 主表**） | value | 差距是否随覆盖要求移动 | 20 | 0 | **20** | 0 |
| GQC-MRG 主实验 | quantile_tied | 差距是否随 MRG/同质化移动 | 40 | 0 | 12 | 28 |
| GQC-MRG 敏感性 | quantile_tied | 差距对参数的敏感性 | 40 | 0 | 12 | 28 |
| GQC-tied 变体 | quantile_tied | tied 桶相对 value 的改进 | 20 | 0 | 4 | 16 |
| GQC-tied 变体 | value | （对照组） | 20 | **5** | 15 | 0 |
| 预算标定 | quantile_tied | 下界随预算档位移移动 | 8 | 0 | 2 | 6 |

**没有一个实验是全干净的。** 两种严重程度要分开看：

* **退化**（`gqc_gap ≡ 0`，致命）：只出现在 `run_gqc_tied.py` 的 value 对照组
  （medium θ≤0.4 共 5 格）。该实验**本来就**用这个对照组展示 value 桶失效，
  所以是预期内的——但确认了失效区确实存在。
* **不可判读**（最小组 < 15%）：普遍存在。最严重的是
  **`run_fairness_tradeoff.py` 的 20/20 格全部不可判读**，最小组份额低到
  **2.5%**（40 人里的 1 个人）。而它正是论文 Fig.3 与 E3.1/E3.2 主表的数据来源，
  且用的是 `GQCParams()` 的默认 `value` 桶（脚本里没显式指定方案）。

**一个新发现（E5.2 没覆盖）**：高同质化（φ=0.95）下连 `quantile_tied` 也救不了。
偏好被同质化后所有人能力趋同 → 大量并列 → tied 桶把绝大多数人压进同一档，
最小组只剩 **2.5%**（`GQC-MRG` 的 medium φ=0.95 格全线如此）。
E5.2 记录的"medium θ=0 低组 17%"是 φ=0 的情形，φ=0.95 下要差一个数量级。

**已做的修复**：`run_fairness_tradeoff.py` 现在在每行结果与两张表里
**强制输出 `组数` 与 `最低组份额`**（低于 15% 直接标 **不可判读**、单组标 **退化**），
让这个问题不能再次隐身。`grouping_is_degenerate()` 可供其他脚本复用。

**待定（需要作者决策）**：Fig.3 的公平面板是否继续用 `gqc_gap`。
若要保留，至少必须：① 显式换 `quantile_tied`；② 把 φ=0.95 的格子
按"不可判读"标注或剔除；③ 在正文说明低组规模，否则 2.5% 的组上算出的
份额偏差无法支撑任何公平性论断。

### 8.9 两种修法哪个更可靠：**连续度量略优，但真正决定可靠性的是选取规模**

§8.8 留下二选一：**A** 保留 `gqc_gap` 但换 `quantile_tied` + 标注/剔除；
**B** 换成不依赖分桶的度量。这不能靠直觉定，用跨种子变异系数（CV = 标准差/均值）
在同一批实例上横向比（`experiments/compare_fairness_metrics.py`，8 种子/格）。

**绝对标准差（份额点，越小越稳）——按选取条数分层：**

| 选取条数 | `gqc_gap`(value) | `gqc_gap`(tied) | `gqc_excess` | `ks_capability` | `\|rank_shift\|` |
|---|---|---|---|---|---|
| n_sel ≤ 6（默认预算） | 0.027\* | 0.098 | 0.101 | 0.094 | **0.056** |
| 7 ≤ n_sel ≤ 12 | 0.027\* | 0.080 | 0.082 | 0.072 | **0.039** |
| n_sel ≥ 13 | 0.013\* | 0.044 | 0.046 | 0.040 | **0.022** |

\* **陷阱**：`gqc_gap(value)` 的标准差最低，是因为它**卡在 0 附近不动**
（medium θ=0 时均值 0.000，即 §8.8 的退化），不是因为稳定。
低方差 ≠ 可靠，退化度量的方差天然小。

**CV（相对离散度，越小越稳）：** 默认预算下 `ks_capability` 0.65 <
`gqc_gap(value)` 0.71 < `gqc_gap(tied)` 0.79 < `|rank_shift|` 0.82
< `gqc_excess` **1.49**；抬到 n_sel ≥ 13 后依次降到 0.47 / 0.59 / 0.64 / 0.61 / 0.89。

**三条结论：**

1. **B 略优于 A**：连续度量（`|rank_shift|`、`ks_capability`）在所有分层上
   都比修好的 `gqc_gap(tied)` 稳 10–40%，且**彻底避开分桶退化**——不需要再
   担心组数塌成一档。这是 B 的主要价值：不是更准，而是**不会失效**。
2. **但没有任何度量在默认预算下算"可靠"**。n_sel ≤ 6 时
   全部度量的跨种子 SD 在 0.05–0.10 个份额点，CV 0.6–1.5。
   换句话说：**换个种子，这个数会跳掉自身量级的一半以上**。
   任何小于 ~0.1 的"公平性差异"在这个规模下都无法与噪声区分。
3. **主导变量是选取规模，不是度量选择。** 把 n_sel 从 ≤6 抬到 ≥13，
   所有度量的 SD 近似**减半**（0.098→0.044）；而换度量最多改善 40%。
   根因很简单：公平性在组层面是用 5 条左右的选取去估份额，
   样本量就那么大，换个统计量救不回来。

**关于 `gqc_excess` 的警告**：扣掉整数下限在理论上更"干净"，实测却**最差**
（CV 1.49）。原因是从两个都有噪声的量里相减，噪声不减反增。
**不要用这个量做跨条件比较。**

**推荐做法（按性价比排序）：**

1. 用 `|rank_shift|` 或 `ks_capability` 替代 `gqc_gap`——避开退化，且更稳。
2. **同时**把预算抬到 x3.5（n_sel ≈ 15–17）——单项收益最大。
3. 种子数 ≥ 30，报**均值 ± 置信区间**，不报单种子数值。
4. 任何仍要用 `gqc_gap` 的地方，先过 `grouping_is_degenerate` 与
   `min_share ≥ 0.15` 两道闸。

**落地状态（已实施）。**

* `aprime.analysis.fairness` 新增 `ks_capability` / `rank_shift` / `bootstrap_ci`，
  并已并入 `fairness_report` 的输出，所有调用方自动获得。
* `run_fairness_tradeoff.py` 新增 `--budget-mult`（默认 1.0，历史结果逐位不变）
  与 95% 自助置信区间表；θ 曲线表新增 `ks` / `rank` 两列与 `组数`/`最低组份额` 诊断列。

**实测验证（small，30 种子，θ=0.60，φ=0）：**

| 量 | 预算×1（n_sel=6.1） | 预算×3.5（n_sel=16.5） | 区间变化 |
|---|---|---|---|
| `gqc`（A′） | 0.194 [0.160, 0.230] | 0.118 [0.093, 0.143] | 0.070 → 0.050 |
| `gqc`（GQC） | 0.133 [0.114, 0.152] | 0.043 [0.035, 0.051] | 0.038 → 0.016 |
| `ks`（A′） | 0.242 [0.203, 0.284] | 0.128 [0.108, 0.149] | 0.081 → 0.041 |
| `ks`（GQC） | 0.189 [0.158, 0.223] | 0.090 [0.074, 0.105] | 0.065 → 0.035 |

区间普遍收窄 29%–58%，且**判读结论确实变了**：`ks` 上 A′ vs GQC 的差异
在 ×1 时两个区间重叠（0.203 vs 0.223，不可区分），在 ×3.5 时分离
（0.108 vs 0.105，可区分）。这正是扩预算要达到的效果。

**必须注意的代价：覆盖率口径变了。** 同一格子的 `cov/预算` 从 0.963 降到 0.455，
因为预算翻了三倍半而绝对覆盖没按比例涨。绝对覆盖其实变好（`jain` 0.153 → 0.414，
覆盖人数显著增加），但**"覆盖/预算"这个比值不能再和旧图直接比**。
Fig.3 若采用 ×3.5，覆盖率子图与正文相关数值需整体重算并说明口径变更，
不能新旧混用。

**复现：**

```powershell
$env:APRIME_ALLOW_SHELVED = '1'
py -3.14 experiments/run_fairness_tradeoff.py --n 30 --phis 0,0.95 `
    --budget-mult 3.5 --tag _mult35_n30
```

---

## 附录 A：本次实测记录（2026-11-13）

| 探测对象 | 结果 |
|---|---|
| `https://github.com` / `api.github.com` | 200，往返正常 |
| `https://archive.ics.uci.edu`（Porto #339 静态 ZIP） | 200 |
| `https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz` | 200 |
| `https://download.microsoft.com/.../Geolife Trajectories 1.3.zip` | 200（298.7 MB） |
| `https://overturemaps.org/download/` | 200 |
| `https://engineering.grab.com/grab-posisi` | 可访问（正文已核对） |
| `https://crowdsensim.gforge.uni.lu/{index,download,documentation,publications}.html` | 可访问（**仅经外部代理**；本机 `curl` 直连 80/443 均超时） |
| `https://huggingface.co/datasets/fsq/places-open` | **401**（托管位置需再确认） |
| `wsl --status`（本机） | 失败：`WSL/CallMsi/Install/REGDB_E_CLASSNOTREG` → 无法用 WSL 挂载 VM 磁盘 |
| `VBoxManage clonemedium ... --format RAW`（20 GiB） | 成功；用于离线查看 VM 磁盘，验证后已清理 |
| CrowdSenSim VM | **未启动**。数据清单来自官方站点文档 + 源码镜像 + `ReadValues.cc` 的读路径，未做客户机内实测 |

> 清理说明：本次为查看 OVA 而临时解出的 `D:\composition\_vmpeek\`（20 GB raw + 5 GB vmdk）已删除、磁盘空间已释放（D: 空闲回到 123 GB）；仅剩一个空目录被系统句柄占用，重启后可删，无其他影响。

## 附录 B：引用要点

- GeoLife：Zheng et al., WWW 2009 / UbiComp 2008 / IEEE DEBull 2010
- Porto Taxi：Moreira-Matias et al., ECML/PKDD 2015
- T-Drive：Yuan et al., SIGSPATIAL 2010
- Overture Maps：Overture Maps Foundation, Places 数据集（按月发布，GeoParquet）
- Foursquare 签到：Yang, Zhang, Qu, TIST 2015 / Yang et al., WWW 2019
- Gowalla：Cho, Myers, Leskovec, ICDM 2011
- CrowdSenSim：Fiandrino et al., IEEE Access 2017；Montori et al., MSWiM 2019（2.0）
- Grab-Posisi：Huang et al., ACM SIGSPATIAL PredictGIS Workshop 2019
