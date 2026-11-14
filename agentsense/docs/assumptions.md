# 假设清单（论文未给出、必须自行确定的自由度）

> 原则：每一条都在代码里有一个明确的开关或注释，便于对照原文复核与二次标定。

## 1. 环境与成本

| # | 假设 | 取值 | 依据 / 影响 |
|---|---|---|---|
| A1 | 任务单元 | 网格中的 `(x, y, t)`，共 `nx·ny·T` 个 | 论文 2.1 定义 1 |
| A2 | 工作者移动 | 每时间步最多移动 `cells_per_step` 格（沿单一轴），**实测值**：8×8→3、16×16→7、32×32→13（T-Drive 相邻时槽位移中位数）；grab 2/4/8 | 附录 B.1 prompt 的 "up to speed grid units"；论文 Fig.2 的四邻域是 `cps=1` 的特例。口径与复现见 `experiments/measure_speed.py`、`config.SETTINGS` |
| A3 | 时间窗 | 每个工作者有 `[t_start, t_end]`，长度 55%–100% 的时域 | 论文未给出，影响可招募人数 |
| A4 | **成本口径** | 被招募路径的时间步数之和 × 1.0 | 论文 4.1.1"每时间步归一化报酬 1.0"；可用 Fig.5 的 `cost 38 / budget 40` 交叉验证 |
| A5 | 预算约束 | `cost ≤ budget`（论文的 `Σ r_i ≤ B`） | 定义 1 |
| A6 | 数据量 Q | 被覆盖的**不同** `(x,y,t)` 单元数（`q_mode="distinct"`） | `cost 38 → log2(38)=5.25`，与 `J≈4.57` 自洽 |
| A7 | 任务激励成本 | 插入路点带来的绕行步数 | 原文未定义；仅影响 TCPG 的排序 |

**交叉锚点**（来自论文图 5/图 10 的示例）：
- T-Drive Small、预算 40 → 基线 `objective 4.568 / cost 38`；
- Grab-Posisi Medium、预算 60 → `objective 4.955 / cost 57`。
本工程在 Small 上得到 `J≈4.6 / cost≈38–40`，与上述一致。

## 2. 目标函数

`J(D) = α·E(D) + (1−α)·log2 Q(D)`，`α = 0.5`（论文式 2）。

`E(D)` 的"层次化熵"定义不在本文中（引自 [18]），因此：

```
E = e_scale × mean( H_time/log2(min(T, obs_t)) , H_space/log2(min(nx·ny, obs_s)) , H_st/log2(min(nx·ny·T, obs_st)) )
```

- `e_scale` 由表 4 的 Small 档（TVPG = 4.565）**单参数标定**得到 `4.2`；
- Medium / Large 不参与拟合，作为留出验证（见 `calibration.md`）；
- 可切换的备选口径：`e_mode = relative | norm | equal | sum | weighted`（见 `EnvConfig`）。

## 3. 数据

| # | 假设 | 说明 |
|---|---|---|
| D1 | 默认使用**合成轨迹** | Grab-Posisi 需 Kaggle 授权，T-Drive 需下载；为保证流水线可复现，默认合成，分布与真实数据同构（随机起终点 + 可达性约束） |
| D2 | T-Drive 接入方式 | `make_instance(..., source="tdrive", tdrive_path=...)`；期望 CSV 格式：`taxi_id,YYYY-MM-DD HH:MM:SS,lon,lat`；bbox 默认 `(116.20, 39.80)–(116.55, 40.05)` |
| D3 | 实例固定 | 每档 20 个实例，`seed = base + scale*100 + idx*7919`，跨机器一致 |
| D4 | 20 次试验 | `config.N_TRIALS = 20`，与论文 SR 的定义一致 |

## 4. 基线（附录 A）

| 基线 | 实现 |
|---|---|
| RN | 随机顺序招募，随机插入 0–3 个可行路点 |
| TVPG | 每步选边际覆盖增益最大的（工作者, 路点） |
| TCPG | 先按绕行步数排序，成本相同再比增益 |
| MSA | 5 类邻域（移动/删除/新增/交换路点、招募/淘汰）+ SA，3 次重启 |
| MSAGI | MSA，但以 TVPG 解初始化 |
| GraphDP | 时间分层网格 DP（`1/(1+count)²` 权重）+ 工作者替换 |

**简化点**：GraphDP 的权重是"边际熵增益"的近似；MSA 的邻域规模小于原论文的完整实现。

## 5. 待 W3 补齐

- 三个 Agent 的提示词细节、工具 JSON schema、记忆库的元操作定义与打分函数 `φ`；
- LLM 选择与温度（论文图 9 显示各模型最优温度不同：GPT-4.1 → 0.5，Gemini-2.0 → 0.1）。
