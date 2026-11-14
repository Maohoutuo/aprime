# AgentSense 复现工程

复现论文 **AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing**
(arXiv:2510.19661v2) 的实验环境。

当前进度：**W1–W4 完成**（环境 + 基线 + 表 4 校准 + 多智能体闭环 + 真实 LLM 跑通）。

```
表 4 复现（T-Drive，n=20）：Small ±3.3% / Medium ±5.1% / Large −6.1%～+6.3%，可行性 100%

真实 LLM（SiliconFlow DeepSeek-V3.2，T-Drive Small，基线 RN）：
  表 3  n=5：单 Agent SR 60% / AIR +0.552%    AgentSense SR 100% / AIR +1.025%
  表 5  n=2：7 类扰动，成功案例 AIR 均 ≥ 0（+0.417% ～ +3.934%）
```

## 安装

```powershell
cd D:\composition\agentsense
py -3.14 -m pip install -e .
py -3.14 -m pytest tests -q          # 20 passed
```

## 快速开始

```powershell
py -3.14 experiments\smoke.py                              # 环境自检：三档规模跑全部基线
py -3.14 experiments\calibrate.py                          # 用表 4 锚点标定 E(D) 的尺度
py -3.14 experiments\run_table4.py --n 5 --out results\table4_n5.md
```

### 真实 LLM（SiliconFlow / OpenAI 兼容）

把 key 写进本地 `.env`（已被 `.gitignore` 排除，不要提交）：

```
SILICONFLOW_API_KEY=sk-...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

```powershell
py -3.14 experiments\siliconflow_smoke.py                  # 1 次调用验证连通性
py -3.14 experiments\llm_e2e.py --scale small --index 0 --max-iter 2 --verbose
py -3.14 experiments\run_llm.py --mode table3 --n 5 --scale small --planner RN --max-iter 3
py -3.14 experiments\run_llm.py --mode table5 --n 2 --scale small --planner RN --max-iter 3
py -3.14 experiments\run_llm.py --mode ablation --n 2 --scale small --planner RN
```

注意：基线越强，LLM 可提升的空间越小。用 DP 基线（TVPG/MSAGI）时 AIR 接近 0；
表 3/表 5 量级的提升需要用较弱的基线（如 RN）才可观察。详见 `docs/llm_experiment.md`。

### 结果网页

`report.html` 是单文件报告页（双击即可打开，无需联网）。

`run_table4.py` / `run_llm.py` 跑完会自动把 `results/*.json` 写回该页面；
也可以手动刷新：

```powershell
py -3.14 experiments\build_report.py            # 汇总并写回 report.html
py -3.14 experiments\build_report.py --dry-run  # 只看会写什么
```

## 目录

```
agentsense/
├── agentsense/
│   ├── config.py            论文表 2 的 6 个配置 + 目标函数口径（EnvConfig）
│   ├── data/instances.py    实例生成（synthetic / T-Drive），固定 seed
│   ├── env/model.py         Worker / Instance / Solution、路径构造、可行性校验
│   ├── env/objective.py     覆盖计数、层次化熵 E(D)、J(D)、成本
│   ├── baselines/planners.py RN / TVPG / TCPG / MSA / MSAGI / GraphDP
│   ├── agents.py            Parser / Solver / Eval / Memory 四个 Agent
│   ├── loop.py              迭代改进循环 + 单 Agent 对照
│   ├── repair.py            确定性可行性最小修复（保留 LLM 路线形状）
│   ├── advice.py            具体可执行的重规划建议
│   ├── tools.py             附录 B 的 5 个工具（含视觉对比）
│   ├── prompts.py           三个 Agent 的系统/用户提示词
│   └── metrics.py           SR / AIR / ANI / ACS（论文 4.1.2）
├── experiments/             smoke / calibrate / run_table4 / run_llm / llm_e2e / build_report
├── tests/                   20 个单元测试
├── docs/                    assumptions · calibration · w3_agents · llm_experiment
└── results/                 实验输出（表格 + json）
```

## 环境模型的约定（详见 docs/assumptions.md）

| 项 | 约定 |
|---|---|
| 任务 | 网格中的 `(x, y, t)` 单元，三档网格见论文表 2 |
| 工作者 | 起点、终点、可用时间窗；每步最多移动 `cells_per_step` 格（实测：8×8→3、16×16→7、32×32→13；grab 代理 2/4/8，见 `experiments/measure_speed.py`） |
| 成本 | 被招募路径的时间步数之和（报酬 1.0/步），`cost ≤ budget` |
| 数据量 Q | 被覆盖的不同 `(x, y, t)` 单元数 |
| 目标函数 | `J = α·E + (1−α)·log2 Q`，`α = 0.5` |
| 层次化熵 E | 时间层 / 空间层 / 时空层的相对归一化熵均值 × 尺度常数（标定得到 4.2） |

## 与论文的已知差距

1. **E(D) 的精确定义不可恢复**：论文只在式 2 给出形式，分层定义在引文 [18] 中，且 [18]/[41] 均未开源。
   这里用单参数标定 E 的尺度，Medium/Large 作为留出验证。
2. **Small/Medium 已在 ±5% 以内**（实测速度口径、n=20：Small +0.7%～+3.3%，Medium −5.7%～+5.3%）；
   **Large 仍低 3–9%**，且 *不可通过继续调参消除*：在相对归一化下 Large 的理论上限
   `J_max ≈ 5.42`（`e_scale 4.2 × 1.0 + 0.5·log2(100)`），低于论文的 5.602，说明
   原文的 E(D) 在大规模档的尺度随规模增长，本工程无法从论文文本推导出该形式。
3. **GraphDP 仍偏低**：Small −1.8% / Medium −5.7% / Large −8.1%（T-Drive，n=20）。论文中
   GraphDP 是最强基线；当前实现用 `1/(1+count)²` 权重近似"边际熵增益"，属近似。
4. **实例来源**：默认合成轨迹（保证可复现）；提供 `instance.source="tdrive"` 接口读取真实
   T-Drive 样本，格式见 docs/assumptions.md。
5. **真实 LLM 的 AIR 量级取决于基线强度**：本工程的 DP 基线（TVPG/MSAGI）在 Small 上已接近最优，
   从 TVPG 出发时 LLM 提升近似为 0；表 3/表 5 的可测提升需从较弱基线（RN）出发。
   详见 `docs/llm_experiment.md` 第 5 节。
6. **历史结果的时效**：`results/_frozen_pre_cps/` 里的表 3/表 4/表 5 结果都是在
   `cells_per_step = 1` 的旧口径下产出的。速度口径已改为实测值并写进 `Worker.speed`，
   表 4（基线）已在实测速度下重跑：`results_speed_v2/table4_{tdrive,grab}_n20.*`；
   `e_scale` 重拟合仍为 4.00，标定常数无需改。表 3/表 5（真实 LLM）待重跑。

## 下一步

- [x] 扰动解析器：7 类扰动 → `{type, description, value}`
- [x] Solver / Eval / Memory 三个 Agent + 5 个工具（附录 B 的 prompt 骨架）
- [x] 迭代改进循环（≤10 轮，首次有效提升即停）
- [x] 接入真实 LLM（SiliconFlow DeepSeek-V3.2）跑通表 3 / 表 5
- [ ] Medium / Large 规模的真实模型实验
- [ ] 图 7 消融的真实模型实验
- [ ] GPT-4.1 / Gemini-2.0-Flash 对照

多智能体层的设计、工具清单与 Mock 结果见 `docs/w3_agents.md`；
真实模型的完整结果、方法修正与已知差距见 `docs/llm_experiment.md`。
