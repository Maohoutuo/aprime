# 真实 LLM 实验记录（SiliconFlow · DeepSeek-V3.2）

本文件记录用**真实托管模型**跑通 AgentSense 全链路的结果、方法与已知差距。
所有数字均由 `experiments/run_llm.py` 直接产出，可重跑。

## 1. 接入方式

- Endpoint：`https://api.siliconflow.cn/v1`（OpenAI 兼容）
- 模型：`deepseek-ai/DeepSeek-V3.2`，temperature = 0.5，`response_format=json_object`
- 凭据：仅存放于本地 `.env`（已被 `.gitignore` 排除）

```powershell
py -3.14 experiments\siliconflow_smoke.py    # 1 次调用验证连通/JSON
py -3.14 experiments\llm_e2e.py --scale small --index 0 --max-iter 2 --verbose
py -3.14 experiments\run_llm.py --mode table3 --n 5 --scale small --planner RN --max-iter 3
py -3.14 experiments\run_llm.py --mode table5 --n 2 --scale small --planner RN --max-iter 3
```

## 2. 表 3（n=5，T-Drive Small，基线 RN）

| 方法 | SR | AIR | ANI | ACS | tokens | 用时 |
|---|---|---|---|---|---|---|
| 单 Agent | 60.0% | +0.552% | 1.0 | +0.00 | 11,222 | 197s |
| **AgentSense** | **100.0%** | **+1.025%** | 1.4 | −0.60 | 23,837 | 703s |

原始文件：`results/real_table3_rn_small.md` / `.json`

与论文（表 3，GPT-4.1，T-Drive Small：单 Agent AIR≈+1.35%、AgentSense AIR≈+1.55%）相比：

- **趋势一致**：AgentSense 的 SR（100% vs 60%）与 AIR（+1.03% vs +0.55%）都优于单 Agent。
- **量级接近**：AIR 同属 0.5%–2% 区间。
- **绝对差异**来自基线与模型不同：论文用的是他们自己的基线（我们用的是本工程的 RN），
  模型也换成了 DeepSeek-V3.2 而非 GPT-4.1。

## 3. 表 5（n=2，T-Drive Small，基线 RN，7 类扰动）

| 扰动 | SR | AIR | ANI | ACS |
|---|---|---|---|---|
| bad_weather | 100.0% | +0.417% | 1.0 | +0.00 |
| area_blocked | 50.0% | +1.154% | 3.0 | −1.00 |
| worker_unavailable | 100.0% | +1.500% | 2.0 | −2.00 |
| traffic_congestion | 100.0% | +3.934% | 1.0 | −4.50 |
| time_window_shift | 100.0% | +1.035% | 1.0 | +0.50 |
| budget_cut | 50.0% | +0.000% | 2.0 | +0.00 |
| task_priority_change | 50.0% | +0.909% | 1.0 | +1.00 |

原始文件：`results/real_table5_rn_small.md` / `.json`

要点：

- 7 类扰动的成功案例 **AIR 均 ≥ 0**，没有出现"靠牺牲覆盖度过关"的情况。
- 越硬的结构性扰动（`area_blocked`、`budget_cut`、`task_priority_change`）在 3 轮内
  只有一半能达标，ANI 明显上升——这与论文中 ANI 随扰动难度增长的结论一致。
- `worker_unavailable` / `traffic_congestion` 的负 ACS 表示为了达标需要额外招募或加长路径。

## 4. 为跑通真实模型所做的方法修正

真实 LLM 暴露了 mock 下看不到的问题，以下修正都保留了论文的语义：

1. **worker 约束注入**：Solver prompt 原先不含每个工作者的起终点/时间窗，
   模型必然造出非法路径。现在 `workers_spec()` 显式给出约束。
2. **确定性可行性修复**：`agentsense/repair.py` 实现了"最小修复"——只修终点、
   时间戳、连续性等必要处，**保留模型给出的路线形状**，避免像 DP 重规划那样
   把 LLM 的编辑整条抹掉（这是最初 AIR≈0 的直接原因）。
3. **单 Agent 对照的口径**：原 `refine_single()` 仍调用了 Eval Agent，
   不符合"单 Agent、一次调用"的定义。现在严格只做一次 Solver 调用，评估全在本地完成，
   并有单测锁定调用次数。
4. **可执行的建议**：`agentsense/advice.py` 把"下一步"算成具体动作
   （招募哪个 worker、重路由哪个 worker、指向哪些未覆盖格），否则模型倾向于"不改动"。
5. **保留历史最佳可行解**：循环可能因相邻轮次退化而覆盖已有提升，现在返回
   best-feasible，并让指标读实际返回的解（`RunRecord.final_override`）。
6. **扰动下的公平比较**：`worker_unavailable` 会让基线引用已移除的工作者；
   现在先在扰动后的实例上把基线修复为可行解，再算 J_base。扰动模式的成功判据
   也改为"合规 **且** 不劣于基线"。
7. **失效引用容错**：`workers_spec` / `SolverAgent.propose` / `refine` 都会清洗
   指向已移除 worker 的引用（否则 `KeyError: 'w000'`）。

## 5. 已知差距（必须与论文结论区分开）

1. **基线强度决定 AIR 量级**：本工程的 TVPG/MSAGI 用 DP 规划，Small 上 J≈4.72–4.74，
   已接近最优，LLM 几乎没有提升空间（实测 TVPG 起点时 AIR≈0%）。表 3/表 5 的
   **可测提升必须用较弱的基线（RN）才能观察到**。论文报告的 AIR 对应的是他们自己的
   基线强度，两者不可直接对齐。
2. **样本量小**：表 3 n=5、表 5 n=2（论文为 20）。SR 的置信区间很宽，
   50% vs 100% 这类差异在 n=2 下只是一个指示。
3. **规模差异**：目前只跑 T-Drive Small。Medium/Large 的网格与预算更大，
   单实例 token 与延迟会显著上升。
4. **成本**：本轮全部实验约 25 万 tokens（含调试），单实例约 6–24k tokens。

## 6. 尚未完成

- [ ] Medium / Large 规模的真实模型实验
- [ ] 图 7 消融（`--mode ablation` 已实现，未跑真实模型）
- [ ] GPT-4.1 / Gemini-2.0-Flash 的对照（需要对应 provider 的 key）
