# W3：多智能体层（对应论文第 3 节 / 附录 B）

## 架构

```
扰动描述文本
   │
   ▼
Disturbance Parser ──► {type, description, value}
   │
   ▼
S₀（基线规划器：RN / TVPG / TCPG / MSA / MSAGI / GraphDP）
   │
   ├──► Solver Agent ── 调用 5 个工具，读 Eval 的 advice 与 Memory 的 top-k
   │        │           输出 {think_process ≤200 词, refined_solution}
   │        ▼
   ├──► Eval Agent ──── 数值 + 视觉（matplotlib 热力图）判定
   │        │           输出 {eval_summary, valid_improvement, advice}
   │        ▼
   ├──► Memory Agent ── 差分 → 元操作 → φ 打分 → top-k 检索
   │
   └──► 有效提升则停，否则 ≤10 轮
```

## 文件

| 文件 | 内容 |
|---|---|
| `agentsense/prompts.py` | 三 Agent 的 system prompt + user 模板（附录 B 五段式补全） |
| `agentsense/agents.py` | `DisturbanceParser` / `SolverAgent` / `EvalAgent` / `MemoryAgent` |
| `agentsense/tools.py` | 5 个工具：validate_worker_paths / compute_objective / compute_cost / check_disturbance_handling / visual_analysis |
| `agentsense/llm.py` | mock / OpenAI 兼容 / Anthropic 三种后端（标准库 urllib） |
| `agentsense/disturbances.py` | 7 类扰动的实例变换 |
| `agentsense/loop.py` | 迭代改进主循环 |
| `experiments/run_mock.py` | 表 5 端到端（Mock LLM） |

## 两种成功判据

- `success_mode="improvement"`（RQ1 主实验，表 3/表 4）：可行 **且** J > J_base；
- `success_mode="complies"`（扰动恢复，表 5）：可行 **且** 处理了扰动。论文表 5 中 AIR 有负值，
  说明此时"有效"只要求恢复可行性/合规，不要求目标函数提升。

## Mock LLM 的意义与局限

`provider="mock"` 用确定性的"修复 + DP 重规划"模拟 LLM 的 edit-and-verify 能力，
**用于在没有 API key 的情况下验证整条闭环**（工具调用、Prompt 拼装、解析、记忆读写、
指标统计），而不用于复现论文的 AIR 数值。

Mock 下的表 5（T-Drive Small，n=2）：

| 扰动 | SR | AIR |
|---|---|---|
| bad_weather | 100% | +1.20% |
| area_blocked | 100% | +0.20% |
| worker_unavailable | 100% | −1.73% |
| traffic_congestion | 100% | −10.04% |
| time_window_shift | 100% | +0.36% |
| budget_cut | 100% | −6.89% |
| task_priority_change | 100% | +0.57% |

这与论文表 5 的**结构**一致（SR 全 100%，AIR 可正可负），但**数值不可比**——
真实 AIR 需要接入 GPT-4.1 / Gemini-2.0-Flash 等模型。

## 接入真实 LLM

```powershell
$env:OPENAI_API_KEY = "sk-..."
py -3.14 experiments\run_mock.py --n 5 --scale small   # 现在默认 mock
```

要切到真实模型，在 `LLMClient(provider="openai", model="gpt-4.1", temperature=0.5)` 处替换即可
（`experiments/run_mock.py` 里 `client = LLMClient("mock")` 一行）。
论文图 9 的最优温度：GPT-4.1 → T=0.5，Gemini-2.0-Flash → T=0.1。

## 尚未做的工作

- 真实 LLM 的 AIR/SR 复现（需要 API key + token 预算）；
- 图 7 的消融（`--no-vision` / `--no-memory` / `--with-history` 已实现开关，未跑真实模型）；
- 记忆库的持久化（当前是进程内列表，跨实例不共享）。
