# -*- coding: utf-8 -*-
"""迭代改进循环（论文 4.1.2 / 图 2）：S0 → S1 → ... → S10。

终止条件：取得"有效提升"（可行 且 J > J_base）即停；否则最多 10 轮。
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import DisturbanceParser, EvalAgent, MemoryAgent, SolverAgent
from .config import DEFAULT_ENV, MAX_ITERATIONS, EnvConfig
from .env.model import Instance, Solution, validate_solution
from .env.objective import evaluate
from .llm import LLMClient
from .metrics import RunRecord


@dataclass
#LoopResult类：存储AgentSense实验结果
class LoopResult:
    solution: Solution
    history: List[dict]
    run: RunRecord
    tokens_used: int = 0

#refine函数：AgentSense实验主函数
def refine(inst: Instance,
           baseline: Solution,
           disturbance: Dict[str, Any],
           client: Optional[LLMClient] = None,
           cfg: EnvConfig = DEFAULT_ENV,
           max_iter: int = MAX_ITERATIONS,
           use_vision: bool = True,
           use_memory: bool = True,
           with_history: bool = False,
           seed: int = 0,
           success_mode: str = "improvement") -> LoopResult:
    """success_mode:
    - "improvement"：需要可行 且 J > J_base（RQ1 主实验）；
    - "complies"   ：只需可行且合规（扰动恢复实验，对应表 5 中 AIR 可为负的情形）。
    """
    client = client or LLMClient("mock")#生成一个LLM客户端实例
    rng = random.Random(seed)#生成一个随机数生成器
    # 扰动（如 worker_unavailable）可能移除工作者，先清洗基线中的失效引用
    valid_ids = {w.wid for w in inst.workers}#生成一个有效工作者ID集合
    baseline = {k: v for k, v in baseline.items() if k in valid_ids}#清洗基线中的失效引用
    solver = SolverAgent(client, cfg)#生成一个SolverAgent实例
    evaluator = EvalAgent(client, cfg)#生成一个EvalAgent实例
    memory = MemoryAgent(client if use_memory else None, k=3)#生成一个MemoryAgent实例

    base_m = evaluate(inst, baseline, cfg)#评估基线性能
    run = RunRecord(base_obj=base_m["objective"], base_cost=base_m["cost"])#生成一个RunRecord实例
    cur = baseline
    advice: List[str] = []#生成一个建议列表
    history: List[dict] = []#生成一个历史记录列表
    best: Optional[Solution] = None#生成一个最佳解决方案
    best_key = None#生成一个最佳解决方案的键

    for t in range(1, max_iter + 1):#迭代次数
        mem = memory.retrieve(inst, disturbance) if use_memory else []#生成一个记忆列表
        if with_history and history:#如果有历史记录
            # 变体：把历史解当静态参考（论文发现它比"无记忆"更差）
            mem = [f"previous attempt #{i+1}: objective={h['objective']:.4f}" for i, h in enumerate(history[-3:])] + mem#生成一个记忆列表
        cand, think = solver.propose(inst, cur, disturbance, advice, mem, rng)#生成一个候选解决方案
        ev = evaluator.evaluate(inst, baseline, cand, disturbance, use_vision=use_vision)#评估候选解决方案性能

        feasible = bool(ev["feasible"] and ev["disturbance_handled"])#生成一个可行性标志
        if success_mode == "improvement":#如果成功模式是改进
            success = bool(ev["valid_improvement"])#生成一个有效提升标志
        else:
            # 扰动模式：既要合规，也不能比基线更差——否则"靠牺牲覆盖度达标"不算有效适应
            success = bool(ev["complies"]
                           and ev["candidate"]["objective"] >= base_m["objective"] - 1e-12)#生成一个合规标志
        if feasible:
            key = (ev["candidate"]["objective"], -ev["candidate"]["cost"])#生成一个键
            if best_key is None or key > best_key:
                best, best_key = cand, key#更新最佳解决方案
        history.append({
            "iteration": t,#更新历史记录
            "objective": ev["candidate"]["objective"],
            "cost": ev["candidate"]["cost"],#更新历史记录
            "q": ev["candidate"]["q"],#更新历史记录
            "feasible": feasible,
            "valid_improvement": success,#更新历史记录
            "advice": ev["advice"],
            "think": think,#更新历史记录
        })
        run.history.append({"obj": ev["candidate"]["objective"], "cost": ev["candidate"]["cost"],
                            "feasible": feasible, "valid": success})#更新历史记录

        if use_memory:
            delta = ev["candidate"]["objective"] - evaluate(inst, cur, cfg)["objective"]#更新历史记录
            memory.record(inst, cur, cand, delta, disturbance)#更新历史记录

        cur = cand
        advice = ev["advice"]#更新历史记录
        if success:
            break

    solution = best if best is not None else cur#更新历史记录
    m = evaluate(inst, solution, cfg)#评估解决方案性能
    run.final_override = {"obj": m["objective"], "cost": m["cost"],
                          "feasible": bool(validate_solution(inst, solution).feasible)}#更新历史记录
    return LoopResult(solution=solution, history=history, run=run,
                      tokens_used=solver.tokens_used + evaluator.tokens_used)#返回AgentSense实验结果


def refine_single(inst: Instance, baseline: Solution, disturbance: Dict[str, Any],
                  client: LLMClient, cfg: EnvConfig = DEFAULT_ENV, seed: int = 0) -> LoopResult:#单Agent实验函数
    """表 3 的"单 Agent"对照：一次 LLM 调用，无工具、无评估反馈、无记忆。"""
    solver = SolverAgent(client, cfg)#生成一个SolverAgent实例
    base_m = evaluate(inst, baseline, cfg)#评估基线性能
    run = RunRecord(base_obj=base_m["objective"], base_cost=base_m["cost"])#生成一个RunRecord实例
    cand, think = solver.propose(inst, baseline, disturbance, [], [], random.Random(seed), use_tools=False)#生成一个候选解决方案
    cand_m = evaluate(inst, cand, cfg)#评估候选解决方案性能
    vr = validate_solution(inst, cand)#验证候选解决方案可行性
    feasible = bool(vr.feasible)#生成一个可行性标志
    valid = bool(feasible and cand_m["objective"] > base_m["objective"] + 1e-12)#生成一个有效提升标志
    run.history.append({"obj": cand_m["objective"], "cost": cand_m["cost"],
                        "feasible": feasible, "valid": valid})#更新历史记录
    return LoopResult(solution=cand,#返回单Agent实验结果
                      history=[{"iteration": 1, "objective": cand_m["objective"],
                                "cost": cand_m["cost"], "q": cand_m["q"],
                                "feasible": feasible, "valid_improvement": valid,
                                "advice": [], "think": think}],
                      run=run, tokens_used=solver.tokens_used)#返回单Agent实验结果


def refine_under_disturbance(inst: Instance, baseline: Solution, disturbance_type: str,
                             client: Optional[LLMClient] = None, cfg: EnvConfig = DEFAULT_ENV,
                             **kwargs) -> LoopResult:#扰动实验函数
    """扰动 → 重新评测基线 → 迭代改进。"""
    from .disturbances import apply_disturbance, disturbance_instruction
    from .repair import repair_solution

    rng = random.Random(inst.seed + zlib.crc32(disturbance_type.encode()) % 1000)#生成一个随机数生成器
    dist_inst = apply_disturbance(inst, disturbance_type, rng)#应用扰动
    instruction = disturbance_instruction(disturbance_type, {"meta": dist_inst.meta})#生成一个扰动描述
    disturbance = {"type": disturbance_type, "description": instruction}#生成一个扰动
    base_feasible, _ = repair_solution(dist_inst, baseline)#修复基线
    return refine(dist_inst, base_feasible, disturbance, client=client, cfg=cfg, **kwargs)#返回扰动实验结果
