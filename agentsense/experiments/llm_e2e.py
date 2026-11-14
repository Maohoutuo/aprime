# -*- coding: utf-8 -*-
"""单实例端到端：真实 LLM 下的 单 Agent vs AgentSense，逐轮打印。

用于在正式跑表 3/表 5 之前评估 token 成本与 JSON 合规率。

    py -3.14 experiments\\llm_e2e.py --scale small --index 0
"""

import argparse
import json
import random
import time

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.env.model import validate_solution
from agentsense.env.objective import evaluate
from agentsense.llm import LLMClient
from agentsense.loop import refine, refine_single


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="siliconflow")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--scale", default="small")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--planner", default="TVPG")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    inst = make_instance(f"tdrive_{args.scale}", args.index)
    base = PLANNERS[args.planner](inst, DEFAULT_ENV, random.Random(1000 + args.index))#生成一个规划器实例
    base_m = evaluate(inst, base, DEFAULT_ENV)#评估规划器性能
    print(f"instance: {inst.setting_name}#{inst.index} grid={inst.grid} slots={inst.n_slots} "
          f"budget={inst.budget:.0f} workers={len(inst.workers)}")#打印实例信息
    print(f"baseline: J={base_m['objective']:.4f} cost={base_m['cost']:.0f} "
          f"Q={base_m['q']:.0f} feasible={validate_solution(inst, base).feasible}")#打印规划器性能

    dist = {"type": "none", "description": "no environmental disturbance"}#环境干扰类型
    client = LLMClient(args.provider, args.model, temperature=args.temperature, verbose=args.verbose)#生成一个LLM客户端实例

    print("\n=== single-agent (1 call, no tools / no feedback / no memory) ===")#打印单Agent实验信息
    t0 = time.time()
    r1 = refine_single(inst, base, dist, client, seed=1000 + args.index)#单Agent实验
    ev = evaluate(inst, r1.solution, DEFAULT_ENV)#评估单Agent实验性能
    print(f"  final J={ev['objective']:.4f} cost={ev['cost']:.0f} "
          f"feasible={validate_solution(inst, r1.solution).feasible} "
          f"delta={(ev['objective']-base_m['objective'])/base_m['objective']*100:+.3f}% "
          f"({time.time()-t0:.0f}s)")#打印单Agent实验性能
    print(f"  usage so far: {client.usage}")#打印LLM使用量

    print("\n=== AgentSense (multi-agent loop) ===")#打印AgentSense实验信息
    t0 = time.time()
    r2 = refine(inst, base, dist, client=client, max_iter=args.max_iter,
                seed=1000 + args.index, success_mode="improvement")#AgentSense实验
    ev2 = evaluate(inst, r2.solution, DEFAULT_ENV)#评估AgentSense实验性能
    for h in r2.history:#打印AgentSense实验历史
        print(f"  iter {h['iteration']}: J={h['objective']:.4f} cost={h['cost']:.0f} "
              f"feasible={h['feasible']} success={h['valid_improvement']} "
              f"| {h['think'][:80]}", flush=True)#打印AgentSense实验历史
        for a in h.get("advice", [])[:2]:
            print(f"        advice: {a[:110]}", flush=True)#打印AgentSense实验历史
    print(f"  final J={ev2['objective']:.4f} cost={ev2['cost']:.0f} "
          f"feasible={validate_solution(inst, r2.solution).feasible} "
          f"delta={(ev2['objective']-base_m['objective'])/base_m['objective']*100:+.3f}% "
          f"({time.time()-t0:.0f}s)")
    print(f"\nTOTAL usage: {client.usage}")


if __name__ == "__main__":
    main()
