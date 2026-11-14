# -*- coding: utf-8 -*-
"""端到端闭环测试（Mock LLM）：模拟表 5 的七类扰动下的自修复能力。

用法：
    python experiments/run_mock.py --n 3 --scale small
"""

import argparse
import json
import os
import random
import time

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.disturbances import DISTURBANCE_TYPES, apply_disturbance, disturbance_instruction
from agentsense.env.objective import evaluate
from agentsense.llm import LLMClient
from agentsense.loop import refine
from agentsense.metrics import summarize, format_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scale", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--planner", default="TVPG")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--with-history", action="store_true")
    ap.add_argument("--success-mode", default="complies", choices=["improvement", "complies"])
    ap.add_argument("--out", default="results/mock_table5.md")
    args = ap.parse_args()

    client = LLMClient("mock")
    rows = {}
    lines = [f"# 表 5 复现（Mock LLM，{args.scale}，n={args.n}，planner={args.planner}）", "",
             "| 扰动 | SR | AIR | ANI | ACS | 用时(s) |", "|---|---|---|---|---|---|"]
    for dtype in DISTURBANCE_TYPES:
        t0 = time.time()
        runs = []
        for i in range(args.n):
            inst = make_instance(f"tdrive_{args.scale}", i)
            base_sol = PLANNERS[args.planner](inst, DEFAULT_ENV, random.Random(1000 + i))
            rng = random.Random(inst.seed + 7)
            dist_inst = apply_disturbance(inst, dtype, rng)
            instruction = disturbance_instruction(dtype, {"meta": dist_inst.meta})
            res = refine(dist_inst, base_sol, {"type": dtype, "description": instruction},
                         client=client, max_iter=args.max_iter,
                         use_vision=not args.no_vision,
                         use_memory=not args.no_memory,
                         with_history=args.with_history, seed=1000 + i,
                         success_mode=args.success_mode)
            runs.append(res.run)
        m = summarize(runs)
        dt = time.time() - t0
        rows[dtype] = m
        line = f"| {dtype} | {m['SR']:.1f}% | {m['AIR']:+.3f}% | {m['ANI']:.1f} | {m['ACS']:+.2f} | {dt:.1f} |"
        lines.append(line)
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
