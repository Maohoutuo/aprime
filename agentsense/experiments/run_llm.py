# -*- coding: utf-8 -*-
"""用真实 LLM 复现表 3（单 Agent vs AgentSense）、表 5（扰动）与图 7（消融）。

用法示例：
    py -3.14 experiments\\run_llm.py --mode table3 --n 2 --scale small
    py -3.14 experiments\\run_llm.py --mode table5 --n 1 --scale small --disturbance bad_weather
    py -3.14 experiments\\run_llm.py --mode ablation --n 1 --scale small
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
from agentsense.loop import refine, refine_single
from agentsense.repair import repair_solution
from agentsense.metrics import summarize


def _baseline(inst, planner, seed):
    return PLANNERS[planner](inst, DEFAULT_ENV, random.Random(seed))


def run_table3(args, client):
    rows = {}
    lines = [f"# 表 3 复现（{args.model}，{args.scale}，n={args.n}，T={args.temperature}）", "",
             "| 方法 | SR | AIR | ANI | ACS | tokens | 用时(s) |", "|---|---|---|---|---|---|---|"]
    for method in ("single-agent", "AgentSense"):
        t0, runs, usage_before = time.time(), [], client.usage["total_tokens"]
        for i in range(args.n):
            inst = make_instance(f"tdrive_{args.scale}", i)
            base = _baseline(inst, args.planner, 1000 + i)
            dist = {"type": "none", "description": "no environmental disturbance"}
            if method == "single-agent":
                res = refine_single(inst, base, dist, client, seed=1000 + i)
            else:
                res = refine(inst, base, dist, client=client, max_iter=args.max_iter,
                             use_vision=True, use_memory=True, seed=1000 + i,
                             success_mode="improvement")
            runs.append(res.run)
            print(f"  [{method}] inst {i}: base={res.run.base_obj:.3f} "
                  f"final={res.run.final['obj']:.3f} iters={len(res.history)}", flush=True)
        m = summarize(runs)
        tokens = client.usage["total_tokens"] - usage_before
        m["tokens"] = tokens
        rows[method] = m
        lines.append(f"| {method} | {m['SR']:.1f}% | {m['AIR']:+.3f}% | {m['ANI']:.1f} | "
                     f"{m['ACS']:+.2f} | {tokens} | {time.time()-t0:.0f} |")
        print(lines[-1], flush=True)
    return rows, lines


def run_table5(args, client):
    rows = {}
    types = [args.disturbance] if args.disturbance else DISTURBANCE_TYPES
    lines = [f"# 表 5 复现（{args.model}，{args.scale}，n={args.n}，T={args.temperature}）", "",
             "| 扰动 | SR | AIR | ANI | ACS | tokens |", "|---|---|---|---|---|---|"]
    for dtype in types:
        runs, usage_before = [], client.usage["total_tokens"]
        for i in range(args.n):
            inst = make_instance(f"tdrive_{args.scale}", i)
            base = _baseline(inst, args.planner, 1000 + i)
            rng = random.Random(inst.seed + 7)
            dist_inst = apply_disturbance(inst, dtype, rng)
            instr = disturbance_instruction(dtype, {"meta": dist_inst.meta})
            # 公平比较：先把基线在扰动后的实例上修复为可行解，再算 J_base
            base_feasible, _ = repair_solution(dist_inst, base)
            res = refine(dist_inst, base_feasible, {"type": dtype, "description": instr},
                         client=client, max_iter=args.max_iter, seed=1000 + i,
                         success_mode="complies")
            runs.append(res.run)
            print(f"  [{dtype}] inst {i}: iters={len(res.history)} "
                  f"final={res.run.final['obj']:.3f}", flush=True)
        m = summarize(runs)
        tokens = client.usage["total_tokens"] - usage_before
        m["tokens"] = tokens
        rows[dtype] = m
        lines.append(f"| {dtype} | {m['SR']:.1f}% | {m['AIR']:+.3f}% | {m['ANI']:.1f} | "
                     f"{m['ACS']:+.2f} | {tokens} |")
        print(lines[-1], flush=True)
    return rows, lines


def run_ablation(args, client):
    variants = {
        "Ours": dict(use_vision=True, use_memory=True, with_history=False),
        "Ours w/o vision": dict(use_vision=False, use_memory=True, with_history=False),
        "Ours w/o memory": dict(use_vision=True, use_memory=False, with_history=False),
        "Ours w/ history": dict(use_vision=True, use_memory=True, with_history=True),
    }
    rows = {}
    lines = [f"# 图 7 消融（{args.model}，{args.scale}，n={args.n}）", "",
             "| 变体 | SR | AIR | ANI | ACS | tokens |", "|---|---|---|---|---|---|"]
    for name, kw in variants.items():
        runs, usage_before = [], client.usage["total_tokens"]
        for i in range(args.n):
            inst = make_instance(f"tdrive_{args.scale}", i)
            base = _baseline(inst, args.planner, 1000 + i)
            dist = {"type": "none", "description": "no environmental disturbance"}
            res = refine(inst, base, dist, client=client, max_iter=args.max_iter,
                         seed=1000 + i, success_mode="improvement", **kw)
            runs.append(res.run)
        m = summarize(runs)
        tokens = client.usage["total_tokens"] - usage_before
        m["tokens"] = tokens
        rows[name] = m
        lines.append(f"| {name} | {m['SR']:.1f}% | {m['AIR']:+.3f}% | {m['ANI']:.1f} | "
                     f"{m['ACS']:+.2f} | {tokens} |")
        print(lines[-1], flush=True)
    return rows, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="table3", choices=["table3", "table5", "ablation"])
    ap.add_argument("--provider", default="siliconflow")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--scale", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--planner", default="TVPG")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--disturbance", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-report", action="store_true", help="跑完不刷新 report.html")
    args = ap.parse_args()

    client = LLMClient(args.provider, args.model, temperature=args.temperature)
    if client.api_key is None and client.kind != "mock":
        raise SystemExit("未找到 API key：请在 .env 中设置 SILICONFLOW_API_KEY 或 OPENAI_API_KEY")

    t0 = time.time()
    if args.mode == "table3":
        rows, lines = run_table3(args, client)
    elif args.mode == "table5":
        rows, lines = run_table5(args, client)
    else:
        rows, lines = run_ablation(args, client)

    out = args.out or f"results/{args.mode}_{args.model.split('/')[-1]}_{args.scale}.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "scale": args.scale, "n": args.n,
                   "temperature": args.temperature, "rows": rows}, fh, ensure_ascii=False, indent=2)
    u = client.usage
    print(f"\nwrote {out}")
    print(f"usage: calls={u['calls']} prompt={u['prompt_tokens']} "
          f"completion={u['completion_tokens']} total={u['total_tokens']} "
          f"elapsed={time.time()-t0:.0f}s")

    if not args.no_report:
        _refresh_report(out)


def _refresh_report(out: str) -> None:
    """把最新结果同步进 report.html（失败不影响实验本身）。"""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_report import rebuild
        stats = rebuild()
        print("report.html 已更新：" + "  ".join(f"{k}={v}" for k, v in stats.items()))
    except Exception as exc:  # noqa: BLE001 - 报告是附属产物，不能因此让实验失败
        print(f"（report.html 未更新：{exc}）")


if __name__ == "__main__":
    main()
