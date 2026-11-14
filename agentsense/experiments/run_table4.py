# -*- coding: utf-8 -*-
"""复现论文表 4（不同基线规划器下的 Base Obj. / AIR）。

用法：
    python experiments/run_table4.py --n 20 --out results/table4.md

注意：AIR 列需要智能体改进后的结果（W3 之后），这里先给出 Base Obj. 与
成本消耗，作为环境正确性的校准表。
"""

import argparse
import json
import os
import random
import time
from statistics import mean

import numpy as np

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.env.model import validate_solution
from agentsense.env.objective import evaluate

PAPER_BASE = {
    "small": {"RN": 4.505, "TVPG": 4.565, "TCPG": 4.588, "MSA": 4.662, "MSAGI": 4.702, "GraphDP": 4.780},
    "medium": {"RN": 4.706, "TVPG": 4.761, "TCPG": 5.118, "MSA": 5.086, "MSAGI": 5.090, "GraphDP": 5.205},
    "large": {"RN": 5.029, "TVPG": 5.602, "TCPG": 5.544, "MSA": 5.631, "MSAGI": 5.682, "GraphDP": 5.701},
}
ORDER = ["RN", "TVPG", "TCPG", "MSA", "MSAGI", "GraphDP"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每个设置的实例数（论文为 20）")
    ap.add_argument("--dataset", default="tdrive", choices=["tdrive", "grab"])
    ap.add_argument("--out", default="results/table4.md")
    ap.add_argument("--no-report", action="store_true", help="跑完不刷新 report.html")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = {}
    lines = [
        f"# 表 4 复现（{args.dataset}，n={args.n}）",
        "",
        "| 基线 | 规模 | 本文 Base Obj. | 论文 Base Obj. | 误差 | 平均成本 | 可行率 | 用时(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for scale in ("small", "medium", "large"):
        insts = [make_instance(f"{args.dataset}_{scale}", i) for i in range(args.n)]
        for name in ORDER:
            t0 = time.time()
            objs, costs, feas = [], [], []
            for i, inst in enumerate(insts):
                sol = PLANNERS[name](inst, DEFAULT_ENV, random.Random(1000 + i))
                ev = evaluate(inst, sol, DEFAULT_ENV)
                objs.append(ev["objective"])
                costs.append(ev["cost"])
                feas.append(validate_solution(inst, sol).feasible)
            dt = time.time() - t0
            m = mean(objs)
            ref = PAPER_BASE[scale][name]
            err = (m - ref) / ref * 100
            rows[f"{scale}/{name}"] = {
                "ours": round(m, 4), "paper": ref, "err_pct": round(err, 2),
                "cost": round(mean(costs), 2), "feasible_rate": round(mean(feas), 3),
                "seconds": round(dt, 2),
            }
            lines.append(f"| {name} | {scale} | {m:.3f} | {ref:.3f} | {err:+.1f}% | "
                         f"{mean(costs):.1f} | {mean(feas)*100:.0f}% | {dt:.1f} |")
            print(lines[-1], flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")

    if not args.no_report:
        _refresh_report()


def _refresh_report() -> None:
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
