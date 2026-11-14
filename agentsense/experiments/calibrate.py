# -*- coding: utf-8 -*-
"""环境标定：用表 4 的 Base Obj. 锚点确定 E(D) 的尺度常数。

论文只给出 J(D)=αE(D)+(1−α)log2Q(D) 的形式，E(D) 的分层定义在其引文 [18]
中，原文未复述。因此 e_scale 是唯一的未知常数：
  * 在 **Small** 档上用 TVPG 的目标值（论文 4.565）做单参数标定；
  * 在 **Medium / Large** 档上做留出验证（论文 4.761 / 5.602），不参与拟合。
"""

import json
import random
from dataclasses import replace

import numpy as np

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.env.objective import evaluate

PAPER = {
    "small": {"RN": 4.505, "TVPG": 4.565, "TCPG": 4.588, "MSA": 4.662, "MSAGI": 4.702, "GraphDP": 4.780},
    "medium": {"RN": 4.706, "TVPG": 4.761, "TCPG": 5.118, "MSA": 5.086, "MSAGI": 5.090, "GraphDP": 5.205},
    "large": {"RN": 5.029, "TVPG": 5.602, "TCPG": 5.544, "MSA": 5.631, "MSAGI": 5.682, "GraphDP": 5.701},
}
TARGET_SMALL_TVPG = PAPER["small"]["TVPG"]


def mean_objective(scale: str, planner: str, e_scale: float, n: int = 3) -> float:
    cfg = replace(DEFAULT_ENV, e_scale=e_scale)
    vals = []
    for i in range(n):
        inst = make_instance(f"tdrive_{scale}", i)
        sol = PLANNERS[planner](inst, cfg, random.Random(1000 + i))
        vals.append(evaluate(inst, sol, cfg)["objective"])
    return float(np.mean(vals))


def main() -> None:
    grid = np.round(np.arange(0.0, 12.001, 0.25), 2)
    curve = [(float(s), mean_objective("small", "TVPG", float(s))) for s in grid]
    best = min(curve, key=lambda kv: abs(kv[1] - TARGET_SMALL_TVPG))
    e_scale = best[0]
    print(f"[calibration] best e_scale = {e_scale:.2f}  (Small TVPG = {best[1]:.3f}, "
          f"paper = {TARGET_SMALL_TVPG:.3f})")
    print("  sweep:", " ".join(f"{s:.1f}:{v:.2f}" for s, v in curve[::4]))

    cfg = replace(DEFAULT_ENV, e_scale=e_scale)
    print("\n[held-out validation]  (拟合只用 Small，Medium/Large 为留出)")
    report = {"e_scale": e_scale, "settings": {}}
    for scale in ("small", "medium", "large"):
        rows = {}
        print(f"\n--- tdrive_{scale} ---")
        print(f"{'planner':8s} {'ours':>7s} {'paper':>7s} {'err%':>7s} {'cost':>6s} {'Q':>5s} {'E':>5s}")
        for name in ("RN", "TVPG", "TCPG", "MSA", "MSAGI", "GraphDP"):
            vals, costs, qs, es = [], [], [], []
            for i in range(3):
                inst = make_instance(f"tdrive_{scale}", i)
                sol = PLANNERS[name](inst, cfg, random.Random(1000 + i))
                ev = evaluate(inst, sol, cfg)
                vals.append(ev["objective"])
                costs.append(ev["cost"])
                qs.append(ev["q"])
                es.append(ev["entropy"])
            m = float(np.mean(vals))
            ref = PAPER[scale][name]
            err = (m - ref) / ref * 100
            rows[name] = {"ours": round(m, 3), "paper": ref, "err_pct": round(err, 1),
                          "cost": round(float(np.mean(costs)), 1), "q": round(float(np.mean(qs)), 1)}
            print(f"{name:8s} {m:7.3f} {ref:7.3f} {err:+7.1f} "
                  f"{np.mean(costs):6.1f} {np.mean(qs):5.1f} {np.mean(es):5.2f}")
        report["settings"][scale] = rows
    with open("docs/calibration_result.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\nwrote docs/calibration_result.json")


if __name__ == "__main__":
    main()
