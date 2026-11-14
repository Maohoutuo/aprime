# -*- coding: utf-8 -*-
r"""图 12：动态扰动下的案例研究（只启用 PRG + NRR，被选中的工作者集合固定）。

案例 (a)：T-Drive，某格发生交通事故，需要绕行；
案例 (b)：Grab-Posisi，某格出现高价值数据需求。

    py -3.14 experiments\run_case_study.py --n 1 --backend surrogate
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import build_llm, write_json
from agentsense.config import DEFAULT_ENV
from mapus.config import MAPUSConfig
from mapus.data.instances import make_mapus_instance
from mapus.pipeline import run_disturbance, run_mapus


def _pick(routes: dict, mode: str, urban):
    from collections import Counter
    if mode == "accident":
        cnt = Counter()
        for path in routes.values():
            for step in path:
                cnt[(int(step[0]), int(step[1]))] += 1
        if not cnt:
            return 0, 0
        cell, _ = cnt.most_common(1)[0]
        return cell
    # priority: 选一个被最多路线"擦肩而过"的格子（距离某路线 ≤ 1 但不落脚），
    # 这样高价值数据需要参与者主动绕行，而不是原地不动。
    covered = {(int(s[0]), int(s[1])) for p in routes.values() for s in p}
    cand = []
    for path in routes.values():
        for step in path:
            x, y = int(step[0]), int(step[1])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if (0 <= nb[0] < urban.nx and 0 <= nb[1] < urban.ny and nb not in covered):
                    cand.append(nb)
    if not cand:
        # 兜底：取所有格子的中心
        return urban.nx // 2, urban.ny // 2
    cell, _ = Counter(cand).most_common(1)[0]
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="跑几个实例（结果取第一个成功案例）")
    ap.add_argument("--backend", default="surrogate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-workers", type=int, default=4)
    args = ap.parse_args()

    cfg = MAPUSConfig()
    llm = build_llm(args.backend, args.model, workers=args.llm_workers)
    payload = {"backend": args.backend, "model": getattr(llm, "name", ""), "cases": {}}

    # 案例 (a)：交通事故封锁一个格子
    inst = make_mapus_instance("tdrive_small", 0)
    run = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm, seed=1000,
                    parallel=args.llm_workers)
    cx, cy = _pick(run.solution, "accident", inst.urban)
    event = f"A serious traffic accident has occurred in grid [{cx}, {cy}]. Please take a detour."
    res = run_disturbance(inst, inst.urban, inst.profiles, run.solution, cfg, DEFAULT_ENV, llm,
                          event=event, blocked=[(cx, cy)], parallel=args.llm_workers)
    payload["cases"]["accident"] = {"event": event, "grid": [cx, cy],
                                    "coverage_before": round(res["coverage_before"], 4),
                                    "coverage_after": round(res["coverage_after"], 4),
                                    "pss_before": round(res["pss_before"], 4),
                                    "pss_after": round(res["pss_after"], 4),
                                    "n_agents": len(run.solution), "n_changed": res["n_changed"],
                                    "decisions": res["decisions"]}
    print(f"[accident] cov {res['coverage_before']:.3f} -> {res['coverage_after']:.3f}  "
          f"pss {res['pss_before']:.3f} -> {res['pss_after']:.3f}  changed {res['n_changed']}"
          f"/{len(run.solution)}", flush=True)

    # 案例 (b)：高价值数据需求
    inst2 = make_mapus_instance("grab_small", 0)
    run2 = run_mapus(inst2, inst2.urban, inst2.profiles, cfg, DEFAULT_ENV, llm, seed=1000,
                     parallel=args.llm_workers)
    tx, ty = _pick(run2.solution, "priority", inst2.urban)
    event2 = (f"Data in grid [{tx}, {ty}] is important and more data needs to be collected.")
    res2 = run_disturbance(inst2, inst2.urban, inst2.profiles, run2.solution, cfg, DEFAULT_ENV, llm,
                           event=event2, targets=[(tx, ty)], parallel=args.llm_workers)
    payload["cases"]["priority"] = {"event": event2, "grid": [tx, ty],
                                    "coverage_before": round(res2["coverage_before"], 4),
                                    "coverage_after": round(res2["coverage_after"], 4),
                                    "pss_before": round(res2["pss_before"], 4),
                                    "pss_after": round(res2["pss_after"], 4),
                                    "n_agents": len(run2.solution), "n_changed": res2["n_changed"],
                                    "decisions": res2["decisions"]}
    print(f"[priority] cov {res2['coverage_before']:.3f} -> {res2['coverage_after']:.3f}  "
          f"pss {res2['pss_before']:.3f} -> {res2['pss_after']:.3f}  changed {res2['n_changed']}"
          f"/{len(run2.solution)}", flush=True)

    path = os.path.join("results", f"case_study_{args.backend}.json")
    write_json(path, payload)
    print("wrote", path)


if __name__ == "__main__":
    main()
