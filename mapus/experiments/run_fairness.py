# -*- coding: utf-8 -*-
r"""图 4：公平感知的参与者选择（30 人固定池 + 60 个传感任务）。

四种策略：
  UR  均匀随机
  LS  最少被选中优先（强公平，忽略效用）
  CG  纯覆盖贪心（β=1，无公平项、无 LLM 平局裁决）
  FPS 本文方法（β=0.5 + 历史次数公平项）

    py -3.14 experiments\run_fairness.py --n-tasks 60 --backend surrogate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor

from common import build_llm, write_json
from agentsense.config import DEFAULT_ENV
from mapus.agents.coordinator import CoordinatorAgent
from mapus.agents.participant import ParticipantAgent
from mapus.config import MAPUSConfig
from mapus.data.instances import make_pool_instance
from mapus.env.metrics_obj import evaluate_mapus
from mapus.pipeline import run_classical

POOL = 30
MAX_PICK = 20


def _gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _random_pick(routes, pool, budget, rng) -> dict:
    order = list(routes)
    rng.shuffle(order)
    chosen, left = {}, budget
    for wid in order:
        cost = pool.worker(wid).window
        if len(chosen) >= MAX_PICK:
            break
        if cost <= left:
            chosen[wid] = routes[wid]
            left -= cost
    return chosen


def _least_selected(routes, pool, budget, hist) -> dict:
    order = sorted(routes, key=lambda w: (hist.get(w, 0), w))
    chosen, left = {}, budget
    for wid in order:
        cost = pool.worker(wid).window
        if len(chosen) >= MAX_PICK:
            break
        if cost <= left:
            chosen[wid] = routes[wid]
            left -= cost
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--backend", default="surrogate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--route-workers", type=int, default=6,
                    help="同一任务内并行生成路线的参与者数（真实 LLM 下主要影响墙钟）")
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = MAPUSConfig()
    llm = build_llm(args.backend, args.model, workers=args.llm_workers)
    strategies = ["UR", "LS", "CG", "FPS"]
    hist = {s: {f"w{i:03d}": 0 for i in range(POOL)} for s in strategies}
    rows = {s: [] for s in strategies}
    t0 = time.time()

    for t in range(args.n_tasks):
        pool = make_pool_instance("tdrive", "medium", pool_size=POOL, task_index=t, seed=args.seed)
        # 同一份候选路线，四种策略在相同候选集上做选择
        routes = {}
        workers = list(pool.workers)

        def _gen(w):
            pa = ParticipantAgent(pool, w, pool.profiles[w.wid], pool.urban, cfg, DEFAULT_ENV, llm,
                                  seed=t * 31 + int(w.wid[1:]))
            return w.wid, pa.generate_route()

        with ThreadPoolExecutor(max_workers=max(1, args.route_workers)) as pool_exec:
            for wid, out in pool_exec.map(_gen, workers):
                if out:
                    routes[wid] = out["route"]
        ca = CoordinatorAgent(pool, pool.urban, cfg, DEFAULT_ENV, llm)
        prefs = pool.prefs

        picked = {}
        picked["UR"] = _random_pick(routes, pool, pool.budget, random.Random(t * 7 + 1))
        picked["LS"] = _least_selected(routes, pool, pool.budget, hist["LS"])
        for name, kwargs in (("CG", dict(beta=1.0)), ("FPS", dict())):
            cfg_use = MAPUSConfig(use_fps=(name == "FPS"), tie_break="rule")
            ca_use = CoordinatorAgent(pool, pool.urban, cfg_use, DEFAULT_ENV, llm)
            picked[name], _ = ca_use.select(routes, pool.profiles, histories=hist[name],
                                            budget=pool.budget, beta=kwargs.get("beta"))
        for name in strategies:
            for wid in picked[name]:
                hist[name][wid] = hist[name].get(wid, 0) + 1
            ev = evaluate_mapus(pool, picked[name], pool.urban, prefs, DEFAULT_ENV,
                                mu=cfg.mu, prior=cfg.pss_prior, scale=cfg.pss_scale)
            rows[name].append({"coverage": ev["coverage"], "pss": ev["pss"],
                               "n": ev["n_selected"]})
        if (t + 1) % 5 == 0 or t == 0:
            print(f"  task {t+1}/{args.n_tasks} done  ({time.time()-t0:.0f}s)",
                  flush=True)

    out = {}
    for name in strategies:
        counts = np.array([hist[name][f"w{i:03d}"] for i in range(POOL)])
        cov = float(np.mean([r["coverage"] for r in rows[name]]))
        pss = float(np.mean([r["pss"] for r in rows[name]]))
        out[name] = {"coverage": round(cov, 4), "pss": round(pss, 4),
                     "counts": counts.tolist(), "gini": round(_gini(counts), 4),
                     "max_count": int(counts.max()), "std_count": round(float(counts.std()), 3)}
        print(f"{name:4s} cov {cov:.3f} pss {pss:.3f} gini {out[name]['gini']:.3f} "
              f"max {counts.max()} std {counts.std():.2f}", flush=True)

    tag = args.tag or args.backend
    write_json(os.path.join("results", f"fairness_{tag}.json"),
               {"pool": POOL, "tasks": args.n_tasks, "backend": args.backend,
                "model": getattr(llm, "name", ""), "strategies": out})
    print("wrote results/fairness.json")


if __name__ == "__main__":
    main()
