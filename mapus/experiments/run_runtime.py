# -*- coding: utf-8 -*-
r"""表 5：各方法在不同规模下的运行时间。

    py -3.14 experiments\run_runtime.py --n 5 --backend surrogate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CLASSICAL, build_llm, write_json
from agentsense.config import DEFAULT_ENV
from mapus.config import MAPUSConfig
from mapus.data.instances import make_mapus_instance
from mapus.pipeline import run_classical, run_mapus, run_single_llm

PAPER_RUNTIME = {  # 论文表 5（秒），用于对照
    "tdrive": {"small": {"RN": 6.8e-4, "TVPG": 0.010, "TCPG": 0.012, "MSA": 0.040, "MSAGI": 0.089,
                         "GraphDP": 0.160, "GPT-4.1-mini": 158.72, "MAPUS": 232.12},
               "medium": {"RN": 4.5e-4, "TVPG": 0.048, "TCPG": 0.112, "MSA": 0.077, "MSAGI": 0.322,
                          "GraphDP": 7.763, "GPT-4.1-mini": 187.15, "MAPUS": 254.66},
               "large": {"RN": 1.1e-3, "TVPG": 0.472, "TCPG": 1.155, "MSA": 0.217, "MSAGI": 2.442,
                         "GraphDP": 147.94, "GPT-4.1-mini": 210.32, "MAPUS": 283.41}},
    "grab": {"small": {"RN": 3.0e-4, "TVPG": 0.0039, "TCPG": 0.0044, "MSA": 0.0319, "MSAGI": 0.0509,
                       "GraphDP": 0.0366, "GPT-4.1-mini": 125.52, "MAPUS": 228.62},
             "medium": {"RN": 9.4e-4, "TVPG": 0.0212, "TCPG": 0.0394, "MSA": 0.0616, "MSAGI": 0.168,
                        "GraphDP": 0.798, "GPT-4.1-mini": 156.38, "MAPUS": 243.06},
             "large": {"RN": 1.6e-3, "TVPG": 0.266, "TCPG": 0.388, "MSA": 0.157, "MSAGI": 1.502,
                       "GraphDP": 15.22, "GPT-4.1-mini": 183.91, "MAPUS": 262.74}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive", choices=["tdrive", "grab"])
    ap.add_argument("--scales", default="small,medium,large")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--backend", default="surrogate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--which", default="all", help="all | classical | llm")
    args = ap.parse_args()

    cfg = MAPUSConfig()
    llm = build_llm(args.backend, args.model, workers=args.llm_workers)
    payload = {"dataset": args.dataset, "n": args.n, "backend": args.backend,
               "model": getattr(llm, "name", ""), "runtime": {}}
    for scale in args.scales.split(","):
        payload["runtime"][scale] = {}
        setting = f"{args.dataset}_{scale}"
        methods = CLASSICAL if args.which == "classical" else (
            ["GPT-4.1-mini", "MAPUS"] if args.which == "llm" else CLASSICAL + ["GPT-4.1-mini", "MAPUS"])
        for name in methods:
            times = []
            for i in range(args.n):
                inst = make_mapus_instance(setting, i)
                t0 = time.time()
                if name in CLASSICAL:
                    run_classical(name, inst, DEFAULT_ENV, seed=1000 + i)
                elif name == "MAPUS":
                    run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm,
                              seed=1000 + i, parallel=args.llm_workers)
                else:
                    run_single_llm(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm, seed=1000 + i)
                times.append(time.time() - t0)
            ref = PAPER_RUNTIME[args.dataset][scale].get(name)
            payload["runtime"][scale][name] = {"mean": round(mean(times), 4)}
            print(f"[{setting}] {name:14s} {mean(times):9.3f}s   paper {ref if ref else '-'}", flush=True)

    tag = args.backend if not args.model else args.model.replace("/", "_")
    path = os.path.join("results", f"runtime_{args.dataset}_{tag}_n{args.n}.json")
    write_json(path, payload)
    print("wrote", path)


if __name__ == "__main__":
    main()
