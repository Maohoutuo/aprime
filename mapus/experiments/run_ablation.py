# -*- coding: utf-8 -*-
r"""图 3 的消融实验：w/o PRG、w/o FPS、w/o NRR、完整模型。

    py -3.14 experiments\run_ablation.py --n 10 --backend surrogate
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import aggregate, build_llm, run_instances, write_json
from agentsense.config import DEFAULT_ENV
from mapus.config import MAPUSConfig
from mapus.data.instances import make_mapus_instance
from mapus.pipeline import run_mapus

VARIANTS = {
    "w/o PRG": dict(use_prg=False),
    "w/o FPS": dict(use_fps=False),
    "w/o NRR": dict(use_nrr=False),
    "Full": dict(),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scales", default="small,medium,large")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--backend", default="surrogate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    llm = build_llm(args.backend, args.model, workers=args.llm_workers)
    payload = {"dataset": args.dataset, "n": args.n, "backend": args.backend,
               "model": getattr(llm, "name", ""), "results": {}}
    for scale in args.scales.split(","):
        payload["results"][scale] = {}
        setting = f"{args.dataset}_{scale}"
        for name, over in VARIANTS.items():
            cfg = MAPUSConfig(**over)
            def _one(i: int, _cfg=cfg, _setting=setting):
                inst = make_mapus_instance(_setting, i)
                run = run_mapus(inst, inst.urban, inst.profiles, _cfg, DEFAULT_ENV,
                                llm, seed=1000 + i, parallel=args.llm_workers)
                out = dict(run.stats)
                out["runtime"] = run.runtimes["total"]
                return out
            rows = run_instances(_one, list(range(args.n)), parallel=args.parallel)
            agg = aggregate(rows, ["coverage", "pss", "avg_overlap", "n_selected", "runtime"])
            payload["results"][scale][name] = {
                "coverage": round(agg["coverage"]["mean"], 4),
                "coverage_std": round(agg["coverage"]["std"], 4),
                "pss": round(agg["pss"]["mean"], 4),
                "pss_std": round(agg["pss"]["std"], 4),
                "overlap": round(agg["avg_overlap"]["mean"], 4),
                "n_selected": round(agg["n_selected"]["mean"], 2),
                "runtime": round(agg["runtime"]["mean"], 2),
            }
            print(f"[{setting}] {name:9s} cov {agg['coverage']['mean']:.3f} "
                  f"pss {agg['pss']['mean']:.3f} overlap {agg['avg_overlap']['mean']:.3f}", flush=True)

    tag = args.tag or args.backend
    path = os.path.join("results", f"ablation_{args.dataset}_{tag}_n{args.n}.json")
    write_json(path, payload)
    print("wrote", path)


if __name__ == "__main__":
    main()
