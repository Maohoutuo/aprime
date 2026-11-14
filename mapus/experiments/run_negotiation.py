# -*- coding: utf-8 -*-
r"""表 4：协商式路线精修（NRR）前后的路线统计。

    py -3.14 experiments\run_negotiation.py --n 20 --backend surrogate
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

KEYS = ["pre_neg_overlap", "avg_overlap", "pre_neg_coverage", "coverage",
        "pre_neg_entropy", "entropy", "pre_neg_count", "count", "pre_neg_pss", "pss",
        "neg_attempts", "neg_accepted", "neg_rejected", "n_selected"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="medium")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backend", default="surrogate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tau", type=float, default=None,
                    help="覆盖重叠阈值 τ（默认取 MAPUSConfig.overlap_tau）；用于速度口径变更后的敏感性对照")
    args = ap.parse_args()

    cfg = MAPUSConfig()
    if args.tau is not None:
        cfg.overlap_tau = float(args.tau)
    llm = build_llm(args.backend, args.model, workers=args.llm_workers)
    payload = {"scale": args.scale, "n": args.n, "backend": args.backend,
               "tau": cfg.overlap_tau,
               "model": getattr(llm, "name", ""), "datasets": {}}
    for dataset in ("tdrive", "grab"):
        setting = f"{dataset}_{args.scale}"
        def _one(i: int, _setting=setting):
            inst = make_mapus_instance(_setting, i)
            return run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm,
                             seed=1000 + i, parallel=args.llm_workers).stats
        rows = run_instances(_one, list(range(args.n)), parallel=args.parallel)
        agg = aggregate(rows, KEYS)
        payload["datasets"][dataset] = {k: round(v["mean"], 4) for k, v in agg.items()}
        d = payload["datasets"][dataset]
        print(f"[{dataset}] overlap {d['pre_neg_overlap']:.3f} -> {d['avg_overlap']:.3f} | "
              f"cov {d['pre_neg_coverage']:.3f} -> {d['coverage']:.3f} | "
              f"H {d['pre_neg_entropy']:.3f} -> {d['entropy']:.3f} | "
              f"count {d['pre_neg_count']:.1f} -> {d['count']:.1f} | "
              f"pss {d['pre_neg_pss']:.3f} -> {d['pss']:.3f} | "
              f"pairs {d['neg_attempts']:.1f} acc {d['neg_accepted']:.1f}", flush=True)

    tag = args.tag or args.backend
    path = os.path.join("results", f"negotiation_{tag}_n{args.n}.json")
    write_json(path, payload)
    print("wrote", path)


if __name__ == "__main__":
    main()
