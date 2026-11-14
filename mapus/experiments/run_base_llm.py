# -*- coding: utf-8 -*-
r"""图 11：不同基座 LLM 下的性能—时延权衡（Medium 设置）。

    py -3.14 experiments\run_base_llm.py --n 2 --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-72B-Instruct
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

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-32B",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--datasets", default="tdrive,grab")
    ap.add_argument("--scale", default="medium")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--llm-workers", type=int, default=6)
    args = ap.parse_args()

    cfg = MAPUSConfig()
    payload = {"scale": args.scale, "n": args.n, "models": {}}
    for model in [m for m in args.models.split(",") if m]:
        llm = build_llm("siliconflow", model, workers=args.llm_workers)
        entry = {}
        for dataset in args.datasets.split(","):
            setting = f"{dataset}_{args.scale}"
            def _one(i: int, _setting=setting):
                inst = make_mapus_instance(_setting, i)
                run = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm,
                                seed=1000 + i, parallel=args.llm_workers)
                out = dict(run.stats)
                out["runtime"] = run.runtimes["total"]
                return out
            rows = run_instances(_one, list(range(args.n)), parallel=args.parallel)
            agg = aggregate(rows, ["coverage", "pss", "runtime"])
            entry[dataset] = {"coverage": round(agg["coverage"]["mean"], 4),
                              "pss": round(agg["pss"]["mean"], 4),
                              "runtime": round(agg["runtime"]["mean"], 2)}
            print(f"[{model}][{dataset}] cov {entry[dataset]['coverage']:.3f} "
                  f"pss {entry[dataset]['pss']:.3f} t {entry[dataset]['runtime']:.1f}s", flush=True)
        payload["models"][model] = entry

    path = os.path.join("results", f"base_llm_n{args.n}.json")
    write_json(path, payload)
    print("wrote", path)


if __name__ == "__main__":
    main()
