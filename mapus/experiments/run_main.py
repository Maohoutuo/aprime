# -*- coding: utf-8 -*-
r"""复现论文表 2（T-Drive）与表 3（Grab-Posisi）的主结果。

    # 离线、确定性代理（不消耗额度，可跑满 20 个实例）
    py -3.14 experiments\run_main.py --dataset tdrive --n 20 --parallel 4

    # 真实模型
    py -3.14 experiments\run_main.py --dataset tdrive --n 3 --backend siliconflow `
        --model Qwen/Qwen3-32B --parallel 3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CLASSICAL, PAPER, aggregate, build_llm, fmt, run_instances, run_method, write_json
from mapus.config import DEFAULT_MAPUS, MAPUSConfig
from mapus.data.instances import data_source_label, make_mapus_instance

METHODS = CLASSICAL + ["GPT-4.1-mini", "MAPUS"]
KEYS = ["coverage", "pss", "runtime", "n_selected", "cost",
        "n_accepted", "n_refused", "llm_calls_gen", "llm_invalid", "llm_repaired",
        "refines_accepted", "sel_llm_ties", "neg_attempts", "neg_accepted",
        "neg_rejected", "neg_failed_pairs", "avg_overlap", "pre_neg_overlap",
        "pre_neg_coverage", "pre_neg_pss", "entropy", "pre_neg_entropy", "count"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive", choices=["tdrive", "grab"])
    ap.add_argument("--scales", default="small,medium,large")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backend", default="surrogate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--parallel", type=int, default=4, help="同时运行的实例数")
    ap.add_argument("--llm-workers", type=int, default=6, help="单个实例内并行的智能体数")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    llm = build_llm(args.backend, args.model, workers=args.llm_workers, no_cache=args.no_cache)
    setattr(llm, "_parallel", args.llm_workers)
    # 真实模型受 TPM 限流约束：同时跑的实例数要收敛（离线代理没有这个问题）
    parallel = args.parallel if args.backend == "surrogate" else min(args.parallel, 2)
    methods = [m for m in args.methods.split(",") if m]
    cfg = MAPUSConfig()

    payload = {"dataset": args.dataset, "n": args.n, "backend": args.backend,
               "model": getattr(llm, "name", ""), "methods": methods,
               "data_source": data_source_label(args.dataset), "scales": {}}
    if args.dataset == "grab" and payload["data_source"] != "grab_official":
        print("[run_main] Grab-Posisi 官方数据尚未获得：本次结果建立在合成城市 + "
              "合成同分布移动性上，只能作为机制验证，不能写成数据集复现。", file=sys.stderr)
    for scale in args.scales.split(","):
        setting = f"{args.dataset}_{scale}"
        payload["scales"][scale] = {}
        for name in methods:
            t0 = os.times()
            def _one(i: int, _name=name, _setting=setting):
                inst = make_mapus_instance(_setting, i)
                return run_method(_name, inst, cfg, llm, seed=1000 + i)
            rows = run_instances(_one, list(range(args.n)), parallel=parallel)
            stats = aggregate(rows, KEYS)
            payload["scales"][scale][name] = {
                "coverage": round(stats["coverage"]["mean"], 4),
                "coverage_std": round(stats["coverage"]["std"], 4),
                "pss": round(stats["pss"]["mean"], 4),
                "pss_std": round(stats["pss"]["std"], 4),
                "runtime": round(stats["runtime"]["mean"], 3),
                "n_selected": round(stats["n_selected"]["mean"], 2),
                "cost": round(stats["cost"]["mean"], 2),
                "detail": {k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                           for k, v in stats.items()},
            }
            ref = PAPER[args.dataset][scale].get(name)
            extra = ""
            if ref:
                extra = (f"  | paper {ref[0]:.3f}/{ref[1]:.3f}  "
                         f"dCov {(stats['coverage']['mean']-ref[0])/ref[0]*100:+.1f}%  "
                         f"dPSS {stats['pss']['mean']-ref[1]:+.3f}")
            print(f"[{args.dataset}/{scale}] {name:14s} "
                  f"cov {stats['coverage']['mean']:.3f}±{stats['coverage']['std']:.3f}  "
                  f"pss {stats['pss']['mean']:.3f}±{stats['pss']['std']:.3f}  "
                  f"n {stats['n_selected']['mean']:.1f}  t {stats['runtime']['mean']:.1f}s{extra}",
                  flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.backend.replace("/", "_")
    path = os.path.join(args.out_dir, f"main_{args.dataset}_{tag}_n{args.n}.json")
    write_json(path, payload)
    _write_markdown(payload, os.path.splitext(path)[0] + ".md")
    print("wrote", path)


def _write_markdown(payload: dict, path: str) -> None:
    ds = payload.get("data_source") or ""
    lines = [f"# 主结果复现（{payload['dataset']}，n={payload['n']}，backend={payload['backend']}"
             f"{'，model=' + str(payload['model']) if payload.get('model') else ''}"
             f"{'，data_source=' + ds if ds else ''}）", ""]
    if ds.startswith("synthetic"):
        lines += [f"> 数据来源：`{ds}`——官方数据集未获得，本表用于机制验证，"
                  "不能作为官方数据集的复现结论。", ""]
    lines.append("| Method | Scale | Coverage | PSS | paper Coverage | paper PSS |")
    lines.append("|---|---|---|---|---|---|")
    for scale, per in payload["scales"].items():
        for name, v in per.items():
            ref = PAPER[payload["dataset"]][scale].get(name)
            lines.append(f"| {name} | {scale} | {v['coverage']:.3f} ± {v['coverage_std']:.3f} | "
                         f"{v['pss']:.3f} ± {v['pss_std']:.3f} | "
                         f"{ref[0]:.3f} | {ref[1]:.3f} |" if ref else
                         f"| {name} | {scale} | {v['coverage']:.3f} ± {v['coverage_std']:.3f} | "
                         f"{v['pss']:.3f} ± {v['pss_std']:.3f} | - | - |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
