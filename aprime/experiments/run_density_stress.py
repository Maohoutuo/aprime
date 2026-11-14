# -*- coding: utf-8 -*-
r"""密度加压：验证 "ρ(θ) ↑ → α ↓" 在**高竞争**下是否真的成立。

    py -3.14 experiments/run_density_stress.py --densities 20,80,320 --thetas 0,0.3,0.6,0.9

回答 Day-1 暴露的问题：小规模（20–60 人 / 预算 40–60 步）下覆盖≈预算、路线几乎不重叠，
α 议题没有实证。方案 E1.2 靠 500→5,000 workers 提高密度；这里先用"同栅格、只放大参与人数"
的加压实例把同样的密度造出来，看三件事：

1. `coverage / budget` 是否随密度下降（重叠真的咬住预算了）；
2. `ρ(θ)`（合法候选两两 Jaccard 重叠）随 θ 是否单调上升（命题 3 的假设 H）；
3. `α_min(θ) = 1/(1+(m−1)ρ)` 是否随 θ 单调下降（定理 2 的单调性）；
4. A′ 的 `coverage / OPT` 是否随之下降（θ 提高的真实代价）。

高密度下不跑 ILP（元素数几千），`--exact-max-elements` 控制，超过就跳过 OPT，只报 A′。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.analysis.submodularity import (alpha_min_bound, apx_bound, overlap_floor,  # noqa: E402
                                          submodularity_ratio)
from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.exact import exact  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


def _m(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 4) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--densities", default="20,80,320")
    ap.add_argument("--thetas", default="0,0.3,0.6,0.9")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--budget", type=float, default=None, help="默认同 SETTINGS 的步数预算")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instances", type=int, default=1, help="每个密度跑几个实例")
    ap.add_argument("--exact-max-elements", type=int, default=400)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("E1.2")  # 真实数据实验搁置中：docs/e0_1_data_access_plan.md §8

    densities = [int(d) for d in args.densities.split(",") if d]
    thetas = [float(t) for t in args.thetas.split(",") if t]
    rng = random.Random(args.seed)
    rows, t0 = [], time.time()

    for n_workers in densities:
        per_theta = []
        for inst_i in range(args.instances):
            inst = make_dense_instance(args.dataset, args.scale, n_workers=n_workers,
                                       budget=args.budget, seed=args.seed, index=inst_i)
            cands = build_candidates(inst, k=args.k, seed=args.seed + inst_i,
                                     lams=(1.0, 0.5, 0.0), n_random=2)
            for theta in thetas:
                prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=args.k))
                sel, _ = solve(prob, with_ls=True)
                rec = {"theta": theta,
                       "coverage": prob.evaluate(sel)["coverage"],
                       "budget": prob.budget,
                       "n_selected": len(sel),
                       "pss": prob.evaluate(sel)["pss_mean"],
                       "eligible_workers": len(prob.eligible_workers()),
                       "n_cand": len(cands),
                       "rho": overlap_floor(prob, samples=300, rng=rng),
                       "am": alpha_min_bound(prob, samples=300, rng=rng),
                       "alpha_hat": submodularity_ratio(prob, samples=200, rng=rng)["alpha"],
                       "opt": None}
                if len(prob.eligible) <= args.exact_max_elements:
                    rec["opt"] = prob.evaluate(exact(prob, time_limit=30.0)[0])["coverage"]
                per_theta.append(rec)

        for theta in thetas:
            rs = [r for r in per_theta if r["theta"] == theta]
            cov = _m([r["coverage"] for r in rs])
            opt = _m([r["opt"] for r in rs])
            row = {
                "n_workers": n_workers,
                "density": round(n_workers / 64.0, 3),
                "theta": theta,
                "coverage": cov,
                "budget": rs[0]["budget"],
                "coverage_ratio_budget": round(cov / rs[0]["budget"], 4),
                "n_selected": _m([r["n_selected"] for r in rs]),
                "pss": _m([r["pss"] for r in rs]),
                "eligible_workers": _m([r["eligible_workers"] for r in rs]),
                "n_cand": rs[0]["n_cand"],
                "rho": _m([r["rho"] for r in rs]),
                "alpha_min": _m([r["am"]["alpha_min"] for r in rs]),
                "m_eff": _m([r["am"]["m"] for r in rs]),
                "alpha_hat": _m([r["alpha_hat"] for r in rs]),
                "opt": opt,
                "ratio": round(cov / opt, 4) if opt else None,
                "apx_bound_min": round(apx_bound(_m([r["am"]["alpha_min"] for r in rs]) or 0.0), 4),
                "elapsed": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"[N={n_workers:4d} θ={theta:.1f}] cov {row['coverage']:.1f}/{row['budget']:.0f} "
                  f"({row['coverage_ratio_budget']:.3f}) | n {row['n_selected']:.1f} "
                  f"| elig {row['eligible_workers']:.0f}/{n_workers} | ρ {row['rho']:.3f} "
                  f"| α_min {row['alpha_min']:.3f} | α̂ {row['alpha_hat']:.3f} "
                  f"| PSS {row['pss']:.3f} | opt {row['opt']} | {row['elapsed']}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"density_stress_{args.dataset}_{args.scale}")
    payload = {"dataset": args.dataset, "scale": args.scale, "k": args.k,
               "budget": args.budget, "seed": args.seed, "instances": args.instances,
               "note": "密度加压实例：同栅格/同时间窗/同速度，只放大参与人数；非规模复现",
               "rows": rows}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    head = ("| N | 密度 | θ | Coverage/预算 | 入选 | 合法参与者 | ρ(θ) | α_min | α̂ | PSS | A′/OPT |")
    lines = [f"# 密度加压（{args.dataset}/{args.scale}，k={args.k}）",
             "",
             "同栅格、同时间窗、同速度口径，只放大参与人数。`ρ(θ)` 与 `α_min` 是本表的重点。",
             "", head, "|" + "---|" * 11]
    for r in rows:
        f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
        lines.append(f"| {r['n_workers']} | {r['density']:.2f} | {r['theta']:.1f} | "
                     f"{f(r['coverage_ratio_budget'])} | {f(r['n_selected'], 1)} | "
                     f"{f(r['eligible_workers'], 0)} | {f(r['rho'])} | {f(r['alpha_min'])} | "
                     f"{f(r['alpha_hat'])} | {f(r['pss'])} | {f(r['ratio'])} |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", stem + ".json")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()
