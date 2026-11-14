# -*- coding: utf-8 -*-
r"""偏好同质性 × θ 的二维扫描：找出"θ↑ → 冗余↑ → α↓"到底在什么人群结构下成立。

    py -3.14 experiments/run_pref_homogeneity.py --phis 0,0.5,0.95 --thetas 0,0.3,0.6,0.9

背景（Day-2 实测）：单纯放大参与人数**不会**提高路线冗余（`results/density_stress_*`）。
真正决定 `ρ(θ)` 的是偏好同质性 φ（`aprime/homogeneity.py`）：

* φ 低（人群偏好分散）→ 高 θ 仍把不同人推向不同区域 → ρ 几乎不动，甚至下降；
* φ 高（偏好同质）→ 高 θ 把所有人推向同一片区域 → ρ↑、α_min↓、覆盖代价变大。

这条如果成立，定理 2 的单调性就不该写成无条件的，而应写成
**"在偏好同质性 φ ≥ φ₀ 的实例族上"** —— 这既更诚实，也更像一篇机制论文该有的刻画。
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
from aprime.analysis.preference import pref_concentration  # noqa: E402
from aprime.analysis.submodularity import (alpha_min_bound, apx_bound, overlap_floor,  # noqa: E402
                                          submodularity_ratio)
from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.homogeneity import with_homogeneity  # noqa: E402
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
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--phis", default="0,0.5,0.95")
    ap.add_argument("--thetas", default="0,0.3,0.6,0.9")
    ap.add_argument("--anchor", default="eco_enthusiast")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", default="0", help="逗号分隔的种子（每个种子一个实例）")
    ap.add_argument("--exact-max-elements", type=int, default=400)
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免不同网格互相覆盖")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("E2.3")  # 真实数据实验搁置中：docs/e0_1_data_access_plan.md §8

    phis = [float(p) for p in args.phis.split(",") if p]
    thetas = [float(t) for t in args.thetas.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    rng = random.Random(0)
    rows, t0 = [], time.time()

    for phi in phis:
        for theta in thetas:
            recs = []
            for s in seeds:
                inst = make_dense_instance(args.dataset, args.scale,
                                           n_workers=args.n_workers, seed=s, index=0)
                inst = with_homogeneity(inst, phi, archetype=args.anchor, seed=s)
                cands = build_candidates(inst, k=args.k, seed=s,
                                         lams=(1.0, 0.5, 0.0), n_random=2)
                prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=args.k))
                sel, _ = solve(prob, with_ls=True)
                ev = prob.evaluate(sel)
                rec = {"coverage": ev["coverage"], "budget": prob.budget,
                       "n_selected": len(sel), "pss": ev["pss_mean"],
                       "pss_min": ev["pss_min"],
                       "eligible_workers": len(prob.eligible_workers()),
                       "rho": overlap_floor(prob, samples=300, rng=rng),
                       "am": alpha_min_bound(prob, samples=300, rng=rng),
                       "alpha_hat": submodularity_ratio(prob, samples=200, rng=rng)["alpha"],
                       "phi_hat": pref_concentration(inst.profiles),
                       "opt": None}
                if len(prob.eligible) <= args.exact_max_elements:
                    rec["opt"] = prob.evaluate(exact(prob, time_limit=30.0)[0])["coverage"]
                recs.append(rec)

            cov = _m([r["coverage"] for r in recs])
            opt = _m([r["opt"] for r in recs])
            row = {
                "phi": phi, "theta": theta,
                "n_workers": args.n_workers, "anchor": args.anchor,
                "coverage": cov,
                "budget": recs[0]["budget"],
                "coverage_ratio_budget": round(cov / recs[0]["budget"], 4),
                "n_selected": _m([r["n_selected"] for r in recs]),
                "pss": _m([r["pss"] for r in recs]),
                "pss_min": _m([r["pss_min"] for r in recs]),
                "eligible_workers": _m([r["eligible_workers"] for r in recs]),
                "rho": _m([r["rho"] for r in recs]),
                "alpha_min": _m([r["am"]["alpha_min"] for r in recs]),
                "m_eff": _m([r["am"]["m"] for r in recs]),
                "alpha_hat": _m([r["alpha_hat"] for r in recs]),
                "phi_cos": _m([r["phi_hat"]["phi_cos"] for r in recs]),
                "top1_share": _m([r["phi_hat"]["top1_share"] for r in recs]),
                "entropy_norm": _m([r["phi_hat"]["entropy_norm"] for r in recs]),
                "opt": opt,
                "ratio": round(cov / opt, 4) if opt else None,
                "apx_bound_min": round(apx_bound(_m([r["am"]["alpha_min"] for r in recs]) or 0.0), 4),
                "elapsed": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"[φ={phi:.2f} θ={theta:.2f}] φ̂_cos {row['phi_cos']:.3f} | cov {cov:.1f}/"
                  f"{row['budget']:.0f} ({row['coverage_ratio_budget']:.3f}) | "
                  f"elig {row['eligible_workers']:.0f}/{args.n_workers} | ρ {row['rho']:.3f} "
                  f"| α_min {row['alpha_min']:.3f} | α̂ {row['alpha_hat']:.3f} "
                  f"| PSS {row['pss']:.3f} | A′/OPT {row['ratio']} | {row['elapsed']}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    stem = os.path.join(args.out_dir, f"pref_homogeneity_{args.dataset}_{args.scale}"
                                      f"_n{args.n_workers}{tag}")
    payload = {"dataset": args.dataset, "scale": args.scale, "n_workers": args.n_workers,
               "k": args.k, "anchor": args.anchor, "seeds": seeds,
               "note": "φ 为偏好同质性（向锚点 archetype 凸组合）；dense 实例只放大参与人数",
               "rows": rows}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    head = ("| φ 旋钮 | φ̂_cos（可测） | θ | Coverage/预算 | 合法参与者 | ρ(θ) | α_min | α̂ | "
            "PSS 均值 | A′/OPT |")
    lines = [f"# 偏好同质性 × θ（{args.dataset}/{args.scale}，N={args.n_workers}，k={args.k}，"
             f"锚点 {args.anchor}）", "",
             "φ 是**旋钮**（向锚点 archetype 凸组合），φ̂_cos 是**可测代理**（参与者两两偏好向量"
             "的平均余弦，见 `aprime/analysis/preference.py`）。论文里要用的是 φ̂_cos，"
             "旋钮只用来证明代理有判别力。", "", head, "|" + "---|" * 10]
    for r in rows:
        f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
        lines.append(f"| {r['phi']:.2f} | {f(r['phi_cos'])} | {r['theta']:.2f} | "
                     f"{f(r['coverage_ratio_budget'])} | {f(r['eligible_workers'], 0)} | "
                     f"{f(r['rho'])} | {f(r['alpha_min'])} | {f(r['alpha_hat'])} | "
                     f"{f(r['pss'])} | {f(r['ratio'])} |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", stem + ".json")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()


