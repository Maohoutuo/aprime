# -*- coding: utf-8 -*-
r"""E1.1 预览：θ-权衡曲线（Fig.3 的数据基础）+ E2.3 的 α(θ) 实测。

    py -3.14 experiments/run_theta_curve.py --setting tdrive_small --n 3

每个实例只生成**一次**与 θ 无关的候选池，θ 扫 10 档全程复用（方案 §7.4 的成本前提）。
每档记录 A′ / 精确最优（可行域含 θ）/ 四个对照 / α(θ) / APX(θ)。
输出 results/theta_curve_<setting>_n<n>.json 与同名 .md。
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

import aprime  # noqa: F401,E402  （把 mapus / agentsense 加进 sys.path）
from aprime.analysis.submodularity import (alpha_min_bound, apx_bound, overlap_stats,  # noqa: E402
                                          submodularity_ratio)
from aprime.candidates import build_candidates  # noqa: E402
from aprime.availability import require_available  # noqa: E402
from aprime.config import THETA_GRID, APrimeConfig  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.baselines import (capacity_greedy, iterative_greedy, random_feasible,  # noqa: E402
                                      sa_greedy, unconstrained_exact)
from aprime.solvers.exact import exact  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402
from mapus.data.instances import data_source_label, make_mapus_instance  # noqa: E402


def _m(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 4) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="tdrive_small")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--budget-kind", default="steps", choices=["steps", "count"])
    ap.add_argument("--theta", default="")
    ap.add_argument("--exact-max-elements", type=int, default=400)
    ap.add_argument("--sa-iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("E1.1")  # 真实数据实验搁置中：docs/e0_1_data_access_plan.md §8

    thetas = [float(t) for t in args.theta.split(",") if t] or THETA_GRID
    rng = random.Random(args.seed)
    rows, t0 = [], time.time()

    for theta in thetas:
        recs = []
        for i in range(args.n):
            inst = make_mapus_instance(args.setting, i)
            cands = build_candidates(inst, k=args.k, seed=args.seed + i)
            prob = make_problem(inst, cands,
                                APrimeConfig(theta=theta, budget_kind=args.budget_kind, k_cand=args.k))
            rec = {"n_eligible_workers": len(prob.eligible_workers())}
            rec["aprime"] = prob.evaluate(solve(prob, with_ls=True)[0])
            rec["exact"] = (prob.evaluate(exact(prob, time_limit=20.0)[0])
                            if len(prob.eligible) <= args.exact_max_elements else None)
            rec["random"] = prob.evaluate(random_feasible(prob, rng, trials=100)[0])
            rec["iter"] = prob.evaluate(iterative_greedy(prob, rng, restarts=6)[0])
            rec["sa"] = prob.evaluate(sa_greedy(prob, rng, iters=args.sa_iters)[0])
            sel_cap, info_cap = capacity_greedy(prob)
            rec["capacity"] = prob.evaluate(sel_cap)
            rec["cap_viol"] = info_cap["violates_theta"]
            if i == 0:
                rec["unconstrained"] = prob.evaluate(unconstrained_exact(prob, time_limit=20.0)[0])
            rec["alpha"] = submodularity_ratio(prob, samples=160, rng=rng)["alpha"]
            rec["ov"] = overlap_stats(prob, samples=200, rng=rng)
            rec["am"] = alpha_min_bound(prob, samples=200, rng=rng)
            recs.append(rec)

        def cov(key):
            return _m([r[key]["coverage"] for r in recs if r.get(key)])

        alpha = _m([r["alpha"] for r in recs])
        row = {
            "theta": theta,
            "coverage_aprime": cov("aprime"),
            "coverage_exact": cov("exact"),
            "coverage_random": cov("random"),
            "coverage_iter": cov("iter"),
            "coverage_sa": cov("sa"),
            "coverage_capacity": cov("capacity"),
            "coverage_unconstrained": cov("unconstrained"),
            "n_selected": _m([r["aprime"]["n_selected"] for r in recs]),
            "pss_mean": _m([r["aprime"]["pss_mean"] for r in recs]),
            "pss_min": _m([r["aprime"]["pss_min"] for r in recs]),
            "capacity_theta_violations": _m([r["cap_viol"] for r in recs]),
            "eligible_workers": _m([r["n_eligible_workers"] for r in recs]),
            "alpha": alpha,
            "apx_bound": round(apx_bound(alpha or 0.0), 4),
            "overlap_floor": _m([r["ov"]["mean"] for r in recs]),
            "rho_p95": _m([r["ov"]["p95"] for r in recs]),
            "alpha_min": _m([r["am"]["alpha_min"] for r in recs]),
            "m_eff": _m([r["am"]["m"] for r in recs]),
            "apx_bound_min": round(apx_bound(_m([r["am"]["alpha_min"] for r in recs]) or 0.0), 4),
            "elapsed": round(time.time() - t0, 1),
        }
        row["ratio"] = (round(row["coverage_aprime"] / row["coverage_exact"], 4)
                        if row["coverage_exact"] else None)
        rows.append(row)
        print(f"[θ={theta:.1f}] A′ {row['coverage_aprime']:.2f} | exact {row['coverage_exact']} "
              f"| ratio {row['ratio']} | rand {row['coverage_random']:.2f} "
              f"| SA {row['coverage_sa']:.2f} | CFI {row['coverage_capacity']:.2f} "
              f"| n {row['n_selected']:.1f} | PSS {row['pss_mean']:.3f} "
              f"| α̂ {row['alpha']:.3f} | α_min {row['alpha_min']:.3f} "
              f"| APX̂ {row['apx_bound']:.3f} | APX_min {row['apx_bound_min']:.3f} "
              f"| ρ̄ {row['overlap_floor']:.3f} | ρ95 {row['rho_p95']:.3f} | "
              f"m {row['m_eff']:.0f} | {row['elapsed']}s",
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"theta_curve_{args.setting}_n{args.n}")
    payload = {"setting": args.setting, "n": args.n, "k": args.k,
               "budget_kind": args.budget_kind,
               "data_source": data_source_label(args.setting.split("_")[0]),
               "seed": args.seed, "rows": rows}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    head = ("| θ | A′ Cov | 精确 Cov | A′/OPT | random | 迭代贪心 | SA | CFI 型 | 选中数 | "
            "PSS 均值 | α̂ 实测 | α_min 下界 | APX̂ | APX_min | ρ̄(θ) | ρ95(θ) | m |")
    lines = [f"# θ-权衡曲线（{args.setting}，n={args.n}，k={args.k}，预算口径 {args.budget_kind}）",
             "",
             "α̂ 是 `submodularity_ratio` 的采样最小值（极值统计，噪声大）；α_min 是 "
             "`alpha_min_bound` 的下界，按 §3.2 用**重叠系数**与 `m=|E(θ)|`（空界，仅作诊断）。"
             "**不要再用 α 论证近似比**，见 `docs/theory_alpha_min.md` §5.4。",
             "", head, "|" + "---|" * 17]
    for r in rows:
        f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
        lines.append(f"| {r['theta']:.1f} | {f(r['coverage_aprime'])} | {f(r['coverage_exact'])} | "
                     f"{f(r['ratio'], 3)} | {f(r['coverage_random'])} | {f(r['coverage_iter'])} | "
                     f"{f(r['coverage_sa'])} | {f(r['coverage_capacity'])} | {f(r['n_selected'], 1)} | "
                     f"{f(r['pss_mean'], 3)} | {f(r['alpha'], 3)} | {f(r['alpha_min'], 3)} | "
                     f"{f(r['apx_bound'], 3)} | {f(r['apx_bound_min'], 3)} | "
                     f"{f(r['overlap_floor'], 3)} | {f(r['rho_p95'], 3)} | {f(r['m_eff'], 0)} |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", stem + ".json")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()




