# -*- coding: utf-8 -*-
r"""E3.1 / E3.2：CFI 型公平的**靶子实验**（定理 3 的直接证据）。

    py -3.14 experiments/run_cfi_target.py --seeds 5

方案 §4.1：**构造占 30% 的低能力者子群**，对照东华 SA 贪心（CFI 型），
验证「最大化 CFI 型公平必然拒绝低能力参与者」。

## 低能力子群怎么定义（必须可复现、不依赖结果）

按 `|F_u(θ)|`（该参与者在该 θ 下还剩的合法路线条数）**升序**排序，取前 30% 为 `low` 组，
其余为 `high` 组。这是数据驱动的、与算法无关的划分，避免"用算法输出定义受害者"的循环论证。
θ 变化时子群成员会变——这本身是现象的一部分（θ 越高，被推进 low 组的人越多），
所以每个 θ 单独划分，并在结果里报出两组人数。

## 指标

* `excl_low` / `excl_high`：两组在**合法池内**的被排除率；
* `excl_gap = excl_low − excl_high`（正值 = 低能力者被排除得更狠）；
* `select_share_low`：名额落到 low 组的比例（应 ≈ 30% 才算配额一致）；
* `gqc_gap`：名额份额 vs 人数份额的最大偏差；
* `theta_violations`：θ 违规数（CFI 型基线的代价）。

输出 `results/cfi_target_<setting>_n<n>.json` / `.md`。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from statistics import mean
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import capability_groups, gqc_gap as _gqc_gap, group_shares  # noqa: E402
from aprime.model import Candidate, SelectionProblem, make_problem  # noqa: E402
from aprime.solvers.baselines import (capacity_greedy, mscta_greedy,  # noqa: E402
                                      sa_caftca, sa_greedy)
from aprime.solvers.gqc import gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve as solve_aprime  # noqa: E402


def _problem(inst, cands, theta: float, budget_steps: float) -> SelectionProblem:
    return make_problem(inst, cands,
                        APrimeConfig(theta=theta, budget=budget_steps,
                                     budget_kind="steps"))


def split_low_high(problem: SelectionProblem, frac: float = 0.30
                   ) -> Tuple[List[str], List[str], Dict[str, int]]:
    """按合法路线条数升序切出低能力子群（只含合法池内的人）。"""
    n_elig = {w: sum(1 for c in problem.by_worker[w] if problem.is_eligible(c))
              for w in problem.by_worker}
    feasible = [w for w, n in n_elig.items() if n >= 1]
    feasible.sort(key=lambda w: (n_elig[w], w))
    k_low = max(1, int(round(len(feasible) * frac)))
    low = feasible[:k_low]
    high = feasible[k_low:]
    return low, high, n_elig


def metrics(problem: SelectionProblem, sel: Sequence[Candidate],
            low: List[str], high: List[str]) -> dict:
    chosen = {c.wid for c in sel}
    lo_sel = sum(1 for w in low if w in chosen)
    hi_sel = sum(1 for w in high if w in chosen)
    excl_low = 1.0 - lo_sel / len(low) if low else 0.0
    excl_high = 1.0 - hi_sel / len(high) if high else 0.0
    total_sel = lo_sel + hi_sel
    group_of = capability_groups(problem)
    return {
        "n_select": len(sel),
        "low_sel": lo_sel, "high_sel": hi_sel,
        "select_share_low": round(lo_sel / total_sel, 4) if total_sel else 0.0,
        "excl_low": round(excl_low, 4), "excl_high": round(excl_high, 4),
        "excl_gap": round(excl_low - excl_high, 4),
        "gqc_gap": round(_gqc_gap(sel, group_of), 4),
        "theta_violations": sum(1 for c in sel if not problem.is_eligible(c)),
        "low_group_frac_feasible": round(len(low) / max(1, len(low) + len(high)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--budget-steps", type=float, default=120.0)
    ap.add_argument("--thetas", default="0.0,0.3,0.5,0.7,0.9")
    ap.add_argument("--low-frac", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("SEC6")  # 真实数据实验搁置中：docs/e0_1_data_access_plan.md §8

    thetas = [float(x) for x in args.thetas.split(",") if x]
    methods = ("aprime", "gqc", "capacity", "mscta", "sa_caftca", "sa")
    rows: List[dict] = []

    for seed in range(args.seeds):
        inst = make_dense_instance(args.dataset, args.scale,
                                   n_workers=args.n_workers, seed=seed, index=0)
        cands = build_candidates(inst, k=args.k, seed=seed,
                                 lams=(1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0),
                                 n_random=16)
        for theta in thetas:
            prob = _problem(inst, cands, theta, args.budget_steps)
            low, high, n_elig = split_low_high(prob, args.low_frac)
            if not low or not high:
                continue
            rng = random.Random(seed * 1000 + int(theta * 100))

            out: Dict[str, Sequence[Candidate]] = {}
            out["aprime"], _ = solve_aprime(prob, with_ls=True)
            try:
                out["gqc"], _ = gqc_solve(prob, rng=rng)
            except Exception:  # noqa: BLE001  配额不可行时跳过
                out["gqc"] = []
            out["capacity"], _ = capacity_greedy(prob, rng=rng)
            out["mscta"], _ = mscta_greedy(prob, rng=rng)
            out["sa_caftca"], _ = sa_caftca(prob, rng=rng, iters=2500)
            out["sa"], _ = sa_greedy(prob, rng=rng, iters=2500)

            for m in methods:
                sel = out.get(m, [])
                rec = metrics(prob, sel, low, high)
                rec.update({
                    "seed": seed, "theta": theta, "method": m,
                    "cov": prob.coverage(sel),
                    "n_low": len(low), "n_high": len(high),
                    "low_cap_median": int(sorted(n_elig[w] for w in low)[len(low) // 2]),
                    "high_cap_median": int(sorted(n_elig[w] for w in high)[len(high) // 2]),
                })
                rows.append(rec)
        print(f"[seed {seed}] done")

    # ---------------------------------------------------------------- 汇总
    def agg(method, theta, field):
        v = [r[field] for r in rows if r["method"] == method and r["theta"] == theta]
        return round(mean(v), 4) if v else None

    print(f"\n===== E3.1 CFI 靶子（低能力子群 = 合法池前 {args.low_frac:.0%}）=====")
    print(f"{'theta':>5s} {'method':10s} {'cov':>5s} {'excl_low':>9s} {'excl_high':>9s} "
          f"{'gap':>7s} {'share_low':>9s} {'gqc':>6s} {'θ违反':>5s}")
    for theta in thetas:
        for m in methods:
            cov = agg(m, theta, "cov")
            if cov is None:
                continue
            print(f"{theta:5.1f} {m:10s} {cov:5.1f} {agg(m,theta,'excl_low'):9.3f} "
                  f"{agg(m,theta,'excl_high'):9.3f} {agg(m,theta,'excl_gap'):+7.3f} "
                  f"{agg(m,theta,'select_share_low'):9.3f} {agg(m,theta,'gqc_gap'):6.3f} "
                  f"{agg(m,theta,'theta_violations'):5.1f}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"cfi_target_{args.dataset}_{args.scale}_n{args.seeds}")
    payload = {
        "dataset": args.dataset, "scale": args.scale, "n_workers": args.n_workers,
        "k": args.k, "budget_steps": args.budget_steps, "thetas": thetas,
        "low_frac": args.low_frac, "seeds": args.seeds,
        "low_group_definition": ("按 |F_u(θ)| 升序取前 30%（只含合法池内的人），"
                                 "每个 θ 单独划分"),
        "rows": rows,
    }
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = [f"# CFI 型公平的靶子实验（E3.1/E3.2，{args.dataset}/{args.scale}）", "",
             f"参与者 {args.n_workers}、k={args.k}、步数预算 {args.budget_steps:.0f}、"
             f"{args.seeds} 个种子。", "",
             f"**低能力子群**：按 `|F_u(θ)|` 升序取前 **{args.low_frac:.0%}**（只含合法池内的人），"
             "每个 θ 单独划分（避免用算法输出定义受害者）。", "",
             "`excl_gap` 正值 = 低能力者被排除得更狠；`share_low` 应 ≈ "
             f"{args.low_frac:.2f} 才算配额一致。", ""]
    for theta in thetas:
        lines += [f"## θ = {theta:.1f}", "",
                  "| 方法 | 覆盖 | excl_low | excl_high | excl_gap | share_low | gqc_gap | θ违规 |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in methods:
            cov = agg(m, theta, "cov")
            if cov is None:
                continue
            lines.append(
                f"| `{m}` | {cov} | {agg(m,theta,'excl_low'):.3f} | "
                f"{agg(m,theta,'excl_high'):.3f} | **{agg(m,theta,'excl_gap'):+.3f}** | "
                f"{agg(m,theta,'select_share_low'):.3f} | {agg(m,theta,'gqc_gap'):.3f} | "
                f"{agg(m,theta,'theta_violations'):.1f} |")
        lines.append("")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {stem}.json\nwrote {stem}.md")


if __name__ == "__main__":
    main()
