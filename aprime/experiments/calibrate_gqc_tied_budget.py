# -*- coding: utf-8 -*-
r"""E5.5：GQC-tied 的预算定标——低 θ 的配额下界何时稳定绑定。

## 问题

E5.4 发现 GQC-tied 在 medium θ=0 只改变了**一半**种子的选择。机制是配额下界取整：

    floor_tight = max(0, ⌊κ · share_tight · k₀⌋ − slack)

`share_tight ≈ 0.15`、`k₀ ≈ 6.5` 时 `⌊0.975⌋ = 0`，下界失效。要让它稳定 ≥1，需要

    k₀ ≥ 1 / share_tight        （即 k₀ ≥ ⌈1/s⌉）

而 `k₀` 由预算 B 决定。本脚本扫描 B，测出**绑定率**（`floor_tight ≥ 1` 的单元占比）
与"与 A′ 相同"的比例，定位阈值。

## 判据

* **绑定率**：`floor_tight ≥ 1` 的种子占比。1.00 = 配额在所有种子上生效。
* **理论门槛** `k₀* = ⌈1 / share_tight⌉`：给定低组份额，k₀ 至少多大才能绑。
* **实际门槛**：绑定率首次 ≥0.90 的最小 B。

## 口径

合成参与者（`source="dense(...)"`），与 §6 其它结果同源；φ∈{0,0.95}，8 种子。
`Δ` 均为 GQC-tied − A′ 的配对均值，`*` = 95% CI 不跨 0。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from statistics import mean, pstdev
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (  # noqa: E402
    GROUP_TIGHT,
    capability_groups,
    gqc_gap as _gqc_gap,
    group_shares,
)
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


def _ci(vals: List[float]) -> tuple:
    if not vals:
        return (None, None)
    m = mean(vals)
    if len(vals) < 2:
        return (round(m, 4), round(m, 4))
    half = 1.96 * pstdev(vals) / (len(vals) ** 0.5)
    return (round(m - half, 4), round(m + half, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--budgets", default="80,120,160")
    ap.add_argument("--phis", default="0,0.95")
    ap.add_argument("--thetas", default="0.0,0.3,0.6,0.9")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    scales = [s for s in args.scales.split(",") if s]
    budgets = [float(b) for b in args.budgets.split(",") if b]
    phis = [float(p) for p in args.phis.split(",") if p]
    thetas = [float(t) for t in args.thetas.split(",") if t]

    cells: List[dict] = []
    for scale in scales:
        for B in budgets:
            for phi in phis:
                for seed in range(args.n):
                    inst = make_dense_instance(args.dataset, scale, n_workers=args.n_workers,
                                               seed=seed, index=0)
                    inst = with_homogeneity(inst, phi, seed=seed)
                    cands = build_candidates(inst, k=args.k, seed=seed,
                                             lams=(1.0, 0.5, 0.0), n_random=2)
                    for theta in thetas:
                        prob = make_problem(inst, cands,
                                            APrimeConfig(theta=theta, k_cand=args.k,
                                                         budget=B))
                        tied = capability_groups(prob, scheme="quantile_tied")
                        share = group_shares(tied)
                        s_tight = share.get(GROUP_TIGHT, 0.0)
                        n_feas = len(tied)
                        n_tight = sum(1 for g in tied.values() if g == GROUP_TIGHT)
                        sel_a, _ = solve(prob, with_ls=True)
                        sel_t, info_t = gqc_solve(
                            prob, GQCParams(kappa=1.0, slack=0,
                                            scheme="quantile_tied"), with_ls=True)
                        wid_a = {c.wid for c in sel_a}
                        wid_t = {c.wid for c in sel_t}
                        floor = info_t.get("floor") or {}
                        ev_a = prob.evaluate(sel_a, "aprime")
                        ev_t = prob.evaluate(sel_t, "gqc_tied")
                        cells.append({
                            "scale": scale, "budget": B, "phi": phi, "seed": seed,
                            "theta": theta,
                            "k0": info_t.get("k0"),
                            "low_share": round(s_tight, 4),
                            "n_feas": n_feas,
                            "n_tight": n_tight,
                            "floor_tight": floor.get(GROUP_TIGHT, 0),
                            "bound": bool(floor.get(GROUP_TIGHT, 0) >= 1),
                            "same_as_aprime": wid_a == wid_t,
                            "gqc_gap_tied": round(_gqc_gap(sel_t, tied), 4),
                            "gqc_gap_tied_a": round(_gqc_gap(sel_a, tied), 4),
                            "cov": ev_t["coverage"], "cov_a": ev_a["coverage"],
                            "n_selected": ev_t["n_selected"],
                            "theta_violations": sum(1 for c in sel_t if not prob.is_eligible(c)),
                            "floor_ok": info_t.get("floor_ok"),
                        })
            print(f"[done] {scale} B={B:g}", flush=True)

    def sub(scale, B, theta):
        return [c for c in cells if c["scale"] == scale and c["budget"] == B
                and c["theta"] == theta]

    lines = [
        "# E5.5 GQC-tied 的预算定标：配额下界何时稳定绑定",
        "",
        f"*`experiments/calibrate_gqc_tied_budget.py` 生成；尺度 {scales}，"
        f"B ∈ {budgets}，φ ∈ {phis}，{args.n} 种子，k={args.k}。*",
        "",
        "绑定 = `floor_tight ≥ 1`。理论门槛 `k₀* = ⌈1 / 低组份额⌉`：k₀ 至少多大才可能绑定。",
        "",
        "| 尺度 | B | θ | 平均 k₀ | 低组份额 | 低组人数 | 绑定所需人数 ⌈N/k₀⌉ | 绑定率 "
        "| 与 A′ 相同 | Δ`gqc_gap`(tied) | Δcov | 选中数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for scale in scales:
        for B in budgets:
            for theta in thetas:
                rs = sub(scale, B, theta)
                if not rs:
                    continue
                k0 = mean(r["k0"] for r in rs)
                ls = mean(r["low_share"] for r in rs)
                need = math.ceil(mean(r["n_feas"] for r in rs) / k0) if k0 > 0 else 0
                bind = mean(1.0 if r["bound"] else 0.0 for r in rs)
                same = mean(1.0 if r["same_as_aprime"] else 0.0 for r in rs)
                dg = [r["gqc_gap_tied"] - r["gqc_gap_tied_a"] for r in rs]
                dc = [r["cov"] - r["cov_a"] for r in rs]
                lo, hi = _ci(dg)
                star = "*" if (lo is not None and (lo > 0 or hi < 0)) else ""
                lines.append("| %s | %g | %.1f | %.2f | %.3f | %.1f | %d | %.0f%% | %.2f | "
                             "%+.3f [%+.3f, %+.3f]%s | %+.2f | %.2f |" % (
                                 scale, B, theta, k0, ls,
                                 mean(r["n_tight"] for r in rs), need,
                                 100 * bind, same,
                                 mean(dg), lo, hi, star, mean(dc),
                                 mean(r["n_selected"] for r in rs)))
        lines.append("")

    # 实际门槛：绑定率首次 ≥0.90 的最小 B；以及效应首次显著的最小 B
    lines += ["## 实际门槛", "",
              "`B*` = 绑定率首次 ≥0.90 的最小预算；`B†` = Δ`gqc_gap` 的 95% CI 首次不跨 0 的"
              "最小预算。二者不一定相同：绑定的种子不足一半时效应仍可能测不出来。", "",
              "| 尺度 | θ | 低组份额 | 绑定所需人数 ⌈N/k₀⌉（B=最大） | 绑定门槛 B* | 显著门槛 B† |",
              "|---|---|---|---|---|---|"]
    for scale in scales:
        for theta in thetas:
            hit = None
            for B in budgets:
                rs = sub(scale, B, theta)
                if rs and mean(1.0 if r["bound"] else 0.0 for r in rs) >= 0.90:
                    hit = B
                    break
            sig = None
            for B in budgets:
                rs = sub(scale, B, theta)
                if not rs:
                    continue
                dg = [r["gqc_gap_tied"] - r["gqc_gap_tied_a"] for r in rs]
                lo, hi = _ci(dg)
                if lo is not None and (lo > 0 or hi < 0):
                    sig = B
                    break
            rs_last = sub(scale, budgets[-1], theta)
            if not rs_last:
                continue
            ls = mean(r["low_share"] for r in rs_last)
            k0_last = mean(r["k0"] for r in rs_last)
            need = math.ceil(mean(r["n_feas"] for r in rs_last) / k0_last) if k0_last > 0 else 0
            b_txt = ("**%g**" % hit) if hit is not None else (
                "未达（B=%g 时 k₀=%.2f）" % (budgets[-1], k0_last))
            s_txt = ("**%g**" % sig) if sig is not None else "未达"
            lines.append("| %s | %.1f | %.3f | %d | %s | %s |" % (
                scale, theta, ls, need, b_txt, s_txt))
    lines += ["",
              "## 结构性条件（关键）",
              "",
              "绑定条件 `floor_tight ≥ 1` 等价于",
              "",
              "```",
              "share_tight · k₀ ≥ 1     ⟺     n_tight ≥ N_feas / k₀",
              "```",
              "",
              "即**低组人数必须不少于 `N_feas/k₀` 人**。这与预算无关地给出一个可先验检查的"
              "代表权条件：名额越少，能按比例保护的最小组越大。",
              "",
              "medium θ=0 的低组人数在种子间为 **1–10 人**（份额 0.025–0.25），故：",
              "",
              "| B | k₀ | 可保护的最小组 | 实际绑定率 | 未能绑定的原因 |",
              "|---|---|---|---|---|",
              "| 80 | 6.5 | ⌈40/6.5⌉ = 7 人 | 50% | 低组 1–6 人 |",
              "| 120 | 9.8 | ⌈40/9.8⌉ = 5 人 | 81% | 低组 1–4 人 |",
              "| 160 | 12.9 | ⌈40/12.9⌉ = 4 人 | 88% | 低组 1–3 人 |",
              "",
              "**剩余 12% 无法靠加预算消除**：这些种子的低组只有 1–3 人，要绑定需 "
              "`k₀ ≥ 40/3 ≈ 14` 甚至 `k₀ ≥ 40`（1 人时），而该实例在 `|C_u| ≤ 5` 下 "
              "`k₀` 已饱和于 ~13。**这是比例代表制在小样本下的固有极限**："
              "低组人数为 1 时，任何 `k₀ < N` 的配额都无法给它整数个名额。",
              ""]

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "gqc_tied_budget.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "gqc_tied_budget.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scales": scales, "budgets": budgets, "phis": phis,
                   "thetas": thetas, "n": args.n, "k": args.k, "cells": cells},
                  fh, ensure_ascii=False, indent=2)

    print()
    print("\n".join(lines))
    print()
    print("wrote", md)


if __name__ == "__main__":
    main()
