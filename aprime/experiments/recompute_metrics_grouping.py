# -*- coding: utf-8 -*-
r"""E5.3：把 `quantile_tied` 分桶套到真实指标上（重跑 selections）。

## 为什么需要重跑

`results/fairness_tradeoff_*.json` 只存聚合指标、**不存选中的人**，因此无法仅凭存档
重算指标。本脚本用与 `run_fairness_tradeoff.py` **完全相同**的实例/候选/求解配置
重跑 A′、GQC、capacity 三者的选择（三者都不依赖 rng，重跑逐位可复现），
再同时计算**旧口径**与**新口径**的公平指标：

* **旧口径**：直接调用 `fairness_report`，即 `gqc_gap` 用值桶、`exclusion_gap` 用
  能力中位数二分。用于**复现校验**——其结果应与存档逐一相等。
* **新口径**：`quantile_tied` 并列一致分桶，"低组 = `tight`，高组 = 其余"，
  与 §6.10 靶子实验"底部 30% vs 其余"的语义一致。

## 定义

`excl_gap = 低组被排除率 − 高组被排除率`，所有量只在合法池内计算。
`低组份额` 是新口径的**有效性门禁**：份额过小说明该格没有可比较的低能力层。

## 口径

合成参与者（`source="dense(...)"`），与 §6 其它结果同源。
`exact / sa / iter / mscta / sa_caftca` 在本脚本中**不运行**——它们不影响
A′/GQC/capacity 的选择，且本实验只需后者。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, pstdev
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.analysis.fairness import fairness_report  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import THETA_GRID, APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (  # noqa: E402
    GROUP_TIGHT,
    capability_groups,
    gqc_gap as _gqc_gap,
    group_shares,
)
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.baselines import capacity_greedy  # noqa: E402
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402

METHODS = ("aprime", "gqc", "capacity")
NEW_SCHEME = "quantile_tied"


def _excl_gap(sel_wids, labels: Dict[str, str]) -> Optional[float]:
    """低组（`tight`）与高组（非 `tight`）被排除率之差。组为空则 None。"""
    low = [w for w, g in labels.items() if g == GROUP_TIGHT]
    high = [w for w, g in labels.items() if g != GROUP_TIGHT]
    if not low or not high:
        return None
    return (1.0 - sum(1 for w in low if w in sel_wids) / len(low)) - \
           (1.0 - sum(1 for w in high if w in sel_wids) / len(high))


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
    ap.add_argument("--scale", default="medium")
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--phis", default="0,0.95")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget-steps", type=float, default=80.0)
    ap.add_argument("--gqc-slack", type=int, default=0)
    ap.add_argument("--verify", default="", help="存档 JSON，用于逐位复现校验")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    phis = [float(p) for p in args.phis.split(",") if p]
    thetas = [float(t) for t in args.thetas.split(",") if t] or THETA_GRID

    cells: List[dict] = []
    for phi in phis:
        for seed in range(args.n):
            inst = make_dense_instance(args.dataset, args.scale, n_workers=args.n_workers,
                                       seed=seed, index=0)
            inst = with_homogeneity(inst, phi, seed=seed)
            cands = build_candidates(inst, k=args.k, seed=seed,
                                     lams=(1.0, 0.5, 0.0), n_random=2)
            for theta in thetas:
                prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=args.k,
                                                              budget=args.budget_steps))
                sels = {
                    "aprime": solve(prob, with_ls=True)[0],
                    "gqc": gqc_solve(prob, GQCParams(kappa=1.0, slack=args.gqc_slack),
                                     with_ls=True)[0],
                    "capacity": capacity_greedy(prob)[0],
                }
                labels = capability_groups(prob, scheme=NEW_SCHEME)
                shares = group_shares(labels)
                low_share = round(shares.get(GROUP_TIGHT, 0.0), 4)
                n_low = sum(1 for g in labels.values() if g == GROUP_TIGHT)
                for method in METHODS:
                    sel = sels[method]
                    wids = {c.wid for c in sel}
                    fr = fairness_report(prob, sel)          # 旧口径（中位数二分 + 值桶）
                    eg = _excl_gap(wids, labels)             # 新口径
                    cells.append({
                        "phi": phi, "seed": seed, "theta": theta, "method": method,
                        "old_excl": fr["exclusion_gap"],
                        "old_gqc": fr["gqc_gap"],
                        "new_excl": None if eg is None else round(eg, 4),
                        "new_gqc": round(_gqc_gap(sel, labels), 4),
                        "low_share": low_share,
                        "n_low": n_low,
                    })
        print(f"[done] phi={phi}", flush=True)

    # ── 复现校验：旧口径的 gqc_gap 应与存档逐一相等
    verify = {"checked": 0, "max_abs_diff": None, "excl_checked": 0,
              "excl_max_abs_diff": None}
    if args.verify and os.path.exists(args.verify):
        with open(args.verify, encoding="utf-8") as fh:
            old = json.load(fh)
        idx = {(r["phi"], r["seed"], r["theta"], r["method"]): r for r in old["rows"]}
        gd, ed = [], []
        for c in cells:
            key = (c["phi"], c["seed"], c["theta"], c["method"])
            if key not in idx:
                continue
            gd.append(abs(idx[key]["gqc_gap"] - c["old_gqc"]))
            ed.append(abs(idx[key]["exclusion_gap"] - c["old_excl"]))
        verify = {
            "checked": len(gd),
            "max_abs_diff": round(max(gd), 8) if gd else None,
            "excl_checked": len(ed),
            "excl_max_abs_diff": round(max(ed), 8) if ed else None,
        }

    def paired(theta: float, old_field: Optional[str], new_field: Optional[str]) -> Optional[dict]:
        vals = []
        for phi in phis:
            for seed in range(args.n):
                a = next((c for c in cells if c["phi"] == phi and c["seed"] == seed
                          and c["theta"] == theta and c["method"] == "aprime"), None)
                g = next((c for c in cells if c["phi"] == phi and c["seed"] == seed
                          and c["theta"] == theta and c["method"] == "gqc"), None)
                if not a or not g:
                    continue
                fa = old_field or new_field
                fg = old_field or new_field
                if a[fa] is None or g[fg] is None:
                    continue
                vals.append(g[fg] - a[fa])
        if not vals:
            return None
        lo, hi = _ci(vals)
        return {"mean": round(mean(vals), 4), "lo": lo, "hi": hi, "n": len(vals)}

    def fmt(p) -> str:
        if p is None:
            return "—"
        return "%+.3f [%+.3f, %+.3f]" % (p["mean"], p["lo"], p["hi"])

    lines = [
        f"# E5.3 `{args.scale}` 上两种分桶的指标对照",
        "",
        f"*`experiments/recompute_metrics_grouping.py` 生成；"
        f"φ ∈ {phis}，种子 {args.n}，B={args.budget_steps:g}，k={args.k}。*",
        "",
        f"**复现校验**：旧口径与存档对比 —— `gqc_gap` {verify['checked']} 条，"
        f"最大绝对差 **{verify['max_abs_diff']}**；`exclusion_gap` "
        f"{verify['excl_checked']} 条，最大绝对差 **{verify['excl_max_abs_diff']}**。"
        + (" 逐位一致，说明重跑的选择与存档完全相同。"
           if verify["max_abs_diff"] == 0 and verify["excl_max_abs_diff"] == 0 else ""),
        "",
        "## 逐 θ 配对差值（GQC − A′）与新口径的低组份额",
        "",
        "低组 = `tight` 桶；高组 = 其余。低组份额是新口径的有效性门禁："
        "小于 0.15 时该格没有可比较的低能力层，指标不可判读。",
        "",
        "| θ | 低组份额 | Δ`gqc_gap`（旧） | Δ`gqc_gap`（并列一致） "
        "| Δ`excl`（旧中位数） | Δ`excl`（并列一致） |",
        "|---|---|---|---|---|---|",
    ]

    old_excl_vals, new_excl_vals, old_gqc_vals, new_gqc_vals = [], [], [], []
    for theta in thetas:
        low_share = [c["low_share"] for c in cells if c["theta"] == theta]
        ls = round(mean(low_share), 3) if low_share else 0.0
        g_old = paired(theta, "old_gqc", None)
        g_new = paired(theta, None, "new_gqc")
        e_old = paired(theta, "old_excl", None)
        e_new = paired(theta, None, "new_excl")
        for p, acc in ((g_old, old_gqc_vals), (g_new, new_gqc_vals),
                       (e_old, old_excl_vals), (e_new, new_excl_vals)):
            if p:
                acc.append(p["mean"])
        lines.append("| %.1f | %.2f | %s | %s | %s | %s |" % (
            theta, ls, fmt(g_old), fmt(g_new), fmt(e_old), fmt(e_new)))

    lines += [
        "",
        "## 全 θ 汇总（逐 θ 均值的平均，仅作趋势参考）",
        "",
        "| 口径 | Δ`gqc_gap` | Δ`excl` |",
        "|---|---|---|",
        "| 旧（值桶 + 中位数二分） | %s | %s |" % (
            "%+.4f" % mean(old_gqc_vals) if old_gqc_vals else "—",
            "%+.4f" % mean(old_excl_vals) if old_excl_vals else "—"),
        "| 新（并列一致桶） | %s | %s |" % (
            "%+.4f" % mean(new_gqc_vals) if new_gqc_vals else "—",
            "%+.4f" % mean(new_excl_vals) if new_excl_vals else "—"),
        "",
        "读法：并列一致桶的 Δ`excl` 只在低组份额 ≥0.15 的 θ 上可解释；"
        "低组份额很小的格子即使有数值，也不得用于比较方法。",
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, f"fairness_quantile_tied_{args.scale}.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, f"fairness_quantile_tied_{args.scale}.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scale": args.scale, "phis": phis, "thetas": thetas,
                   "n": args.n, "budget_steps": args.budget_steps, "k": args.k,
                   "verify": verify, "cells": cells}, fh, ensure_ascii=False, indent=2)

    print()
    print("\n".join(lines))
    print()
    print("wrote", md)


if __name__ == "__main__":
    main()
