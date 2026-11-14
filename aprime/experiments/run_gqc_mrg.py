# -*- coding: utf-8 -*-
r"""E5.7：MRG 的过度代表上限——"要么代表，要么显式声明不可表示"。

## 三种配置

| 方法 | `min_rep` | `mrg_cap` | 行为 |
|---|---|---|---|
| `gqc_tied` | 0 | — | 纯比例（无兜底） |
| `gqc_mrg_all` | 1 | 0（关闭） | 保护**所有**非空组，过度代表最高 `1/(s·k₀)` 倍 |
| `gqc_mrg_cap` | 1 | **2.0** | 只在过度代表 ≤2× 时兜底（= Pareto 分界 `s·k₀ ≥ 0.5`） |

## 为什么 `cap=2` 是 Pareto 分界

兜底把组的名额从 0 抬到 1，名额份额偏差从 `s` 变为 `1/k₀ − s`（因 `s·k₀ < 1`）。
偏差变小 ⟺ `1/k₀ − s < s` ⟺ `s·k₀ > 0.5` ⟺ 过度代表倍数 `1/(s·k₀) < 2`。
故 `mrg_cap=2.0` **保证兜底不会抬高 `gqc_gap`**。

## 核心判据：`all_accounted`

超限的组不被兜底，但会进入 `mrg_blocked`（显式标记"不可表示"）。
故理想结果是 **`n_unrepr == n_blocked`**——每个没拿到名额的组都**被明确记账**，
不存在静默排除。这是本设计的替代性保证：放弃"人人有份"，换取"无人被无声丢弃"。

## 口径

合成参与者（`source="dense(...)"`），与 §6 其它结果同源；φ∈{0,0.95}，8 种子。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, pstdev
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402

METHODS = ("gqc_tied", "gqc_mrg_all", "gqc_mrg_cap")


def _excl_gap(sel_wids, labels: Dict[str, str]) -> Optional[float]:
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
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--budgets", default="80,120")
    ap.add_argument("--phis", default="0,0.95")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-rep", type=int, default=1)
    ap.add_argument("--mrg-cap", type=float, default=2.0)
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",") if t] or THETA_GRID
    scales = [s for s in args.scales.split(",") if s]
    budgets = [float(b) for b in args.budgets.split(",") if b]
    phis = [float(p) for p in args.phis.split(",") if p]

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
                        n_tight = sum(1 for g in tied.values() if g == GROUP_TIGHT)
                        base = dict(kappa=1.0, slack=0, scheme="quantile_tied")
                        sels = {
                            "gqc_tied": gqc_solve(prob, GQCParams(**base), with_ls=True),
                            "gqc_mrg_all": gqc_solve(
                                prob, GQCParams(**base, min_rep=args.min_rep, mrg_cap=0.0),
                                with_ls=True),
                            "gqc_mrg_cap": gqc_solve(
                                prob, GQCParams(**base, min_rep=args.min_rep,
                                                mrg_cap=args.mrg_cap), with_ls=True),
                        }
                        for method in METHODS:
                            sel, info = sels[method]
                            wids = {c.wid for c in sel}
                            k_sel = len(sel)
                            cnt: Dict[str, int] = {}
                            for c in sel:
                                g = tied.get(c.wid)
                                if g is not None:
                                    cnt[g] = cnt.get(g, 0) + 1
                            n_unrep = sum(1 for g in share if cnt.get(g, 0) == 0)
                            blocked = info.get("mrg_blocked") or []
                            over = None
                            if k_sel > 0:
                                ratios = [(cnt.get(g, 0) / k_sel) / s
                                          for g, s in share.items() if cnt.get(g, 0) > 0]
                                over = round(max(ratios), 3) if ratios else None
                            ev = prob.evaluate(sel, method)
                            eg = _excl_gap(wids, tied)
                            cells.append({
                                "scale": scale, "budget": B, "phi": phi, "seed": seed,
                                "theta": theta, "method": method,
                                "k0": info.get("k0"), "n_sel": k_sel,
                                "n_tight": n_tight,
                                "low_share": round(share.get(GROUP_TIGHT, 0.0), 4),
                                "rep_ok": n_unrep == 0,
                                "n_unrep": n_unrep,
                                "n_blocked": len(blocked),
                                "all_accounted": n_unrep <= len(blocked),
                                "over_rep": over,
                                "gqc_gap_tied": round(_gqc_gap(sel, tied), 4),
                                "excl_gap_tied": None if eg is None else round(eg, 4),
                                "coverage": ev["coverage"],
                                "floor_ok": info.get("floor_ok"),
                                "quota_ok": info.get("quota_ok"),
                                "mrg_active": info.get("mrg_active"),
                                "theta_violations": sum(1 for c in sel
                                                        if not prob.is_eligible(c)),
                            })
            print(f"[done] {scale} B={B:g}", flush=True)

    def sub(scale, B, theta, method):
        return [c for c in cells if c["scale"] == scale and c["budget"] == B
                and c["theta"] == theta and c["method"] == method]

    def pair_vals(scale, B, hi, lo, field):
        vals = []
        for theta in thetas:
            for phi in phis:
                for seed in range(args.n):
                    a = next((c for c in cells if c["scale"] == scale and c["budget"] == B
                              and c["phi"] == phi and c["seed"] == seed
                              and c["theta"] == theta and c["method"] == hi), None)
                    b = next((c for c in cells if c["scale"] == scale and c["budget"] == B
                              and c["phi"] == phi and c["seed"] == seed
                              and c["theta"] == theta and c["method"] == lo), None)
                    if not a or not b or a[field] is None or b[field] is None:
                        continue
                    vals.append(a[field] - b[field])
        return vals

    def fmt(vals):
        if not vals:
            return "—"
        lo, hi = _ci(vals)
        star = "*" if (lo is not None and (lo > 0 or hi < 0)) else ""
        return "%+.3f [%+.3f, %+.3f]%s" % (mean(vals), lo, hi, star)

    lines = [
        "# E5.7 MRG 的过度代表上限",
        "",
        f"*`experiments/run_gqc_mrg.py` 生成；尺度 {scales}，B ∈ {budgets}，"
        f"φ ∈ {phis}，{args.n} 种子，k={args.k}，min_rep={args.min_rep}，"
        f"mrg_cap={args.mrg_cap:g}。*",
        "",
        "`rep` = 所有非空组都有名额；`blk` = 因超过过度代表上限而被**显式标记**的组数；"
        "`acct` = `未表示组数 ≤ blk`（不存在**静默排除**：每个没拿到名额的组都被记账）。",
        "",
    ]

    for scale in scales:
        for B in budgets:
            lines += [f"## {scale}，B={B:g}", "",
                      "| θ | 低组人数 | 方法 | rep | 未表示 | blk | acct | over "
                      "| `gqc_gap` | `excl_gap` | 覆盖 |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for theta in thetas:
                for method in METHODS:
                    rs = sub(scale, B, theta, method)
                    if not rs:
                        continue
                    lines.append(
                        "| %.1f | %.1f | %s | %.0f%% | %.2f | %.2f | %s | %s | %.3f | %s | %.1f |" % (
                            theta, mean(r["n_tight"] for r in rs), method,
                            100 * mean(1.0 if r["rep_ok"] else 0.0 for r in rs),
                            mean(r["n_unrep"] for r in rs),
                            mean(r["n_blocked"] for r in rs),
                            ("%.0f%%" % (100 * mean(1.0 if r["all_accounted"] else 0.0
                                                    for r in rs))),
                            ("%.2f" % mean(r["over_rep"] for r in rs
                                           if r["over_rep"] is not None))
                            if any(r["over_rep"] is not None for r in rs) else "—",
                            mean(r["gqc_gap_tied"] for r in rs),
                            ("%+.3f" % mean(r["excl_gap_tied"] for r in rs
                                            if r["excl_gap_tied"] is not None))
                            if any(r["excl_gap_tied"] is not None for r in rs) else "—",
                            mean(r["coverage"] for r in rs)))
            lines.append("")

    lines += ["## 汇总：相对纯比例（`gqc_tied`）的配对净效应", "",
              "| 尺度 | B | 配置 | Δrep（pp） | Δ`gqc_gap` | Δ`excl_gap` | Δ覆盖 "
              "| acct |",
              "|---|---|---|---|---|---|---|---|"]
    for scale in scales:
        for B in budgets:
            for method, tag in (("gqc_mrg_all", "无上限（保护全部）"),
                                ("gqc_mrg_cap", "cap=%.1f" % args.mrg_cap)):
                drep = pair_vals(scale, B, method, "gqc_tied", "rep_ok")
                drep200 = []
                for theta in thetas:
                    for phi in phis:
                        for seed in range(args.n):
                            a = next((c for c in cells if c["scale"] == scale
                                      and c["budget"] == B and c["phi"] == phi
                                      and c["seed"] == seed and c["theta"] == theta
                                      and c["method"] == method), None)
                            b = next((c for c in cells if c["scale"] == scale
                                      and c["budget"] == B and c["phi"] == phi
                                      and c["seed"] == seed and c["theta"] == theta
                                      and c["method"] == "gqc_tied"), None)
                            if a and b:
                                drep200.append(int(a["rep_ok"]) - int(b["rep_ok"]))
                rs_m = [c for c in cells if c["scale"] == scale and c["budget"] == B
                        and c["method"] == method]
                if not rs_m or not drep200:
                    continue
                lines.append("| %s | %g | %s | %+.1f | %s | %s | %s | %.0f%% |" % (
                    scale, B, tag,
                    100 * mean(float(x) for x in drep200),
                    fmt(pair_vals(scale, B, method, "gqc_tied", "gqc_gap_tied")),
                    fmt(pair_vals(scale, B, method, "gqc_tied", "excl_gap_tied")),
                    fmt(pair_vals(scale, B, method, "gqc_tied", "coverage")),
                    100 * mean(1.0 if r["all_accounted"] else 0.0 for r in rs_m)))
    lines += ["", "`*` = 95% CI 不跨 0。", ""]

    lines += ["## 安全性", "", "| 方法 | θ 违规 | `floor_ok` | `quota_ok` |",
              "|---|---|---|---|"]
    for method in METHODS:
        rs = [c for c in cells if c["method"] == method]
        fo = [c["floor_ok"] for c in rs if c["floor_ok"] is not None]
        qo = [c["quota_ok"] for c in rs if c["quota_ok"] is not None]
        lines.append("| %s | %d | %s | %s |" % (
            method, sum(c["theta_violations"] for c in rs),
            ("%.0f%%" % (100 * mean(1.0 if x else 0.0 for x in fo))) if fo else "—",
            ("%.0f%%" % (100 * mean(1.0 if x else 0.0 for x in qo))) if qo else "—"))
    lines.append("")

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "gqc_mrg_cap.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "gqc_mrg_cap.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scales": scales, "budgets": budgets, "phis": phis,
                   "thetas": thetas, "n": args.n, "k": args.k,
                   "min_rep": args.min_rep, "mrg_cap": args.mrg_cap, "cells": cells},
                  fh, ensure_ascii=False, indent=2)

    print()
    print("\n".join(lines))
    print()
    print("wrote", md)


if __name__ == "__main__":
    main()
