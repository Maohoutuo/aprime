# -*- coding: utf-8 -*-
r"""E5.4：GQC-tied —— 把并列一致分桶放进 GQC 内部（算法变体）。

## 动机

E5.3 证明：只把**指标**换成并列一致分桶救不回 medium 的低 θ——因为 GQC 与 A′ 在该区间
**选择逐位相同**（能力饱和使三组塌成单组，`floor = ceil = k₀`，配额退化为"恰好选 k₀ 个"）。
要让配额机制在低 θ 激活，必须把分桶方案换到**算法内部**。

`GQCParams.scheme="quantile_tied"` 即该变体（下称 GQC-tied）：medium θ=0 时低组占 17%，
名额下界 `⌊0.17·k₀⌋ = 1`，配额从"无约束"变成"至少 1 个名额"。

## 本实验回答

1. GQC-tied 是否真的在低 θ 改变了选择（与 A′ 不再相同）？
2. 改变的方向是否压低了 `gqc_gap`（它被设计来压的量）？
3. 代价是多少（覆盖、选中数）？`floor_ok` / `quota_ok` 是否仍兑现？
4. 与现行 GQC（值桶）相比，高 θ 上是否退步？

## 口径

* `gqc_gap` 同时报两种分桶下的值：`tied` 是并列一致（新口径、指标 of record），
  `value` 是值桶（旧口径，便于与 §6.5 对照）。
* `excl_gap` 用并列一致口径（低组 = `tight`，高组 = 其余）。
* 合成参与者（`source="dense(...)"`），与 §6 其它结果同源。
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
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402

METHODS = ("aprime", "gqc", "gqc_tied")


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
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--phis", default="0,0.95")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget-steps", type=float, default=80.0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    phis = [float(p) for p in args.phis.split(",") if p]
    thetas = [float(t) for t in args.thetas.split(",") if t] or THETA_GRID
    scales = [s for s in args.scales.split(",") if s]

    cells: List[dict] = []
    for scale in scales:
        for phi in phis:
            for seed in range(args.n):
                inst = make_dense_instance(args.dataset, scale, n_workers=args.n_workers,
                                           seed=seed, index=0)
                inst = with_homogeneity(inst, phi, seed=seed)
                cands = build_candidates(inst, k=args.k, seed=seed,
                                         lams=(1.0, 0.5, 0.0), n_random=2)
                for theta in thetas:
                    prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=args.k,
                                                                  budget=args.budget_steps))
                    tied = capability_groups(prob, scheme="quantile_tied")
                    value = capability_groups(prob, scheme="value")
                    low_share = group_shares(tied).get(GROUP_TIGHT, 0.0)

                    sel_a, _ = solve(prob, with_ls=True)
                    sel_g, info_g = gqc_solve(prob, GQCParams(kappa=1.0, slack=0,
                                                              scheme="value"), with_ls=True)
                    sel_t, info_t = gqc_solve(prob, GQCParams(kappa=1.0, slack=0,
                                                              scheme="quantile_tied"),
                                              with_ls=True)
                    sels = {"aprime": (sel_a, {}), "gqc": (sel_g, info_g),
                            "gqc_tied": (sel_t, info_t)}
                    wid_a = {c.wid for c in sel_a}
                    for method in METHODS:
                        sel, info = sels[method]
                        wids = {c.wid for c in sel}
                        ev = prob.evaluate(sel, method)
                        fr = fairness_report(prob, sel)
                        eg = _excl_gap(wids, tied)
                        cells.append({
                            "scale": scale, "phi": phi, "seed": seed, "theta": theta,
                            "method": method,
                            "coverage": ev["coverage"], "n_selected": ev["n_selected"],
                            "theta_violations": sum(1 for c in sel if not prob.is_eligible(c)),
                            "gqc_gap_tied": round(_gqc_gap(sel, tied), 4),
                            "gqc_gap_value": round(_gqc_gap(sel, value), 4),
                            "excl_gap_tied": None if eg is None else round(eg, 4),
                            "excl_gap_median": fr["exclusion_gap"],
                            "low_share": round(low_share, 4),
                            "floor_ok": info.get("floor_ok"),
                            "quota_ok": info.get("quota_ok"),
                            "kappa_eff": info.get("kappa_eff"),
                            "k0": info.get("k0"),
                            "floor": info.get("floor"),
                            "same_as_aprime": wids == wid_a,
                        })
            print(f"[done] {scale} phi={phi}", flush=True)

    def agg(scale: str, theta: float, method: str, field: str) -> Optional[float]:
        vals = [c[field] for c in cells if c["scale"] == scale and c["theta"] == theta
                and c["method"] == method and c[field] is not None]
        return round(mean(vals), 4) if vals else None

    def paired(scale: str, theta: float, hi: str, lo: str, field: str) -> Optional[dict]:
        vals = []
        for phi in phis:
            for seed in range(args.n):
                a = next((c for c in cells if c["scale"] == scale and c["phi"] == phi
                          and c["seed"] == seed and c["theta"] == theta
                          and c["method"] == hi), None)
                b = next((c for c in cells if c["scale"] == scale and c["phi"] == phi
                          and c["seed"] == seed and c["theta"] == theta
                          and c["method"] == lo), None)
                if not a or not b or a[field] is None or b[field] is None:
                    continue
                vals.append(a[field] - b[field])
        if not vals:
            return None
        lo_, hi_ = _ci(vals)
        return {"mean": round(mean(vals), 4), "lo": lo_, "hi": hi_, "n": len(vals)}

    def fmt(p) -> str:
        if p is None:
            return "—"
        star = "*" if (p["lo"] is not None and (p["lo"] > 0 or p["hi"] < 0)) else ""
        return "%+.3f [%+.3f, %+.3f]%s" % (p["mean"], p["lo"], p["hi"], star)

    lines = [
        "# E5.4 GQC-tied：把并列一致分桶放进 GQC 内部",
        "",
        f"*`experiments/run_gqc_tied.py` 生成；尺度 {scales}，φ ∈ {phis}，"
        f"种子 {args.n}，B={args.budget_steps:g}，k={args.k}。*",
        "",
        "`gqc_gap` 的 `tied` 列为并列一致分桶（新口径），`value` 为值桶（旧口径）。"
        "`same` = GQC 的选择与 A′ 逐位相同的种子占比。`floor_ok` 是配额下界兑现率。",
        "",
    ]

    for scale in scales:
        lines += [f"## {scale}", "",
                  "| θ | 低组份额 | GQC-tied 与 A′ 相同 | Δ`gqc_gap`(tied) GQC-tied−A′ "
                  "| Δ`gqc_gap`(tied) GQC−A′ | Δcov GQC-tied−A′ | floor_ok | κ_eff |",
                  "|---|---|---|---|---|---|---|---|"]
        for theta in thetas:
            sub = [c for c in cells if c["scale"] == scale and c["theta"] == theta]
            ls = round(mean([c["low_share"] for c in sub]), 3) if sub else 0.0
            same = mean([1.0 if c["same_as_aprime"] else 0.0
                         for c in sub if c["method"] == "gqc_tied"]) if sub else 0.0
            fo = [c["floor_ok"] for c in sub if c["method"] == "gqc_tied"
                  and c["floor_ok"] is not None]
            ke = [c["kappa_eff"] for c in sub if c["method"] == "gqc_tied"
                  and c["kappa_eff"] is not None]
            lines.append("| %.1f | %.2f | %.2f | %s | %s | %s | %s | %s |" % (
                theta, ls, same,
                fmt(paired(scale, theta, "gqc_tied", "aprime", "gqc_gap_tied")),
                fmt(paired(scale, theta, "gqc", "aprime", "gqc_gap_tied")),
                fmt(paired(scale, theta, "gqc_tied", "aprime", "coverage")),
                ("%.0f%%" % (100 * mean(1.0 if x else 0.0 for x in fo))) if fo else "—",
                ("%.2f" % mean(ke)) if ke else "—"))
        lines.append("")

    lines += ["## 汇总（全 θ 平均）", "",
              "| 尺度 | Δ`gqc_gap`(tied) GQC-tied−A′ | Δ`gqc_gap`(tied) GQC−A′ "
              "| Δcov GQC-tied−A′ | Δcov GQC−A′ |",
              "|---|---|---|---|---|"]
    for scale in scales:
        def overall(hi: str, lo: str, field: str) -> Optional[float]:
            per = [paired(scale, t, hi, lo, field) for t in thetas]
            vals = [p["mean"] for p in per if p]
            return round(mean(vals), 4) if vals else None

        def f2(v):
            return "—" if v is None else "%+.4f" % v

        lines.append("| %s | %s | %s | %s | %s |" % (
            scale,
            f2(overall("gqc_tied", "aprime", "gqc_gap_tied")),
            f2(overall("gqc", "aprime", "gqc_gap_tied")),
            f2(overall("gqc_tied", "aprime", "coverage")),
            f2(overall("gqc", "aprime", "coverage"))))
    lines += ["", "`*` 表示 95% CI 不跨 0。", ""]

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "gqc_tied.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "gqc_tied.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scales": scales, "phis": phis, "thetas": thetas, "n": args.n,
                   "budget_steps": args.budget_steps, "k": args.k, "cells": cells},
                  fh, ensure_ascii=False, indent=2)

    print()
    print("\n".join(lines))
    print()
    print("wrote", md)


if __name__ == "__main__":
    main()
