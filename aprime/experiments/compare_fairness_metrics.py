# -*- coding: utf-8 -*-
r"""哪种公平度量更可靠：修分组后的 `gqc_gap`，还是直接报组份额 / 连续度量？

## 争论不能靠直觉，要靠变异系数

可靠性至少包含两件事：

1. **跨种子稳定**：同一实验条件换个种子，数值不能大幅漂移，否则无法支撑
   "公平性随 X 变化"这类比较。
2. **不被分组规模伪影主导**：`gqc_gap` 是"各组偏差取最大值"，而名额是整数，
   所以它天然带一个**结构下限** `min_gap_floor`（见 `aprime.groups`）。
   组越偏斜，下限越高，与算法好坏无关。

本脚本把候选度量放进**同一批实例**上算，用跨种子变异系数（CV = 标准差/均值）
横向比较。同时把**选中条数**单独作为一个轴拉出来——因为 n_sel 只有 4–6 条时，
任何份额型度量都会被"0 条还是 1 条"的二值跳跃支配，这是度量选择救不了的。

## 候选度量

* ``gqc_gap(value)``：历史口径，三档按合法条数切。
* ``gqc_gap(tied)``：并列一致口径。
* ``gqc_excess``：``gqc_gap(tied) − min_gap_floor``，扣掉整数伪影后算法该负责的部分。
* ``ks_capability``：选中者的能力分布与合法池能力分布的两样本 KS 距离。
  连续、不分桶，方差由样本量决定而不是由最小组规模决定。
* ``|rank_shift|``：选中者能力百分位均值相对池均值 0.5 的偏移绝对值。

## 判读

CV 越小越可靠。若某个度量随 n_sel 增大而 CV 明显下降，说明它的噪声主要来自
**选取规模**而非度量本身——那结论应是"先扩大选取规模"，而不是"换个度量"。

用法::

    $env:APRIME_ALLOW_SHELVED = '1'
    py -3.14 experiments/compare_fairness_metrics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, pstdev
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (capability_groups, group_shares, gqc_gap,  # noqa: E402
                           grouping_is_degenerate, min_gap_floor)
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.gqc import gqc_solve  # noqa: E402

METRICS = ("gqc_gap(value)", "gqc_gap(tied)", "gqc_excess", "ks_capability", "|rank_shift|")


def _ks_distance(a: List[float], b: List[float]) -> float:
    """两样本 KS 距离：max|F_a − F_b|。不看分桶，只看分布。"""
    if not a or not b:
        return 0.0
    xs = sorted(set(a) | set(b))
    na, nb = len(a), len(b)
    sa, sb = sorted(a), sorted(b)
    best = 0.0
    for x in xs:
        fa = sum(1 for v in sa if v <= x) / na
        fb = sum(1 for v in sb if v <= x) / nb
        best = max(best, abs(fa - fb))
    return best


def _rank_shift(sel_caps: List[float], pool_caps: List[float]) -> float:
    """选中者能力在池内的平均百分位，相对池中心 0.5 的偏移。"""
    if not sel_caps or not pool_caps:
        return 0.0
    n = len(pool_caps)
    ps = []
    for c in sel_caps:
        below = sum(1 for v in pool_caps if v < c)
        equal = sum(1 for v in pool_caps if v == c)
        ps.append((below + 0.5 * equal) / n)  # 并列取中点，避免系统性偏置
    return abs(mean(ps) - 0.5)


def measure(scale: str, theta: float, phi: float, budget_mult: float,
            seed: int, k: int, n_workers: int) -> Dict[str, float]:
    inst = make_dense_instance("tdrive", scale, n_workers=n_workers, seed=seed, index=0)
    if phi:
        inst = with_homogeneity(inst, phi, seed=seed)
    budget = inst.budget * budget_mult
    cands = build_candidates(inst, k=k, seed=seed, lams=(1.0, 0.5, 0.0), n_random=2)
    prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=k, budget=budget))

    caps = {w: sum(1 for c in cs if prob.is_eligible(c)) for w, cs in prob.by_worker.items()}
    pool_caps = [v for v in caps.values() if v >= 1]
    if not pool_caps:
        return {}
    sel = gqc_solve(prob)[0]
    if not sel:
        return {}
    sel_caps = [float(caps[c.wid]) for c in sel]

    gv = capability_groups(prob, scheme="value")
    gt = capability_groups(prob, scheme="quantile_tied")
    sizes = {g: n for g, n in
             {g: sum(1 for v in gt.values() if v == g) for g in set(gt.values())}.items() if n > 0}
    gap_t = gqc_gap(sel, gt)
    floor = min_gap_floor(sizes, len(sel)) if sizes else 0.0

    return {
        "n_sel": len(sel),
        "n_groups": len(sizes),
        "min_share": min(group_shares(gt).values()) if sizes else 0.0,
        "degenerate": float(grouping_is_degenerate(gt)),
        "gqc_gap(value)": gqc_gap(sel, gv),
        "gqc_gap(tied)": gap_t,
        "gqc_excess": gap_t - floor,
        "ks_capability": _ks_distance(sel_caps, pool_caps),
        "|rank_shift|": _rank_shift(sel_caps, pool_caps),
    }


def cv(xs: List[float]) -> float:
    """变异系数。均值接近 0 时返回 inf（该度量在此条件下没有可比的信号）。"""
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    if abs(m) < 1e-9:
        return float("inf")
    return pstdev(xs) / abs(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--thetas", default="0.0,0.6")
    ap.add_argument("--phis", default="0.0,0.95")
    ap.add_argument("--budget-mults", default="1,2,3.5")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--out", default="results/fairness_metric_reliability.md")
    args = ap.parse_args()

    require_available("SEC6")
    scales = [s for s in args.scales.split(",") if s]
    thetas = [float(t) for t in args.thetas.split(",") if t]
    phis = [float(p) for p in args.phis.split(",") if p]
    mults = [float(m) for m in args.budget_mults.split(",") if m]

    rows: List[dict] = []
    for scale in scales:
        for theta in thetas:
            for phi in phis:
                for mult in mults:
                    recs = [measure(scale, theta, phi, mult, s, args.k, args.n_workers)
                            for s in range(args.seeds)]
                    recs = [r for r in recs if r]
                    if len(recs) < 2:
                        continue
                    row = {"scale": scale, "theta": theta, "phi": phi, "mult": mult,
                           "n_sel": mean(r["n_sel"] for r in recs),
                           "n_groups": mean(r["n_groups"] for r in recs),
                           "min_share": mean(r["min_share"] for r in recs),
                           "degen_rate": mean(r["degenerate"] for r in recs)}
                    for m in METRICS:
                        vals = [r[m] for r in recs]
                        row[f"mean:{m}"] = mean(vals)
                        row[f"sd:{m}"] = pstdev(vals)
                        row[f"cv:{m}"] = cv(vals)
                    rows.append(row)

    def avg(pred, key):
        xs = [r[key] for r in rows if pred(r) and r[key] == r[key]]
        xs = [x for x in xs if x != float("inf")]
        return mean(xs) if xs else float("nan")

    lines = ["# 公平度量可靠性：`gqc_gap` 修正版 vs 组份额/连续度量", "",
             f"*`experiments/compare_fairness_metrics.py` 生成；{args.seeds} 种子/格，"
             f"{args.n_workers} 人，k={args.k}。*", "",
             "CV = 跨种子标准差 / |均值|。**CV 越小越可靠**；`inf` 表示均值≈0"
             "（该度量在此条件下没有可比的信号）。", "",
             "## 1. 按选取规模分组看 CV（越小越好）", "",
             "| 选取条数 n_sel | 格数 | " + " | ".join(METRICS) + " |",
             "|---|---|" + "|".join("---" for _ in METRICS) + "|"]

    buckets = [("n_sel ≤ 6（默认预算）", lambda r: r["n_sel"] <= 6),
               ("7 ≤ n_sel ≤ 12", lambda r: 6 < r["n_sel"] <= 12),
               ("n_sel ≥ 13", lambda r: r["n_sel"] >= 13)]
    for name, pred in buckets:
        n = sum(1 for r in rows if pred(r))
        if not n:
            continue
        cells = ["%.2f" % avg(pred, f"cv:{m}") for m in METRICS]
        lines.append("| %s | %d | %s |" % (name, n, " | ".join(cells)))

    lines += ["", "## 2. 逐格明细", "",
              "| 尺度 | θ | φ | 预算× | n_sel | 组数 | 最低组份额 | 退化率 | "
              + " | ".join("CV " + m for m in METRICS) + " |",
              "|---|---|---|---|---|---|---|---|" + "|".join("---" for _ in METRICS) + "|"]
    for r in rows:
        cvs = []
        for m in METRICS:
            v = r[f"cv:{m}"]
            cvs.append("—" if v != v else ("inf" if v == float("inf") else "%.2f" % v))
        lines.append("| %s | %.1f | %.2f | %.1f | %.1f | %.1f | %.3f | %.0f%% | %s |" % (
            r["scale"], r["theta"], r["phi"], r["mult"], r["n_sel"], r["n_groups"],
            r["min_share"], 100 * r["degen_rate"], " | ".join(cvs)))

    # ---- 结论性对比
    small = lambda r: r["n_sel"] <= 6  # noqa: E731
    big = lambda r: r["n_sel"] >= 13  # noqa: E731

    lines += ["", "## 3. 绝对标准差（份额单位，越小越可靠）", "",
              "CV 对均值接近 0 的度量会虚高（尤其 `gqc_excess`）。"
              "`gqc_gap` 与 `ks_capability` 都在 [0,1] 份额尺度上，"
              "直接比标准差更公平——它表示「换个种子，这个数会跳多少」。", "",
              "| 选定条数 | " + " | ".join(METRICS) + " |",
              "|---|" + "|".join("---" for _ in METRICS) + "|"]
    for name, pred in buckets:
        if not any(pred(r) for r in rows):
            continue
        cells = ["%.3f" % avg(pred, f"sd:{m}") for m in METRICS]
        lines.append("| %s | %s |" % (name, " | ".join(cells)))

    lines += ["", "## 4. 结论", "",
              "**固定默认预算（n_sel ≤ 6）时各度量的平均 CV：**", ""]
    for m in METRICS:
        lines.append("* %s：%.2f" % (m, avg(small, f"cv:{m}")))
    lines += ["", "**把选取条数抬到 ≥13 后：**", ""]
    for m in METRICS:
        lines.append("* %s：%.2f" % (m, avg(big, f"cv:{m}")))
    lines.append("")

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[-40:]))
    print("\n写出 %s" % out)


if __name__ == "__main__":
    main()
