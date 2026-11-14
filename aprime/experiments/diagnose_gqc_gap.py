# -*- coding: utf-8 -*-
r"""拆解公平差距 `gqc_gap`：多少是**结构下限**，多少是**算法超额**。

## 为什么必须拆

`gqc_gap` 定义为「选中份额与人数份额的最大偏差」。但它有个被忽略的性质：
**选取数是整数**，所以即使算法完美，偏差也降不到 0。

例：某小组占 1/60 = 1.67%，而本次共选 5 条路线。该组能拿到的份额只能是
0/5=0%、1/5=20%…… 于是最小可能偏差是 1.67%（给 0 条），再往下不可能。
这个下限**与算法无关**，只由「组规模 × 选取数」决定。

如果不拆开，就会把"OSM 用地让组规模更偏斜 → 整数下限抬高"误读成
"OSM 用地让算法更不公平"。两者是完全不同的结论。

## 定义

* ``gap_floor``：在**给定组规模与选中条数**下，枚举所有整数分配
  （c_tight + c_two + c_ample = n_sel，且 c_i ≤ 组规模）能得到的最小偏差。
* ``excess = gap_actual - gap_floor``：算法相对结构性最优的退步，
  这才是"公平性"该看的量。

## 用法

```powershell
$env:APRIME_ALLOW_SHELVED = '1'
py -3.14 experiments/diagnose_gqc_gap.py --scales small,medium --theta-mode relative
```
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (GROUP_AMPLE, GROUP_TIGHT, GROUP_TWO,  # noqa: E402
                           capability_groups, gqc_gap)
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.gqc import gqc_solve  # noqa: E402

GROUPS = (GROUP_TIGHT, GROUP_TWO, GROUP_AMPLE)
COMBOS = (("synthetic", "synthetic", "全合成"),
          ("synthetic", "osm", "仅OSM用地"))


def gap_floor(group_sizes: Dict[str, int], n_sel: int) -> Tuple[float, Tuple[int, ...]]:
    """给定组规模与选中条数，枚举整数分配求**最小可能**的 `gqc_gap`。

    返回 ``(最小偏差, 对应的各组选中数)``。组只有 3 个、n_sel 通常 4–8，
    枚举 ``O(n_sel²)`` 足够，不需要上 LP。
    """
    total = sum(group_sizes.values())
    if total <= 0 or n_sel <= 0:
        return 0.0, tuple()
    ids = [g for g in GROUPS if g in group_sizes]
    share = {g: group_sizes[g] / total for g in ids}
    best, best_alloc = float("inf"), tuple()
    ranges = [range(0, min(group_sizes[g], n_sel) + 1) for g in ids]
    for alloc in itertools.product(*ranges):
        if sum(alloc) != n_sel:
            continue
        dev = max((abs(alloc[i] / n_sel - share[ids[i]]) for i in range(len(ids))),
                  default=0.0)
        if dev < best - 1e-12:
            best, best_alloc = dev, alloc
    return (0.0, tuple()) if best == float("inf") else (best, best_alloc)


def run(scales: List[str], seeds: int, thetas: List[float], k: int,
        mode: str, n_workers: int, scheme: str = "value") -> List[dict]:
    rows: List[dict] = []
    for scale in scales:
        for wsrc, usrc, label in COMBOS:
            for seed in range(seeds):
                inst = make_dense_instance("tdrive", scale, n_workers=n_workers,
                                           seed=seed, index=seed,
                                           source=wsrc, urban_source=usrc)
                cands = build_candidates(inst, k=k, seed=seed)
                for theta in thetas:
                    prob = make_problem(inst, cands,
                                        APrimeConfig(theta=theta, budget_kind="steps",
                                                     k_cand=k, theta_mode=mode))
                    group_of = capability_groups(prob, scheme=scheme)
                    sizes = {g: sum(1 for v in group_of.values() if v == g) for g in GROUPS}
                    sizes = {g: n for g, n in sizes.items() if n > 0}
                    sel = gqc_solve(prob)[0]
                    if not sel or not sizes:
                        continue
                    total = sum(sizes.values())
                    pop = {g: sizes[g] / total for g in sizes}
                    cnt = {g: sum(1 for c in sel if group_of.get(c.wid) == g) for g in sizes}
                    n_sel = len(sel)
                    dev = {g: cnt[g] / n_sel - pop[g] for g in sizes}
                    worst = max(dev, key=lambda g: abs(dev[g]))
                    floor, _ = gap_floor(sizes, n_sel)
                    rows.append({
                        "scale": scale, "worker_source": wsrc, "urban_source": usrc,
                        "label": label, "seed": seed, "theta": theta,
                        "n_sel": n_sel, "n_groups": len(sizes),
                        "sizes": sizes, "gap": round(gqc_gap(sel, group_of), 4),
                        "floor": round(floor, 4),
                        "excess": round(gqc_gap(sel, group_of) - floor, 4),
                        "worst_group": worst,
                        "worst_dev": round(dev[worst], 4),
                    })
    return rows


def _med(xs: List[float]) -> float:
    return float(median(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--thetas", default="0.0,0.3,0.6")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=60)
    ap.add_argument("--theta-mode", default="relative", choices=["absolute", "relative"])
    ap.add_argument("--scheme", default="value",
                    choices=["value", "quantile", "quantile_tied"],
                    help="能力分组方案。value 按合法条数切 1/2/3+，在'几乎全员 ample'时会"
                         "退化成一档、gap 恒为 0；quantile_tied 按能力排名切且不拆并列，"
                         "退化时仍保留可解释的组份额")
    ap.add_argument("--out", default="results/gqc_gap_decomposition.md")
    args = ap.parse_args()

    require_available("CROSS")
    thetas = [float(t) for t in args.thetas.split(",") if t]
    rows = run([s for s in args.scales.split(",") if s], args.seeds, thetas,
               args.k, args.theta_mode, args.n_workers, args.scheme)

    def sub(scale: str, usrc: str) -> List[dict]:
        return [r for r in rows if r["scale"] == scale and r["urban_source"] == usrc]

    lines = ["# 公平差距拆解：结构下限 vs 算法超额", "",
             f"*`experiments/diagnose_gqc_gap.py` 生成；θ 口径 = **{args.theta_mode}**，"
             f"分组方案 = **{args.scheme}**，"
             f"{args.n_workers} 人，k={args.k}，{args.seeds} 种子 × θ∈{thetas}。*", "",
             "`gap` 是实测差距，`floor` 是**整数选取本身造成的不可消除下限**，"
             "`excess = gap − floor` 才是算法该负责的部分。", "",
             "> **组数 = 1 时 `gqc_gap` 恒为 0 且无信息量**（份额双方都是 100%）。"
             "看到 0 不要当成\"完全公平\"，要当成\"指标退化\"。", "",
             "| 尺度 | 组合 | 差距 gap | 结构下限 floor | 算法超额 excess | 选中条数 | 组数 |",
             "|---|---|---|---|---|---|---|"]

    for scale in [s for s in args.scales.split(",") if s]:
        for _, usrc, label in COMBOS:
            s = sub(scale, usrc)
            if not s:
                continue
            n_groups = _med([r["n_groups"] for r in s])
            degenerated = n_groups < 1.5
            lines.append("| %s | %s | %s | %s | %s | %.1f | %.1f%s |" % (
                scale, label,
                "%.4f" % _med([r["gap"] for r in s]),
                "%.4f" % _med([r["floor"] for r in s]),
                "%.4f" % _med([r["excess"] for r in s]),
                _med([r["n_sel"] for r in s]),
                n_groups,
                " ⚠️退化" if degenerated else ""))

    lines += ["", "## 组规模分布（人数）", "",
              "| 尺度 | 组合 | tight(1条) | two(2条) | ample(3+条) | 最坏组 |",
              "|---|---|---|---|---|---|"]
    for scale in [s for s in args.scales.split(",") if s]:
        for _, usrc, label in COMBOS:
            s = sub(scale, usrc)
            if not s:
                continue
            sizes = {g: _med([r["sizes"].get(g, 0) for r in s]) for g in GROUPS}
            worst = {}
            for r in s:
                worst[r["worst_group"]] = worst.get(r["worst_group"], 0) + 1
            top = max(worst, key=worst.get) if worst else "—"
            lines.append("| %s | %s | %.0f | %.0f | %.0f | %s |" % (
                scale, label, sizes[GROUP_TIGHT], sizes[GROUP_TWO],
                sizes[GROUP_AMPLE], top))

    # ---- 归因：gap 的上升来自 floor 还是 excess
    lines += ["", "## 归因", ""]
    for scale in [s for s in args.scales.split(",") if s]:
        a, b = sub(scale, "synthetic"), sub(scale, "osm")
        if not a or not b:
            continue
        dg = _med([r["gap"] for r in b]) - _med([r["gap"] for r in a])
        df = _med([r["floor"] for r in b]) - _med([r["floor"] for r in a])
        de = _med([r["excess"] for r in b]) - _med([r["excess"] for r in a])
        which = "**结构下限**" if abs(df) >= abs(de) else "**算法超额**"
        lines.append("* **%s**：Δgap = %+.4f，其中 Δfloor = %+.4f、Δexcess = %+.4f "
                     "→ 主要由 %s 驱动" % (scale, dg, df, de, which))
    lines.append("")

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("\n写出 %s" % out)


if __name__ == "__main__":
    main()
