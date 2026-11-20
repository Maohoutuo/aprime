# -*- coding: utf-8 -*-
r"""CROSS 试跑：把「轨迹来源」「用地来源」「犯罪来源」三个维度拆开，看各自对满意度那一路的影响。

## 为什么这么设计

`CROSS` 原本依赖 Grab-Posisi 的官方轨迹 + 属性层，而那份数据实际拿不到。
现在换成三个**可独立开关**的替代源：

* 轨迹侧：合成参与者 / T-Drive / GeoLife（`data/geolife/`）
* 用地侧：手绘高斯团块 / OSM 实测（`data/osm_attrs/`）
* 犯罪侧：合成热点 / Zhang et al. 2025 北京犯罪栅格（`data/crime/`）

拆成多因子而不是"一把全换"，是因为**只看终点看不出归因**——
如果 PSS 变了，到底是轨迹变了、用地变了还是犯罪层变了？分因子才能回答。

## 关键约束：种子与规模必须逐位相同

所有组合都用同一个 `(seed, index, n_workers, k, θ)`，唯一的自变量是数据来源。
否则差异里混进了抽样噪声。

## 注意

本脚本调用 `require_available("CROSS")`，而 CROSS 在 E0.1 解冻前是 `deferred`。
试跑需显式放行：

```powershell
$env:APRIME_ALLOW_SHELVED = '1'
py -3.14 experiments/run_cross_trial.py
```

放行**不改变** `availability.py` 的状态判定，只是不拦——这些数字属于
"解冻前试跑"，**不得**当作 CROSS 的正式结果写进论文。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from statistics import mean
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import capability_groups, gqc_gap  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.gqc import gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402
from mapus.config import LAND_USE  # noqa: E402

#: (轨迹来源, 用地来源, 犯罪来源, 短标签)。只换这三个维度，其余全固定。
#: 最后一行是"实测城市属性"组：用地与犯罪均为实测，用于主结果表。
COMBOS = (
    ("synthetic", "synthetic", "synthetic", "全合成"),
    ("tdrive", "synthetic", "synthetic", "仅T-Drive轨迹"),
    ("geolife", "synthetic", "synthetic", "仅GeoLife轨迹"),
    ("synthetic", "osm", "synthetic", "仅OSM用地"),
    ("synthetic", "synthetic", "zhang2025", "仅犯罪实测"),
    ("geolife", "osm", "synthetic", "轨迹+用地均实测"),
    ("geolife", "osm", "zhang2025", "轨迹+用地+犯罪均实测"),
)


def _m(xs: List[float]) -> float:
    return float(mean(xs)) if xs else float("nan")


def _lu_entropy(land_use: np.ndarray) -> float:
    """逐格土地利用分布的平均香农熵（nats）。用来解释"为什么结果不同"。"""
    p = np.clip(land_use, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=2)
    return float(h.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--thetas", default="0.0,0.3,0.6")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--theta-mode", default="absolute",
                    choices=["absolute", "relative"],
                    help="θ 的尺度口径。absolute 是历史行为；relative 用 '达到本人"
                         "最佳满意度的 θ 倍'，跨数据源可比（见 aprime.config）")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    require_available("CROSS")

    thetas = [float(t) for t in args.thetas.split(",") if t]
    combos = list(itertools.product(range(args.seeds), thetas, COMBOS))
    rows = []
    t0 = time.time()
    print("共 %d 个求解单元（%d 种子 × %d θ × %d 组合）"
          % (len(combos), args.seeds, len(thetas), len(COMBOS)))

    for n, (seed, theta, (wsrc, usrc, csrc, label)) in enumerate(combos, start=1):
        inst = make_dense_instance(
            "tdrive", args.scale, n_workers=args.n_workers, budget=args.budget,
            seed=seed, index=seed, source=wsrc, urban_source=usrc,
            crime_source=csrc)
        cands = build_candidates(inst, k=args.k, seed=seed)
        prob = make_problem(inst, cands,
                            APrimeConfig(theta=theta, budget_kind="steps", k_cand=args.k,
                                         theta_mode=args.theta_mode))
        group_of = capability_groups(prob)

        sel_a = solve(prob, with_ls=True)[0]
        sel_g = gqc_solve(prob)[0]
        ea, eg = prob.evaluate(sel_a), prob.evaluate(sel_g)

        sats = np.array([c.sat for c in cands], dtype=float)
        best = {}
        for c in cands:
            best[c.wid] = max(best.get(c.wid, -1e9), c.sat)
        best_arr = np.array(list(best.values()), dtype=float)

        rows.append({
            "seed": seed, "theta": theta,
            "worker_source": wsrc, "urban_source": usrc, "crime_source": csrc,
            "label": label,
            "lu_entropy": round(_lu_entropy(inst.urban.land_use), 4),
            "urban_coverage": round(float(inst.urban.coverage), 4),
            "crime_nonzero": int((inst.urban.crime > 0).sum()),
            # 满意度**尺度**：这是 θ 跨数据集不可比的根源，必须逐组合记录。
            "sat_p05": round(float(np.percentile(sats, 5)), 4),
            "sat_p50": round(float(np.percentile(sats, 50)), 4),
            "sat_p95": round(float(np.percentile(sats, 95)), 4),
            "sat_best_med": round(float(np.median(best_arr)), 4),
            # 该 θ 在本次候选池里覆盖的满意度区间比例（尺度无关的可比量）
            "theta_covers": round(float((sats >= best_arr.mean() - theta).mean()), 4),
            "eligible": len(prob.eligible_workers()),
            "pss_mean_a": ea["pss_mean"], "pss_min_a": ea["pss_min"],
            "cov_a": ea["coverage"], "n_a": ea["n_selected"],
            "gap_a": round(gqc_gap(sel_a, group_of), 4),
            "pss_mean_g": eg["pss_mean"], "pss_min_g": eg["pss_min"],
            "cov_g": eg["coverage"], "n_g": eg["n_selected"],
            "gap_g": round(gqc_gap(sel_g, group_of), 4),
            "viol_a": sum(1 for c in sel_a if not prob.is_eligible(c)),
            "viol_g": sum(1 for c in sel_g if not prob.is_eligible(c)),
        })
        if n % 5 == 0 or n == len(combos):
            print("  %d/%d  %.0fs" % (n, len(combos), time.time() - t0))

    def sub(wsrc, usrc, csrc, theta=None) -> List[dict]:
        out = [r for r in rows if r["worker_source"] == wsrc
               and r["urban_source"] == usrc and r["crime_source"] == csrc]
        if theta is not None:
            out = [r for r in out if abs(r["theta"] - theta) < 1e-9]
        return out

    def agg(wsrc, usrc, csrc, key):
        return _m([r[key] for r in sub(wsrc, usrc, csrc)])

    def _elig(wsrc, usrc, csrc, t):
        return _m([r["eligible"] for r in sub(wsrc, usrc, csrc, theta=t)])

    # ---- 汇总
    lines = ["# CROSS 试跑：轨迹来源 × 用地来源 × 犯罪来源 的因子对照", "",
             f"*`experiments/run_cross_trial.py` 生成；tdrive/{args.scale}，"
             f"{args.n_workers} 人，k={args.k}，{args.seeds} 种子 × θ∈{thetas}，"
             f"θ 口径 = **{args.theta_mode}**。*", "",
             "**这是解冻前试跑（`APRIME_ALLOW_SHELVED=1`），不是 CROSS 的正式结果。**", "",
             "## 1. 主表", "",
             "| 组合 | 用地熵 | 犯罪非零格 | 合法池 | PSS 均值(A′) | PSS 最小值(A′) "
             "| 覆盖(A′) | `gqc_gap`(A′) | PSS 均值(GQC) | 覆盖(GQC) | `gqc_gap`(GQC) | θ 违规 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    for wsrc, usrc, csrc, label in COMBOS:
        rows_c = sub(wsrc, usrc, csrc)
        lines.append("| %s | %.3f | %.0f | %.0f | **%.4f** | %.4f | %.1f | **%.4f** "
                     "| %.4f | %.1f | %.4f | %d |" % (
                         label,
                         _m([r["lu_entropy"] for r in rows_c]),
                         _m([r["crime_nonzero"] for r in rows_c]),
                         _m([r["eligible"] for r in rows_c]),
                         _m([r["pss_mean_a"] for r in rows_c]),
                         _m([r["pss_min_a"] for r in rows_c]),
                         _m([r["cov_a"] for r in rows_c]),
                         _m([r["gap_a"] for r in rows_c]),
                         _m([r["pss_mean_g"] for r in rows_c]),
                         _m([r["cov_g"] for r in rows_c]),
                         _m([r["gap_g"] for r in rows_c]),
                         sum(r["viol_a"] + r["viol_g"] for r in rows_c)))

    # ---- 满意度尺度：θ 跨数据集不可比的根源
    hdr = " | ".join("θ=%.1f 合法池" % t for t in thetas)
    sep = "|".join("---" for _ in thetas)
    lines += ["", "## 2. 满意度尺度（θ 跨数据集不可比的根源）", "",
              "| 组合 | 最佳满意度中位 | 全部候选 P5 | P50 | P95 |",
              "|---|---|---|---|---|"]
    for wsrc, usrc, csrc, label in COMBOS:
        rows_c = sub(wsrc, usrc, csrc)
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (
            label,
            _m([r["sat_best_med"] for r in rows_c]),
            _m([r["sat_p05"] for r in rows_c]),
            _m([r["sat_p50"] for r in rows_c]),
            _m([r["sat_p95"] for r in rows_c])))

    lines += ["", "同一批实例下，**合法池随 θ 的收缩速度**：", "",
              "| 组合 | %s |" % hdr, "|---|%s|" % sep]
    for wsrc, usrc, csrc, label in COMBOS:
        cells = ["%.0f" % _elig(wsrc, usrc, csrc, t) for t in thetas]
        lines.append("| %s | %s |" % (label, " | ".join(cells)))

    base = ("synthetic", "synthetic", "synthetic")
    t_hi = max(thetas)
    lines += ["",
              "**这两张表是本试跑最重要的产出。** θ 的合法条件是 `S_u(r) ≥ θ`"
              "（**满意度下限**，不是与本人最佳之差），所以 θ 是**绝对**阈值。"
              "换到 OSM 用地后满意度同时发生两件事："
              "**水平下移** %.2f（最佳中位 %.2f → %.2f）与**离散变宽**"
              "（P5–P95 宽度 %.2f → %.2f）。两者叠加，同一个 θ 切掉的下尾比例就完全不同："
              "θ=0 时各组合都接近满池，θ=%.1f 时合成池仍有 %.0f 人合法、"
              "OSM 池只剩 %.0f 人（用地+犯罪均实测时 %.0f 人）。"
              % (agg(*base, key="sat_best_med") - agg("synthetic", "osm", "synthetic", "sat_best_med"),
                 agg(*base, key="sat_best_med"), agg("synthetic", "osm", "synthetic", "sat_best_med"),
                 agg(*base, key="sat_p95") - agg(*base, key="sat_p05"),
                 agg("synthetic", "osm", "synthetic", "sat_p95") - agg("synthetic", "osm", "synthetic", "sat_p05"),
                 t_hi, _elig(*base, t=t_hi), _elig("synthetic", "osm", "synthetic", t_hi),
                 _elig("geolife", "osm", "zhang2025", t_hi)),
              "",
              "> 因此**不能把不同数据源上的固定 θ 网格当成同一实验条件比较**。"
              "跨数据集报告时应改用尺度无关的量（如把 θ 归一化到该实例的满意度极差）。"
              "这一条直接影响论文里所有跨数据集对比，建议在处理 CROSS 时必须先解决。"]

    # ---- 逐因子归因
    lines += ["", "## 3. 逐因子归因（相对全合成基线）", "",
              "| 改动 | ΔPSS(A′) | ΔPSS最小值 | Δ覆盖 | Δ`gqc_gap` |",
              "|---|---|---|---|---|"]
    for wsrc, usrc, csrc, label in COMBOS[1:]:
        lines.append("| %s | %+.4f | %+.4f | %+.1f | %+.4f |" % (
            label,
            agg(wsrc, usrc, csrc, "pss_mean_a") - agg(*base, key="pss_mean_a"),
            agg(wsrc, usrc, csrc, "pss_min_a") - agg(*base, key="pss_min_a"),
            agg(wsrc, usrc, csrc, "cov_a") - agg(*base, key="cov_a"),
            agg(wsrc, usrc, csrc, "gap_a") - agg(*base, key="gap_a")))

    # ---- 交互项：两个因子一起换，是否等于各自效应之和
    def d(wsrc, usrc, csrc, key):
        return agg(wsrc, usrc, csrc, key) - agg(*base, key=key)

    for key, name in (("pss_mean_a", "PSS 均值"), ("cov_a", "覆盖"),
                      ("gap_a", "`gqc_gap`")):
        only_t = d("geolife", "synthetic", "synthetic", key)
        only_u = d("synthetic", "osm", "synthetic", key)
        only_c = d("synthetic", "synthetic", "zhang2025", key)
        both_tu = d("geolife", "osm", "synthetic", key)
        both_all = d("geolife", "osm", "zhang2025", key)
        lines += ["", f"**{name}**：仅换轨迹 {only_t:+.4f}、仅换用地 {only_u:+.4f}、"
                       f"仅换犯罪 {only_c:+.4f}；轨迹+用地 {both_tu:+.4f}、"
                       f"三者同换 {both_all:+.4f} → 交互项 **{both_all - only_t - only_u - only_c:+.4f}**"
                       f"{'（基本可加）' if abs(both_all - only_t - only_u - only_c) < 0.01 else '（**不可加，需分开报告**）'}"]

    lines += ["", "## 4. 判读", "",
              "* **θ 违规全为 0** 才说明该组合下约束仍然成立；非 0 表示解本身不合法，"
              "该组合的数字不可用。",
              "* **合法池**收缩不是噪声：θ=0 时各组合都是满池，θ 越大差异越大"
              "（见 §2）。收缩的通道是「满意度尺度下移 → 同一 θ 切掉更多下尾」，"
              "不是参与者来源改变了可行域本身。",
              "* **交互项**不能只看点估计：本表只给单尺度的差，"
              "必须做跨尺度 + 标准误检验（`experiments/compare_cross_scales.py`）。"
              "试跑中交互项的 z 值仅 0.6–2.0，**不足以支持\"两个因子不可加\"**，"
              "别凭这一张表写进论文。",
              "* **犯罪非零格**是犯罪层的实测覆盖：合成热点几乎填满网格，"
              "Zhang 2025 栅格按附录 B.1 剔除 <10 条格子后只剩 %d 格（8×8）。"
              "这会让 `risk(r_u)` 在边远格上为 0，进而改变偏好路线的风险规避行为。"
              % int(_m([r["crime_nonzero"] for r in sub("geolife", "osm", "zhang2025")])),
              "* **用地熵**只作描述，不要当解释：OSM 的熵系统性高于合成布局"
              "（本次 %.3f vs %.3f）是因为本层把面积与兴趣点各取一半，"
              "这个合成动作会把分布推向均匀；而满意度下降的主因是"
              "**绝对水平下移**，两者不是同一件事。"
              % (agg("synthetic", "osm", "synthetic", "lu_entropy"),
                 agg(*base, key="lu_entropy")), ""]

    os.makedirs(args.out_dir, exist_ok=True)
    # 文件名带上口径，否则 absolute / relative 两次跑会互相覆盖
    tag = "%s%d_%s" % (args.scale, args.n_workers, args.theta_mode)
    md = os.path.join(args.out_dir, "cross_trial_%s.md" % tag)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "cross_trial_%s.json" % tag)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scale": args.scale, "n_workers": args.n_workers,
                   "thetas": thetas, "seeds": args.seeds, "k": args.k,
                   "theta_mode": args.theta_mode,
                   "land_use_order": list(LAND_USE), "rows": rows},
                  fh, ensure_ascii=False, indent=2)
    print("\n写出 %s\n     %s" % (md, js))


if __name__ == "__main__":
    main()
