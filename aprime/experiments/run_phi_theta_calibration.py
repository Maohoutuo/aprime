# -*- coding: utf-8 -*-
r"""E5.11：偏好异质性 φ 与个性化强度 θ 的联合定标。

## 要回答的问题

§6.8.11 开放问题 #3：本文只做了 φ 的端点对照（0 / 0.95），没有给出
"给定实测的偏好异质性 φ̂，θ 该取多少"的建议。本实验补上这条规则。

## 核心假设：θ 应按 `theta_all(φ)` 归一化

`theta_all = min_u max_r S_u(r)` 是命题 4 的下界，且**随 φ 单调上升**（E5.10 已见：
small 0.429 → 0.829，medium 0.839 → 1.034）。直觉：φ 越大，每人的最优路线越贴合其偏好，
"人人都还留有选择"的阈值就越高。

于是 θ 的绝对大小没有跨 φ 的可比性。定义无量纲量

    ρ = θ / theta_all(φ)

若**覆盖损失与 dropout 对 ρ 崩溃成一条曲线**（而非对 θ 崩溃），则定标规则为

    θ_rec = ρ* · theta_all(φ̂)

其中 ρ* 是唯一需要凭经验选的常数，`theta_all` 由候选池**闭式算出**（O(|C|)），
不需要任何额外标注。这就是可操作的规则。

## 崩溃检验怎么算

对每个分箱 `[ρ_lo, ρ_hi)`，取该箱内所有 (φ, seed, θ) 的覆盖损失均值；
再看这组均值**跨 φ 的离散度**（max−min）。同样的箱数下，按 ρ 分箱的离散度若明显小于
按 θ 分箱，说明 ρ 才是正确的归一化量。这是一个可证伪的比较，不是画图观感。

## 口径

合成参与者（`source="dense(...)"`），与 §6 其它结果同源；B=80；8 种子。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import capability_groups, gqc_gap  # noqa: E402
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402

PHIS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
THETAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: ρ 分箱上界（含首尾）
RHO_EDGES = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 4.0)
THETA_EDGES = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05)
NW = {"small": 160, "medium": 40}


def _bin_of(x: float, edges: Tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=float, default=80.0)
    ap.add_argument("--source", default="synthetic", choices=("synthetic", "tdrive"),
                    help="参与者来源：synthetic=随机采样，tdrive=真实 T-Drive 出行段")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    scales = [s for s in args.scales.split(",") if s]
    rows: List[dict] = []

    for scale in scales:
        for phi in PHIS:
            for seed in range(args.n):
                inst = make_dense_instance(args.dataset, scale, n_workers=NW[scale],
                                           seed=seed, index=0, source=args.source)
                inst = with_homogeneity(inst, phi, seed=seed)
                cands = build_candidates(inst, k=args.k, seed=seed,
                                         lams=(1.0, 0.5, 0.0), n_random=2)
                p0 = make_problem(inst, cands,
                                  APrimeConfig(theta=0.0, k_cand=args.k, budget=args.budget))
                th = p0.theta_thresholds()
                n_pool = len(p0.by_worker)
                base_cov = p0.coverage(solve(p0, with_ls=True)[0])
                for theta in THETAS:
                    prob = make_problem(inst, cands,
                                        APrimeConfig(theta=theta, k_cand=args.k,
                                                     budget=args.budget))
                    sel, _ = solve(prob, with_ls=True)
                    tied = capability_groups(prob, scheme="quantile_tied")
                    cov = prob.coverage(sel)
                    n_feas = len({c.wid for c in sel})
                    rows.append({
                        "scale": scale, "phi": phi, "seed": seed, "theta": theta,
                        "theta_all": round(th["theta_all"], 4),
                        "theta_any": round(th["theta_any"], 4),
                        "rho": round(theta / th["theta_all"], 4) if th["theta_all"] > 0 else 0.0,
                        "coverage": cov,
                        "dcov": cov - base_cov,
                        "dcov_rel": round((cov - base_cov) / base_cov, 4) if base_cov else 0.0,
                        "dropout_frac": round((n_pool - len(prob.eligible_workers())) / n_pool, 4)
                        if n_pool else 0.0,
                        "n_sel": len(sel),
                        "gqc_gap": round(gqc_gap(sel, tied), 4) if tied else 0.0,
                        "pss_mean": round(mean(c.sat for c in sel), 4) if sel else 0.0,
                    })
            print(f"[done] {scale} phi={phi}", flush=True)

    # ---------------- 表 1：theta_all(φ) ----------------
    fits: Dict[str, dict] = {}
    lines = [
        "# E5.11 φ 与 θ 的联合定标",
        "",
        f"*`experiments/run_theta_bounds.py` 同类口径；尺度 {scales}，φ ∈ {list(PHIS)}，"
        f"{args.n} 种子，B={args.budget:g}，θ ∈ {list(THETAS)}；"
        f"参与者来源 **{args.source}**。*",
        "",
        "## 1. `theta_all` 随 φ 的移动（定标的分母）",
        "",
        "| 尺度 | φ | `theta_all` | `theta_any` | θ=0 覆盖 |",
        "|---|---|---|---|---|",
    ]
    for scale in scales:
        xs, ys = [], []
        for phi in PHIS:
            base = [r for r in rows if r["scale"] == scale and r["phi"] == phi
                    and abs(r["theta"]) < 1e-9]
            ta = mean(r["theta_all"] for r in base)
            ty = mean(r["theta_any"] for r in base)
            cov0 = mean(r["coverage"] for r in base)
            xs.append(phi)
            ys.append(ta)
            lines.append("| %s | %.2f | %.4f | %.4f | %.1f |" % (scale, phi, ta, ty, cov0))
        coef = np.polyfit(xs, ys, 1)
        pred = np.polyval(coef, xs)
        ss_res = float(np.sum((np.array(ys) - pred) ** 2))
        ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fits[scale] = {"slope": round(float(coef[0]), 4),
                       "intercept": round(float(coef[1]), 4), "r2": round(r2, 4)}
        lines.append("")
        lines.append("线性拟合（%s）：`theta_all ≈ %.4f + %.4f·φ`，R² = %.4f。"
                     % (scale, coef[1], coef[0], r2))
        lines.append("")

    # ---------------- 表 2：按 ρ 分箱 ----------------
    def bin_mean(scale, edge, key, use_rho):
        edges = RHO_EDGES if use_rho else THETA_EDGES
        out = {}
        for phi in PHIS:
            rs = [r for r in rows if r["scale"] == scale and r["phi"] == phi
                  and _bin_of(r["rho"] if use_rho else r["theta"], edges) == edge]
            out[phi] = mean(r[key] for r in rs) if rs else None
        return out

    lines += ["## 2. 崩溃检验：按 ρ 分箱 vs 按 θ 分箱", "",
              "每格是该箱内全部 (seed, θ) 的均值；`离散度` = 跨 φ 的 max−min。",
              "**按 ρ 分箱的离散度应显著更小**，否则 ρ 不是正确的归一化量。", ""]

    collapse = {}
    for scale in scales:
        for key, label in (("dcov_rel", "相对覆盖损失"), ("dropout_frac", "dropout")):
            lines += ["### %s · %s" % (scale, label), "",
                      "| 箱 | " + " | ".join("φ=%.2f" % p for p in PHIS) + " | 离散度 |",
                      "|---|" + "---|" * len(PHIS) + "---|"]
            spreads_rho, spreads_theta = [], []
            for edge in range(len(RHO_EDGES) - 1):
                vals_r = bin_mean(scale, edge, key, True)
                got_r = [v for v in vals_r.values() if v is not None]
                sp_r = (max(got_r) - min(got_r)) if len(got_r) > 1 else 0.0
                spreads_rho.append(sp_r)
                lines.append("| ρ∈[%.2f,%.2f) | %s | %.4f |" % (
                    RHO_EDGES[edge], RHO_EDGES[edge + 1],
                    " | ".join(("%.4f" % vals_r[p]) if vals_r[p] is not None else "—"
                               for p in PHIS), sp_r))
            for edge in range(len(THETA_EDGES) - 1):
                vals_t = bin_mean(scale, edge, key, False)
                got_t = [v for v in vals_t.values() if v is not None]
                sp_t = (max(got_t) - min(got_t)) if len(got_t) > 1 else 0.0
                spreads_theta.append(sp_t)
            mr = mean(spreads_rho) if spreads_rho else 0.0
            mt = mean(spreads_theta) if spreads_theta else 0.0
            collapse[(scale, key)] = {"spread_rho": round(mr, 5),
                                      "spread_theta": round(mt, 5),
                                      "ratio": round(mt / mr, 3) if mr > 1e-9 else None}
            lines += ["",
                      "平均跨 φ 离散度：**按 ρ 分箱 %.4f** vs 按 θ 分箱 %.4f → 比值 %.2f×。"
                      % (mr, mt, (mt / mr) if mr > 1e-9 else 0.0), ""]

    # ---------------- 表 3：推荐 ρ* ----------------
    lines += ["## 3. 定标规则：`θ_rec = ρ* · theta_all(φ̂)`", "",
              "取覆盖损失 ≤5% 与 dropout ≤5% 两个约束下可用的最大 ρ。", "",
              "| 尺度 | 约束 | 满足约束的最大 ρ | 对应 θ（φ=0 / 0.9） |",
              "|---|---|---|---|"]
    rho_star: Dict[str, dict] = {}
    for scale in scales:
        all_rho = sorted({r["rho"] for r in rows if r["scale"] == scale})
        def max_rho_ok(key, limit):
            ok = None
            for rho in all_rho:
                rs = [r for r in rows if r["scale"] == scale and abs(r["rho"] - rho) < 1e-9]
                if not rs:
                    continue
                v = mean(r[key] for r in rs)
                # 覆盖损失是负数，故用 ≥ -limit
                good = (v >= -limit) if key == "dcov_rel" else (v <= limit)
                if good:
                    ok = rho
                else:
                    break
            return ok
        rc = max_rho_ok("dcov_rel", 0.05)
        rd = max_rho_ok("dropout_frac", 0.05)
        star = min(x for x in (rc, rd) if x is not None) if (rc or rd) else 0.0
        rho_star[scale] = {"rho_cov5": rc, "rho_drop5": rd, "rho_star": star}
        f = fits[scale]
        lines.append("| %s | 覆盖损失 ≤5%% | %s | — |" % (
            scale, ("%.2f" % rc) if rc is not None else "无"))
        lines.append("| %s | dropout ≤5%% | %s | — |" % (
            scale, ("%.2f" % rd) if rd is not None else "无"))
        lines.append("| %s | **取 min → ρ\\*** | **%.2f** | %.3f / %.3f |" % (
            scale, star,
            star * (f["intercept"] + f["slope"] * 0.0),
            star * (f["intercept"] + f["slope"] * 0.9)))
    lines.append("")

    # ---------------- 表 4：规则的自检 ----------------
    lines += ["## 4. 规则自检：把 ρ* 代回去，各 φ 的实测表现", ""]
    for scale in scales:
        star = rho_star[scale]["rho_star"]
        lines += ["### %s（ρ\\* = %.2f）" % (scale, star), "",
                  "| φ | `theta_all` | 推荐 θ | 实测 θ | 实测覆盖损失 | 实测 dropout |",
                  "|---|---|---|---|---|---|"]
        f = fits[scale]
        for phi in PHIS:
            ta = f["intercept"] + f["slope"] * phi
            rec = star * ta
            nearest = min(THETAS, key=lambda t: abs(t - rec))
            rs = [r for r in rows if r["scale"] == scale and r["phi"] == phi
                  and abs(r["theta"] - nearest) < 1e-9]
            if not rs:
                continue
            lines.append("| %.2f | %.4f | %.3f | %.1f | %+.2f%% | %.2f%% |" % (
                phi, ta, rec, nearest,
                100 * mean(r["dcov_rel"] for r in rs),
                100 * mean(r["dropout_frac"] for r in rs)))
        lines.append("")

    lines += ["## 5. 判读", ""]
    ratios = [v["ratio"] for v in collapse.values() if v["ratio"]]
    if ratios and mean(ratios) > 1.5:
        lines += ["**ρ = θ/theta_all 是正确的归一化量**：按 ρ 分箱后跨 φ 的离散度平均降到"
                  "按 θ 分箱的 **1/%.2f**（各格比值 %s）。" % (
                      mean(ratios), ", ".join("%.2f" % x for x in ratios)), ""]
    else:
        lines += ["**ρ 归一化未能显著改善跨 φ 的一致性**（比值 %s），"
                  "说明覆盖损失对 φ 的依赖不止通过 `theta_all` 一条通道。" % (
                      ", ".join("%.2f" % x for x in ratios) if ratios else "—"), ""]
    lines += [
        "* `theta_all(φ)` 随 φ 单调上升，线性拟合 R² 分别为 %s。"
        % ", ".join("%s %.4f" % (s, fits[s]["r2"]) for s in scales),
        "* 定标规则：**`θ = ρ*·theta_all(φ̂)`**，`theta_all` 由候选池闭式算出（O(|C|)），"
        "无需额外标注；ρ\\* 为唯一经验常数。",
        "* 这与“θ 应固定取某个值”的直觉相反：偏好越异质，同样的 θ 越不苛刻"
        "（因为人人都有更贴合偏好的最优路线），故 θ 必须随 φ 放大。",
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.source == "synthetic" else "_" + args.source
    md = os.path.join(args.out_dir, "phi_theta_calibration%s.md" % suffix)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "phi_theta_calibration%s.json" % suffix)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"phis": list(PHIS), "thetas": list(THETAS), "n": args.n,
                   "budget": args.budget, "source": args.source,
                   "fits": fits, "collapse": 
                   {"%s|%s" % k: v for k, v in collapse.items()},
                   "rho_star": rho_star, "rows": rows},
                  fh, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print("\nwrote", md)


if __name__ == "__main__":
    main()
