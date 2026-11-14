# -*- coding: utf-8 -*-
r"""E5.8：MRG 上限的敏感性实验与稳健性验证。

## 要回答的三个问题

1. **`cap=2` 的优势是不是小样本假象？** —— 种子从 8 提到 30/50，用配对 CI、
   逐种子记账、bootstrap 三重检验，确认不是少数种子驱动。
2. **`cap` 的效应形状是否符合理论？** —— 理论预测：`cap ≤ 2` 时所有兜底都在 Pareto
   区间内（`q_g ≥ min_rep/2`），故 `gqc_gap` 随 cap 增大而**改善**；`cap > 2` 起把
   `q_g < 0.5` 的组也抬上来，`gqc_gap` 转而**恶化**。即曲线应在 `cap=2` 处出现**极值**。
3. **有没有引入回归？** —— θ 违规、`floor_ok`/`quota_ok`、覆盖代价、无静默排除。

## 配置

| 名称 | `min_rep` | `mrg_cap` | 语义 |
|---|---|---|---|
| `base` | 0 | — | 纯比例（基线） |
| `cap1.5` | 1 | 1.5 | 仅 `q_g ≥ 0.667` 的组兜底（保守） |
| `cap2.0` | 1 | 2.0 | **推荐**：仅 `q_g ≥ 0.5`（Pareto 分界） |
| `cap3.0` | 1 | 3.0 | 仅 `q_g ≥ 0.333`（放宽） |
| `open` | 1 | 0（关） | 保护所有非空组（无上限） |

## 稳健性检验（三重）

* **配对 CI**：`Δ` 的 95% 正态近似区间。
* **逐种子记账**：把每个种子的 `Δ` 先按 θ 求均值，再看有多少种子改善/持平/恶化
  ——若优势只由 1–2 个种子驱动，这里会立刻暴露。
* **bootstrap**：对 `Δ` 做 2000 次有放回重采样，取 2.5%/97.5% 分位，
  不依赖正态假设（`gqc_gap` 分布有界且偏斜）。

## 口径

合成参与者（`source="dense(...)"`），与 §6 其它结果同源；φ∈{0,0.95}。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

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

# (名称, min_rep, mrg_cap)
CONFIGS: Tuple[Tuple[str, int, float], ...] = (
    ("base", 0, 2.0),
    ("cap1.5", 1, 1.5),
    ("cap2.0", 1, 2.0),
    ("cap3.0", 1, 3.0),
    ("open", 1, 0.0),
)
CAP_OF = {name: cap for name, _mr, cap in CONFIGS if name != "base"}


def _excl_gap(sel_wids, labels: Dict[str, str]) -> Optional[float]:
    low = [w for w, g in labels.items() if g == GROUP_TIGHT]
    high = [w for w, g in labels.items() if g != GROUP_TIGHT]
    if not low or not high:
        return None
    return (1.0 - sum(1 for w in low if w in sel_wids) / len(low)) - \
           (1.0 - sum(1 for w in high if w in sel_wids) / len(high))


def _ci(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return (None, None)
    m = mean(vals)
    if len(vals) < 2:
        return (round(m, 5), round(m, 5))
    half = 1.96 * pstdev(vals) / (len(vals) ** 0.5)
    return (round(m - half, 5), round(m + half, 5))


def _boot(vals: List[float], rng: random.Random, n_boot: int = 2000) -> Tuple[float, float]:
    if len(vals) < 2:
        m = mean(vals) if vals else 0.0
        return (m, m)
    out = []
    n = len(vals)
    for _ in range(n_boot):
        out.append(mean([vals[rng.randrange(n)] for _ in range(n)]))
    out.sort()
    return (round(out[int(0.025 * n_boot)], 5), round(out[int(0.975 * n_boot)], 5))


def _stars(lo: Optional[float], hi: Optional[float]) -> str:
    return "*" if (lo is not None and hi is not None and (lo > 0 or hi < 0)) else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--budgets", default="80,120")
    ap.add_argument("--phis", default="0,0.95")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--boot", type=int, default=2000)
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
                        for name, min_rep, cap in CONFIGS:
                            sel, info = gqc_solve(
                                prob, GQCParams(kappa=1.0, slack=0,
                                                scheme="quantile_tied",
                                                min_rep=min_rep, mrg_cap=cap),
                                with_ls=True)
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
                                over = round(max(ratios), 4) if ratios else None
                            eg = _excl_gap(wids, tied)
                            cells.append({
                                "scale": scale, "budget": B, "phi": phi, "seed": seed,
                                "theta": theta, "config": name,
                                "k0": info.get("k0"), "n_sel": k_sel,
                                "n_tight": n_tight,
                                "low_share": round(share.get(GROUP_TIGHT, 0.0), 4),
                                "rep_ok": n_unrep == 0,
                                "n_unrep": n_unrep,
                                "n_blocked": len(blocked),
                                "all_accounted": n_unrep <= len(blocked),
                                "over_rep": over,
                                "mrg_active": bool(info.get("mrg_active")),
                                "gqc_gap": round(_gqc_gap(sel, tied), 5),
                                "excl_gap": None if eg is None else round(eg, 5),
                                "coverage": prob.evaluate(sel, name)["coverage"],
                                "floor_ok": info.get("floor_ok"),
                                "quota_ok": info.get("quota_ok"),
                                "theta_violations": sum(1 for c in sel
                                                        if not prob.is_eligible(c)),
                            })
            print(f"[done] {scale} B={B:g} (n={args.n}, configs={len(CONFIGS)})",
                  flush=True)

    rng = random.Random(12345)

    def get(scale, B, phi, seed, theta, name):
        return next((c for c in cells if c["scale"] == scale and c["budget"] == B
                     and c["phi"] == phi and c["seed"] == seed
                     and c["theta"] == theta and c["config"] == name), None)

    def deltas(scale, B, name, field):
        vals = []
        for phi in phis:
            for seed in range(args.n):
                for theta in thetas:
                    a = get(scale, B, phi, seed, theta, name)
                    b = get(scale, B, phi, seed, theta, "base")
                    if a and b and a[field] is not None and b[field] is not None:
                        vals.append(a[field] - b[field])
        return vals

    def seed_deltas(scale, B, name, field):
        """逐种子：先把该种子在各 θ 上的 Δ 求均值。"""
        out = []
        for phi in phis:
            for seed in range(args.n):
                vals = []
                for theta in thetas:
                    a = get(scale, B, phi, seed, theta, name)
                    b = get(scale, B, phi, seed, theta, "base")
                    if a and b and a[field] is not None and b[field] is not None:
                        vals.append(a[field] - b[field])
                if vals:
                    out.append(mean(vals))
        return out

    lines = [
        "# E5.8 MRG 上限的敏感性与稳健性验证",
        "",
        f"*`experiments/run_gqc_mrg_sensitivity.py` 生成；尺度 {scales}，B ∈ {budgets}，"
        f"φ ∈ {phis}，**{args.n} 种子**，θ 全集 {len(thetas)} 点，bootstrap {args.boot} 次。*",
        "",
        "配置：`base` = 纯比例；`capX` = `min_rep=1, mrg_cap=X`；`open` = `mrg_cap=0`（保护全部）。",
        "",
    ]

    # ---------- 表 1：上限敏感性（Δ 相对 base） ----------
    lines += ["## 表 E5.8-1　上限敏感性（Δ 相对纯比例，配对）", "",
              "| 尺度 | B | 配置 | Δ`gqc_gap`（正态 95% CI） | Δ`gqc_gap`（bootstrap） "
              "| Δ`excl_gap` | Δ覆盖 | 激活率 | blocked | rep | over |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for scale in scales:
        for B in budgets:
            for name, _mr, _cap in CONFIGS[1:]:
                dg = deltas(scale, B, name, "gqc_gap")
                de = deltas(scale, B, name, "excl_gap")
                dc = deltas(scale, B, name, "coverage")
                if not dg:
                    continue
                gl, gh = _ci(dg)
                bl, bh = _boot(dg, rng, args.boot)
                rs = [c for c in cells if c["scale"] == scale and c["budget"] == B
                      and c["config"] == name]
                act = mean(1.0 if c["mrg_active"] else 0.0 for c in rs)
                blk = mean(c["n_blocked"] for c in rs)
                rep = mean(1.0 if c["rep_ok"] else 0.0 for c in rs)
                ov = [c["over_rep"] for c in rs if c["over_rep"] is not None]
                el, eh = _ci(de)
                cl, ch = _ci(dc)
                lines.append("| %s | %g | %s | %+.4f [%+.4f, %+.4f]%s | %+.4f [%+.4f, %+.4f]%s "
                             "| %+.4f [%+.4f, %+.4f]%s | %+.3f [%+.3f, %+.3f]%s "
                             "| %.0f%% | %.2f | %.0f%% | %s |" % (
                                 scale, B, name, mean(dg), gl, gh, _stars(gl, gh),
                                 mean(dg), bl, bh, _stars(bl, bh),
                                 mean(de), el, eh, _stars(el, eh),
                                 mean(dc), cl, ch, _stars(cl, ch),
                                 100 * act, blk, 100 * rep,
                                 ("%.2f" % mean(ov)) if ov else "—"))
            lines.append("")
    lines += ["`*` = 95% 区间不含 0。`激活率` = `floor` 被兜底抬升的单元占比。", ""]

    # ---------- 表 2：cap → gqc_gap 的极值形状 ----------
    lines += ["## 表 E5.8-2　`cap` 的效应形状（检验 `cap=2` 处的极值预测）", "",
              "理论：`cap ≤ 2` 时兜底全在 Pareto 区间内（`q_g ≥ min_rep/2`），`gqc_gap` 随 cap "
              "增大而改善；`cap > 2` 起把 `q_g < 0.5` 的组也抬上来，`gqc_gap` 转而恶化。"
              "故应在 `cap=2` 处取得最小值。", "",
              "| 尺度 | B | base（绝对值） | cap1.5 | cap2.0 | cap3.0 | open | 最小值位置 |",
              "|---|---|---|---|---|---|---|---|"]
    for scale in scales:
        for B in budgets:
            base_abs = mean(c["gqc_gap"] for c in cells
                            if c["scale"] == scale and c["budget"] == B
                            and c["config"] == "base")
            row = {"base": base_abs}
            for name in ("cap1.5", "cap2.0", "cap3.0", "open"):
                rs = [c["gqc_gap"] for c in cells if c["scale"] == scale
                      and c["budget"] == B and c["config"] == name]
                row[name] = mean(rs) if rs else None
            cand = {k: v for k, v in row.items() if v is not None}
            best = min(cand, key=lambda k: cand[k])
            lines.append("| %s | %g | %.4f | %s |" % (
                scale, B, row["base"],
                " | ".join(("%.4f" % row[k]) if row[k] is not None else "—"
                           for k in ("cap1.5", "cap2.0", "cap3.0", "open")) +
                " | **%s** |" % best))
    lines.append("")

    # ---------- 表 3：逐种子记账 ----------
    lines += ["## 表 E5.8-3　逐种子记账（排除少数种子驱动）", "",
              "每个种子先在 θ 上求平均 Δ，再统计改善/持平/恶化。`改善` 比例应远大于 `恶化`，"
              "且 `最强单种子贡献` 不应接近总效应——否则结论由离群种子决定。", "",
              "| 尺度 | B | 配置 | 改善 / 持平 / 恶化（种子数） | 改善比例 "
              "| 均值 Δ | 不含最强种子后均值 Δ |",
              "|---|---|---|---|---|---|"]
    seed_summary = {}
    for scale in scales:
        for B in budgets:
            for name in ("cap2.0", "open"):
                sd = seed_deltas(scale, B, name, "gqc_gap")
                if not sd:
                    continue
                better = sum(1 for x in sd if x < -1e-9)
                tie = sum(1 for x in sd if abs(x) <= 1e-9)
                worse = sum(1 for x in sd if x > 1e-9)
                sd_sorted = sorted(sd)
                # 去掉对均值贡献最强（方向一致且绝对值最大）的一个种子
                cand = [x for x in sd if x < 0] or sd
                drop = min(cand) if cand else 0.0
                rest = [x for x in sd if x != drop]
                mu = mean(sd)
                mu_rest = mean(rest) if rest else mu
                seed_summary[(scale, B, name)] = {
                    "better": better, "tie": tie, "worse": worse,
                    "frac_better": round(better / len(sd), 3),
                    "mean": round(mu, 5), "mean_drop_best": round(mu_rest, 5),
                    "n_seeds": len(sd),
                }
                lines.append("| %s | %g | %s | %d / %d / %d | %.0f%% | %+.4f | %+.4f |" % (
                    scale, B, name, better, tie, worse,
                    100 * better / len(sd), mu, mu_rest))
    lines.append("")

    # ---------- 表 4：E5.8 验收 ----------
    lines += ["## 表 E5.8-4　验收清单（`min_rep=1, mrg_cap=2`）", "",
              "| 判据 | 阈值 | 实测（medium） | 结论 |", "|---|---|---|---|"]
    checks = []
    med_b80 = [c for c in cells if c["scale"] == "medium" and c["budget"] == 80
               and c["config"] == "cap2.0"]
    med_b120 = [c for c in cells if c["scale"] == "medium" and c["budget"] == 120
                and c["config"] == "cap2.0"]
    all_cap = [c for c in cells if c["config"] == "cap2.0"]

    def chk(label, threshold, value, ok):
        checks.append((label, threshold, value, ok))
        lines.append("| %s | %s | %s | %s |" % (
            label, threshold, value, "✅" if ok else "❌"))

    tv = sum(c["theta_violations"] for c in all_cap)
    chk("θ 违规总数", "= 0", str(tv), tv == 0)
    fo = [c["floor_ok"] for c in all_cap if c["floor_ok"] is not None]
    qo = [c["quota_ok"] for c in all_cap if c["quota_ok"] is not None]
    fok = 100 * mean(1.0 if x else 0.0 for x in fo) if fo else 100.0
    qok = 100 * mean(1.0 if x else 0.0 for x in qo) if qo else 100.0
    chk("`floor_ok`", "= 100%", "%.0f%%" % fok, fok >= 99.999)
    chk("`quota_ok`", "= 100%", "%.0f%%" % qok, qok >= 99.999)
    acct = 100 * mean(1.0 if c["all_accounted"] else 0.0 for c in all_cap)
    chk("无静默排除（`n_unrep ≤ n_blocked`）", "= 100%", "%.0f%%" % acct, acct >= 99.999)

    dg80 = deltas("medium", 80, "cap2.0", "gqc_gap")
    dg120 = deltas("medium", 120, "cap2.0", "gqc_gap")
    l80, h80 = _ci(dg80)
    l120, h120 = _ci(dg120)
    chk("Δ`gqc_gap`（medium B=80）", "≤ 0 且 CI 上界 < 0",
        ("%+.4f [%+.4f, %+.4f]" % (mean(dg80), l80, h80)) if dg80 else "未运行",
        h80 is not None and h80 < 0)
    chk("Δ`gqc_gap`（medium B=120）", "≤ 0 且 CI 上界 < 0",
        ("%+.4f [%+.4f, %+.4f]" % (mean(dg120), l120, h120)) if dg120 else "未运行",
        h120 is not None and h120 < 0)
    de80 = deltas("medium", 80, "cap2.0", "excl_gap")
    el80, eh80 = _ci(de80)
    chk("Δ`excl_gap`（medium B=80）", "< 0 且 CI 上界 < 0",
        ("%+.4f [%+.4f, %+.4f]" % (mean(de80), el80, eh80)) if de80 else "未运行",
        eh80 is not None and eh80 < 0)
    dc80 = deltas("medium", 80, "cap2.0", "coverage")
    chk("Δ覆盖（medium B=80）", "绝对值 ≤ 0.5 步",
        ("%+.3f" % mean(dc80)) if dc80 else "未运行",
        bool(dc80) and abs(mean(dc80)) <= 0.5)
    sd = seed_summary.get(("medium", 80, "cap2.0"), {})
    if sd:
        chk("逐种子改善比例（medium B=80）", "≥ 15%",
            "%.0f%%（%d/%d 种子）" % (100 * sd["frac_better"], sd["better"],
                                      sd["n_seeds"]), sd["frac_better"] >= 0.15)
        chk("去掉最强种子后效应仍为负",
            "Δ < 0", "%+.5f" % sd["mean_drop_best"], sd["mean_drop_best"] < 0)
    # 极值位置
    shape_ok = True
    shape_txt = []
    for B in budgets:
        base_abs = mean(c["gqc_gap"] for c in cells if c["scale"] == "medium"
                        and c["budget"] == B and c["config"] == "base")
        vals = {"base": base_abs}
        for name in ("cap1.5", "cap2.0", "cap3.0", "open"):
            rs = [c["gqc_gap"] for c in cells if c["scale"] == "medium"
                  and c["budget"] == B and c["config"] == name]
            if rs:
                vals[name] = mean(rs)
        best = min(vals, key=lambda k: vals[k])
        shape_txt.append("B=%g→%s" % (B, best))
        shape_ok = shape_ok and best in ("base", "cap1.5", "cap2.0")
    chk("`cap` 极值不落在 `open`（medium）", "argmin ≠ open",
        "；".join(shape_txt), shape_ok)
    lines.append("")

    passed = sum(1 for _l, _t, _v, ok in checks if ok)
    lines += ["**结论：%d/%d 项通过%s**" % (
        passed, len(checks), "" if passed == len(checks) else "（存在未通过项，见上表）"), ""]

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "gqc_mrg_sensitivity.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "gqc_mrg_sensitivity.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scales": scales, "budgets": budgets, "phis": phis,
                   "thetas": thetas, "n": args.n, "k": args.k,
                   "configs": [{"name": n, "min_rep": mr, "mrg_cap": cp}
                               for n, mr, cp in CONFIGS],
                   "seed_summary": {"%s|%g|%s" % k: v
                                    for k, v in seed_summary.items()},
                   "checks": [{"label": l, "threshold": t, "value": v, "ok": ok}
                              for l, t, v, ok in checks],
                   "cells": cells},
                  fh, ensure_ascii=False, indent=2)

    print()
    print("\n".join(lines))
    print()
    print("wrote", md)


if __name__ == "__main__":
    main()
