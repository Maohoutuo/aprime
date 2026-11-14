# -*- coding: utf-8 -*-
r"""E5.15：偏好先验的敏感性界 —— 在缺少调查数据时唯一能主张的东西。

**问题**：``mapus.profiles`` 里 5 个 archetype 的偏好向量是手写的，人群构成用
``rng.choice`` **等概率**抽样。这两件事**都没有任何公开调查数据支撑**
（见 ``PREF_MIXTURES`` 的说明）。审稿人会问：那你的结论是不是这个先验的产物？

**做法**：把"未经标定"的部分全部当成自变量扫一遍——

* ``mixture`` ∈ 5 种人群构成先验（等概率 + 4 种"某一类占 40%"的极端替代）
* ``jitter`` ∈ {0.15, 0.35, 0.55}（偏好向量的个体扰动幅度）
* ``mix_uniform`` ∈ {0.0, 0.05, 0.15}（向均匀分布收缩的比例）

对每个组合重算三条论文真正依赖的量：**平均 PSS**（``docs/calibration.md`` 用它
对标论文表 2 的 0.6349 锚点）、**覆盖**、**``gqc_gap``**，以及 MRG 是否绑定。

**判读原则**：若某条结论在全部组合下符号/序关系不变，则缺失的调查数据**不构成威胁**；
若某条随先验大幅移动，则它必须降级为"依赖于先验"的结论，不能作为主结论。

用法::

    py -3.14 experiments/run_pref_mixture_sensitivity.py
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
from statistics import mean, pstdev
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (  # noqa: E402
    GROUP_ORDER,
    capability_groups,
    group_shares,
    gqc_gap as _gqc_gap,
)
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402
from aprime.homogeneity import with_homogeneity  # noqa: E402
from mapus.profiles import PREF_MIXTURES, build_profiles  # noqa: E402

#: ``docs/calibration.md`` 用来标定 ``pss_scale`` 的锚点（论文表 2 六个基线的均值）。
PSS_ANCHOR = 0.6349
PSS_ANCHOR_TOL = 0.05

#: 北京 2022 年第二产业就业占比的两个口径（见 ``data/context/README.md`` §二）。
#: 常住就业人口（年鉴表 3-9）186/1132.1 = 0.1643；法人单位从业人员（表 3-14）
#: 159.3/1052.6 = 0.1513。**两者不能混用**，同时保留是为了量化口径选择的影响。
BJ_SECONDARY = {"resident": 0.1643, "firm": 0.1513}

_SERVICE_ARCHES = ("eco_enthusiast", "city_dweller", "community_helper", "explorer")


def _calibrated_space() -> Dict[str, dict]:
    """**边缘校准后**的先验空间：第二产业权重钉死，其余 4 项自由。

    这是 ``data/context/README.md`` 结论的直接落地——公开能拿到的只有边缘分布，
    故只能钉住 1 个权重。此空间里的每个先验都满足"第二产业 = 官方数字"这一约束，
    用来回答：**校准之后还剩多少不确定性？**
    """
    out: Dict[str, dict] = {}
    for tag, ind in BJ_SECONDARY.items():
        rest = 1.0 - ind
        base_src = ("边缘校准(%s)：industrial_worker=%.4f 钉死（北京 2022 第二产业就业占比），"
                    % (tag, ind))
        w = {k: rest / 4.0 for k in _SERVICE_ARCHES}
        w["industrial_worker"] = ind
        out["cal_%s_equal" % tag] = {
            "weights": w, "source": base_src + "其余 4 项等分余量（无依据）"}
        for dom in _SERVICE_ARCHES:
            w = {k: rest * 0.4 / 3.0 for k in _SERVICE_ARCHES if k != dom}
            w[dom] = rest * 0.6
            w["industrial_worker"] = ind
            out["cal_%s_%s60" % (tag, dom[:4])] = {
                "weights": w, "source": base_src + "%s 占余量的 60%%（无依据）" % dom}
    return out


def _run(inst, cands, theta: float, budget: float) -> dict:
    prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=8, budget=budget))
    tied = capability_groups(prob, scheme="value")
    share = group_shares(tied)
    sel_a, _ = solve(prob, with_ls=True)
    base = dict(kappa=1.0, slack=0, scheme="value")
    sel_p, info_p = gqc_solve(prob, GQCParams(**base), with_ls=True)
    sel_m, info_m = gqc_solve(prob, GQCParams(**base, min_rep=1, mrg_cap=2.0), with_ls=True)
    f_p, f_m = info_p.get("floor") or {}, info_m.get("floor") or {}
    bumped = any(f_m.get(g, 0) > f_p.get(g, 0) for g in GROUP_ORDER)
    changed = {c.wid for c in sel_p} != {c.wid for c in sel_m}
    return {
        "theta": theta,
        "k0": len(sel_a),
        "n_groups": len([g for g in GROUP_ORDER if share.get(g, 0) > 0]),
        "coverage_a": prob.coverage(sel_a),
        "coverage_gqc": prob.coverage(sel_p),
        "coverage_mrg": prob.coverage(sel_m),
        "gap_a": round(_gqc_gap(sel_a, tied), 4),
        "gap_gqc": round(_gqc_gap(sel_p, tied), 4),
        "gap_mrg": round(_gqc_gap(sel_m, tied), 4),
        # 标定锚点 0.6349 指的是**算法产出的解**的平均 PSS，故这里取被选中路线，
        # 而不是候选池（候选池含 λ=1.0 的偏好最优路线，按构造必然偏高）。
        "sat_sel": round(mean(c.sat for c in sel_a), 4) if sel_a else 0.0,
        "mrg_state": ("惰性" if not bumped else ("激活且绑定" if changed else "激活但不绑定")),
        "theta_all": round(prob.theta_thresholds()["theta_all"], 4),
        "theta_violations": sum(1 for c in sel_m if not prob.is_eligible(c)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=20)
    ap.add_argument("--budget", type=float, default=80.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--thetas", default="0.0,0.3,0.6")
    ap.add_argument("--jitters", default="0.15,0.35,0.55")
    ap.add_argument("--mix-uniforms", default="0.0,0.05,0.15")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--reuse-json", action="store_true",
                    help="复用已有 JSON 的 rows 重新生成报告（不重算）")
    ap.add_argument("--space", default="free", choices=("free", "calibrated"),
                    help="free=全部先验；calibrated=固定第二产业权重的边缘校准空间")
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",") if t]
    jitters = [float(j) for j in args.jitters.split(",") if j]
    mix_us = [float(m) for m in args.mix_uniforms.split(",") if m]
    space = PREF_MIXTURES if args.space == "free" else _calibrated_space()
    combos = list(itertools.product(space, jitters, mix_us))

    rows: List[dict] = []
    js_path = os.path.join(args.out_dir, "pref_mixture_sensitivity_%s%d_%s.json"
                           % (args.scale, args.n_workers, args.space))
    if args.reuse_json and os.path.exists(js_path):
        with open(js_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        rows = prev["rows"]
        print("[reuse] 复用 %s 的 %d 行" % (js_path, len(rows)))
    # 实例只依赖 seed，与先验无关——构造一次复用，否则 45 个组合会把
    # ``tdrive_workers`` 的轨迹筛选重复 135 遍。
    base_by_seed = {
        seed: make_dense_instance(args.dataset, args.scale, n_workers=args.n_workers,
                                  seed=seed, index=0)
        for seed in ([] if rows else range(args.seeds))
    }
    for mixture, jitter, mix_u in ([] if rows else combos):
        for seed in range(args.seeds):
            inst = dataclasses.replace(
                base_by_seed[seed],
                profiles=build_profiles(base_by_seed[seed].base, seed=20260325 + seed,
                                        jitter=jitter, mix_uniform=mix_u,
                                        mixture=space[mixture]))
            inst = with_homogeneity(inst, 0.0, seed=seed)
            cands = build_candidates(inst, k=8, seed=seed,
                                     lams=(1.0, 0.5, 0.0), n_random=2)
            mean_sat = mean(c.sat for c in cands)
            for theta in thetas:
                cell = _run(inst, cands, theta, args.budget)
                cell.update({"mixture": mixture, "jitter": jitter,
                             "mix_uniform": mix_u, "seed": seed,
                             "mean_sat": round(mean_sat, 4)})
                rows.append(cell)
        sub = [r for r in rows if r["mixture"] == mixture and r["jitter"] == jitter
               and r["mix_uniform"] == mix_u]
        print("[%s j=%.2f u=%.2f] PSS %.3f | cov %.1f | gap %.4f→%.4f | MRG %s" % (
            mixture, jitter, mix_u, mean(r["sat_sel"] for r in sub),
            mean(r["coverage_a"] for r in sub),
            mean(r["gap_a"] for r in sub), mean(r["gap_gqc"] for r in sub),
            "/".join(sorted({r["mrg_state"] for r in sub}))))

    # ---------------------------------------------------------------- 报告
    def _slice(theta: float) -> List[dict]:
        return [r for r in rows if abs(r["theta"] - theta) < 1e-9]

    lines = ["# E5.15 偏好先验的敏感性界", "",
             f"*`experiments/run_pref_mixture_sensitivity.py` 生成；"
             f"{args.dataset}/{args.scale}（{args.n_workers} 人，B={args.budget:g}），"
             f"{args.seeds} 种子 × {len(combos)} 个先验组合"
             f"（{len(space)} 种人群构成 × jitter {jitters} × mix_uniform {mix_us}）。*", "",
             ("**空间：calibrated（边缘校准后）。** 第二产业就业占比已钉死为北京 2022 官方数字"
              "（常住 16.43% / 法人单位 15.13%，见 `data/context/README.md` §二），"
              "其余 4 项权重自由。本节回答的是：**校准 1 个边缘之后，还剩多少不确定性。**"
              if args.space == "calibrated" else
              "**空间：free（未校准）。** 全部 5 个权重均无调查数据支撑，"
              "本节回答的是：**在完全没有校准的情况下，结论对先验有多敏感。**"), "",
             "**这不是标定，是敏感性分析。** 本工程没有任何公开偏好调查数据，"
             "``ARCHETYPES`` 的偏好向量与人群构成权重都是设计选择。"
             "本节量化的是“结论对这个设计选择有多敏感”，"
             "从而界定哪些主张可以写、哪些必须降级。", ""]

    # 1. 平均 PSS vs 标定锚点
    lines += ["## 1. 平均 PSS 与 `docs/calibration.md` 的锚点", "",
              f"`pss_scale=1.30` 是**用论文表 2 六个基线的均值 {PSS_ANCHOR} 单参数标定**出来的。"
              "若平均 PSS 随人群构成大幅移动，则该标定只在默认先验下成立。"
              "这里统计的是**被选中路线**的平均 PSS（锚点的口径），不是候选池均值。", "",
              "| 人群构成 | 平均 PSS（jitter 0.35, mix_uniform 0.05） | 与锚点差 | 落在 ±"
              f"{PSS_ANCHOR_TOL} 内 |", "|---|---|---|---|"]
    present = [m for m in space if any(r["mixture"] == m for r in rows)]
    for mixture in present:
        sub = [r for r in rows if r["mixture"] == mixture and r["jitter"] == 0.35
               and r["mix_uniform"] == 0.05]
        if not sub:
            continue
        v = mean(r["sat_sel"] for r in sub)
        d = v - PSS_ANCHOR
        lines.append("| %s | %.4f | %+.4f | %s |" % (
            mixture, v, d, "是" if abs(d) <= PSS_ANCHOR_TOL else "**否**"))
    all_pss = [mean(r["sat_sel"] for r in rows if r["mixture"] == m)
               for m in present]
    _pss_span = max(all_pss) - min(all_pss)
    lines += ["", "跨全部先验组合的平均 PSS：min %.4f / max %.4f，极差 **%.4f**"
              "（锚点容差 ±%.2f）→ %s" % (
                  min(all_pss), max(all_pss), _pss_span, PSS_ANCHOR_TOL,
                  "**先验不改变 PSS 水平**（极差在容差内）。" if _pss_span < PSS_ANCHOR_TOL
                  else "**PSS 随先验移动且超出容差**，该尺度上的标定只在特定先验下成立。"), ""]

    # 2. 覆盖与 gqc_gap 的散布
    lines += ["## 2. 核心指标的跨先验散布", "",
              "| θ | 覆盖 A′（min–max） | 覆盖极差 | `gqc_gap`(A′)（min–max） | "
              "`gqc_gap` 极差 | Δ`gqc_gap`(GQC−A′)（min–max） | GQC 未变差 |",
              "|---|---|---|---|---|---|---|"]
    for theta in thetas:
        sub = _slice(theta)
        ca = [r["coverage_a"] for r in sub]
        ga = [r["gap_a"] for r in sub]
        dl = [r["gap_gqc"] - r["gap_a"] for r in sub]
        notworse = sum(1 for x in dl if x <= 1e-4)
        lines.append("| %.1f | %.1f–%.1f | %.1f | %.4f–%.4f | %.4f | %+.4f–%+.4f | %d/%d |" % (
            theta, min(ca), max(ca), max(ca) - min(ca),
            min(ga), max(ga), max(ga) - min(ga),
            min(dl), max(dl), notworse, len(dl)))
    lines.append("")

    # 3. MRG 状态
    lines += ["## 3. MRG 状态跨先验的稳定性", "",
              "| θ | 惰性 | 激活但不绑定 | 激活且绑定 | 绑定率 |", "|---|---|---|---|---|"]
    for theta in thetas:
        sub = _slice(theta)
        c = {"惰性": 0, "激活但不绑定": 0, "激活且绑定": 0}
        for r in sub:
            c[r["mrg_state"]] = c.get(r["mrg_state"], 0) + 1
        lines.append("| %.1f | %d | %d | %d | %.0f%% |" % (
            theta, c["惰性"], c["激活但不绑定"], c["激活且绑定"],
            100.0 * c["激活且绑定"] / len(sub)))
    lines.append("")

    # 4. 判读
    cov_spread = max(max(r["coverage_a"] for r in _slice(t))
                     - min(r["coverage_a"] for r in _slice(t)) for t in thetas)
    gap_spread = max(max(r["gap_a"] for r in _slice(t))
                     - min(r["gap_a"] for r in _slice(t)) for t in thetas)
    deltas = [r["gap_gqc"] - r["gap_a"] for r in rows]
    n_worse = sum(1 for x in deltas if x > 1e-4)
    worst = max(deltas) if deltas else 0.0
    better = sum(1 for x in deltas if x < -1e-4)
    tie = len(deltas) - n_worse - better
    viol = sum(r["theta_violations"] for r in rows)
    n_bind = sum(1 for r in rows if r["mrg_state"] == "激活且绑定")
    degenerate = sum(1 for r in rows if r["n_groups"] <= 2)
    worse_cells = [r for r in rows if r["gap_gqc"] - r["gap_a"] > 1e-4]
    deg_share = (sum(1 for r in worse_cells if r["n_groups"] <= 2) / len(worse_cells)
                 if worse_cells else 0.0)
    pss_spread = max(mean(r["sat_sel"] for r in rows if r["mixture"] == m)
                     for m in present) - \
        min(mean(r["sat_sel"] for r in rows if r["mixture"] == m) for m in present)
    lines += ["## 4. 判读：哪些主张可以写", ""]
    lines += [
        "* **平均 PSS**：跨先验极差 **%.4f**（容差 ±%.2f）→ %s" % (
            pss_spread, PSS_ANCHOR_TOL,
            "**对先验不敏感**，故 `pss_scale` 的标定不因人群构成而失效。"
            if pss_spread < PSS_ANCHOR_TOL else
            "**随先验移动**，标定只在默认先验下成立。"),
        "* **覆盖**：跨先验极差最大 **%.1f 格**。%s" % (
            cov_spread,
            "先验不是覆盖的主因（<2 格），覆盖结论稳健。" if cov_spread < 2.0
            else "**覆盖对先验敏感**，报告时必须给出先验与极差。"),
        "* **`gqc_gap`(A′)**：跨先验极差最大 **%.4f**。%s" % (
            gap_spread,
            "先验不改变量级。" if gap_spread < 0.05
            else "**量级随先验移动**，故 `gqc_gap` 的绝对值不能当先验无关的量报告。"),
        "* **GQC 相对 A′ 的改善**：%d 格改善、%d 格持平、%d 格**变差**"
        "（最大变差 %+.4f）。%s" % (
            better, tie, n_worse, worst,
            "方向在全部格上成立，是先验无关的结论。"
            if n_worse == 0 else
            "**并非始终成立**，必须报例外。观察到的相关量：变差格全部是 `n_groups=%s`、"
            "覆盖同时下降 %d–%d 格、`k₀` 在 %d–%d 之间；**成因未定位**，"
            "本实验只能确认它与偏好先验**无关**（%d 格分散在 %d 种先验上），"
            "不能据此断言是配额取整还是 GQC 目标函数所致——这是一个待查的开放项。"
            % ("/".join(sorted({str(r["n_groups"]) for r in worse_cells})),
               min(r["coverage_gqc"] - r["coverage_a"] for r in worse_cells),
               max(r["coverage_gqc"] - r["coverage_a"] for r in worse_cells),
               min(r["k0"] for r in worse_cells), max(r["k0"] for r in worse_cells),
               len(worse_cells), len({r["mixture"] for r in worse_cells}))),
        "* **θ 违规**：全部格合计 **%d**，%s" % (
            viol, "A′/GQC 的可行性在任一先验下都不被破坏——这是**先验无关**的结论。"
            if viol == 0 else "存在违规，需排查。"),
        "* **MRG 绑定**：%d/%d 格发生绑定。%s" % (
            n_bind, len(rows),
            "绑定不是先验制造的假象，但也只在少数格发生，不能当普适效果。"
            if 0 < n_bind < len(rows) else
            ("MRG 在所有先验下都绑定。" if n_bind else "MRG 在所有先验下都不绑定。")),
        "",
        "**结论**：本节的用途是**界定**而非**消除**不确定性。它能把“先验无关的结论”"
        "（GQC 改善方向、θ 可行性）与“依赖先验的结论”（`gqc_gap` 绝对值、覆盖绝对值）分开。"
        "要真正消除，仍需真实偏好调查数据——那一步不能靠敏感性分析替代。",
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "%s%d_%s" % (args.scale, args.n_workers, args.space)
    md = os.path.join(args.out_dir, "pref_mixture_sensitivity_%s.md" % tag)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "pref_mixture_sensitivity_%s.json" % tag)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"dataset": args.dataset, "scale": args.scale,
                   "n_workers": args.n_workers, "budget": args.budget,
                   "seeds": args.seeds, "thetas": thetas,
                   "jitters": jitters, "mix_uniforms": mix_us,
                   "mixtures": list(space), "pss_anchor": PSS_ANCHOR,
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print("\nwrote", md)


if __name__ == "__main__":
    main()
