# -*- coding: utf-8 -*-
r"""主权衡图数据：θ × 方法 × (覆盖 / 满意度 / 公平)。

    py -3.14 experiments/run_fairness_tradeoff.py --phis 0,0.95 --n 3

这是论文 Fig.3 的数据来源，也是 E3.1/E3.2 的主表。回答三个问题：

1. **A′ 自己的权衡**：θ↑ 时覆盖掉多少、满意度涨多少、公平指标怎么动？
2. **CFI 型 / MSCTA 型"公平"到底公平给谁**：`cfi_gap` 小（名额∝能力的自洽度高）
   但 `exclusion_gap` 大（低能力者被挤出）——这个"对着干"是 E3.1 的核心展示。
3. **θ 一致性的代价**：A′ 严格满足 `S ≥ θ`（`theta_violations = 0`），东华式基线
   忽略了 θ，因此它们的"高覆盖/高 CFI"是靠牺牲个性化换来的。

Day-6 新增的 `gqc` 是创新点 3 的算法：名额 ∝ **合法池内人数份额**（一人一票），
对照它的 `gqc_gap`（名额份额 vs 人数份额）与 `cfi_gap`（名额份额 vs 能力份额）。
**两者按设计互相冲突**：GQC 压 `gqc_gap` 会抬高 `cfi_gap`，这不是缺陷而是论点本身。

输出长表 `results/fairness_tradeoff_<setting>.json` 与两张 `.md`：
按 θ 的 A′ 曲线表 + 代表 θ 下的方法对比表。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.analysis.fairness import bootstrap_ci, fairness_report  # noqa: E402
from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import THETA_GRID, APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import capability_groups, group_shares  # noqa: E402
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.baselines import (capacity_greedy, iterative_greedy,  # noqa: E402
                                      mscta_greedy, sa_caftca, sa_greedy)
from aprime.solvers.exact import exact  # noqa: E402
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


def _m(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 4) if vals else None


METHODS = ("aprime", "gqc", "exact", "sa", "iter", "capacity", "mscta", "sa_caftca")


def _grouping_diag(prob) -> dict:
    """把分组是否退化写进结果行，而不是等到事后才发现。

    `gqc_gap` 在只剩一档时恒为 0；低组过小（<15%，E5.2 门槛）时份额估计不可信。
    这两种格子不该被读成"公平"或"不公平"。
    """
    groups = capability_groups(prob, scheme="value")
    share = group_shares(groups) if groups else {}
    return {
        "n_groups": len(share),
        "min_group_share": round(min(share.values()), 4) if share else 0.0,
        "grouping_degenerate": len(share) < 2,
    }


def _run(prob, rng, sa_iters, exact_max, gqc_params=None):
    out, infos = {}, {}
    sel, _ = solve(prob, with_ls=True)
    out["aprime"] = sel
    sel_gqc, info_gqc = gqc_solve(prob, gqc_params or GQCParams(), with_ls=True)
    out["gqc"] = sel_gqc
    infos["gqc"] = info_gqc
    out["exact"] = exact(prob, time_limit=20.0)[0] if len(prob.eligible) <= exact_max else None
    out["sa"] = sa_greedy(prob, rng, iters=sa_iters)[0]
    out["iter"] = iterative_greedy(prob, rng, restarts=6)[0]
    out["capacity"] = capacity_greedy(prob)[0]
    out["mscta"] = mscta_greedy(prob, cap_per_worker=2)[0]
    out["sa_caftca"] = sa_caftca(prob, rng, iters=max(1500, sa_iters // 2))[0]
    return out, infos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=40)
    ap.add_argument("--phis", default="0,0.95")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=3, help="每个 φ 的种子数")
    ap.add_argument("--sa-iters", type=int, default=2000)
    ap.add_argument("--exact-max-elements", type=int, default=400)
    ap.add_argument("--representative-theta", type=float, default=0.6)
    ap.add_argument("--budget-steps", type=float, default=None,
                    help="覆盖步数预算（None = 用数据集默认；D7 定标用来把名额数抬到 12–15）")
    ap.add_argument("--budget-mult", type=float, default=1.0,
                    help="在上面的预算上再乘一个倍数。公平度量的跨种子噪声主要由"
                         "「选中条数」决定（n_sel 4–6 时 CV≈0.8），×3.5 可把"
                         "n_sel 抬到约 15–17、噪声近似减半。默认 1.0 保持历史结果不变")
    ap.add_argument("--gqc-slack", type=int, default=0,
                    help="GQC 下界松弛量：0 = 严格比例，>0 允许覆盖优先")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免多组实验互相覆盖")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("SEC6")  # 真实数据实验搁置中：docs/e0_1_data_access_plan.md §8

    phis = [float(p) for p in args.phis.split(",") if p]
    thetas = [float(t) for t in args.thetas.split(",") if t] or THETA_GRID
    rows, t0, thresholds = [], time.time(), []

    for phi in phis:
        for seed in range(args.n):
            inst = make_dense_instance(args.dataset, args.scale, n_workers=args.n_workers,
                                       seed=seed, index=0)
            inst = with_homogeneity(inst, phi, seed=seed)
            # 预算倍数：公平度量的跨种子噪声与「选中条数」强相关（n_sel 4–6 时
            # 变异系数约 0.8），把预算放大能近似线性地压低噪声。
            # 默认 1.0 保持历史结果逐位可复现；推荐值见 docs §8.9。
            budget = (args.budget_steps if args.budget_steps is not None
                      else inst.budget) * args.budget_mult
            cands = build_candidates(inst, k=args.k, seed=seed,
                                     lams=(1.0, 0.5, 0.0), n_random=2)
            prob0 = make_problem(inst, cands, APrimeConfig(theta=0.0, k_cand=args.k,
                                                           budget=budget))
            thresholds.append({"phi": phi, "seed": seed, **prob0.theta_thresholds()})
            grid = (inst.nx, inst.ny)
            for theta in thetas:
                prob = make_problem(inst, cands,
                                    APrimeConfig(theta=theta, k_cand=args.k,
                                                 budget=budget))
                rng = random.Random(7 * seed + int(theta * 100))
                sels, infos = _run(prob, rng, args.sa_iters, args.exact_max_elements,
                                   gqc_params=GQCParams(kappa=1.0, slack=args.gqc_slack))
                for method in METHODS:
                    sel = sels[method]
                    if sel is None:
                        continue
                    ev = prob.evaluate(sel, method)
                    fr = fairness_report(prob, sel, grid=grid,
                                         covered_cells=prob.covered(sel))
                    extra = {}
                    if method in infos:
                        info = infos[method]
                        extra = {"quota_ok": info.get("quota_ok"),
                                 "floor_ok": info.get("floor_ok"),
                                 "ceil_ok": info.get("ceil_ok"),
                                 "floor": info.get("floor"),
                                 "final_counts": info.get("final_counts"),
                                 "k0": info.get("k0")}
                    rows.append({"phi": phi, "seed": seed, "theta": theta, "method": method,
                                 "coverage": ev["coverage"], "budget": ev["budget"],
                                 "n_selected": ev["n_selected"],
                                 "pss_mean": ev["pss_mean"], "pss_min": ev["pss_min"],
                                 # 分组诊断：`gqc_gap` 只在至少两档、且低组份额够大时才有意义。
                                 # 只报 gap 不报这个，会让"分组塌成一档"的格子看起来像"零差距"。
                                 **_grouping_diag(prob),
                                 **extra,
                                 **{k: v for k, v in fr.items() if k != "n_selected"}})

    def agg(phi, theta, method):
        rs = [r for r in rows if r["phi"] == phi and r["theta"] == theta and r["method"] == method]
        if not rs:
            return None
        keys = ("coverage", "pss_mean", "pss_min", "gini_pss", "jain_selected", "jain_feasible",
                "feasible_rate", "select_rate", "exclusion_gap", "cfi_gap", "gqc_gap",
                "cap_corr", "spatial_gini", "theta_margin",
                "ks_capability", "rank_shift", "rank_shift_abs")
        out = {"method": method, "theta": theta, "phi": phi,
               "coverage_ratio_budget": round(_m([r["coverage"] / r["budget"] for r in rs]), 4),
               "n_selected": _m([r["n_selected"] for r in rs])}
        for k in keys:
            out[k] = _m([r[k] for r in rs])
        # 95% 自助置信区间：这些度量的跨种子噪声很大，只报点估计会把噪声当效应
        for k in ("gqc_gap", "ks_capability", "rank_shift_abs"):
            lo, hi = bootstrap_ci([r[k] for r in rs])
            out[k + "_ci"] = (round(lo, 4), round(hi, 4))
        out["n_groups"] = _m([r["n_groups"] for r in rs])
        out["min_group_share"] = _m([r["min_group_share"] for r in rs])
        return out

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"fairness_tradeoff_{args.dataset}_{args.scale}" + args.tag)
    payload = {"dataset": args.dataset, "scale": args.scale, "n_workers": args.n_workers,
               "k": args.k, "phis": phis, "thetas": thetas, "n": args.n,
               "budget_steps": args.budget_steps, "budget_mult": args.budget_mult,
               "gqc_slack": args.gqc_slack,
               "note": "东华式基线（capacity/mscta/sa_caftca）忽略 θ，见 docs/baselines_assumptions.md",
               "theta_thresholds": thresholds,
               "rows": rows}

    # ── 表 1：A′ 的 θ 曲线（每个 φ 一段）
    lines = [f"# 主权衡图数据（{args.dataset}/{args.scale}，N={args.n_workers}，k={args.k}）", "",
             "生成脚本 `experiments/run_fairness_tradeoff.py`。"
             "`cov/预算` 是覆盖率占预算步数的比例；`excl` = 低能力组被排除率 − 高能力组被排除率"
             "（负值说明低能力者更容易被排除）；`cfi` = 名额份额与**能力**份额的最大偏差"
             "（东华 CFI 型口径）；`gqc` = 名额份额与**人数**份额的最大偏差（我们的口径）。"
             "**两者按设计互相冲突**：GQC 压 `gqc` 必然抬高 `cfi`。", "",
             "> **`gqc` 的可读性取决于分组**：`组数 = 1` 时它恒为 0（标 **退化**，"
             "该格不可用于任何差距比较）；`最低组份额` 带 `†` 表示低于 15%"
             "（E5.2 门槛）——这通常反映**人群结构**（低能力者本就只占一成），"
             "不是估计精度问题，但意味着该组的名额分配分辨率有限，"
             "解读时要结合下面的置信区间。见 docs §8.7–8.9。", "",
             "**`ks` / `rank` 是不分桶的替代度量，不受分组退化影响**："
             "`ks` = 选中者能力分布与合法池能力分布的 KS 距离（0 = 分布一致）；"
             "`rank` = 选中者能力百分位均值相对池中心 0.5 的偏移"
             "（负 = 偏向低能力者，正 = 偏向高能力者）。"
             "实测二者跨种子噪声低于 `gqc`，推荐作为主报告量。", ""]
    for phi in phis:
        lines += [f"## A′ 的 θ 曲线（φ = {phi:.2f}）", "",
                  "| θ | cov/预算 | 选中 | 可行率 | PSS 均值 | PSS 最小 | gini | jain | 入选率 | excl | cfi | gqc | 组数 | 最低组份额 | ks | rank | cap_corr |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for theta in thetas:
            r = agg(phi, theta, "aprime")
            if not r:
                continue
            f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
            g_diag = ("**退化**" if r["n_groups"] < 1.5
                      else "%.3f†" % r["min_group_share"]
                      if r["min_group_share"] < 0.15
                      else f(r["min_group_share"]))
            lines.append(f"| {theta:.1f} | {f(r['coverage_ratio_budget'])} | "
                         f"{f(r['n_selected'], 1)} | {f(r['feasible_rate'])} | {f(r['pss_mean'])} | "
                         f"{f(r['pss_min'])} | {f(r['gini_pss'])} | {f(r['jain_feasible'])} | "
                         f"{f(r['select_rate'])} | {f(r['exclusion_gap'])} | {f(r['cfi_gap'])} | "
                         f"{f(r['gqc_gap'])} | {f(r['n_groups'], 1)} | {g_diag} | "
                         f"{f(r['ks_capability'])} | {f(r['rank_shift'], 2)} | "
                         f"{f(r['cap_corr'])} |")

    # ── 表 2：代表 θ 下的方法对比
    rep = args.representative_theta
    lines += ["", f"## 方法对比（θ = {rep:.2f}）", "",
              "| 方法 | cov/预算 | 可行率 | PSS 均值 | gini | jain | 入选率 | excl | cfi | gqc | cap_corr | θ 违规 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for phi in phis:
        for method in METHODS:
            r = agg(phi, rep, method)
            if not r:
                continue
            # θ 违规数：东华式基线不看 θ，单独统计
            rs = [x for x in rows if x["phi"] == phi and x["theta"] == rep and x["method"] == method]
            viol = _m([x["theta_violations"] for x in rs])
            f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
            lines.append(f"| {method} (φ={phi:.2f}) | {f(r['coverage_ratio_budget'])} | "
                         f"{f(r['feasible_rate'])} | {f(r['pss_mean'])} | {f(r['gini_pss'])} | "
                         f"{f(r['jain_feasible'])} | {f(r['select_rate'])} | "
                         f"{f(r['exclusion_gap'])} | {f(r['cfi_gap'])} | {f(r['gqc_gap'])} | "
                         f"{f(r['cap_corr'])} | {f(viol, 1)} |")

    # ── 表 2b：公平度量的 95% 置信区间（跨种子）
    #   这些度量的跨种子噪声极大（n_sel≈5 时 CV≈0.8），只看点估计会把噪声当效应。
    lines += ["", "## 公平度量与 95% 置信区间（跨种子自助法）", "",
              "区间宽 = 该差异与噪声不可区分。若两个条件的区间重叠，**不要**声称二者公平性不同。", "",
              "| φ | θ | 方法 | gqc [95% CI] | ks [95% CI] | |rank| [95% CI] | n_sel |",
              "|---|---|---|---|---|---|---|"]
    for phi in phis:
        for theta in thetas:
            for method in ("aprime", "gqc"):
                r = agg(phi, theta, method)
                if not r:
                    continue
                def ci(k):
                    lo, hi = r[k + "_ci"]
                    if lo != lo:
                        return "n/a"
                    return "%.3f [%.3f, %.3f]" % (r[k], lo, hi)
                lines.append("| %.2f | %.1f | %s | %s | %s | %s | %.1f |" % (
                    phi, theta, method, ci("gqc_gap"), ci("ks_capability"),
                    ci("rank_shift_abs"), r["n_selected"]))

    # ── 表 3：GQC vs A′ 的配对差值（同一 (φ, seed, θ) 实例上对比，消掉实例噪声）
    def _row(phi, seed, theta, method):
        for x in rows:
            if (x["phi"] == phi and x["seed"] == seed and x["theta"] == theta
                    and x["method"] == method):
                return x
        return None

    pairs = [("gqc", "aprime"), ("gqc", "capacity"), ("gqc", "mscta"), ("gqc", "sa_caftca")]
    pair_stats = {}
    for a, b in pairs:
        diffs = {k: [] for k in ("coverage", "gqc_gap", "cfi_gap", "exclusion_gap", "pss_mean")}
        for phi in phis:
            for seed in range(args.n):
                for theta in thetas:
                    ra, rb = _row(phi, seed, theta, a), _row(phi, seed, theta, b)
                    if ra is None or rb is None:
                        continue
                    for k in diffs:
                        diffs[k].append(ra[k] - rb[k])
        st = {}
        for k, v in diffs.items():
            if not v:
                st[k] = None
                continue
            mu = mean(v)
            # 配对差值自带实例配对，标准误按样本标准差 / sqrt(n)
            var = sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)
            se = (var / len(v)) ** 0.5
            st[k] = round(mu, 4)
            st[k + "_se"] = round(se, 4)
            st[k + "_ci"] = [round(mu - 1.96 * se, 4), round(mu + 1.96 * se, 4)]
        st["n_pairs"] = len(diffs["coverage"])
        pair_stats[f"{a}-{b}"] = st

    lines += ["", "## 配对差值（同一实例上 `方法A − 方法B` 的均值 ± 95% CI，全 θ × 全 seed）", "",
              "正/负号直接对应变好/变差：`cov` 正=A 覆盖更多；`gqc`/`cfi`/`excl` 正=**A 更差**。", "",
              "| 对比 | n 对 | Δcov | ΔPSS | Δgqc_gap | Δcfi_gap | Δexcl |",
              "|---|---|---|---|---|---|---|"]
    for key, st in pair_stats.items():
        def cell(k, d=3):
            if st.get(k) is None:
                return "n/a"
            ci = st.get(k + "_ci")
            return f"{st[k]:+.{d}f}" + (f" [{ci[0]:+.{d}f},{ci[1]:+.{d}f}]" if ci else "")
        lines.append(f"| {key} | {st['n_pairs']} | {cell('coverage', 2)} | {cell('pss_mean')} | "
                     f"{cell('gqc_gap')} | {cell('cfi_gap')} | {cell('exclusion_gap')} |")

    payload["paired"] = pair_stats
    payload["elapsed"] = round(time.time() - t0, 1)
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    for key, st in pair_stats.items():
        print(f"[配对 {key}] n={st['n_pairs']} Δcov {st['coverage']} ΔPSS {st['pss_mean']} "
              f"Δgqc {st['gqc_gap']} Δcfi {st['cfi_gap']} Δexcl {st['exclusion_gap']}", flush=True)
    gqc_rates = _m([r["quota_ok"] for r in rows if r["method"] == "gqc" and r.get("quota_ok") is not None])
    gqc_floor_rate = _m([r["floor_ok"] for r in rows if r["method"] == "gqc" and r.get("floor_ok") is not None])
    if gqc_rates is not None:
        print(f"[GQC] quota_ok rate = {gqc_rates:.3f}, floor_ok = {gqc_floor_rate:.3f}", flush=True)
    print("wrote", stem + ".json")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()





