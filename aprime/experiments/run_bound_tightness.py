# -*- coding: utf-8 -*-
r"""E2.1/E2.2：界到底紧不紧——把"损耗"拆成**引理 2** 与**近似比公式**两环。

    py -3.14 experiments/run_bound_tightness.py --instances 5

对每个小实例（元素数 ≤ 9，可枚举）算四个量：

* `OPT`、`greedy`（纯贪心，无局部搜索）→ 实测比值 `r = greedy/OPT`；
* `α_true`：**精确**次模比（`analysis/alpha_exact.py`，枚举全部 S,T）；
* `α_min`：Lemma 2 的下界，ρ 取**精确最坏**成对 Jaccard（不是采样、更不是均值）；
* 四个候选界：`α²(1−e^{−α})` 与 `α(1−e^{−α})/2`，分别用 α_true 与 α_min。

判定标准（这是本次实验的唯一目的）：**有效界必须满足 bound ≤ r**。哪个界在任何实例上
被违反（bound > r），它就**不是定理**，只是猜想。然后看剩下那个界的松紧程度：

* `α_min ≪ α_true` ⇒ 损耗在引理 2（成对重叠太粗）；
* `α_true` 不小但界仍远低于 r ⇒ 损耗在公式本身（那个 α² 的多余因子）。

实例分两类：**随机**（扫 routes/热度/预算，找最坏比值）与**手工构造**（贪心的经典失效例、
完全重叠、完全不重叠三个端点）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import types
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.analysis.alpha_exact import exact_submodularity_ratio  # noqa: E402
from aprime.availability import require_available  # noqa: E402
from aprime.analysis.submodularity import alpha_min_bound, apx_bound  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.model import Candidate, SelectionProblem  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


def _cand(wid: str, cells) -> Candidate:
    """抽象格子：整数 c 编码成 (c, 0)，复用真实的 Candidate/SelectionProblem 代码路径。"""
    xy = [(int(c), 0) for c in cells]
    return Candidate(wid=wid, path=tuple((x, y, i) for i, (x, y) in enumerate(xy)),
                     cells=frozenset(xy), sat=1.0)


def _problem(cands: List[Candidate], m: int) -> SelectionProblem:
    cfg = APrimeConfig(theta=0.0, budget_kind="count", budget=float(m))
    return SelectionProblem(instance=types.SimpleNamespace(budget=float(m)),
                            candidates=list(cands), cfg=cfg)


def _opt(problem: SelectionProblem) -> int:
    """穷举最优（每人至多一条 + 基数预算），元素数小，直接枚举子集。"""
    elems = problem.eligible
    best = 0
    for r in range(len(elems) + 1):
        for combo in itertools.combinations(range(len(elems)), r):
            sel = [elems[i] for i in combo]
            if not problem.feasible(sel):
                continue
            best = max(best, problem.coverage(sel))
    return best


def rho_stats_exact(problem: SelectionProblem) -> Dict[str, float]:
    """精确最坏成对重叠：**Jaccard**（看趋势）与**重叠系数**（算界必须用这个）。"""
    el = problem.eligible
    jac = oc = 0.0
    for i in range(len(el)):
        for j in range(i + 1, len(el)):
            a, b = set(el[i].cells), set(el[j].cells)
            u = len(a | b)
            inter = len(a & b)
            if u:
                jac = max(jac, inter / u)
            lo = min(len(a), len(b))
            if lo:
                oc = max(oc, inter / lo)
    return {"jac_max": jac, "oc_max": oc}


def alpha_crossover(target: float = 0.5, lo: float = 0.0, hi: float = 1.0) -> float:
    """解 `α²(1−e^−α) = target`：α 低于这个值时，我们的界**打不过**经典 1/2 界。"""
    for _ in range(80):
        mid = (lo + hi) / 2
        if apx_bound(mid) < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def random_instance(rng: random.Random, n_workers: int, routes: int,
                    universe: int, w: int, hot: float) -> List[Candidate]:
    cells = list(range(universe))
    hot_cells = cells[: max(1, int(round(universe * hot)))] if hot > 0 else cells
    out: List[Candidate] = []
    for u in range(n_workers):
        for r in range(routes):
            pool = hot_cells if (r > 0 and rng.random() < max(hot, 0.0)) else cells
            if len(pool) < w:
                pool = cells
            out.append(_cand(f"u{u}", rng.sample(pool, min(w, len(pool)))))
    return out


def crafted() -> Dict[str, List[Candidate]]:
    """手工端点 + 贪心经典失效例 + **把引理 2 打死**的反例。"""
    out = {
        # 完全不重叠：α 应为 1，贪心最优
        "disjoint": [_cand(f"u{i}", {3 * i, 3 * i + 1, 3 * i + 2}) for i in range(3)],
        # 同一批格子上的 3 条路线（完全重叠）：α 应为 1/3 左右
        "identical": [_cand(f"u{i}", {1, 2, 3}) for i in range(3)],
        # 覆盖函数的经典贪心失效：A 最大，但 B+C > A+任意
        "classic_greedy_fail": [
            _cand("u1", {1, 2, 3}),          # A
            _cand("u2", {1, 4, 5}),          # B
            _cand("u3", {2, 6, 7}),          # C
        ],
        # 划分拟阵挤压：u1 只有一条"略好"的路线，u2/u3 各自有两条互补路线
        "partition_squeeze": [
            _cand("u1", {1, 2, 3, 4}),       # 贪心先吃这条
            _cand("u2", {1, 2, 5, 6}),
            _cand("u2", {7, 8, 9}),
            _cand("u3", {3, 4, 10}),
            _cand("u3", {11, 12}),
        ],
    }
    # ── 嵌套覆盖压力例（Day-5）：试图打死引理 2，**没打成**，但暴露了它的"救命机制" ──
    # 构造：m 条"几乎互不相交、但都含同一格 x"的长路线 R_i = {x} ∪ B_i ∪ {z_i}，
    # 再配 m 条辅助路线 H_i = B_i ∪ {z_i}（**嵌套在 R_i 里**）。取 S = {H_i}, T = {R_i}：
    #   f(S∪T) − f(S) = 1（只剩 x），Σ_{e∈T}[f(S∪e) − f(S)] = m ⇒ α ≤ 1/m。
    # 直觉上这该让 `1/(1+(m总−1)ρ)` 算大（ρ 是 R_i 之间的 1/|R|，很小），从而 α_min > α_true。
    # **实测没有**：因为 H_i ⊂ R_i ⇒ 重叠系数 = 1 ⇒ ρ_oc = 1 ⇒ α_min 掉到 1/|E|，反而更保守。
    # 结论（写进 docs §3）：取"最坏重叠系数"这一步本身就把嵌套结构罚到了底，
    # 这正是引理 2 至今 0/106 违例的原因；但它的**推导**仍缺 γ 条件（见文档）。
    b, m_routes = 4, 4
    cells = [[1 + i * b + j for j in range(b)] for i in range(m_routes)]
    z = [1 + m_routes * b + i for i in range(m_routes)]
    for i in range(m_routes):
        out.setdefault("nested_cover_stress", []).append(
            _cand(f"r{i}", {0, *cells[i], z[i]}))     # R_i
        out["nested_cover_stress"].append(
            _cand(f"h{i}", {*cells[i], z[i]}))        # H_i
    return out


def analyse(cands: List[Candidate], m: int, name: str) -> Optional[dict]:
    prob = _problem(cands, m)
    a = exact_submodularity_ratio(prob, max_elements=11)
    if a is None:
        return None
    opt = _opt(prob)
    if opt <= 0:
        return None
    sel, _ = solve(prob, with_ls=False)
    g = prob.coverage(sel)
    r = g / opt

    rs = rho_stats_exact(prob)
    am = alpha_min_bound(prob, rho=rs["oc_max"], rho_stat="oc_max")
    a_true, a_min = a["alpha"], am["alpha_min"]
    return {
        "name": name, "m": m, "n_elements": a["n"], "opt": opt, "greedy": g, "ratio": round(r, 4),
        "alpha_true": round(a_true, 4), "alpha_min": a_min,
        "rho_jac": round(rs["jac_max"], 4), "rho_oc": round(rs["oc_max"], 4),
        "m_eff": am["m"], "n_selected": len(sel),
        # 引理 2 的有效性：α_min 必须 ≤ α_true
        "lemma_violation": bool(a_min > a_true + 1e-9),
        # 候选界（含经典 matroid 贪心 1/2 作参照）
        "b_classic_half": 0.5,
        "b_true_sq": round(apx_bound(a_true), 4),
        "b_true_half": round(a_true * (1 - math.exp(-a_true)) / 2, 4),
        "b_min_sq": round(apx_bound(a_min), 4),
        "b_min_half": round(a_min * (1 - math.exp(-a_min)) / 2, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=5, help="每个配置的随机实例数")
    ap.add_argument("--universe", type=int, default=8)
    ap.add_argument("--route-len", type=int, default=3)
    ap.add_argument("--ms", default="2,3")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("E2.1")  # 抽象实例，不受真实数据搁置影响（这里也显式声明）

    rows: List[dict] = []
    for name, cands in crafted().items():
        for m in [int(x) for x in args.ms.split(",")]:
            rec = analyse(cands, m, f"crafted:{name}")
            if rec:
                rows.append(rec)

    configs = []
    for n_workers, routes in ((3, 1), (3, 2), (3, 3)):
        for hot in (0.0, 0.5):
            for m in [int(x) for x in args.ms.split(",")]:
                if m > n_workers:
                    continue
                configs.append((n_workers, routes, hot, m))

    for n_workers, routes, hot, m in configs:
        for seed in range(args.instances):
            rng = random.Random(hash((n_workers, routes, hot, m, seed)) & 0xFFFFFFFF)
            cands = random_instance(rng, n_workers, routes, args.universe,
                                    args.route_len, hot)
            rec = analyse(cands, m, f"rand:w{n_workers}/r{routes}/hot{hot}/m{m}/s{seed}")
            if rec:
                rec.update({"n_workers": n_workers, "routes": routes, "hot": hot})
                rows.append(rec)

    bounds = ("b_classic_half", "b_true_sq", "b_true_half", "b_min_sq", "b_min_half")
    viol = {b: sum(1 for row in rows if row[b] > row["ratio"] + 1e-9) for b in bounds}
    lemma_viol = sum(1 for row in rows if row["lemma_violation"])
    ac = alpha_crossover(0.5)
    print(f"实例数 {len(rows)}；α²(1−e^−α) 与经典 1/2 的交点 α₀ = {ac}"
          f"（α < α₀ 时我们的界比 1/2 更差）")
    print(f"引理 2 有效性违例（α_min > α_true）：{lemma_viol}/{len(rows)}")
    print("各候选界的违例数（bound > greedy/OPT 即为无效界）：")
    for b in bounds:
        print(f"  {b:14s} 违例 {viol[b]:3d}/{len(rows)}")
    worst = sorted(rows, key=lambda r: r["ratio"])[:5]
    print("\n最坏比值实例：")
    for row in worst:
        print(f"  {row['name']:32s} m={row['m']} r={row['ratio']:.3f} "
              f"α_true={row['alpha_true']:.3f} α_min={row['alpha_min']:.3f} "
              f"ρ_oc={row['rho_oc']:.3f} b_true_sq={row['b_true_sq']:.3f} "
              f"b_min_sq={row['b_min_sq']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "bound_tightness")
    payload = {"instances": args.instances, "universe": args.universe,
               "route_len": args.route_len, "ms": args.ms,
               "alpha_crossover_vs_half": ac,
               "lemma_violations": lemma_viol,
               "violations": viol, "rows": rows}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = ["# 界的紧性检验（E2.1/E2.2，抽象实例）", "",
             f"实例数 {len(rows)}；`universe={args.universe}`、`w={args.route_len}`、`m∈{args.ms}`。", "",
             f"**引理 2 有效性**：α_min > α_true 的实例 **{lemma_viol}/{len(rows)}**"
             "（>0 就说明这条下界不成立）。", "",
             f"**对照经典 matroid 贪心的 1/2 界**：`α²(1−e^−α) = 0.5` 在 α₀ = {ac}，"
             f"所以 **α < {ac} 时我们这套界比 1/2 还差**。", "",
             "| 候选界 | 形式 | 违例数 |", "|---|---|---|"]
    forms = {"b_classic_half": "1/2（经典）", "b_true_sq": "α_true²(1−e^−α)",
             "b_true_half": "α_true(1−e^−α)/2", "b_min_sq": "α_min²(1−e^−α)",
             "b_min_half": "α_min(1−e^−α)/2"}
    for b in bounds:
        lines.append(f"| `{b}` | {forms[b]} | **{viol[b]}** / {len(rows)} |")
    lines += ["", "## 最坏比值的实例", "",
              "| 实例 | m | greedy/OPT | α_true | α_min | ρ_oc | ρ_jac | b_true_sq | b_min_sq |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in worst:
        lines.append(f"| {row['name']} | {row['m']} | **{row['ratio']:.3f}** | "
                     f"{row['alpha_true']:.3f} | {row['alpha_min']:.3f} | {row['rho_oc']:.3f} | "
                     f"{row['rho_jac']:.3f} | {row['b_true_sq']:.3f} | {row['b_min_sq']:.3f} |")
    lines += ["", "## 手工构造实例", "",
              "| 实例 | m | greedy/OPT | α_true | α_min | ρ_oc | b_true_sq | b_min_sq |",
              "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["name"].startswith("crafted"):
            lines.append(f"| {row['name']} | {row['m']} | {row['ratio']:.3f} | "
                         f"{row['alpha_true']:.3f} | {row['alpha_min']:.3f} | {row['rho_oc']:.3f} | "
                         f"{row['b_true_sq']:.3f} | {row['b_min_sq']:.3f} |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote", stem + ".json")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()
