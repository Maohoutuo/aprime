# -*- coding: utf-8 -*-
"""公平性度量（E3.1 / E3.2 / 主权衡图的第三、四面板）。

设计原则：**区分"被选中的人内部是否公平"和"谁被排除在外"**。这两件事在 MCS 里
经常被混为一谈，而方案 §2.5 的主张恰恰是"CFI 型公平只看前者，会结构性挤出低能力者"。

指标一览（`fairness_report` 一次算全）：

| 键 | 含义 | 方向 |
|---|---|---|
| ``feasible_rate`` | 有 ≥1 条合法路线的参与者比例（θ 上升时必然下降，**这是个性化的代价，不是选择规则的不公平**） | — |
| ``gini_pss`` | 入选者 PSS 的基尼系数 | 越小越公平 |
| ``jain_selected`` | 入选者 PSS 的 Jain 指数 | 越大越公平 |
| ``jain_feasible`` | **合法池**上的 Jain（未入选=0，入选=S/max S）；分母只算有合法路线的人 | 越大越公平 |
| ``exclusion_gap`` | 低能力组被排除率 − 高能力组被排除率，**只在合法池内计算**（E3.1 的靶子指标） | 负值 = 低能力者更易被挤掉 |
| ``select_rate`` | 合法池内的入选率 | — |
| ``cfi_gap`` | 名额份额与能力份额的最大偏差（CFI 型公平的自洽度） | 越小越"CFI 式公平" |
| ``cap_corr`` | 名额与能力的 Spearman 相关（越高越"能力主义"） | 视角不同 |
| ``spatial_gini`` | 覆盖在网格分块上的基尼（空间均衡） | 越小越均衡 |
| ``theta_margin`` | 入选者的 `min(S_u − θ_u)` | ≥0 才算可行 |

**为什么必须把"合法池"单独拎出来**：θ 一提高，很多参与者**根本没有**满足下界的路线，
他们被排除是因为"能力不够"而不是"规则不公平"。把这两件事混在一起算，会让
`exclusion_gap` 在高 θ 端出现虚假的"更公平"（Day-4 实测踩过：φ=0、θ=0.9 时
`exclusion_gap` 跳到 +0.30）。分开之后，`feasible_rate` 报代价、`exclusion_gap` 报规则。

说明：`exclusion_gap` 与 `cfi_gap` 方向相反——CFI 型方法会把 `cfi_gap` 压到很小，
但 `exclusion_gap` 反而更大。这个"对着干"正是论文要展示的。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..groups import capability_groups, gqc_gap as _gqc_gap
from ..model import Candidate, SelectionProblem


def _gini(x: Sequence[float]) -> float:
    v = np.sort(np.asarray([max(0.0, float(t)) for t in x], dtype=float))
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def _jain(x: Sequence[float]) -> float:
    v = np.asarray([max(0.0, float(t)) for t in x], dtype=float)
    if len(v) == 0 or (v ** 2).sum() <= 0:
        return 0.0
    return float(v.sum() ** 2 / (len(v) * (v ** 2).sum()))


def _rank_corr(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def capability(problem: SelectionProblem) -> Dict[str, float]:
    """能力代理：该参与者候选池里**达到 θ** 的路线条数（θ 越高越低能力者越先归零）。"""
    cap: Dict[str, int] = {w: 0 for w in problem.by_worker}
    best: Dict[str, float] = {w: 0.0 for w in problem.by_worker}
    for c in problem.candidates:
        if problem.is_eligible(c):
            cap[c.wid] += 1
            best[c.wid] = max(best[c.wid], c.sat)
    return {"n_eligible_routes": cap, "max_pss": best}


def fairness_report(problem: SelectionProblem, sel: Sequence[Candidate],
                    grid: Optional[tuple] = None, covered_cells=None) -> dict:
    """算全套公平性指标。``grid=(nx, ny)`` 时额外给空间均衡；否则 spatial_gini=0。

    所有"排除/公平"类指标都只在**合法池**（有 ≥1 条合法路线的人）内计算，
    `feasible_rate` 单独报个性化的代价。
    """
    cap = capability(problem)
    workers: List[str] = sorted(problem.by_worker)
    feasible = [w for w in workers if cap["n_eligible_routes"][w] >= 1]
    chosen = {c.wid: c for c in sel}

    # 名额与能力（合法池内）
    cnt = np.array([1.0 if w in chosen else 0.0 for w in feasible])
    caps = np.array([float(cap["n_eligible_routes"][w]) for w in feasible])
    pss_sel = [c.sat for c in sel]
    attain = []
    for w in feasible:
        if w in chosen:
            top = cap["max_pss"][w]
            attain.append(chosen[w].sat / top if top > 0 else 0.0)
        else:
            attain.append(0.0)

    # 低/高能力组的被排除率（合法池内按能力中位数切成两半）
    excl_gap = 0.0
    select_rate = float(cnt.mean()) if len(cnt) else 0.0
    if len(feasible) >= 4 and caps.sum() > 0:
        order = np.argsort(caps)
        half = len(order) // 2
        low, high = order[:half], order[half:]
        low_excl = 1.0 - cnt[low].mean() if len(low) else 0.0
        high_excl = 1.0 - cnt[high].mean() if len(high) else 0.0
        excl_gap = float(low_excl - high_excl)

    # CFI 型自洽度：名额份额与能力份额的偏差（合法池内）
    cfi_gap = 0.0
    if cnt.sum() > 0 and caps.sum() > 0:
        cfi_gap = float(np.max(np.abs(cnt / cnt.sum() - caps / caps.sum())))

    spatial_gini = 0.0
    if grid is not None and covered_cells is not None:
        nx, ny = int(grid[0]), int(grid[1])
        qx, qy = max(1, nx // 4), max(1, ny // 4)
        blocks: Dict[tuple, int] = {}
        for (x, y) in covered_cells:
            blocks[(x // qx, y // qy)] = blocks.get((x // qx, y // qy), 0) + 1
        spatial_gini = _gini(list(blocks.values())) if blocks else 0.0

    margins = [c.sat - problem.theta_of(c.wid) for c in sel]
    cap_groups = capability_groups(problem)
    gqc = _gqc_gap(sel, cap_groups)
    return {
        "n_selected": len(sel),
        "feasible_rate": round(len(feasible) / len(workers), 4) if workers else 0.0,
        "select_rate": round(select_rate, 4),
        "gini_pss": round(_gini(pss_sel), 4),
        "jain_selected": round(_jain(pss_sel), 4),
        "jain_feasible": round(_jain(attain), 4),
        "min_pss": round(min(pss_sel), 4) if pss_sel else 0.0,
        "max_pss": round(max(pss_sel), 4) if pss_sel else 0.0,
        "pss_spread": round((max(pss_sel) - min(pss_sel)) / max(pss_sel), 4)
        if pss_sel and max(pss_sel) > 0 else 0.0,
        "exclusion_gap": round(excl_gap, 4),
        "cfi_gap": round(cfi_gap, 4),
        "gqc_gap": round(gqc, 4),
        "cap_corr": round(_rank_corr(cnt, caps), 4),
        "spatial_gini": round(spatial_gini, 4),
        "theta_margin": round(min(margins), 4) if margins else 0.0,
        "theta_violations": sum(1 for c in sel if not problem.is_eligible(c)),
        # ---- 不分桶的表示度（见 `representation_metrics` 的说明）
        **representation_metrics(problem, sel),
    }


def ks_capability(problem: SelectionProblem, sel: Sequence[Candidate]) -> float:
    """选中者的能力分布 vs 合法池能力分布的**两样本 KS 距离**。

    不用分桶：取的是一整条经验分布曲线的最大差。因此不存在"组数塌成一档"
    或"最小组太小"这两种让 `gqc_gap` 失效的情形——方差由样本量决定，
    而不是由最小组规模决定。实测跨种子标准差比 `gqc_gap` 低约 10–40%
    （见 ``experiments/compare_fairness_metrics.md``）。
    """
    cap = capability(problem)["n_eligible_routes"]
    pool = [float(cap[w]) for w in problem.by_worker if cap[w] >= 1]
    selc = [float(cap[c.wid]) for c in sel if cap.get(c.wid, 0) >= 1]
    if not pool or not selc:
        return 0.0
    xs = sorted(set(pool) | set(selc))
    np_, ns = len(pool), len(selc)
    best = 0.0
    for x in xs:
        f_pool = sum(1 for v in pool if v <= x) / np_
        f_sel = sum(1 for v in selc if v <= x) / ns
        best = max(best, abs(f_pool - f_sel))
    return float(best)


def rank_shift(problem: SelectionProblem, sel: Sequence[Candidate]) -> float:
    """选中者能力在合法池内的平均百分位，相对池中心 0.5 的偏移（**带符号**）。

    ``0`` = 选中者在能力上完全中位；负 = 偏向低能力者；正 = 偏向高能力者。
    并列取中点，避免名次破平局带来的系统性偏置。
    """
    cap = capability(problem)["n_eligible_routes"]
    pool = [float(cap[w]) for w in problem.by_worker if cap[w] >= 1]
    selc = [float(cap[c.wid]) for c in sel if cap.get(c.wid, 0) >= 1]
    if not pool or not selc:
        return 0.0
    n = len(pool)
    ps = []
    for c in selc:
        below = sum(1 for v in pool if v < c)
        equal = sum(1 for v in pool if v == c)
        ps.append((below + 0.5 * equal) / n)
    return float(sum(ps) / len(ps) - 0.5)


def representation_metrics(problem: SelectionProblem,
                           sel: Sequence[Candidate]) -> dict:
    """不分桶的表示度：`ks_capability` 与 `rank_shift`（含绝对值版）。

    为什么要有这两个：``gqc_gap`` 依赖"能力分桶"，当分组塌成一档时恒为 0、
    当最小组过小时份额估计不可信（`docs/data_alternatives_grab_crowdsensim.md`
    §8.7–8.9）。两者都是**在能力轴上直接比较分布**，不经过任何桶划分，
    因此不会失效。
    """
    rs = rank_shift(problem, sel)
    return {
        "ks_capability": round(ks_capability(problem, sel), 4),
        "rank_shift": round(rs, 4),
        "rank_shift_abs": round(abs(rs), 4),
    }


def bootstrap_ci(values: Sequence[float], seed: int = 0, n_boot: int = 2000,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """百分位自助法置信区间（默认 95%）。

    为什么必须报区间：这些公平度量的**跨种子噪声极大**——实测 `gqc_gap`
    在默认预算下变异系数（SD/均值）约 0.8，意味着"换个种子就会跳掉自身量级
    的一半以上"（``experiments/compare_fairness_metrics.py``）。
    只报点估计会把噪声当成效应。区间比点估计更能反映"这个差异是否真实存在"。
    """
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    boots = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (float(lo), float(hi))
