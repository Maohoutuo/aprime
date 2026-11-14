# -*- coding: utf-8 -*-
r"""E5.1：`exclusion_gap` 的统计功效——为什么 medium 上会"反转"。

## 问题

`analysis/fairness.py::fairness_report` 把合法池按能力 `|F_u(θ)|` 的**中位数**二分成
"低能力组 / 高能力组"，再比较两组的被排除率之差。这个定义隐含一个前提：能力有分层。
一旦能力高度并列（例如 medium 网格上多数参与者都触顶到同一 `|F_u|`），中位数切分就会
把**同一能力的人**分到两侧，此时：

* "高能力组"可能根本不存在（无人严格高于中位数）；
* 哪一半被算作"低能力"退化成 **wid 排序** 的产物，而不是能力差异。

于是观测到的 `exclusion_gap` 变成噪声，甚至符号翻转。这正是 §6.11 开放问题 1
（"medium 上 capacity 不造成挤出"）的真实原因——不是挤出消失，而是指标不可测。

## 本实验给出三件事

1. **并列率** `modal_cap_frac`：合法池内最常见能力值的占比；
2. **置换零分布**：把"哪些人被选中"在合法池内随机置换（保持选中人数不变），
   重算 `exclusion_gap`，得到该实例上指标的噪声带；
3. **统计功效** `power`：观测值的双侧 p 值 < 0.05 的单元占比。

判据：`power` 低的实例上，`exclusion_gap` **不可用于比较方法**；应在论文里
同时报告 `modal_cap_frac`，或改用对并列稳健的分组。

口径说明：全部为合成参与者（`source="dense(...)"`），与 §6 其它结果同源；
真实数据版本见 `docs/e0_1_data_access_plan.md` §8。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import aprime  # noqa: F401,E402
from aprime.analysis.fairness import capability, fairness_report  # noqa: E402
from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.solvers.baselines import capacity_greedy  # noqa: E402
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402

METHODS = ("capacity", "aprime", "gqc")

#: 候选池"上界"诊断：`|C_u|` 的最大值 = len(lams) + n_random，与 k 无关。
#: 用递增的尝试次数检查并列到底是"采样不足"还是"能力真的没分化"。
CEILING_CONFIGS = (
    ((1.0, 0.5, 0.0), 2, "3λ + 2随机（上界 5，现用配置）"),
    ((1.0, 0.75, 0.5, 0.25, 0.0), 6, "5λ + 6随机（上界 11）"),
    ((1.0, 0.75, 0.5, 0.25, 0.0), 12, "5λ + 12随机（上界 17）"),
)


def ceiling_diagnostic(scale, seeds=3, k=99):
    """`|C_u|` 的并列是不是因为生成器上界太低（删失）？

    `|C_u|` 统计的是"固定次数的生成尝试里成功几条"，所以它的取值被
    ``len(lams) + n_random`` 硬性截断。若并列率随上界抬高而显著下降，
    说明是删失；若仍然堆积在上界，说明该实例上可行域饱和、能力真的没有分化。
    """
    rows = []
    for lams, n_random, label in CEILING_CONFIGS:
        ceiling = len(lams) + n_random
        caps = []
        for seed in range(seeds):
            inst = make_dense_instance("tdrive", scale, n_workers=40, seed=seed, index=0)
            cands = build_candidates(inst, k=k, seed=seed, lams=lams, n_random=n_random)
            per = {}
            for c in cands:
                per[c.wid] = per.get(c.wid, 0) + 1
            caps += [per.get(w.wid, 0) for w in inst.workers]
        arr = np.array(caps, float)
        vals, counts = np.unique(arr, return_counts=True)
        rows.append({
            "config": label, "ceiling": ceiling, "max": int(arr.max()),
            "modal_frac": round(float(counts.max() / len(arr)), 4),
            "at_ceiling_frac": round(float((arr >= ceiling).mean()), 4),
        })
    return rows


def _solve_all(prob):
    out = {}
    out["aprime"] = solve(prob, with_ls=True)[0]
    out["gqc"] = gqc_solve(prob, GQCParams(), with_ls=True)[0]
    out["capacity"] = capacity_greedy(prob)[0]
    return out


def _median_split_gap(caps: np.ndarray, chosen: set, pool, min_pool: int = 4):
    """**逐字复刻** fairness.py 的中位数二分，供置换检验复用。

    返回 (gap, n_low, n_high)；池太小或能力全等时返回 (None, ...)。
    """
    if len(pool) < min_pool or caps.sum() <= 0:
        return None, 0, 0
    order = np.argsort(caps)
    half = len(order) // 2
    low = [pool[i] for i in order[:half]]
    high = [pool[i] for i in order[half:]]
    if not low or not high:
        return None, len(low), len(high)
    le = 1.0 - mean(1.0 if w in chosen else 0.0 for w in low)
    he = 1.0 - mean(1.0 if w in chosen else 0.0 for w in high)
    return le - he, len(low), len(high)


def _null_distribution(caps, n_chosen, pool, perms, rng):
    """保持选中人数不变，随机置换"谁被选中"，返回零分布样本。"""
    n = len(pool)
    if n < 4 or caps.sum() <= 0:
        return np.array([0.0])
    idx = np.arange(n)
    out = []
    for _ in range(perms):
        pick = set(rng.sample(range(n), min(n_chosen, n)))
        chosen = {pool[i] for i in pick}
        g, _, _ = _median_split_gap(caps, chosen, pool)
        out.append(0.0 if g is None else g)
    return np.array(out)


def _modal_frac(caps: np.ndarray) -> float:
    if len(caps) == 0:
        return 0.0
    vals, counts = np.unique(caps, return_counts=True)
    return float(counts.max() / len(caps))


def build(scale, seed, k):
    """实例与候选池都与 budget / θ 无关，按 (scale, seed) 只建一次。"""
    inst = make_dense_instance("tdrive", scale, n_workers=40, seed=seed, index=0)
    cands = build_candidates(inst, k=k, seed=seed, lams=(1.0, 0.5, 0.0), n_random=2)
    return inst, cands


def run_cell(inst, cands, scale, budget, seed, theta, k, perms, rng):
    prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=k, budget=budget))
    sels = _solve_all(prob)

    cap = capability(prob)["n_eligible_routes"]
    pool = sorted(prob.by_worker)
    # fairness_report 只在**合法池**（|F_u| ≥ 1）内做中位数二分，这里必须一致
    feasible = [w for w in pool if cap[w] >= 1]
    caps = np.array([float(cap[w]) for w in feasible])
    modal = _modal_frac(caps)

    rows = []
    for m in METHODS:
        sel = sels[m]
        chosen = {c.wid for c in sel}
        rep = fairness_report(prob, sel, grid=(inst.nx, inst.ny), covered_cells=prob.covered(sel))
        obs = rep["exclusion_gap"]
        # 自检：复刻的中位数切分必须与 fairness_report 一致
        g, n_low, n_high = _median_split_gap(caps, chosen, feasible)
        assert g is None or abs(g - obs) < 1e-3, (m, g, obs)
        n_chosen_feasible = sum(1 for w in feasible if w in chosen)
        null = _null_distribution(caps, n_chosen_feasible, feasible, perms, rng)
        p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
        rows.append({
            "method": m, "exclusion_gap": obs, "modal_cap_frac": round(modal, 4),
            "n_selected": len(sel), "n_distinct_workers": len(chosen),
            "null_sd": round(float(null.std()), 4),
            "null_q95": round(float(np.quantile(np.abs(null), 0.95)), 4),
            "p_value": round(p, 4),
            "detectable": bool(p < 0.05),
        })
    density = len(sels["capacity"]) / len(pool)
    for r in rows:
        r.update({"scale": scale, "budget": budget, "seed": seed, "theta": theta,
                  "select_density": round(density, 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--budgets", default="40,80,160")
    ap.add_argument("--thetas", default="0.0,0.3,0.6,0.9")
    ap.add_argument("--n", type=int, default=8, help="种子数")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    require_available("E5.1")  # 合成机制实验：docs/e0_1_data_access_plan.md §8

    scales = [s for s in args.scales.split(",") if s]
    budgets = [float(b) for b in args.budgets.split(",") if b]
    thetas = [float(t) for t in args.thetas.split(",") if t]

    rows = []
    for scale in scales:
        for seed in range(args.n):
            inst, cands = build(scale, seed, args.k)
            for budget in budgets:
                for theta in thetas:
                    rng = random.Random(1_000_003 * seed + int(theta * 1000) + int(budget))
                    rows.extend(run_cell(inst, cands, scale, budget, seed, theta,
                                         args.k, args.perms, rng))
            print(f"[done] {scale} seed={seed}", flush=True)

    ceil_diag = {scale: ceiling_diagnostic(scale) for scale in scales}

    # ---- 汇总：每个 (scale, budget) × method 一行
    lines = [
        "# `exclusion_gap` 的统计功效（E5.1）",
        "",
        "生成脚本 `experiments/run_excl_gap_power.py`。回答 §6.11 开放问题 1："
        "medium 上 `exclusion_gap` 为何反转。",
        "",
        "**指标**：`modal_cap_frac` = 合法池内最常见能力 `|F_u(θ)|` 的占比；"
        "`null_sd` = 置换零分布（随机指派同样多的名额）的标准差；"
        "`power` = 观测值双侧 p<0.05 的单元占比。",
        "",
        "> 判据：`power` 低时 `exclusion_gap` **不可用于比较方法**。",
        "",
        "> 全部为合成参与者结果，真实数据版本见 `docs/e0_1_data_access_plan.md` §8。",
        "",
        "| 尺度 | 预算 | 方法 | 并列率 | `exclusion_gap` | 零分布 sd | 零分布 95%分位 | 功效 power | 平均选中人数 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for scale in scales:
        for budget in budgets:
            for m in METHODS:
                sub = [r for r in rows if r["scale"] == scale and r["budget"] == budget
                       and r["method"] == m]
                if not sub:
                    continue
                lines.append("| %s | %.0f | %s | %.2f | %+.3f | %.3f | %.3f | %.2f | %.1f |" % (
                    scale, budget, m, mean(r["modal_cap_frac"] for r in sub),
                    mean(r["exclusion_gap"] for r in sub),
                    mean(r["null_sd"] for r in sub),
                    mean(r["null_q95"] for r in sub),
                    mean(1.0 if r["detectable"] else 0.0 for r in sub),
                    mean(r["n_distinct_workers"] for r in sub)))

    lines += [
        "",
        "读法：**并列率是主判据，`power` 只作必要性检查。** 并列率 → 1（众数 = 最大值）时，"
        "按排名切出的'低能力组'全部落在并列之内，而 `capacity` 又按 wid 破平局"
        "（`key=(-len, wid)`），两者同序 ⇒ 指标测的是'wid 字母序吻合度'；此时 `power` 反而**虚高**"
        "（medium 换富候选池后：并列率 1.00、excl −0.49、power 1.00，全是伪相关）。",
        "",
        "## 为什么并列：`|C_u|` 的上界是生成器的产量，不是参与者的属性",
        "",
        "`|C_u|` 统计的是\"固定次数的生成尝试里成功几条\"，因此取值被 "
        "`len(lams) + n_random` 硬性截断（现用配置 = 3 + 2 = **5**，`k=8` 从不生效）。"
        "`θ=0` 时中位数若等于这个上界，\"严格高于中位数\"的集合就是空的。",
        "",
        "用递增的上界检查并列是\"采样不足（删失）\"还是\"能力真的没分化（饱和）\"：",
        "",
        "| 尺度 | 生成配置 | 上界 | 实测最大 | 并列率 | 顶在上界的比例 |",
        "|---|---|---|---|---|---|",
    ]
    for scale in scales:
        for d in ceil_diag[scale]:
            lines.append("| %s | %s | %d | %d | %.2f | %.2f |" % (
                scale, d["config"], d["ceiling"], d["max"],
                d["modal_frac"], d["at_ceiling_frac"]))
    lines += [
        "",
        "读法：并列率随上界抬高而**大幅下降** ⇒ 删失（多生成几条路线即可修）；"
        "仍**堆积在上界** ⇒ 饱和（可行域已经全通，`|C_u|` 退化为常数，该实例上能力无分化）。",
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "excl_gap_power")
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump({"scales": scales, "budgets": budgets, "thetas": thetas,
                   "n": args.n, "k": args.k, "perms": args.perms,
                   "ceiling_diagnostic": ceil_diag, "rows": rows},
                  fh, ensure_ascii=False, indent=2)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()
