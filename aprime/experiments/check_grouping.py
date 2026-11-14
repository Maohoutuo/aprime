# -*- coding: utf-8 -*-
r"""E5.2：能力分桶方案的失效区与自适应判据。

## 问题

`groups.capability_groups` 原实现按 `|F_u(θ)|` 的**值**分桶（1 / 2 / ≥3）。
这在 θ 高时很准确，但在 θ 低、可行域饱和时有对称的失效区：当 `|F_u|` 全部 ≥3 时
`tight`/`two` 为空，"低能力组"根本不存在，`gqc_gap` 对任何方法都恒为 0。

把它改成分位桶（底部 30% / 中间 40% / 顶部 30%）能修好低 θ，但会在**高 θ 端退化**：
此时 `|F_u|` 集中在 {1,2}，分位桶会把同质的"只够一条"人群硬切开；而且它会**拆开并列**，
把同能力者分到两侧，本身就不成立。

因此本实验对比第三种方案 `quantile_tied`：按**能力值**的累积分布切分，同能力者必同组。
它永不拆并列、任何格子都有定义，退化程度由**组份额**如实暴露。

## 结论（先给）

**并列一致桶（`quantile_tied`）把每个格子都变成干净切分**：杂质率在全部 8 个
`(尺度, θ)` 格上都是 0.00，且低组与高组都非空。值桶在 medium θ≤0.6、small θ≤0.3
上"低组为空"（`gqc_gap` 恒为 0），分位桶则会拆开并列（medium θ=0 杂质 0.45）。

代价是**份额失衡**：medium θ=0 的低组只有 17%，`two` 常为空（实际退化为二分）。
因此判据应改用**低组份额**（如 <15% 视为不可判读），而不是继续找更平衡的分桶——
能力并列时"份额平衡"与"不拆并列"不可兼得。

「自适应判据」方案（逐种子在值桶/分位桶间切换）实测分组定义随种子翻转，已弃用。


## 度量

对每个 `(尺度, θ)`、每种方案，报告：

* **组份额** `tight/two/ample`：三组是否都非空；
* **杂质率** `impurity`：`tight` 组中能力 **≥** `ample` 组最小能力的人数占比。
  `0` = 切分干净；接近 `1` = "低能力组"实际与高能力组同能力，指标无意义；
  `tight` 为空时记作"低组为空"（无法切分，不是 0）。
* **能力值个数**：`|F_u|` 的不同取值个数（分辨率）。
* **众数在顶端**：`_tie_dominated` 的判据，决定 `adaptive` 是否切到分位桶。

## 口径

全部为合成参与者（`source="dense(...)"`），与 §6 其它结果同源。
本实验只做**分组层**诊断，不调用求解器；真实数据版本见
`docs/e0_1_data_access_plan.md` §8。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.candidates import build_candidates  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (  # noqa: E402
    GROUP_AMPLE,
    GROUP_ORDER,
    GROUP_TIGHT,
    GROUP_TWO,
    _tie_dominated,
    capability_groups,
    group_shares,
)
from aprime.model import APrimeConfig, make_problem  # noqa: E402

SCHEMES = ("value", "quantile", "quantile_tied")
LABELS = {GROUP_TIGHT: "tight", GROUP_TWO: "two", GROUP_AMPLE: "ample"}


def _caps(problem) -> Dict[str, int]:
    return {w: sum(1 for c in problem.by_worker[w] if problem.is_eligible(c))
            for w in problem.by_worker}


def _impurity(caps: Dict[str, int], labels: Dict[str, str]) -> Optional[float]:
    """`tight` 组里能力 ≥ `ample` 组最小能力的人数占比。`tight` 空则返回 None。"""
    low = [w for w, g in labels.items() if g == GROUP_TIGHT]
    high = [w for w, g in labels.items() if g == GROUP_AMPLE]
    if not low or not high:
        return None
    thr = min(caps[w] for w in high)
    return mean(1.0 if caps[w] >= thr else 0.0 for w in low)


def _cell(scale: str, theta: float, seeds: int, k: int, budget: float) -> dict:
    per_scheme: Dict[str, dict] = {}
    tie_flag: List[bool] = []
    modal_frac: List[float] = []
    n_distinct: List[int] = []

    for seed in range(seeds):
        inst = make_dense_instance("tdrive", scale, n_workers=40, seed=seed, index=0)
        cands = build_candidates(inst, k=k, seed=seed, lams=(1.0, 0.5, 0.0), n_random=2)
        prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=k, budget=budget))
        caps = _caps(prob)
        vals = [n for n in caps.values() if n >= 1]
        counts: Dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        modal_frac.append(max(counts.values()) / len(vals) if vals else 1.0)
        n_distinct.append(len(counts))
        tie_flag.append(_tie_dominated(caps))

        for scheme in SCHEMES:
            labels = capability_groups(prob, scheme=scheme)
            shares = group_shares(labels)
            rec = per_scheme.setdefault(
                scheme, {"shares": [], "impurity": [], "low_empty": 0, "high_empty": 0})
            rec["shares"].append([shares.get(g, 0.0) for g in GROUP_ORDER])
            n_low = sum(1 for g in labels.values() if g == GROUP_TIGHT)
            n_high = sum(1 for g in labels.values() if g == GROUP_AMPLE)
            if n_low == 0:
                rec["low_empty"] += 1
            if n_high == 0:
                rec["high_empty"] += 1
            imp = _impurity(caps, labels)
            if imp is not None:
                rec["impurity"].append(imp)

    out = {
        "scale": scale,
        "theta": theta,
        "modal_cap_frac": round(mean(modal_frac), 4),
        "n_distinct": round(mean(n_distinct), 2),
        "tie_dominated_frac": round(mean(1.0 if t else 0.0 for t in tie_flag), 4),
        "schemes": {},
    }
    for scheme in SCHEMES:
        rec = per_scheme[scheme]
        share_mean = [round(mean(s[i] for s in rec["shares"]), 3) for i in range(len(GROUP_ORDER))]
        out["schemes"][scheme] = {
            "shares": share_mean,
            "low_empty_seeds": rec["low_empty"],
            "high_empty_seeds": rec["high_empty"],
            "impurity": (round(mean(rec["impurity"]), 4) if rec["impurity"] else None),
        }
    return out


def _fmt_imp(rec: dict, seeds: int) -> str:
    low_gone = rec["low_empty_seeds"] == seeds
    high_gone = rec["high_empty_seeds"] == seeds
    if low_gone and high_gone:
        return "单组（无法切分）"
    if low_gone:
        return "低组为空"
    if high_gone:
        return "高组为空"
    if rec["impurity"] is None:
        return "—"
    txt = "%.2f" % rec["impurity"]
    notes = []
    if rec["low_empty_seeds"]:
        notes.append("低组空 %d/%d" % (rec["low_empty_seeds"], seeds))
    if rec["high_empty_seeds"]:
        notes.append("高组空 %d/%d" % (rec["high_empty_seeds"], seeds))
    if notes:
        txt += " (" + ", ".join(notes) + ")"
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="small,medium")
    ap.add_argument("--thetas", default="0.0,0.3,0.6,0.9")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=float, default=80.0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    scales = [s for s in args.scales.split(",") if s]
    thetas = [float(x) for x in args.thetas.split(",") if x]

    rows: List[dict] = []
    for scale in scales:
        for theta in thetas:
            row = _cell(scale, theta, args.seeds, args.k, args.budget)
            rows.append(row)
            print(f"[done] {scale} theta={theta}", flush=True)

    lines = [
        "# E5.2 能力分桶的失效区与并列一致方案",
        "",
        "*由 `experiments/check_grouping.py` 生成；不调用求解器，只做分组层诊断。*",
        "",
        "## 组份额与杂质率",
        "",
        "杂质率 = `tight` 组中能力 ≥ `ample` 组最小能力的人数占比；"
        "`0` 为干净切分，越接近 1 说明\"低能力组\"越无意义。"
        "`低组为空` 表示该方案在该格上根本无法切分。",
        "",
        "| 尺度 | θ | 众数占比 | 值个数 | 众数在顶端 | 值桶份额 | 值桶杂质 "
        "| 分位桶份额 | 分位桶杂质 | 并列一致份额 | 并列一致杂质 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def cell(scheme: str) -> tuple:
            rec = r["schemes"][scheme]
            sh = "/".join("%.2f" % x for x in rec["shares"])
            return sh, _fmt_imp(rec, args.seeds)

        v_sh, v_imp = cell("value")
        q_sh, q_imp = cell("quantile")
        t_sh, t_imp = cell("quantile_tied")
        lines.append("| %s | %.1f | %.2f | %.1f | %.2f | %s | %s | %s | %s | %s | %s |" % (
            r["scale"], r["theta"], r["modal_cap_frac"], r["n_distinct"],
            r["tie_dominated_frac"], v_sh, v_imp, q_sh, q_imp, t_sh, t_imp))

    lines += [
        "",
        "## 读法",
        "",
        "1. **值桶在低 θ 失效**：`众数在顶端 = 1.00` 的格子里值桶的 `tight` 为空，"
        "`gqc_gap` 对任何方法恒为 0——不是\"公平\"，是\"无低能力组可挤\"。",
        "2. **分位桶在高 θ 失效**：能力集中在 {1,2} 时按排名硬切会把同质人群分开，"
        "杂质率反而高于值桶（medium θ=0.9：0.40 vs 0.00）。",
        "3. **并列一致桶**（`quantile_tied`）永不切开同能力者，任何格子都有定义；"
        "其退化程度由**组份额**暴露：低组份额过小即说明该格没有可比较的低能力层。",
        "4. **并列一致桶把所有格子变成干净切分**（杂质 0.00），代价是**份额失衡**："
        "medium θ=0 的低组只有 17%、`two` 常为空（实际退化为二分）。"
        "因此判据应以**低组份额**为准（如低组 <15% 视为不可判读），"
        "而不是继续寻找更\"平衡\"的分桶——能力并列时平衡与不拆并列不可兼得。",
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "grouping_check.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    js = os.path.join(args.out_dir, "grouping_check.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"scales": scales, "thetas": thetas, "seeds": args.seeds,
                   "k": args.k, "budget": args.budget, "rows": rows},
                  fh, ensure_ascii=False, indent=2)

    print()
    print("\n".join(lines))
    print()
    print("wrote", md)


if __name__ == "__main__":
    main()
