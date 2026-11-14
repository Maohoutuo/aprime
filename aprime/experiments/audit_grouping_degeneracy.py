# -*- coding: utf-8 -*-
r"""审计：论文里用 `gqc_gap` 论证"公平性随某因素变化"的实验，基线分组是否退化。

## 为什么要有这个

`gqc_gap` 在**分组塌成一档**时恒为 0 且不带信息（见 `groups.grouping_is_degenerate`
与 `docs/data_alternatives_grab_crowdsensim.md` §8.7）。此时拿两组数据源的
`gqc_gap` 做差，得到的是"一边退化、一边没退化"的伪差异，不是公平性差异。

CROSS 试跑已经因此作废过一条结论。本脚本把其余实验的配置网格扫一遍，
逐个格子报**组数**与**最小组份额**，把退化区标出来。

## 判据

* ``n_groups < 2`` → **退化**：`gqc_gap ≡ 0`，该格不可用于任何"差距变化"的论证。
* ``min_share < 0.15`` → **不可判读**：E5.2 定下的门槛（低组过小，份额估计不可信）。
* 其余 → 可用。

只统计分组，**不调用求解器**，所以很快。

## 用法

```powershell
py -3.14 experiments/audit_grouping_degeneracy.py
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import THETA_GRID, APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (GROUP_AMPLE, GROUP_TIGHT, GROUP_TWO,  # noqa: E402
                           capability_groups, group_shares)
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import make_problem  # noqa: E402

MIN_SHARE = 0.15  # E5.2 的"低组过小即不可判读"门槛

#: (实验标签, 分组方案, 尺度, θ 网格, φ 网格, 人数, k, 它论证的"因素")
#: φ 是偏好同质化强度（with_homogeneity）；None 表示该实验不扫 φ。
REGISTRY: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[float, ...],
                      Optional[Tuple[float, ...]], int, int, str], ...] = (
    ("E5.15 偏好混合敏感性", "value", ("small",), (0.0, 0.3, 0.6), None, 20, 8,
     "公平差距是否随偏好先验移动"),
    ("公平-覆盖权衡", "value", ("small",), tuple(THETA_GRID), (0.0, 0.95), 40, 8,
     "公平差距是否随覆盖要求移动（用的是 GQCParams 默认 value 桶）"),
    ("GQC-MRG 主实验", "quantile_tied", ("small", "medium"), tuple(THETA_GRID),
     (0.0, 0.95), 40, 8, "公平差距是否随 MRG 开关/同质化移动"),
    ("GQC-MRG 敏感性", "quantile_tied", ("small", "medium"), tuple(THETA_GRID),
     (0.0, 0.95), 40, 8, "公平差距对参数的敏感性"),
    ("GQC-tied 变体", "quantile_tied", ("small", "medium"), tuple(THETA_GRID),
     None, 40, 8, "tied 桶相对 value 桶的改进"),
    ("GQC-tied 变体", "value", ("small", "medium"), tuple(THETA_GRID),
     None, 40, 8, "（对照组：value 桶本身）"),
    ("预算标定", "quantile_tied", ("small", "medium"), (0.0, 0.3, 0.6, 0.9),
     None, 40, 8, "名额下界随预算档位移动"),
)


def audit_cell(scale: str, theta: float, phi: Optional[float], scheme: str,
               n_workers: int, k: int, seed: int) -> Dict[str, object]:
    inst = make_dense_instance("tdrive", scale, n_workers=n_workers, seed=seed, index=0)
    if phi is not None:
        inst = with_homogeneity(inst, phi, seed=seed)
    cands = build_candidates(inst, k=k, seed=seed, lams=(1.0, 0.5, 0.0), n_random=2)
    prob = make_problem(inst, cands, APrimeConfig(theta=theta, k_cand=k))
    groups = capability_groups(prob, scheme=scheme)
    share = group_shares(groups) if groups else {}
    n_groups = len(share)
    min_share = min(share.values()) if share else 0.0
    if n_groups < 2:
        flag = "退化"
    elif min_share < MIN_SHARE:
        flag = "不可判读"
    else:
        flag = "可用"
    return {"n_groups": n_groups, "min_share": round(min_share, 4), "flag": flag,
            "shares": {g: round(v, 4) for g, v in share.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/grouping_degeneracy_audit.md")
    args = ap.parse_args()

    results = []
    for label, scheme, scales, thetas, phis, nw, k, claim in REGISTRY:
        for scale in scales:
            for phi in (phis if phis else (None,)):
                for theta in thetas:
                    cells = [audit_cell(scale, theta, phi, scheme, nw, k, s)
                             for s in range(args.seeds)]
                    flags = [c["flag"] for c in cells]
                    worst = ("退化" if "退化" in flags
                             else "不可判读" if "不可判读" in flags else "可用")
                    results.append({
                        "experiment": label, "claim": claim, "scheme": scheme,
                        "scale": scale, "phi": phi, "theta": theta,
                        # 用**最坏种子**报告：判定按最坏种子取，数字也必须对齐，
                        # 否则会出现「均值 0.183 却标不可判读」这种自相矛盾的行。
                        "n_groups": min(c["n_groups"] for c in cells),
                        "min_share": min(c["min_share"] for c in cells),
                        "flag": worst,
                    })

    def count(key: str, val) -> int:
        return sum(1 for r in results if r[key] == val)

    lines = ["# 分组退化审计：`gqc_gap` 的论证是否建立在退化基线上", "",
             f"*`experiments/audit_grouping_degeneracy.py` 生成；{args.seeds} 种子/格。*", "",
             f"判据：组数 < 2 → **退化**（`gqc_gap ≡ 0`）；最小组份额 < {MIN_SHARE:.0%} "
             "→ **不可判读**（E5.2 门槛）；否则 **可用**。", "",
             "## 总览", "",
             "| 实验 | 方案 | 它论证的因素 | 格数 | 退化 | 不可判读 | 可用 | 结论 |",
             "|---|---|---|---|---|---|---|---|"]

    seen = []
    for label, scheme, scales, thetas, phis, nw, k, claim in REGISTRY:
        if (label, scheme) in seen:
            continue
        seen.append((label, scheme))
        sub = [r for r in results if r["experiment"] == label and r["scheme"] == scheme]
        n_bad = sum(1 for r in sub if r["flag"] == "退化")
        n_thin = sum(1 for r in sub if r["flag"] == "不可判读")
        n_ok = sum(1 for r in sub if r["flag"] == "可用")
        if n_bad:
            verdict = "**有退化格，结论不可直接用于跨条件比较**"
        elif n_thin:
            verdict = "无退化但有不可判读格，需按格剔除"
        else:
            verdict = "全格可用"
        lines.append("| %s | %s | %s | %d | %d | %d | %d | %s |" % (
            label, scheme, claim, len(sub), n_bad, n_thin, n_ok, verdict))

    # 逐格明细（只列有问题的）
    lines += ["", "## 问题格明细", ""]
    bad = [r for r in results if r["flag"] != "可用"]
    if not bad:
        lines.append("无。全部格子分组都非退化且最小组份额达标。")
    else:
        lines += ["| 实验 | 方案 | 尺度 | φ | θ | 组数 | 最小组份额 | 判定 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in bad:
            lines.append("| %s | %s | %s | %s | %.1f | %.1f | %.3f | **%s** |" % (
                r["experiment"], r["scheme"], r["scale"],
                "—" if r["phi"] is None else "%.2f" % r["phi"],
                r["theta"], r["n_groups"], r["min_share"], r["flag"]))
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
