# -*- coding: utf-8 -*-
"""能力分组与配额（创新点 3 的共同底座，供 `solvers/gqc.py` 与 `analysis/fairness.py` 共用）。

## 为什么按"合法路线条数"分组

`|F_u(θ)|` = 该参与者在 θ 下**还剩几条能达标的路线**，这是"个性化强度把谁挤出去"的
直接物理量：

| 组 | 含义 | 在 θ↑ 时的命运 |
|---|---|---|
| `tight`（=1） | 刚好达标，只有一条退路 | 最先被 θ 挤成"无可行路" |
| `two`（=2） | 有两条备选 | 中等 |
| `ample`（≥3） | 有余量，θ 高了也不慌 | 结构性受益者 |

按**值**分桶（而不是分位数）是刻意的：分位数在 θ 高时会把"1/1/1/2"硬切成三组，
把本来同质的"只够一条"人群拆开，反而看不出结构。值分桶的解释也更直白。

但这条理由只覆盖了**高 θ 端**。低 θ 端值分桶有一个对称的失效区：`|F_u|` 堆积在顶端
（如 medium 上 ≥3 占 82%）时 `tight`/`two` 为空，"低能力组"根本不存在。
两种失效区互补，因此提供 `scheme="adaptive"`：按 `|F_u|` 的分布形状（众数是否为最大值）
在值桶与分位桶之间自动切换，判据可先验计算。GQC 算法本身仍用 `"value"`（改组会改变算法）。

## 为什么配额按"人数份额"而不是"能力份额"

东华 CFI 型公平要求 **名额 ∝ 能力**（`{cap}_u`），这在 MCS 里等于"能者多劳多占"。
但 Day-4 实测（`results/fairness_tradeoff_tdrive_small.md`）：
* `cfi_gap` 在所有方法上都落在 **0.141–0.180**，**几乎不区分优劣**；
* 而 CFI 型 capacity 方法的 `exclusion_gap` 最大（0.139–0.240），覆盖最低。

原因是能力份额本身被 θ 扭曲过：θ 越高，"只够一条"的人越多、能力份额越集中，
按能力配额就会把**能力稀缺**放大成**代表权稀缺**。

GQC 改成 **名额 ∝ 合法池内人数份额**（一人一票的代表权），
于是要压的量也从 `cfi_gap` 换成 `gqc_gap`：

```
quota_g = |G_g ∩ 合法池| / |合法池|          （人数份额）
gqc_gap = max_g | cnt_g/|Ω| − quota_g |      （名额份额与人数份额的最大偏差）
```

**注意这两套公平观是不兼容的**（这正是创新点 3 的论点）：
GQC 会把 `gqc_gap` 压小，代价是 `cfi_gap` 变大（因为它故意不按能力分配名额）。
在论文里要明写这一点，否则会被问"为什么 GQC 的 cfi 指标变差了"。
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

from .model import Candidate, SelectionProblem

GROUP_TIGHT = "tight"     # 值桶：|F_u| = 1；分位桶：能力排名底部 30%
GROUP_TWO = "two"         # 值桶：|F_u| = 2；分位桶：中间 40%
GROUP_AMPLE = "ample"     # 值桶：|F_u| >= 3；分位桶：顶部 30%
GROUP_ORDER = (GROUP_TIGHT, GROUP_TWO, GROUP_AMPLE)

#: 分桶方案。
#:
#: * ``value``         —— 按 ``|F_u|`` 的值分桶（原实现，GQC 算法默认使用）；
#: * ``quantile``      —— 按 ``(|F_u|, wid)`` 排名切 30/40/30，**会切开并列**；
#: * ``quantile_tied`` —— 按**能力值**的累积分布切分，同能力者必同组。
#:
#: ``value`` 在低 θ 失效（``|F_u|`` 堆积在顶端时 ``tight``/``two`` 为空，"低能力组"不存在）；
#: ``quantile`` 在高 θ 失效（``|F_u|`` 集中在 {1,2} 时按排名硬切会把同质人群分开）。
#: 曾尝试"众数在顶端则切分位、否则切值"的逐种子自适应方案，实测其分组定义会**随种子翻转**
#: （medium θ=0.6 有 40% 种子切值桶、60% 切分位桶），指标无法跨种子平均，故已弃用。
#: ``quantile_tied`` 永不切开并列，任何格子上都有定义；退化程度由组份额如实暴露
#: （低组过小即说明该格不可判读），诊断见 ``experiments/check_grouping.py``。
GROUP_SCHEMES = ("value", "quantile", "quantile_tied")



def _label(n_routes: int) -> Optional[str]:
    if n_routes <= 0:
        return None          # 无合法路线 → 不在合法池里，不参与配额
    if n_routes == 1:
        return GROUP_TIGHT
    if n_routes == 2:
        return GROUP_TWO
    return GROUP_AMPLE


def capability_groups(problem: SelectionProblem,
                      workers: Optional[Sequence[str]] = None,
                      scheme: str = "value") -> Dict[str, str]:
    """`wid -> 组标签`，只收录**合法池内**的人（|F_u(θ)| ≥ 1）。

    ``scheme`` 见 :data:`GROUP_SCHEMES`。三种方案产出的**标签集合相同**，
    因此 :func:`group_shares` / :func:`group_counts` / :func:`gqc_gap` / :func:`quotas`
    无需改动。默认 ``"value"`` 与历史行为逐位一致。
    """
    if scheme not in GROUP_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {GROUP_SCHEMES}")
    pool = workers if workers is not None else sorted(problem.by_worker)
    caps = {w: _n_eligible(problem, w) for w in pool}
    if scheme == "quantile":
        return _quantile_labels(caps)
    if scheme == "quantile_tied":
        return _tie_consistent_labels(caps)
    return _value_labels(caps)


def _n_eligible(problem: SelectionProblem, wid: str) -> int:
    return sum(1 for c in problem.by_worker[wid] if problem.is_eligible(c))


def _value_labels(caps: Dict[str, int]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for w, n in caps.items():
        lab = _label(n)
        if lab is not None:
            out[w] = lab
    return out


def _quantile_labels(caps: Dict[str, int],
                     edges: Sequence[float] = (0.30, 0.70)) -> Dict[str, str]:
    """按排名切档：底部 30% → `tight`，中间 40% → `two`，顶部 30% → `ample`。

    并列按 wid 破平局，因此同能力的人可能跨两个档——这正是"杂质率"要衡量的东西。
    """
    feas = sorted((w for w, n in caps.items() if n >= 1), key=lambda w: (caps[w], w))
    total = len(feas)
    out: Dict[str, str] = {}
    for i, w in enumerate(feas):
        q = i / max(1, total)
        out[w] = (GROUP_TIGHT if q < edges[0]
                  else GROUP_TWO if q < edges[1]
                  else GROUP_AMPLE)
    return out


def _tie_consistent_labels(caps: Dict[str, int],
                           edges: Sequence[float] = (0.30, 0.70)) -> Dict[str, str]:
    """按**能力值**的累积质量切分，**同一能力值绝不拆到不同组**。

    单个值的归属由它覆盖的累积区间决定：

    * 累积区间触及上分位（`cum_after ≥ edges[1]`）→ `ample`；
    * 完全位于下分位之下（`cum_before ≤ edges[0]`）→ `tight`；
    * 其余 → `two`。

    这样"并列质量"归到**上层**：medium θ=0 时 82% 的人同处最高能力值，
    他们整体进 `ample`，严格更低的那 18% 进 `tight`——两侧都非空且不同能力，
    正是"低能力 vs 高能力"该有的比较。代价是组份额可能严重不均，
    这恰是如实暴露"该格能力无分化"的方式。
    """
    vals = [n for n in caps.values() if n >= 1]
    if not vals:
        return {}
    total = len(vals)
    hist: Dict[int, int] = {}
    for v in vals:
        hist[v] = hist.get(v, 0) + 1
    bucket_of: Dict[int, str] = {}
    cum = 0
    for v in sorted(hist):
        before = cum / total
        after = (cum + hist[v]) / total
        bucket_of[v] = (GROUP_AMPLE if after >= edges[1]
                        else GROUP_TIGHT if before <= edges[0]
                        else GROUP_TWO)
        cum += hist[v]
    return {w: bucket_of[n] for w, n in caps.items() if n >= 1}


def _tie_dominated(caps: Dict[str, int]) -> bool:
    """众数能力值是否**等于最大能力值**（能力堆积在顶端）。

    若成立，值分桶会把所有人塞进 `ample`（低组为空），此时应改判为排名分桶。
    典型：medium 网格 θ≤0.4 时 ``|F_u|`` 全部 ≥3。
    """
    vals = [n for n in caps.values() if n >= 1]
    if not vals:
        return False
    cnt: Dict[int, int] = {}
    for v in vals:
        cnt[v] = cnt.get(v, 0) + 1
    modal = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return modal == max(vals)


def group_shares(group_of: Dict[str, str]) -> Dict[str, float]:
    """合法池内各组的人数份额（和为 1；池空则返回空字典）。"""
    if not group_of:
        return {}
    total = float(len(group_of))
    cnt: Dict[str, int] = {}
    for g in group_of.values():
        cnt[g] = cnt.get(g, 0) + 1
    return {g: cnt[g] / total for g in GROUP_ORDER if g in cnt}


def group_counts(sel: Sequence[Candidate], group_of: Dict[str, str]) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for c in sel:
        g = group_of.get(c.wid)
        if g is not None:
            cnt[g] = cnt.get(g, 0) + 1
    return cnt


def quotas(share: Dict[str, float], k: int, kappa: float = 1.0) -> Dict[str, float]:
    """目标名额数 `κ·share_g·k`（不取整，取整交给调用方）。"""
    return {g: kappa * s * k for g, s in share.items()}


def gqc_gap(sel: Sequence[Candidate], group_of: Dict[str, str]) -> float:
    """名额份额与人数份额的最大偏差。池空 / 未选出任何合法元素时返回 0。"""
    if not sel or not group_of:
        return 0.0
    share = group_shares(group_of)
    cnt = group_counts(sel, group_of)
    total = sum(cnt.values())
    if total <= 0:
        return 0.0
    return max((abs(cnt.get(g, 0) / total - s) for g, s in share.items()), default=0.0)


def grouping_is_degenerate(group_of: Dict[str, str]) -> bool:
    """分组是否退化到**少于两档**——此时 :func:`gqc_gap` 恒为 0 且无信息量。

    这不是"完全公平"，而是"没有可比较的组"：只剩一档时人数份额与选中份额
    双方都是 100%，差值必然是 0。用 0 去论证"公平"会把指标退化当成结论。

    这是实测中反复踩到的坑：``scheme="value"`` 在 θ 较松时会让几乎所有人
    都落进 ``ample``，于是**基线恒为 0、任何使分组分化的改动都显得"抬高差距"**。
    比较两组数据源的差距前，必须先对两边都跑一遍本函数。
    """
    return len(set(group_of.values())) < 2


def min_gap_floor(group_sizes: Dict[str, int], n_sel: int) -> float:
    """给定组规模与选中条数，:func:`gqc_gap` 的**结构性下限**（整数分配下的最小偏差）。

    名额是整数，所以即使算法完美，份额也凑不齐：某组占 1/60=1.67%、本次共选 5 条，
    它只能拿 0 或 1 条（0% 或 20%），最小偏差就是 1.67%。这个下限**与算法无关**，
    只由"组规模 × 选中数"决定。

    实测偏差 = 下限 + 算法超额。不做这个拆分，就会把"组规模变偏斜抬高下限"
    误读成"算法更不公平"（见 ``experiments/diagnose_gqc_gap.py``）。

    组只有三档、``n_sel`` 通常 4–29，枚举 ``O(n_sel²)`` 足够，不需要 LP。
    """
    total = sum(group_sizes.values())
    if total <= 0 or n_sel <= 0:
        return 0.0
    ids = [g for g in group_sizes if group_sizes[g] > 0]
    if not ids:
        return 0.0
    share = {g: group_sizes[g] / total for g in ids}
    best = float("inf")
    ranges = [range(0, min(group_sizes[g], n_sel) + 1) for g in ids]
    for alloc in itertools.product(*ranges):
        if sum(alloc) != n_sel:
            continue
        dev = max((abs(alloc[i] / n_sel - share[ids[i]]) for i in range(len(ids))),
                  default=0.0)
        if dev < best - 1e-12:
            best = dev
    return 0.0 if best == float("inf") else float(best)
