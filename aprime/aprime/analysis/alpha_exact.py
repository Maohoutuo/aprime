# -*- coding: utf-8 -*-
"""**精确**计算次模比 α（小实例枚举，E2.1 的基准）。

定义（与 `submodularity.py` 的采样估计、`docs/theory_alpha_min.md` §2 一致）：

    α_f = min_{S, T}  [ f(S∪T) − f(S) ] / Σ_{e∈T\\S} [ f(S∪{e}) − f(S) ]

枚举全部 ``(S, T)`` 组合（把元素当位集，``3^n`` 种指派），用覆盖函数的位运算算 f。
``n ≤ 11`` 时 ``3^11 ≈ 1.8e5``，配合 ``f`` 的查表（``2^n``）可以秒级完成。

为什么需要它：采样估计（`submodularity_ratio`）是极值统计，噪声大；而 Lemma 2 的
``α_min`` 又只是**下界**。只有拿到真 α，才能判断"界松"到底是松在哪一环：

* ``α_min`` 显著小于 ``α_true`` ⇒ 损耗在 **引理 2**（成对重叠这步太粗）；
* ``α_true`` 已经够大（≈0.8）但 ``α_true²(1−e^{−α_true})`` 仍远低于实测 greedy/OPT
  ⇒ 损耗在 **近似比公式本身**（那个 α² 的多余因子）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..model import Candidate, SelectionProblem


def _coverage_table(elems: List[Candidate]):
    """返回 f 的查表：f_mask[mask] = |∪_{i∈mask} cells(e_i)|。"""
    idx: Dict[tuple, int] = {}
    masks: List[int] = []
    for e in elems:
        m = 0
        for cell in e.cells:
            j = idx.get(cell)
            if j is None:
                j = len(idx)
                idx[cell] = j
            m |= 1 << j
        masks.append(m)
    n = len(elems)
    cover = [0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        cover[mask] = cover[mask ^ lsb] | masks[lsb.bit_length() - 1]
    return [c.bit_count() for c in cover], len(idx)


def exact_submodularity_ratio(problem: SelectionProblem,
                              max_elements: int = 11) -> Optional[dict]:
    """精确 α。元素数超过 ``max_elements`` 时返回 None（枚举会爆）。"""
    elems = problem.eligible
    n = len(elems)
    if n < 2:
        return {"alpha": 1.0, "n": n, "n_cells": 0, "pairs": 0, "worst": None}
    if n > max_elements:
        return None

    f, n_cells = _coverage_table(elems)
    full = (1 << n) - 1
    best, worst = 1.0, None
    pairs = 0

    for T in range(1, full + 1):
        comp = full ^ T          # S 只能取 T 的补集的子集
        S = comp
        while True:
            num = f[S | T] - f[S]
            if num > 0:
                den = 0
                tt = T
                while tt:
                    lsb = tt & -tt
                    den += f[S | lsb] - f[S]
                    tt ^= lsb
                if den > 0:
                    pairs += 1
                    r = num / den
                    if r < best:
                        best, worst = r, (S, T)
            if S == 0:
                break
            S = (S - 1) & comp

    return {"alpha": float(best), "n": n, "n_cells": n_cells, "pairs": pairs,
            "worst": worst}
