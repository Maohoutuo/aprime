# -*- coding: utf-8 -*-
"""A′ 主算法骨架：贪心 + 局部搜索（方案 W1「B：主算法骨架」）。

定理 2 的算法部分只需要「可行性保持 + 单调覆盖增益」，因此骨架刻意保持朴素：

1. **贪心**：每轮在「合法候选集」里取边际覆盖增益最大者（可选按步数做密度比
   ``Δf/|r|``，对应预算为步数时的标准做法）；
2. **局部搜索**：1-swap（换掉一条已选路线）与 drop-refill，只在严格提升覆盖且
   仍可行时接受。

θ 只影响 ``problem.eligible``，算法本身对 θ 无感知——这正是方案 §2.4「LLM/参数
无法破坏保证」的同款结构：约束进可行域，算法只做贪心。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..model import Candidate, SelectionProblem


def greedy(problem: SelectionProblem, ratio: Optional[bool] = None,
           rng: Optional[random.Random] = None) -> Tuple[List[Candidate], dict]:
    """按边际增益（或增益/步数密度）贪心。返回 (选中集合, 统计)。"""
    kind = problem.cfg.budget_kind
    if ratio is None:
        ratio = kind != "count"
    elems = problem.eligible
    used_w: set = set()
    sel: List[Candidate] = []
    covered: set = set()
    cost = 0.0
    trace: List[float] = []

    while True:
        best, best_key = None, None
        for e in elems:
            if e.wid in used_w:
                continue
            c = 1.0 if kind == "count" else float(e.steps)
            if cost + c > problem.budget + 1e-9:
                continue
            gain = len(e.cells - covered)
            if gain <= 0:
                continue
            key = (gain / c) if (ratio and c > 0) else float(gain)
            if best is None or key > best_key:
                best, best_key = e, key
        if best is None:
            break
        sel.append(best)
        used_w.add(best.wid)
        covered |= best.cells
        cost += 1.0 if kind == "count" else float(best.steps)
        trace.append(float(len(covered)))

    return sel, {"solver": "greedy", "ratio": bool(ratio), "iteration_gains": trace}


def _refill(problem: SelectionProblem, base: Sequence[Candidate],
            elems: Sequence[Candidate]) -> List[Candidate]:
    """在 base 上按纯增益贪心补齐（用于 drop-refill 移动）。"""
    trial = list(base)
    covered = set()
    for c in trial:
        covered |= c.cells
    used = {c.wid for c in trial}
    cost = problem.cost(trial)
    while True:
        best, best_key = None, None
        for e in elems:
            if e.wid in used:
                continue
            unit = 1.0 if problem.cfg.budget_kind == "count" else float(e.steps)
            if cost + unit > problem.budget + 1e-9:
                continue
            gain = len(e.cells - covered)
            if gain <= 0:
                continue
            if best is None or gain > best_key:
                best, best_key = e, gain
        if best is None:
            break
        trial.append(best)
        used.add(best.wid)
        covered |= best.cells
        cost += 1.0 if problem.cfg.budget_kind == "count" else float(best.steps)
    return trial


def local_search(problem: SelectionProblem, sel: Sequence[Candidate],
                 max_rounds: int = 200) -> Tuple[List[Candidate], dict]:
    """1-swap + drop-refill，仅接受严格改进且可行的移动。"""
    cur = list(sel)
    cur_cov = problem.coverage(cur)
    moves = 0
    elems = problem.eligible

    for _ in range(max_rounds):
        improved = False
        # (a) 1-swap
        for i, old in enumerate(cur):
            rest = [c for j, c in enumerate(cur) if j != i]
            for new in elems:
                if new in cur or new.wid in {c.wid for c in rest}:
                    continue
                trial = rest + [new]
                if not problem.feasible(trial):
                    continue
                cov = problem.coverage(trial)
                if cov > cur_cov:
                    cur, cur_cov, improved, moves = trial, cov, True, moves + 1
                    break
            if improved:
                break
        if improved:
            continue
        # (b) drop-refill
        for i in range(len(cur)):
            rest = [c for j, c in enumerate(cur) if j != i]
            trial = _refill(problem, rest, elems)
            cov = problem.coverage(trial)
            if cov > cur_cov and problem.feasible(trial):
                cur, cur_cov, improved, moves = trial, cov, True, moves + 1
                break
        if not improved:
            break

    return cur, {"solver": "local_search", "moves": moves}


def solve(problem: SelectionProblem, with_ls: bool = True,
          rng: Optional[random.Random] = None) -> Tuple[List[Candidate], dict]:
    """A′ 主算法：贪心 + 局部搜索。"""
    sel, info = greedy(problem, rng=rng)
    if with_ls and sel:
        sel, ls_info = local_search(problem, sel)
        info = {**info, "local_search": ls_info}
    info["solver"] = "aprime(greedy+ls)" if with_ls else "aprime(greedy)"
    return sel, info
