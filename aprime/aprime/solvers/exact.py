# -*- coding: utf-8 -*-
"""精确求解器（ILP）与暴力枚举（小规模验算）。

用途有两个：
* **B 的「精确求解器」**：E2.1「小规模枚举验近似比」需要真 OPT 作分母；
* **图 3 的上界**：θ=0 时的 ILP 最优就是"无约束最优"，用来画 θ-权衡曲线的参照线。

模型（与 ``SelectionProblem`` 一一对应，变量全部 0/1）：

    max  Σ_c y_c
    s.t. x_e ≤ y_c                 ∀e, ∀c ∈ cells(e)
         Σ_{e: u(e)=u} x_e ≤ 1     ∀u        （每人至多一条）
         Σ_e cost_e · x_e ≤ B                  （步数或人数预算）
         x, y ∈ {0,1}

未来若要在 A′ 主算法里替换 milp，可直接换 HiGHS/Gurobi 后端，模型不变。
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..model import Candidate, SelectionProblem


def exact(problem: SelectionProblem, time_limit: float = 30.0,
          mip_rel_gap: float = 1e-4) -> Tuple[List[Candidate], dict]:
    """在**当前 θ 的可行域**上求精确最优。"""
    elems = problem.eligible
    info: Dict[str, object] = {"solver": "milp", "n_elements": len(elems)}
    if not elems:
        info["status"] = "infeasible:no-eligible-route"
        return [], info

    cell_idx = problem.cell_index(elems)
    n_e, n_c = len(elems), len(cell_idx)
    n_var = n_e + n_c

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    r = 0

    # 约束：y_c ≤ Σ_{e ∋ c} x_e（选了覆盖 c 的路线才允许点亮 c；
    # 目标最大化 Σ y_c，因此 y_c 恰好等于"c 被覆盖"的指示变量）
    for cell, j in cell_idx.items():
        rows.append(r)
        cols.append(n_e + j)
        vals.append(1.0)
        for i, e in enumerate(elems):
            if cell in e.cells:
                rows.append(r)
                cols.append(i)
                vals.append(-1.0)
        lb.append(-np.inf)
        ub.append(0.0)
        r += 1

    # (2) 每人至多一条
    groups: Dict[str, List[int]] = {}
    for i, e in enumerate(elems):
        groups.setdefault(e.wid, []).append(i)
    for idxs in groups.values():
        for i in idxs:
            rows.append(r)
            cols.append(i)
            vals.append(1.0)
        lb.append(-np.inf)
        ub.append(1.0)
        r += 1

    # (3) 预算
    for i, e in enumerate(elems):
        rows.append(r)
        cols.append(i)
        vals.append(1.0 if problem.cfg.budget_kind == "count" else float(e.steps))
    lb.append(-np.inf)
    ub.append(float(problem.budget))
    r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    c = np.concatenate([np.zeros(n_e), -np.ones(n_c)])

    res = milp(
        c=c,
        constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
        integrality=np.ones(n_var),
        bounds=Bounds(np.zeros(n_var), np.ones(n_var)),
        options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap),
                 "disp": False},
    )

    if res.x is None:
        info["status"] = f"failed:{res.message}"
        return [], info

    sel = [elems[i] for i in range(n_e) if res.x[i] > 0.5]
    info.update({
        "status": str(res.message),
        "obj": float(-res.fun),
        "coverage": float(len(problem.covered(sel))),
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "n_vars": int(n_var),
        "n_rows": int(r),
    })
    return sel, info


def brute_force(problem: SelectionProblem, max_elements: int = 18) -> Tuple[List[Candidate], dict]:
    """穷举（仅用于测试 / E2.1 的小实例复核）。元素数超过阈值直接拒绝。"""
    elems = problem.eligible
    if len(elems) > max_elements:
        raise ValueError(f"brute_force 只支持 ≤{max_elements} 个元素，当前 {len(elems)}")
    best: List[Candidate] = []
    best_val = -1
    for r in range(len(elems) + 1):
        for combo in combinations(range(len(elems)), r):
            sel = [elems[i] for i in combo]
            if not problem.feasible(sel):
                continue
            val = problem.coverage(sel)
            if val > best_val:
                best_val, best = val, sel
    return best, {"solver": "brute_force", "obj": float(max(best_val, 0)),
                  "n_elements": len(elems)}
