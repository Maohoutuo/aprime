# -*- coding: utf-8 -*-
"""确定性可行性修复（对应论文 B.2 的 validate_worker_paths + Solver 自校正）。

LLM 返回的解经常出现：终点不匹配、时间戳越界、超出预算、引用不可用工作者。
论文的做法是 Solver 调用 validate_worker_paths，发现违规后迭代校准。
这里把这层"确定性兜底"做成独立模块。

单 Agent 基线**不使用**本修复，以保持论文里"单 Agent 可行性差"的对照。
"""

from __future__ import annotations

from typing import List, Tuple

from .config import DEFAULT_ENV
from .env.model import Instance, Solution, validate_solution
from .env.objective import Coverage


def _obj(inst: Instance, sol: Solution) -> float:
    cov = Coverage(inst)
    for p in sol.values():
        cov.add_path(p)
    return cov.objective(DEFAULT_ENV)


def _minimal_fix_path(inst: Instance, w, path, blocked: set):
    """尽量保留模型给出的路线形状，只修必要处（不像 DP 重规划那样把编辑抹掉）。"""
    from .env.model import manhattan, shortest_path, steps_for

    cells = []
    for p in path:
        x, y = int(p[0]), int(p[1])
        if not (0 <= x < inst.nx and 0 <= y < inst.ny) or (x, y) in blocked:
            x, y = (cells[-1] if cells else w.origin)
        cells.append((x, y))
    if not cells:
        return None
    # 从起点开始，强制连续（超出速度上限的跳变退化为停留）
    cells[0] = w.origin
    for i in range(1, len(cells)):
        if manhattan(cells[i - 1], cells[i]) > w.speed:
            cells[i] = cells[i - 1]
    # 收尾：找到能按时抵达终点的最长前缀，再补最短路
    cut = None
    for i in range(len(cells) - 1, -1, -1):
        if steps_for(cells[i], w.dest, w.speed) <= w.window - 1 - i:
            cut = i
            break
    if cut is None:
        return None
    cells = cells[: cut + 1] + shortest_path(cells[cut], w.dest, w.speed)[1:]
    if len(cells) > w.window:
        return None
    # 用停留补齐到窗口长度
    pad = w.window - len(cells)
    if pad > 0:
        mid = max(1, len(cells) // 2)
        cells = cells[:mid] + [cells[mid - 1]] * pad + cells[mid:]
    return [[c[0], c[1], w.t_start + i] for i, c in enumerate(cells)]


def _is_valid_path(inst: Instance, w, path, blocked: set) -> bool:
    from .env.model import manhattan

    if len(path) != w.window:
        return False
    if (int(path[0][0]), int(path[0][1])) != w.origin:
        return False
    if (int(path[-1][0]), int(path[-1][1])) != w.dest:
        return False
    for i, p in enumerate(path):
        if int(p[2]) != w.t_start + i:
            return False
        x, y = int(p[0]), int(p[1])
        if not (0 <= x < inst.nx and 0 <= y < inst.ny) or (x, y) in blocked:
            return False
        if i and manhattan((int(path[i - 1][0]), int(path[i - 1][1])), (x, y)) > w.speed:
            return False
    return True


def _dp_replan(inst: Instance, wid: str, cov: Coverage):
    from .baselines.planners import _dp_path, _weights_from, path_to_solution

    w = inst.worker(wid)
    if w.min_steps > w.window - 1:
        return None
    cells = _dp_path(inst, w, _weights_from(inst, cov))
    if cells is None or len(cells) != w.window:
        return None
    return path_to_solution(w, cells)


def repair_solution(inst: Instance, solution: Solution) -> Tuple[Solution, List[str]]:
    """尽量把解修复到可行；返回 (修复后的解, 剩余违规列表)。"""
    if not solution:
        return {}, []
    blocked = inst.meta.get("blocked_cells", set())
    valid_ids = {w.wid for w in inst.workers}
    out: Solution = {}
    notes: List[str] = []

    for wid, path in solution.items():
        if wid not in valid_ids:
            notes.append(f"dropped_unknown_worker:{wid}")
            continue
        w = inst.worker(wid)
        if _is_valid_path(inst, w, path, blocked):
            out[wid] = [[int(p[0]), int(p[1]), int(p[2])] for p in path]
            continue
        fixed = _minimal_fix_path(inst, w, path, blocked)
        if fixed is not None and _is_valid_path(inst, w, fixed, blocked):
            out[wid] = fixed
            notes.append(f"minimal_fix:{wid}")
            continue
        notes.append(f"replanned:{wid}")

    for wid in list(solution.keys()):
        if wid in out or wid not in valid_ids:
            continue
        cov = Coverage(inst)
        for p in out.values():
            cov.add_path(p)
        new_path = _dp_replan(inst, wid, cov)
        if new_path is not None:
            out[wid] = new_path
        else:
            notes.append(f"unrecoverable:{wid}")

    while out and sum(len(p) for p in out.values()) > inst.budget + 1e-9:
        base_obj = _obj(inst, out)
        worst, worst_loss = None, float("inf")
        for wid in out:
            trial = {k: v for k, v in out.items() if k != wid}
            loss = base_obj - _obj(inst, trial)
            if loss < worst_loss:
                worst, worst_loss = wid, loss
        if worst is None:
            break
        del out[worst]
        notes.append(f"budget_trim:{worst}")

    vr = validate_solution(inst, out)
    return out, vr.violations
