# -*- coding: utf-8 -*-
"""七类扰动（论文表 5 / 图 5、6）的定义与实例变换。

每种扰动都返回一个**新的 Instance**（不修改原实例），供 Eval Agent 校验、
Solver Agent 重新规划使用。AIR 的计算方式：基线解在扰动后的实例上重新评测
（J_base），Agent 改进后的解再评测（J_refined），AIR = (J_refined-J_base)/J_base。

注意：论文未给出扰动的精确参数，这里取合理默认值并在 docs/assumptions.md 中记录。
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np

from .config import SETTINGS
from .data.instances import make_instance
from .env.model import Instance, Worker


DISTURBANCE_TYPES = [
    "bad_weather",
    "area_blocked",
    "worker_unavailable",
    "traffic_congestion",
    "time_window_shift",
    "budget_cut",
    "task_priority_change",
]


def _copy_instance(inst: Instance, **overrides) -> Instance:
    new = Instance(
        setting_name=inst.setting_name,
        index=inst.index,
        grid=inst.grid,
        n_slots=inst.n_slots,
        budget=overrides.get("budget", inst.budget),
        workers=overrides.get("workers", [Worker(w.wid, w.origin, w.dest, w.t_start, w.t_end, w.speed) for w in inst.workers]),
        seed=inst.seed,
        meta=dict(inst.meta),
    )
    new.meta.update(overrides.get("meta", {}))
    return new


def apply_disturbance(inst: Instance, dtype: str, rng: random.Random) -> Instance:
    nx, ny = inst.nx, inst.ny
    if dtype == "bad_weather":
        # 30% 的格子变为"低价值"（权重 0.3），模拟天气恶化
        w = np.ones((nx, ny), dtype=float)
        mask = np.array([[rng.random() < 0.30 for _ in range(ny)] for _ in range(nx)])
        w[mask] = 0.3
        return _copy_instance(inst, meta={"cell_weights": w})
    if dtype == "area_blocked":
        # 10% 的格子被封锁（不可进入）
        blocked = set()
        n_block = max(1, int(0.10 * nx * ny))
        cells = [(x, y) for x in range(nx) for y in range(ny)]
        rng.shuffle(cells)
        blocked = set(cells[:n_block])
        return _copy_instance(inst, meta={"blocked_cells": blocked})
    if dtype == "worker_unavailable":
        # 2-3 名工作者不可用
        n_remove = min(3, max(1, len(inst.workers) // 8))
        keep = list(inst.workers)
        rng.shuffle(keep)
        keep = keep[n_remove:]
        return _copy_instance(inst, workers=keep)
    if dtype == "traffic_congestion":
        # 时间窗缩短 30%（移动变慢）
        new_workers = []
        for w in inst.workers:
            new_win = max(3, int(w.window * 0.7))
            new_workers.append(Worker(w.wid, w.origin, w.dest, w.t_start, w.t_start + new_win - 1, w.speed))
        return _copy_instance(inst, workers=new_workers)
    if dtype == "time_window_shift":
        # 所有时间窗延后 2 个 slot（部分工作者超出时域）
        new_workers = []
        for w in inst.workers:
            ts = min(w.t_start + 2, inst.n_slots - w.window)
            new_workers.append(Worker(w.wid, w.origin, w.dest, ts, ts + w.window - 1, w.speed))
        return _copy_instance(inst, workers=new_workers)
    if dtype == "budget_cut":
        return _copy_instance(inst, budget=inst.budget * 0.75)
    if dtype == "task_priority_change":
        # α 从 0.5 变为 0.3（更重视数据量）
        return _copy_instance(inst, meta={"alpha_override": 0.3})
    raise ValueError(f"unknown disturbance: {dtype}")


def disturbance_instruction(dtype: str, disturbance: Dict[str, Any]) -> str:
    """把扰动转成 Solver Agent 可读的 instruction（附录 B.1 的形式）。"""
    templates = {
        "bad_weather": "Bad weather has reduced the sensing value of {n} grid cells by 70%. Re-route workers to high-value cells and report the impact on overall data quality.",
        "area_blocked": "{n} grid cells are now blocked and cannot be entered. Re-route affected worker paths around the blocked area.",
        "worker_unavailable": "{n} workers are no longer available. Reassign their tasks to remaining workers within budget.",
        "traffic_congestion": "Traffic congestion has reduced worker speed; time windows shrink by 30%. Re-plan paths to fit the shorter windows.",
        "time_window_shift": "All worker time windows shift forward by 2 slots. Adjust paths to the new availability.",
        "budget_cut": "Budget is cut by 25%. Reduce recruitment while preserving coverage as much as possible.",
        "task_priority_change": "Task priority changed: data volume (Q) now weighs more than balance (E). Re-optimize accordingly.",
    }
    t = templates.get(dtype, "A disturbance occurred: {dtype}. Re-adapt the solution.")
    if dtype == "bad_weather":
        n = int((disturbance.get("meta", {}).get("cell_weights", np.ones((1, 1))) < 1).sum()) if isinstance(disturbance.get("meta"), dict) else 0
    elif dtype == "area_blocked":
        n = len(disturbance.get("meta", {}).get("blocked_cells", []))
    elif dtype == "worker_unavailable":
        n = disturbance.get("removed", 0)
    else:
        n = 0
    return t.format(dtype=dtype, n=n)
