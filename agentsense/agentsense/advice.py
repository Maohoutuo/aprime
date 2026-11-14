# -*- coding: utf-8 -*-
"""把"下一步该做什么"算成具体、可执行的建议，喂给 Solver Agent。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import DEFAULT_ENV, EnvConfig
from .env.model import Instance, Solution
from .env.objective import Coverage


def _contributions(inst: Instance, solution: Solution, cfg: EnvConfig) -> Dict[str, float]:
    total = Coverage(inst)
    for p in solution.values():
        total.add_path(p)
    base = total.objective(cfg)
    out = {}
    for wid in solution:
        trial = Coverage(inst)
        for k, p in solution.items():
            if k != wid:
                trial.add_path(p)
        out[wid] = base - trial.objective(cfg)
    return out


def suggest_actions(inst: Instance, solution: Solution, cfg: EnvConfig = DEFAULT_ENV,
                    k: int = 3) -> List[str]:
    cov = Coverage(inst)
    for p in solution.values():
        cov.add_path(p)
    cost = sum(len(p) for p in solution.values())
    slack = inst.budget - cost
    contrib = _contributions(inst, solution, cfg) if solution else {}
    advice: List[Tuple[float, str]] = []

    if slack > 0:
        free = [w for w in inst.workers if w.wid not in solution and w.window <= slack]
        free.sort(key=lambda w: -w.window)
        if free:
            w = free[0]
            advice.append((1e9, f"recruit {w.wid} (window {w.window}, {w.origin}->{w.dest}, "
                                f"fits slack {slack:.0f}) and route it through low-coverage cells"))
    free_all = [w for w in inst.workers if w.wid not in solution]

    if solution:
        worst = min(contrib, key=lambda w: contrib[w]) if contrib else None
        if worst is not None:
            coord_sum = cov.counts.sum(axis=2)
            xs, ys = np.unravel_index(np.argsort(coord_sum, axis=None)[:5], coord_sum.shape)
            targets = [f"({int(x)},{int(y)})" for x, y in zip(xs, ys)]
            advice.append((1e8, f"reroute {worst} (contribution {contrib[worst]:+.3f}) through "
                                f"uncovered cells such as {', '.join(targets[:3])}"))
            # 预算重构：换一个窗口更大的工作者，即使当前 slack 不够
            freed = inst.worker(worst).window
            better = [w for w in free_all if w.window <= freed + slack]
            better.sort(key=lambda w: -w.window)
            if better and better[0].window > (free[0].window if free else 0):
                advice.append((1e7, f"drop {worst} (window {freed}) and recruit {better[0].wid} "
                                    f"(window {better[0].window}): cost stays "
                                    f"<= {inst.budget:.0f}"))

    per_slot = cov.counts.sum(axis=(0, 1))
    if per_slot.size and (per_slot == 0).any():
        empty = int(np.argmin(per_slot))
        advice.append((1e6, f"no data collected in time slot {empty}; assign a worker there"))

    advice.sort(key=lambda x: -x[0])
    return [a for _, a in advice[:k]]
