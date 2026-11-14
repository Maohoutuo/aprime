# -*- coding: utf-8 -*-
"""AgentSense 的五个工具（论文附录 B.2 的 Solver/Eval 工具集）。

所有工具都返回可 JSON 序列化的 dict，供 Solver Agent / Eval Agent 调用。
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import numpy as np

from .config import EnvConfig, DEFAULT_ENV
from .data.instances import make_instance
from .env.model import Instance, Solution, validate_solution
from .env.objective import Coverage, evaluate, coverage_of


def _to_instance(inst_or_name) -> Instance:
    if isinstance(inst_or_name, Instance):
        return inst_or_name
    if isinstance(inst_or_name, dict):
        return make_instance(inst_or_name["setting"], inst_or_name["index"])
    raise TypeError(f"无法转换为 Instance: {type(inst_or_name)}")


# --------------------------------------------------------------------------- 1. validate_worker_paths
def validate_worker_paths(instance, solution: Solution) -> Dict[str, Any]:
    inst = _to_instance(instance)
    vr = validate_solution(inst, solution)
    return {"overall_feasible": vr.feasible, "cost": vr.cost, "violations": vr.violations,
            "validation_results": {wid: {"feasible": True} for wid in solution}}


# --------------------------------------------------------------------------- 2. compute_objective
def compute_objective(instance, solution: Solution, cfg: Optional[EnvConfig] = None) -> float:
    inst = _to_instance(instance)
    return float(evaluate(inst, solution, cfg or DEFAULT_ENV)["objective"])


# --------------------------------------------------------------------------- 3. compute_cost
def compute_cost(instance, solution: Solution, cfg: Optional[EnvConfig] = None) -> float:
    inst = _to_instance(instance)
    return float(evaluate(inst, solution, cfg or DEFAULT_ENV)["cost"])


# --------------------------------------------------------------------------- 4. check_disturbance_handling
def check_disturbance_handling(instance, solution: Solution, disturbance: Dict[str, Any]) -> Dict[str, Any]:
    """检查解是否处理了给定扰动（7 类扰动的语义见 disturbances.py）。"""
    inst = _to_instance(instance)
    d_type = disturbance.get("type", "unknown")
    violations = []
    handled = True
    if d_type == "worker_unavailable":
        wid = disturbance.get("worker_id")
        if wid and wid in solution:
            handled = False
            violations.append(f"unavailable_worker_still_used:{wid}")
    elif d_type == "area_blocked":
        cells = disturbance.get("blocked_cells", [])
        for path in solution.values():
            for step in path:
                if (int(step[0]), int(step[1])) in cells:
                    handled = False
                    violations.append("path_enters_blocked_area")
                    break
    elif d_type == "budget_cut":
        new_budget = disturbance.get("new_budget", inst.budget)
        cost = compute_cost(inst, solution)
        if cost > new_budget + 1e-9:
            handled = False
            violations.append(f"cost {cost:.1f} exceeds cut budget {new_budget}")
    elif d_type == "time_window_shift":
        # 重新校验起终点/时间戳
        vr = validate_solution(inst, solution)
        if not vr.feasible:
            handled = False
            violations.extend(vr.violations)
    elif d_type == "bad_weather":
        # 降低覆盖要求：仍要求可行
        vr = validate_solution(inst, solution)
        if not vr.feasible:
            handled = False
            violations.extend(vr.violations)
    elif d_type in ("traffic_congestion", "task_priority_change"):
        vr = validate_solution(inst, solution)
        if not vr.feasible:
            handled = False
            violations.extend(vr.violations)
    else:
        vr = validate_solution(inst, solution)
        if not vr.feasible:
            handled = False
            violations.extend(vr.violations)
    return {"handled": handled, "violations": violations, "disturbance": d_type}


# --------------------------------------------------------------------------- 5. visual_analysis
def visual_analysis(instance, baseline: Solution, refined: Solution, out_path: Optional[str] = None,
                    cfg: Optional[EnvConfig] = None) -> Dict[str, Any]:
    """生成覆盖热力图对比（PNG）+ 文本摘要。"""
    inst = _to_instance(instance)
    cfg = cfg or DEFAULT_ENV
    cov_b = coverage_of(inst, baseline)
    cov_r = coverage_of(inst, refined)
    delta = cov_r.counts.astype(int) - cov_b.counts.astype(int)
    gain = int((delta > 0).sum())
    loss = int((delta < 0).sum())
    unchanged = int((delta == 0).sum())

    summary = {
        "baseline_objective": float(cov_b.objective(cfg)),
        "refined_objective": float(cov_r.objective(cfg)),
        "improvement": float(cov_r.objective(cfg) - cov_b.objective(cfg)),
        "gain_cells": gain,
        "loss_cells": loss,
        "unchanged_cells": unchanged,
        "png": None,
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, arr, title in zip(axes,
                                   [cov_b.counts.sum(axis=2), cov_r.counts.sum(axis=2), delta.sum(axis=2)],
                                   ["Baseline", "Refined", "Delta"]):
            im = ax.imshow(arr.T, origin="lower", cmap="viridis" if title != "Delta" else "RdBu")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Instance {inst.setting_name}#{inst.index}  "
                     f"J: {summary['baseline_objective']:.3f} -> {summary['refined_objective']:.3f}")
        if out_path is None:
            out_path = os.path.join("results", "visual", f"{inst.setting_name}_{inst.index}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        summary["png"] = out_path
    except Exception as e:
        summary["png_error"] = str(e)
    return summary
