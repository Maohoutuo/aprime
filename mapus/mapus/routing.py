# -*- coding: utf-8 -*-
"""路线层：TSPTW 基线路径、可行性校验、把 LLM 输出投影回可行空间。

论文算法 1 的第一行是 ``R0 ← TSPTW_SOLVER(Σ, T)``：先用经典规划器拿到一条
**保证可行**的初始路线，再由 LLM 用剩余步数插入/调整路点。本文件实现这个经典
部分（最短路径 + 贪心路点插入 + 停留补齐），以及算法 1 第 6 行的 ``VALID(·)``。
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agentsense.config import EnvConfig
from agentsense.env.model import (Instance, Worker, build_path, manhattan,
                                  pad_to_window, path_to_solution, shortest_path,
                                  steps_for)
from agentsense.env.objective import Coverage

from .config import MAPUSConfig
from .env.metrics_obj import PathSatisfaction, path_cells, pss_of, route_coverage_utility
from .env.urban import UrbanGrid

Step = List[int]


# --------------------------------------------------------------------------- 可行性
def validate_route(worker: Worker, path: Sequence[Sequence[int]], nx: int, ny: int,
                   blocked: Optional[Iterable[Tuple[int, int]]] = None,
                   cells_per_step: Optional[int] = None) -> Tuple[bool, List[str]]:
    """算法 1 第 6 行的 VALID(R_i, Σ, T)：起点/终点/时序/速度/边界/封锁。

    ``cells_per_step`` 缺省时取 ``worker.speed``（唯一真源），避免调用方传错。
    """
    speed = int(cells_per_step if cells_per_step is not None else worker.speed)
    v: List[str] = []
    if not path:
        return False, ["empty"]
    if len(path) > worker.window:
        v.append(f"window_overflow:{len(path)}>{worker.window}")
    for i, s in enumerate(path):
        if len(s) != 3:
            v.append(f"bad_step:{i}")
            continue
        x, y, t = int(s[0]), int(s[1]), int(s[2])
        if not (0 <= x < nx and 0 <= y < ny):
            v.append(f"out_of_grid:{i}")
        if t != worker.t_start + i:
            v.append(f"bad_time:{i}")
        if blocked and (x, y) in blocked:
            v.append(f"blocked:{i}")
    if (int(path[0][0]), int(path[0][1])) != worker.origin:
        v.append("bad_origin")
    if (int(path[-1][0]), int(path[-1][1])) != worker.dest:
        v.append("bad_dest")
    for i in range(1, len(path)):
        d = manhattan((int(path[i - 1][0]), int(path[i - 1][1])),
                      (int(path[i][0]), int(path[i][1])))
        if d > speed:
            v.append(f"illegal_move:{i}")
    return (not v), v


# --------------------------------------------------------------------------- 组装
def cells_to_path(worker: Worker, cells: Sequence[Tuple[int, int]],
                  pref=None, urban: UrbanGrid = None, pref_pad: bool = False) -> List[Step]:
    if pref is not None and urban is not None and pref_pad:
        return path_to_solution(worker, pad_pref(cells, worker, pref, urban))
    return path_to_solution(worker, pad_to_window(cells, worker))


def pad_pref(cells: Sequence[Tuple[int, int]], worker: Worker, pref: np.ndarray,
             urban: UrbanGrid) -> List[Tuple[int, int]]:
    """偏好感知的停留补齐。

    论文允许"staying in place"，且 h(r_u) 按**时间步**统计土地利用，所以把剩余
    步数停留在与本人偏好最契合的格子上，是提升 PSS 的直接手段（同时会被覆盖项
    惩罚，形成 λ 权衡）。
    """
    cells = list(cells)
    residual = worker.window - len(cells)
    if residual <= 0 or not cells:
        return cells
    scores = [float(pref[urban.category[x, y]]) if (0 <= x < urban.nx and 0 <= y < urban.ny) else -1.0
              for x, y in cells]
    extra = [0] * len(cells)
    for _ in range(residual):
        i = max(range(len(cells)), key=lambda k: (scores[k], -extra[k]))
        extra[i] += 1
    out: List[Tuple[int, int]] = []
    for c, e in zip(cells, extra):
        out.extend([c] * (1 + e))
    return out[: worker.window]


def route_cells(worker: Worker, waypoints: Sequence[Tuple[int, int]],
                blocked: Optional[Iterable[Tuple[int, int]]] = None) -> Optional[List[Tuple[int, int]]]:
    cells = build_path(worker, list(waypoints))
    if cells is None:
        return None
    if blocked and any(c in set(blocked) for c in cells):
        return None
    return cells


# --------------------------------------------------------------------------- 效用
def route_utility(instance: Instance, path: Sequence[Sequence[int]], pref: np.ndarray,
                  urban: UrbanGrid, cfg: MAPUSConfig, env_cfg: EnvConfig) -> float:
    """算法 1 的 UTILITY(R; T, P, G, λ) = λ·φ_route + (1−λ)·S(r)。"""
    phi = route_coverage_utility(instance, list(path), env_cfg)
    pss = pss_of(urban, cfg).score(pref, path_cells(path))
    return cfg.lam * phi + (1.0 - cfg.lam) * pss


class _Incremental:
    """在增量覆盖张量上评估"插入一个路点"的收益，避免反复重建 Coverage。"""

    def __init__(self, instance: Instance, env_cfg: EnvConfig):
        self.instance = instance
        self.env_cfg = env_cfg
        self.cov = Coverage(instance)

    def objective_with(self, path: Sequence[Sequence[int]]) -> float:
        self.cov.add_path(path)
        val = self.cov.objective(self.env_cfg)
        self.cov.add_path(path, sign=-1)
        return val


def _reachable_cells(instance: Instance, w: Worker,
                     blocked: Optional[Iterable[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    cache = instance.meta.setdefault("_reach_cache", {})
    key = w.wid
    if key in cache:
        return cache[key]
    blocked = set(blocked or ())
    cells = []
    for x in range(instance.nx):
        for y in range(instance.ny):
            if (x, y) in blocked:
                continue
            if steps_for(w.origin, (x, y), w.speed) + steps_for((x, y), w.dest, w.speed) <= w.window - 1:
                cells.append((x, y))
    cache[key] = cells
    return cells


# --------------------------------------------------------------------------- TSPTW 基线
def baseline_route(instance: Instance, worker: Worker, pref: np.ndarray, urban: UrbanGrid,
                   cfg: MAPUSConfig, env_cfg: EnvConfig, rng: random.Random,
                   max_insert: int = 6, candidates: int = 24,
                   blocked: Optional[Iterable[Tuple[int, int]]] = None,
                   lam: float = 1.0) -> List[Step]:
    """构造可行初始路线 R0（算法 1 第 1 行）。

    1. 起点→终点最短路（保证 O–D 一致与时间窗可行）；
    2. 用剩余步数贪心插入路点，最大化 λ·φ_route + (1−λ)·S(r)；
       默认 λ=1：**经典规划器只优化覆盖**，偏好由后面的 LLM 精修引入
       （与论文"classical planner 保证可行、LLM 负责个性化"的分工一致）；
    3. 余下步数用"停留"补齐（停留在沿途格子上，等价于更长时间采样）。
    """
    sat = pss_of(urban, cfg)
    base_cells = route_cells(worker, [], blocked)
    if base_cells is None:
        return []

    def _util(path) -> float:
        phi = route_coverage_utility(instance, list(path), env_cfg)
        pss = sat.score(pref, path_cells(path))
        return lam * phi + (1.0 - lam) * pss

    waypoints: List[Tuple[int, int]] = []
    cells = base_cells
    best_cells = cells
    best_util = _util(path_to_solution(worker, pad_to_window(cells, worker)))
    reach = _reachable_cells(instance, worker, blocked)
    for _ in range(max_insert):
        trial_pool = reach if len(reach) <= candidates * 4 else rng.sample(reach, candidates * 4)
        pool = rng.sample(trial_pool, min(candidates, len(trial_pool)))
        improved = False
        for cand in pool:
            new_cells = route_cells(worker, waypoints + [cand], blocked)
            if new_cells is None:
                continue
            util = _util(path_to_solution(worker, pad_to_window(new_cells, worker)))
            if util > best_util + 1e-9:
                best_util, best_cells, best_wp, improved = util, new_cells, list(waypoints) + [cand], True
        if improved:
            waypoints = best_wp
            cells = best_cells
        else:
            break
    return cells_to_path(worker, best_cells)


# --------------------------------------------------------------------------- LLM 输出修复
def repair_from_llm(worker: Worker, raw: Sequence[Sequence[int]], nx: int, ny: int,
                    blocked: Optional[Iterable[Tuple[int, int]]] = None,
                    pref=None, urban: UrbanGrid = None) -> Optional[List[Step]]:
    """把 LLM 给出的 (x, y, t) 序列投影成可行路线。

    做法：保留其经过的格子顺序（去掉连续重复），用最短路径把路点连起来；
    若超时窗则从中段逐步删减路点；最后用停留补齐。返回 None 表示无法修复。
    """
    blocked = set(blocked or ())
    seq: List[Tuple[int, int]] = []
    for s in raw or []:
        try:
            x, y = int(s[0]), int(s[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= x < nx and 0 <= y < ny) or (x, y) in blocked:
            continue
        if not seq or seq[-1] != (x, y):
            seq.append((x, y))
    if not seq:
        return None
    # 端点纠正：强制从 origin 出发、在 dest 结束
    body = [c for c in seq if c != worker.origin and c != worker.dest]
    waypoints = list(body)
    while True:
        cells = build_path(worker, waypoints)
        if cells is not None and not (blocked & set(cells)):
            return cells_to_path(worker, cells, pref=pref, urban=urban, pref_pad=True)
        if not waypoints:
            break
        waypoints.pop(len(waypoints) // 2)  # 从中间删，保住两端意图
    return None
