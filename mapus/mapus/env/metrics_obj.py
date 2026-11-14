# -*- coding: utf-8 -*-
"""MAPUS 的两个评价量：覆盖效用 φ(A) 与路径满意分 PSS。

* 覆盖效用直接复用 agentsense 的 ``Coverage``（式 12：φ = αE + (1-α)log Q），
  该口径已用论文表 4 的 Base Obj. 锚点标定，跨论文可比。
* PSS 按式 (2)：S(r_u) = sim(p_u, h(r_u)) − μ·risk(r_u)，sim 为两个归一化
  分布的余弦相似度，h(r_u) 为式 (3) 的经验土地利用分布。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agentsense.config import EnvConfig
from agentsense.env.model import Instance, Solution, manhattan
from agentsense.env.objective import Coverage

from .urban import N_CAT, UrbanGrid


# --------------------------------------------------------------------------- PSS
def _normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class PathSatisfaction:
    """式 (2)：S(r_u) = a·cos(p_u, h(r_u)) − μ·risk(r_u)。

    两个常数：
    * ``prior``（κ）—— h(r_u) 的 Laplace 平滑强度。参与者路线只有 5–15 步，
      目标卖验分布在有 6 个类别时往往退化成一热向量（cos 随之被噪声支配），
      用一个向全局用地占比收缩的先验是标准做法；
    * ``scale``（a）与 ``mu`` —— 论文未给出 PSS 的绝对尺度与 μ，二者由
      docs/calibration.md 的锚点拟合得到（与 E(D) 的 e_scale 标定同一思路）。
    """

    def __init__(self, urban: UrbanGrid, mu: float = 0.5, prior: float = 6.0, scale: float = 1.0):
        self.urban = urban
        self.mu = float(mu)
        self.prior = float(prior)
        self.scale = float(scale)

    def land_use(self, cells: Sequence[Tuple[int, int]]) -> np.ndarray:
        return self.urban.route_land_use(cells, prior=self.prior)

    def risk(self, cells: Sequence[Tuple[int, int]]) -> float:
        return self.urban.route_risk(cells)

    def score(self, pref: np.ndarray, cells: Sequence[Tuple[int, int]]) -> float:
        h = self.land_use(cells)
        return self.scale * cosine(_normalize(pref.astype(np.float64)), h) - self.mu * self.risk(cells)

    def explain(self, pref: np.ndarray, cells) -> dict:
        h = self.land_use(cells)
        return {"similarity": round(cosine(_normalize(pref.astype(np.float64)), h), 4),
                "risk": round(self.risk(cells), 4),
                "land_use": {i: round(float(h[i]), 4) for i in range(N_CAT)}}


def path_cells(path: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in path]


def pss_of(urban: UrbanGrid, cfg) -> "PathSatisfaction":
    """按配置构造 PSS（统一的 μ / prior / scale 口径）。"""
    return PathSatisfaction(urban, mu=cfg.mu, prior=cfg.pss_prior, scale=cfg.pss_scale)


def route_pss(instance: Instance, path, pref, urban: UrbanGrid, mu: float) -> float:
    return PathSatisfaction(urban, mu).score(pref, path_cells(path))


# --------------------------------------------------------------------------- 覆盖
def coverage_utility(instance: Instance, solution: Solution, env_cfg: EnvConfig) -> float:
    cov = Coverage(instance)
    for path in solution.values():
        cov.add_path(path)
    return cov.objective(env_cfg)


def route_coverage_utility(instance: Instance, path, env_cfg: EnvConfig) -> float:
    """单个参与者自己的数据张量 A_{r_u} 上的覆盖效用（算法 1 的 φ_route）。"""
    cov = Coverage(instance)
    cov.add_path(path)
    return cov.objective(env_cfg)


# --------------------------------------------------------------------------- 式 (7)
def route_overlap(cells_a: Iterable[Tuple[int, int]], cells_b: Iterable[Tuple[int, int]]) -> float:
    """Overlap(r_u, r_v) = Σ min / Σ max，在网格单元集合上计算。"""
    a, b = set(cells_a), set(cells_b)
    if not a and not b:
        return 0.0
    inter, union = len(a & b), len(a | b)
    return inter / union if union else 0.0


def pairwise_overlap(solution: Solution) -> Dict[Tuple[str, str], float]:
    ids = list(solution.keys())
    out = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            out[(ids[i], ids[j])] = route_overlap(
                path_cells(solution[ids[i]]), path_cells(solution[ids[j]]))
    return out


# --------------------------------------------------------------------------- 汇总
def evaluate_mapus(instance: Instance, solution: Solution, urban: UrbanGrid,
                   prefs: Dict[str, np.ndarray], env_cfg: EnvConfig,
                   mu: float = 0.5, prior: float = 6.0, scale: float = 1.0) -> dict:
    """论文表 2/3 的 Coverage 与 PSS，外加重叠/熵/被覆盖单元数（表 4 用）。"""
    cov = Coverage(instance)
    for path in solution.values():
        cov.add_path(path)
    sat = PathSatisfaction(urban, mu=mu, prior=prior, scale=scale)
    pss_values = [sat.score(prefs[wid], path_cells(path))
                  for wid, path in solution.items() if wid in prefs]
    overlaps = list(pairwise_overlap(solution).values())
    return {
        "coverage": cov.objective(env_cfg),
        "pss": float(np.mean(pss_values)) if pss_values else 0.0,
        "pss_std": float(np.std(pss_values)) if pss_values else 0.0,
        "cost": float(sum(len(p) for p in solution.values()) * env_cfg.reward_per_step),
        "n_selected": len(solution),
        "entropy": cov.entropy(env_cfg.e_mode, env_cfg.e_weights, env_cfg.e_scale),
        "count": cov.n_distinct(),
        "avg_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "max_overlap": float(np.max(overlaps)) if overlaps else 0.0,
        "q": cov.q(env_cfg.q_mode),
    }
