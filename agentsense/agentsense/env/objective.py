# -*- coding: utf-8 -*-
"""数据覆盖度与目标函数 J(D) = α·E(D) + (1−α)·log2 Q(D)。

论文正文只给出了公式形式（式 2），E(D) 的"层次化熵"定义在文献 [18]
（Ji et al., UbiComp'16）中，原文未复述。这里实现一个等价结构：
分别统计时间层、空间层、时空层的访问分布，计算各层 Shannon 熵（bit），
再按 EnvConfig.e_mode 聚合。口径差异通过 e_mode/q_mode 暴露，便于标定。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..config import EnvConfig
from .model import Instance, Solution


def _entropy(counts: np.ndarray) -> float:#_entropy函数
    """Shannon 熵（bit），忽略零元素。"""
    c = counts.astype(np.float64).ravel()
    c = c[c > 0]
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c / total
    return float(-(p * np.log2(p)).sum())


def _norm_entropy(counts: np.ndarray, n_bins: int) -> float:#_norm_entropy函数
    """归一化到 [0,1] 的熵：H / log2(#bins)。"""
    if n_bins <= 1:
        return 0.0
    return _entropy(counts) / math.log2(n_bins)


def _rel_entropy(counts: np.ndarray, n_bins: int) -> float:#_rel_entropy函数
    """相对归一化熵：H / log2(min(#bins, 观测到的样本数))。"""
    c = counts.astype(np.float64).ravel()
    n_obs = int((c > 0).sum())
    denom_bins = max(2, min(n_bins, n_obs))
    return _entropy(counts) / math.log2(denom_bins)


class Coverage:#Coverage类
    """覆盖计数张量：counts[x, y, t] = 访问次数。"""

    __slots__ = ("instance", "counts")

    def __init__(self, instance: Instance):#__init__方法
        self.instance = instance#生成一个实例
        self.counts = np.zeros((instance.nx, instance.ny, instance.n_slots), dtype=np.int32)#生成一个计数张量

    def add_path(self, path, sign: int = 1) -> None:#add_path方法
        for step in path:
            x, y, t = int(step[0]), int(step[1]), int(step[2])#生成一个x, y, t
            if 0 <= x < self.instance.nx and 0 <= y < self.instance.ny and 0 <= t < self.instance.n_slots:
                self.counts[x, y, t] += sign#更新计数

    def copy(self) -> "Coverage":
        c = Coverage(self.instance)#生成一个覆盖计数张量
        c.counts[...] = self.counts
        return c#返回一个覆盖计数张量

    # ---- 统计量
    def q(self, q_mode: str = "distinct") -> float:#q方法
        inst = self.instance
        w = inst.meta.get("cell_weights", None)#生成一个权重
        if w is None:
            w = np.ones((inst.nx, inst.ny))#生成一个权重
        else:
            w = np.asarray(w, dtype=float)#生成一个权重
        if q_mode == "distinct":
            return float(sum(w[x, y] for x in range(inst.nx) for y in range(inst.ny)#生成一个权重
                             for t in range(inst.n_slots) if self.counts[x, y, t] > 0))
        total = 0.0#生成一个总和
        for x in range(inst.nx):
            for y in range(inst.ny):
                for t in range(inst.n_slots):
                    total += self.counts[x, y, t] * w[x, y]#更新总和
        return float(total)

    def entropy(self, e_mode: str = "relative", weights=(1 / 3, 1 / 3, 1 / 3),
                e_scale: float = 4.2) -> float:#entropy方法
        inst = self.instance
        w = inst.meta.get("cell_weights", None)#生成一个权重
        if w is None:
            w = np.ones((inst.nx, inst.ny))#生成一个权重
        else:
            w = np.asarray(w, dtype=float)
        # 加权计数张量
        weighted = self.counts.astype(np.float64) * w[:, :, None]#生成一个加权计数张量
        h_t_raw = weighted.sum(axis=(0, 1))#生成一个时间层熵
        h_s_raw = weighted.sum(axis=2)#生成一个空间层熵
        h_st_raw = weighted#生成一个时空层熵
        n_bins_t = inst.n_slots#生成一个时间层bin数
        n_bins_s = inst.nx * inst.ny#生成一个空间层bin数
        n_bins_st = inst.nx * inst.ny * inst.n_slots#生成一个时空层bin数
        if e_mode == "relative":
            h_t = _rel_entropy(h_t_raw, n_bins_t)#生成一个时间层相对熵
            h_s = _rel_entropy(h_s_raw, n_bins_s)#生成一个空间层相对熵
            h_st = _rel_entropy(h_st_raw, n_bins_st)#生成一个时空层相对熵
            return e_scale * (h_t + h_s + h_st) / 3.0#返回一个相对熵
        if e_mode == "norm":
            h_t = _norm_entropy(h_t_raw, n_bins_t)#生成一个时间层归一化熵
            h_s = _norm_entropy(h_s_raw, n_bins_s)#生成一个空间层归一化熵
            h_st = _norm_entropy(h_st_raw, n_bins_st)#生成一个时空层归一化熵
            return e_scale * (h_t + h_s + h_st) / 3.0#返回一个归一化熵
        h_time = _entropy(h_t_raw)#生成一个时间层熵
        h_space = _entropy(h_s_raw)#生成一个空间层熵
        h_st_e = _entropy(h_st_raw)#生成一个时空层熵
        if e_mode == "sum":
            return h_time + h_space + h_st_e#返回一个熵
        if e_mode == "weighted":
            return weights[0] * h_time + weights[1] * h_space + weights[2] * h_st_e#返回一个熵
        return (h_time + h_space + h_st_e) / 3.0#返回一个熵

    def objective(self, cfg: EnvConfig) -> float:#objective方法
        inst = self.instance
        q = self.q(cfg.q_mode)#生成一个q
        if q <= 0:
            return 0.0
        alpha = inst.meta.get("alpha_override", cfg.alpha)#生成一个alpha
        e = self.entropy(cfg.e_mode, cfg.e_weights, cfg.e_scale)#生成一个熵
        return alpha * e + (1.0 - alpha) * math.log2(q)#返回一个目标函数

    def n_distinct(self) -> int:
        return int((self.counts > 0).sum())#返回一个不同的任务数量


def coverage_of(instance: Instance, solution: Solution) -> Coverage:
    cov = Coverage(instance)#生成一个覆盖计数张量       
    for path in solution.values():
        cov.add_path(path)#添加一个路径
    return cov


def compute_objective(instance: Instance, solution: Solution, cfg: EnvConfig) -> float:
    return coverage_of(instance, solution).objective(cfg)#返回一个目标函数


def compute_cost(instance: Instance, solution: Solution, cfg: EnvConfig) -> float:
    return float(sum(len(p) for p in solution.values())) * cfg.reward_per_step#返回一个成本


def evaluate(instance: Instance, solution: Solution, cfg: EnvConfig) -> dict:
    cov = coverage_of(instance, solution)#生成一个覆盖计数张量      
    q = cov.q(cfg.q_mode)
    return {
        "objective": cov.objective(cfg),
        "cost": compute_cost(instance, solution, cfg),
        "q": q,
        "distinct_tasks": cov.n_distinct(),
        "entropy": cov.entropy(cfg.e_mode, cfg.e_weights, cfg.e_scale),
        "n_workers": len(solution),
    }
