# -*- coding: utf-8 -*-
"""选择模型：路线级、每人至多一条、满意度下界作为硬约束。

对应方案 §2.1：

    max |∪_{u∈Ω} cells(r_u)|        （栅格单元覆盖率）
    s.t. S(r_u) ≥ θ_u               （个性化满意度下界，**硬约束**）
         每人至多一条路线
         Σ|r_u| ≤ B 或 |Ω| ≤ B     （预算，见 APrimeConfig.budget_kind）

与 MAPUS 的关键差别：MAPUS 把 PSS 放进目标函数做加权和（软目标），这里它是约束，
因此可行域会随 θ 增大而**碎片化**（定理 1 的现象），必须有可行性判定与退化路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import APrimeConfig

Step = Tuple[int, int, int]
Cell = Tuple[int, int, int]
XY = Tuple[int, int]


@dataclass(frozen=True)
class Candidate:
    """一条候选路线 = 参与者 u 的一个可选元素 e = (u, r)。

    ``cells`` 是**空间**栅格单元 (x, y) 的集合——覆盖函数 f 就在它上面定义
    （方案 §2.1「选中路线覆盖的栅格单元总数」）。若按 (x, y, t) 计，不同参与者的
    时间轴几乎不重合，覆盖会恒等于步数预算、问题退化成"填满预算"，
    θ-曲线也就没有意义了（Day-1 实测确认过这一点）。

    ``space_time`` 保留 (x, y, t) 三元组，仅用于审计与后续的时空熵分析。
    """

    wid: str
    path: Tuple[Step, ...]
    cells: FrozenSet[XY]
    sat: float
    tag: str = ""
    space_time: FrozenSet[Cell] = frozenset()

    @property
    def steps(self) -> int:
        return len(self.path)

    @property
    def route(self) -> List[Tuple[int, int]]:
        """(x, y) 序列，供 PSS 计算用（保留停留重复）。"""
        return [(x, y) for (x, y, _) in self.path]


@dataclass
class SelectionProblem:
    """一个实例 + 一份与 θ 无关的候选池 + 一个 θ。"""

    instance: object
    candidates: List[Candidate]
    cfg: APrimeConfig
    #: PSS 已在候选生成时算好（θ 无关），这里只存一个查表，便于报告
    meta: Dict[str, float] = None

    def __post_init__(self) -> None:
        self.by_worker: Dict[str, List[Candidate]] = {}
        for c in self.candidates:
            self.by_worker.setdefault(c.wid, []).append(c)
        if self.meta is None:
            self.meta = {}
        self.cfg.validate()
        self._rel_theta = self._resolve_relative_theta()

    def _resolve_relative_theta(self) -> Dict[str, float]:
        """``theta_mode="relative"`` 时把 θ 折成逐工作者的绝对阈值 ``θ·best_u``。

        必须在这里算：相对口径需要知道每人的最佳满意度，而那只有拿到候选池
        之后才知道。缓存成字典而不是改 ``cfg``，避免同一份 config 被多个问题对象
        复用时互相污染。
        """
        if self.cfg.theta_mode == "absolute":
            return {}
        out: Dict[str, float] = {}
        for wid, cands in self.by_worker.items():
            if not cands:
                continue
            # θ·best_u ≤ best_u（θ ≤ 1 时），故每个人的最优路线永远合法。
            # best_u 为负也成立（θ<1 时 θ·best_u 更接近 0，即更宽松）。
            out[wid] = self.cfg.theta * max(c.sat for c in cands)
        return out

    # ------------------------------------------------------------------ 可行性
    def theta_of(self, wid: str) -> float:
        if wid in self._rel_theta:
            return self._rel_theta[wid]
        return self.cfg.theta_of(wid)

    def is_eligible(self, c: Candidate) -> bool:
        return c.sat >= self.theta_of(c.wid) - 1e-12

    @property
    def eligible(self) -> List[Candidate]:
        return [c for c in self.candidates if self.is_eligible(c)]

    def eligible_workers(self) -> List[str]:
        return sorted({c.wid for c in self.eligible})

    def theta_thresholds(self) -> Dict[str, float]:
        """命题 4 的两个**闭式**阈值，限定在候选池非空的 `U⁺` 上。

        * ``theta_all = min_{u∈U⁺} max_{r∈R_u} S_u(r)``：`θ ≤ theta_all` ⇔ 每个人都还有得选；
        * ``theta_any = max_{u∈U⁺} max_{r∈R_u} S_u(r)``：`θ > theta_any` ⇔ 可行域为空。

        必须限定 `U⁺`：若某 `u` 的 `R_u = ∅`，`max_{r∈R_u} S = −∞` 会把 `theta_all` 拉成 `−∞`。
        两个阈值都与 `θ` 无关（只取决于候选池与每人阈值 `θ_u`），因此可对整条 θ 曲线画成竖线。

        若 ``theta_mode="relative"``，阈值已在「θ 的倍数」尺度上：此时
        ``theta_all == theta_any == 1.0``（每个人的最优路线对 θ=1 仍合法），
        即相对口径下可行域在 θ∈[0,1] 内必然非空。
        """
        if self.cfg.theta_mode == "relative":
            # 相对口径下 θ_u = θ·best_u；最优路线始终合法（best_u ≥ θ·best_u ⟺ θ≤1）
            return {"theta_all": 1.0, "theta_any": 1.0, "n_workers": len(self.by_worker)}
        best = {wid: max(c.sat for c in cands) for wid, cands in self.by_worker.items() if cands}
        if not best:
            return {"theta_all": float("inf"), "theta_any": float("-inf"), "n_workers": 0}
        return {
            "theta_all": float(min(best.values())),
            "theta_any": float(max(best.values())),
            "n_workers": len(best),
        }

    @property
    def budget(self) -> float:
        return self.cfg.budget_of(self.instance)

    def cost(self, sel: Iterable[Candidate]) -> float:
        if self.cfg.budget_kind == "count":
            return float(sum(1 for _ in sel))
        return float(sum(c.steps for c in sel))

    def feasible(self, sel: Sequence[Candidate]) -> bool:
        seen = set()
        for c in sel:
            if c.wid in seen:
                return False
            seen.add(c.wid)
            if not self.is_eligible(c):
                return False
        return self.cost(sel) <= self.budget + 1e-9

    # ------------------------------------------------------------------ 目标
    @staticmethod
    def covered(sel: Iterable[Candidate]) -> FrozenSet[XY]:
        out: set = set()
        for c in sel:
            out |= c.cells
        return frozenset(out)

    def coverage(self, sel: Iterable[Candidate]) -> int:
        return len(self.covered(sel))

    def marginal(self, sel: Sequence[Candidate], c: Candidate) -> int:
        return len(c.cells - self.covered(sel))

    def evaluate(self, sel: Sequence[Candidate], label: str = "") -> dict:
        cells = self.covered(sel)
        sats = [c.sat for c in sel]
        st: set = set()
        for c in sel:
            st |= c.space_time
        return {
            "label": label,
            "theta": float(self.cfg.theta),
            "budget_kind": self.cfg.budget_kind,
            "budget": round(self.budget, 3),
            "coverage": len(cells),
            "coverage_st": len(st),
            "n_selected": len(sel),
            "cost": round(self.cost(sel), 3),
            "budget_used": round(self.cost(sel) / self.budget, 4) if self.budget else 0.0,
            "pss_mean": round(float(np.mean(sats)), 4) if sats else 0.0,
            "pss_min": round(float(np.min(sats)), 4) if sats else 0.0,
            "feasible": self.feasible(sel),
            "n_eligible_workers": len(self.eligible_workers()),
            "n_workers": len(self.by_worker),
            "workers": [c.wid for c in sel],
        }

    # ------------------------------------------------------------------ 便于求解器
    def elements(self, eligible_only: bool = True) -> List[Candidate]:
        return self.eligible if eligible_only else list(self.candidates)

    def cell_index(self, elems: Sequence[Candidate]) -> Dict[XY, int]:
        idx: Dict[XY, int] = {}
        for c in elems:
            for cell in c.cells:
                if cell not in idx:
                    idx[cell] = len(idx)
        return idx

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"<SelectionProblem θ={self.cfg.theta} workers={len(self.by_worker)} "
                f"cand={len(self.candidates)} budget={self.budget}({self.cfg.budget_kind})>")


def make_problem(instance, candidates: Sequence[Candidate], cfg: APrimeConfig) -> SelectionProblem:
    return SelectionProblem(instance=instance, candidates=list(candidates), cfg=cfg)

