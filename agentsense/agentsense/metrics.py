# -*- coding: utf-8 -*-
"""论文 4.1.2 的四个指标：SR / AIR / ANI / ACS。

* SR  (Success Rate)          20 次试验中，在 ≤10 轮迭代内相对基线取得"有效提升"的比例
* AIR (Average Improvement Rate) ((J' − J_base) / J_base) × 100，仅统计成功实例
* ANI (Average Number of iterations) 首次取得有效提升所需的平均迭代数
* ACS (Average Cost Savings)  cost_base − cost_final（绝对量，负值表示成本上升）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .config import MAX_ITERATIONS


@dataclass
class RunRecord:
    """单次试验（一个实例 + 一条方法链）的轨迹。"""

    base_obj: float
    base_cost: float
    history: List[dict] = field(default_factory=list)   # 每轮 {"obj":..., "cost":..., "feasible":...}
    final_override: Optional[dict] = None               # 实际返回解（可能优于最后一轮）

    def first_valid_improvement(self, max_iter: int = MAX_ITERATIONS) -> Optional[int]:
        for i, h in enumerate(self.history[:max_iter]):
            ok = h["valid"] if "valid" in h else (
                h.get("feasible", True) and h["obj"] > self.base_obj + 1e-12)
            if ok:
                return i + 1
        return None

    @property
    def final(self) -> dict:
        if self.final_override is not None:
            return self.final_override
        return self.history[-1] if self.history else {"obj": self.base_obj, "cost": self.base_cost, "feasible": True}


def summarize(runs: Sequence[RunRecord], max_iter: int = MAX_ITERATIONS) -> dict:
    n = len(runs)
    if n == 0:
        return {"SR": 0.0, "AIR": 0.0, "ANI": 0.0, "ACS": 0.0, "n": 0}
    succ, airs, anis, acss = 0, [], [], []
    for r in runs:
        it = r.first_valid_improvement(max_iter)
        if it is not None:
            succ += 1
            airs.append((r.final["obj"] - r.base_obj) / r.base_obj * 100.0)
            anis.append(it)
            acss.append(r.base_cost - r.final["cost"])
    return {
        "SR": succ / n * 100.0,
        "AIR": sum(airs) / len(airs) if airs else 0.0,
        "ANI": sum(anis) / len(anis) if anis else 0.0,
        "ACS": sum(acss) / len(acss) if acss else 0.0,
        "n": n,
    }


def format_row(name: str, m: dict) -> str:
    return (f"{name:12s} SR={m['SR']:5.1f}%  AIR={m['AIR']:+7.3f}%  "
            f"ANI={m['ANI']:4.1f}  ACS={m['ACS']:+6.2f}  (n={m['n']})")
