# -*- coding: utf-8 -*-
"""α(θ) 的**实测**估计器（支撑 E2.3「α(θ) 实测 vs 理论下界」）。

定义（与 docs/theory_alpha_min.md 一致）：

    α(θ) = min_{S,T}  [ f(S ∪ T) − f(S) ] / Σ_{e∈T\\S} [ f(S ∪ {e}) − f(S) ]

其中 f 是覆盖函数（|∪ 栅格单元|），S、T 取自 θ 下的合法候选集，且要求分母 > 0。

* α = 1 ⇔ 该实例上 f 对 T 内元素"可加"（无收益递减以外的损失）；
* α < 1 ⇔ 存在显著冗余（路线重叠），贪心的近似比随之下调；
* α 只依赖 (实例, θ)，不依赖算法——所以它是"问题的属性"，可以做进图 3/图 6。

实测用随机采样（S 取随机合法子集，T 取 ≤ t_max 个不重复参与者的元素），
取所有样本的最小值作为经验下界；``samples`` 越大越接近真 α。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

from ..model import Candidate, SelectionProblem


def _f(problem: SelectionProblem, sel: Sequence[Candidate]) -> int:
    return len(problem.covered(sel))


def submodularity_ratio(problem: SelectionProblem, samples: int = 300,
                        rng: Optional[random.Random] = None,
                        t_max: int = 4) -> Dict[str, float]:
    """采样估计 α(θ)。返回最小比值与若干分位数（便于画置信带）。"""
    rng = rng or random.Random(0)
    elems = problem.eligible
    if len(elems) < 2:
        return {"alpha": 1.0, "samples": 0, "p05": 1.0, "p50": 1.0, "n_elements": len(elems)}

    by_w: Dict[str, List[int]] = {}
    for i, e in enumerate(elems):
        by_w.setdefault(e.wid, []).append(i)

    ratios: List[float] = []
    for _ in range(samples):
        ws = list(by_w)
        rng.shuffle(ws)
        # S：随机一半参与者各取一条
        s_ws = ws[: max(0, int(rng.random() * len(ws) * 0.5))]
        S = [elems[rng.choice(by_w[w])] for w in s_ws]
        # T\S：剩下的参与者里随机挑 2..t_max 个
        rest = [w for w in ws if w not in set(s_ws)]
        if len(rest) < 2:
            continue
        k = rng.randint(2, min(t_max, len(rest)))
        T_only = [elems[rng.choice(by_w[w])] for w in rng.sample(rest, k)]

        base = problem.covered(S)
        f_s = len(base)
        f_st = len(base | problem.covered(T_only))
        denom = 0
        for e in T_only:
            denom += len(e.cells - base)
        if denom <= 0:
            continue
        num = f_st - f_s
        if num <= 0:
            continue
        ratios.append(num / denom)

    if not ratios:
        return {"alpha": 1.0, "samples": 0, "p05": 1.0, "p50": 1.0, "n_elements": len(elems)}

    ratios.sort()
    return {
        "alpha": float(min(1.0, ratios[0])),
        "samples": len(ratios),
        "p05": float(ratios[int(0.05 * (len(ratios) - 1))]),
        "p50": float(ratios[len(ratios) // 2]),
        "n_elements": len(elems),
    }


def apx_bound(alpha: float) -> float:
    """方案 §2.2 的近似比表达式 APX(θ) = α² · (1 − e^(−α))，α ∈ (0, 1]。"""
    a = max(0.0, min(1.0, float(alpha)))
    return a * a * (1.0 - math.exp(-a)) if a > 0 else 0.0


def overlap_stats(problem: SelectionProblem, samples: int = 400,
                  rng: Optional[random.Random] = None) -> Dict[str, float]:
    """合法候选路线两两重叠的分布。**两种口径，用途不同，别混用**：

    * ``mean`` / ``p95`` / ``max``：**Jaccard**（交/并）。直观、适合看趋势（θ 走向、φ 走向），
      但**不能**拿它算 Lemma 2 的界。
    * ``mean_oc`` / ``p95_oc`` / ``max_oc``：**重叠系数**（交/min(|A|,|B|)）。
      引理 2 的推导要求 `|A_e ∩ A_e'| ≤ ρ·min(|A_e|,|A_e'|)`，只有这个口径满足；
      用 Jaccard 会算出**超过真值**的 α_min（Day-3 实测抓到：α_min=0.333 > α_true=0.250），
      也就是说界会失效。
    * ``se`` 是 Jaccard 均值的标准误，用来判断趋势是否只是噪声。
    """
    rng = rng or random.Random(0)
    elems = problem.eligible
    empty = {"mean": 0.0, "p95": 0.0, "max": 0.0, "se": 0.0,
             "mean_oc": 0.0, "p95_oc": 0.0, "max_oc": 0.0, "n": 0}
    if len(elems) < 2:
        return empty
    jac: List[float] = []
    oc: List[float] = []
    for _ in range(samples):
        a, b = rng.sample(range(len(elems)), 2)
        ca, cb = set(elems[a].cells), set(elems[b].cells)
        if not ca and not cb:
            continue
        union = len(ca | cb)
        inter = len(ca & cb)
        jac.append(inter / union if union else 1.0)
        lo = min(len(ca), len(cb))
        oc.append(inter / lo if lo else 0.0)
    if not jac:
        return empty
    jac.sort()
    oc.sort()
    n = len(jac)
    mean = sum(jac) / n
    se = (sum((v - mean) ** 2 for v in jac) / max(1, n - 1)) ** 0.5 / (n ** 0.5)
    q = lambda vals: float(vals[int(0.95 * (len(vals) - 1))])  # noqa: E731
    return {"mean": float(mean), "p95": q(jac), "max": float(jac[-1]), "se": float(se),
            "mean_oc": float(sum(oc) / n), "p95_oc": q(oc), "max_oc": float(oc[-1]),
            "n": n}


def overlap_floor(problem: SelectionProblem, samples: int = 400,
                  rng: Optional[random.Random] = None) -> float:
    """两两 **Jaccard** 的均值（只用于看趋势，不用于算界，见 ``overlap_stats``）。"""
    return float(overlap_stats(problem, samples, rng)["mean"])


def alpha_min_bound(problem: SelectionProblem, rho: Optional[float] = None,
                    samples: int = 400, rng: Optional[random.Random] = None,
                    rho_stat: str = "max_oc", m_mode: str = "eligible") -> Dict[str, float]:
    """docs/theory_alpha_min.md 的 Lemma 2：``α_min(θ) = 1 / (1 + (m−1)·ρ(θ))``。

    ``rho_stat`` 默认 ``max_oc``：必须用**重叠系数**（交/最小尺寸），不能用 Jaccard；
    必须取最坏，不能取均值。

    ⚠️ ``m_mode`` 是这条引理的**生死点**（Day-3 实测）：

    * ``"eligible"``（默认，**唯一合法**）：`m = |E(θ)|`。因为 α 的定义里 `T` 可以是
      **任意**子集、不受预算限制，所以 `max|T\\S| = |E(θ)|`。取别的值会让
      `α_min > α_true`（实测 70/104 个实例违例），界直接失效。
      代价：真实实例 `|E|=240`、`ρ_oc≈0.6` ⇒ `α_min ≈ 0.007`，**空界**。
    * ``"budget"``：`m = ⌊B/min|r|⌋`。数字好看但**不成立**，只保留用于复现"为什么不能用"。

    结论：两头堵 ⇒ 成对重叠这条路走不通，见 docs/theory_alpha_min.md §3.2。
    """
    rng = rng or random.Random(0)
    elems = problem.eligible
    if len(elems) < 2:
        return {"rho": 0.0, "rho_stat": rho_stat, "m_mode": m_mode,
                "m": 1.0, "w_min": 0.0, "alpha_min": 1.0}
    if rho is None:
        st = overlap_stats(problem, samples=samples, rng=rng)
        rho = st[rho_stat] if rho_stat in st else st["mean_oc"]
    else:
        rho = float(rho)
    w_min = max(1, min(len(c.cells) for c in elems))
    if m_mode == "budget":
        m_cap = (int(problem.budget) if problem.cfg.budget_kind == "count"
                 else int(problem.budget // w_min))
        m = max(2, min(len(elems), m_cap))
    else:
        m = len(elems)
    return {"rho": round(rho, 4), "rho_stat": rho_stat, "m_mode": m_mode, "m": float(m),
            "w_min": float(w_min),
            "alpha_min": round(1.0 / (1.0 + (m - 1) * rho), 4)}
