# -*- coding: utf-8 -*-
"""对照方法（方案 §4.1 E1.3 的方法清单里，A′ 之外的部分）。

| 本文件函数 | 对应方案里的对照 | 说明 |
|---|---|---|
| ``random_feasible`` | random | 随机可行解，多次取最好 |
| ``iterative_greedy`` | 迭代贪心 | 随机重启的贪心，取最好 |
| ``sa_greedy`` | SA 贪心（东华 CAFCTA 的做法） | 在选中集合上做退火 |
| ``capacity_greedy`` | CFI 型能力比例（东华） | **忽略 θ**，按能力排序取人 → E3.1 的靶子 |
| ``unconstrained_exact`` | 无约束最优（图 3 参照线） | θ=0 的 ILP 最优 |

注意 ``capacity_greedy`` 是**故意**违反满意度下界的：它模仿东华「分配数 ∝ 能力」的
公平观，用于 E3.1/E3.2 展示「CFI 型公平会把低能力者结构性挤出」。
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..model import Candidate, SelectionProblem
from .exact import exact
from .greedy import greedy


# --------------------------------------------------------------------- random
def random_feasible(problem: SelectionProblem, rng: Optional[random.Random] = None,
                    trials: int = 300) -> Tuple[List[Candidate], dict]:
    rng = rng or random.Random(0)
    elems = problem.eligible
    best: List[Candidate] = []
    best_cov = -1
    for _ in range(trials):
        order = list(elems)
        rng.shuffle(order)
        sel: List[Candidate] = []
        used: set = set()
        for e in order:
            if e.wid in used:
                continue
            trial = sel + [e]
            if problem.feasible(trial):
                sel = trial
                used.add(e.wid)
        cov = problem.coverage(sel)
        if cov > best_cov:
            best, best_cov = sel, cov
    return best, {"solver": "random", "trials": trials}


# ------------------------------------------------------------- 迭代贪心
def iterative_greedy(problem: SelectionProblem, rng: Optional[random.Random] = None,
                     restarts: int = 12) -> Tuple[List[Candidate], dict]:
    """随机首元素 + 贪心补齐，多次重启取最好（对应"迭代贪心"基线）。"""
    rng = rng or random.Random(0)
    elems = problem.eligible
    best, best_cov = [], -1
    for k in range(restarts):
        forced: List[Candidate] = []
        if k > 0 and elems:
            forced = [rng.choice(elems)]
            if not problem.feasible(forced):
                forced = []
        sel, _ = greedy(problem, rng=rng)
        if forced:
            used = {forced[0].wid}
            sel = [forced[0]] + [c for c in sel if c.wid not in used]
            if not problem.feasible(sel):
                continue
        cov = problem.coverage(sel)
        if cov > best_cov:
            best, best_cov = sel, cov
    return best, {"solver": "iterative_greedy", "restarts": restarts}


# ------------------------------------------------------------- SA 贪心
def sa_greedy(problem: SelectionProblem, rng: Optional[random.Random] = None,
              iters: int = 6000, t0: float = 2.0, t_end: float = 0.02,
              init: Optional[Sequence[Candidate]] = None) -> Tuple[List[Candidate], dict]:
    """模拟退火（SA 贪心基线）：在选中集合上做 add / remove / swap。"""
    rng = rng or random.Random(0)
    elems = problem.eligible
    if not elems:
        return [], {"solver": "sa", "iters": 0}
    cur = list(init) if init else greedy(problem)[0]
    cur_cov = problem.coverage(cur)
    best, best_cov = list(cur), cur_cov
    alpha = (t_end / t0) ** (1.0 / max(1, iters))

    for it in range(iters):
        temp = t0 * (alpha ** it)
        move = rng.random()
        trial = list(cur)
        if move < 0.4 or not trial:
            trial.append(rng.choice(elems))
        elif move < 0.7:
            trial.pop(rng.randrange(len(trial)))
        else:
            i = rng.randrange(len(trial))
            trial[i] = rng.choice(elems)
        if not problem.feasible(trial):
            continue
        cov = problem.coverage(trial)
        delta = cov - cur_cov
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            cur, cur_cov = trial, cov
            if cov > best_cov:
                best, best_cov = list(trial), cov
    return best, {"solver": "sa", "iters": iters}


# ----------------------------------------------------- CFI 型能力比例（忽略 θ）
def capacity_greedy(problem: SelectionProblem,
                    rng: Optional[random.Random] = None) -> Tuple[List[Candidate], dict]:
    """东华 CFI 型：能力 ∝ 候选集规模，按能力从高到低取人，**不检查 θ**。

    用 ``Problem.candidates``（θ 之前的全池）来排名与取人，因此会选中
    S(r_u) < θ_u 的路线——这正是 E3.1 要展示的结构性挤出。
    """
    rng = rng or random.Random(0)
    by_w: Dict[str, List[Candidate]] = {}
    for c in problem.candidates:
        by_w.setdefault(c.wid, []).append(c)
    # 能力代理：可行候选路线条数（低能力者的可行集天然更小）
    ranked = sorted(by_w.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    sel: List[Candidate] = []
    for wid, cands in ranked:
        best = max(cands, key=lambda c: (len(c.cells), c.sat))
        trial = sel + [best]
        if problem.cost(trial) <= problem.budget + 1e-9:
            sel = trial
    # 用剩余预算给每人补一条次优路线（CFI 的"能力容量"语义：强者可占多个名额）
    for wid, cands in ranked:
        for c in sorted(cands, key=lambda c: -len(c.cells)):
            trial = sel + [c]
            if problem.cost(trial) <= problem.budget + 1e-9:
                sel = trial
                break
    return sel, {"solver": "capacity(CFI-like)",
                 "violates_theta": sum(1 for c in sel if not problem.is_eligible(c))}


# ------------------------------------------------------- 无约束最优（图 3 参照）
def unconstrained_exact(problem: SelectionProblem, time_limit: float = 30.0):
    """θ=0 的 ILP 最优：不受满意度下界限制，只受预算限制。"""
    cfg0 = dataclasses.replace(problem.cfg, theta=0.0, per_worker_theta=None)
    relaxed = SelectionProblem(instance=problem.instance,
                               candidates=list(problem.candidates), cfg=cfg0)
    sel, info = exact(relaxed, time_limit=time_limit)
    info["solver"] = "exact(theta=0)"
    return sel, info


# ------------------------------------------- MSCTA_Greedy（东华式，无个性化、多名额）
def mscta_greedy(problem: SelectionProblem, cap_per_worker: int = 2,
                 rng: Optional[random.Random] = None) -> Tuple[List[Candidate], dict]:
    """MSCTA_Greedy 式基线：**按边际覆盖**贪心，但允许同一参与者占多个名额。

    ⚠️ **假设声明**（`docs/baselines_assumptions.md`）：东华原文未获得，这里按其方案描述
    复现"最大传感覆盖 + 每人容量上限 `cnt_j < cap_j`"的最简忠实版本：

    * 不看 θ（东华无个性化满意度下界）；
    * 不限每人一条（允许 `cap_per_worker` 条，这是与 A′ 的核心结构差异）；
    * 目标只有覆盖，没有公平项。

    与 ``capacity_greedy`` 的区别：那个按**能力**排序取人（CFI 语义），这个按**边际覆盖**
    排序取人（覆盖语义）。两者共用"忽略 θ"的前提。
    """
    rng = rng or random.Random(0)
    elems = list(problem.candidates)          # 故意用全池（θ 之前）
    kind = problem.cfg.budget_kind
    used: Dict[str, int] = {}
    sel: List[Candidate] = []
    covered: set = set()
    cost = 0.0
    while True:
        best, best_key = None, None
        for e in elems:
            if used.get(e.wid, 0) >= cap_per_worker:
                continue
            unit = 1.0 if kind == "count" else float(e.steps)
            if cost + unit > problem.budget + 1e-9:
                continue
            gain = len(e.cells - covered)
            if gain <= 0:
                continue
            if best is None or gain > best_key:
                best, best_key = e, gain
        if best is None:
            break
        sel.append(best)
        used[best.wid] = used.get(best.wid, 0) + 1
        covered |= best.cells
        cost += 1.0 if kind == "count" else float(best.steps)
    return sel, {"solver": "mscta_greedy(cap=%d)" % cap_per_worker,
                 "ignores_theta": True,
                 "max_slots_per_worker": max(used.values()) if used else 0}


# ------------------------------------------- SA + CFI 惩罚（东华式公平目标）
def sa_caftca(problem: SelectionProblem, rng: Optional[random.Random] = None,
              iters: int = 4000, lam: float = 0.5, t0: float = 2.0,
              t_end: float = 0.02) -> Tuple[List[Candidate], dict]:
    """CAFCTA 式：**覆盖 − λ·CFI 偏差** 上的退火（东华"贪心 + 模拟退火"的对应物）。

    目标：`f(Ω) = coverage(Ω) − λ · N_cells · cfi_gap(Ω)`，其中
    `cfi_gap` 是"名额份额与能力份额的最大偏差"。同样**不看 θ**、允许每人多个名额。
    初值取 ``capacity_greedy``（东华式贪心初始化）。
    """
    rng = rng or random.Random(0)
    elems = list(problem.candidates)
    by_w: Dict[str, List[Candidate]] = {}
    for c in elems:
        by_w.setdefault(c.wid, []).append(c)
    caps = {w: float(len(v)) for w, v in by_w.items()}
    cap_sum = sum(caps.values()) or 1.0
    n_cells = max(1, len(problem.covered(elems)))

    def score(sel: Sequence[Candidate]) -> float:
        cnt: Dict[str, int] = {}
        for c in sel:
            cnt[c.wid] = cnt.get(c.wid, 0) + 1
        total = max(1, len(sel))
        gap = max((abs(cnt.get(w, 0) / total - caps[w] / cap_sum) for w in by_w), default=0.0)
        return problem.coverage(sel) - lam * n_cells * gap

    def cost_of(sel: Sequence[Candidate]) -> float:
        return (float(len(sel)) if problem.cfg.budget_kind == "count"
                else float(sum(c.steps for c in sel)))

    cur, _ = capacity_greedy(problem)
    if cost_of(cur) > problem.budget:      # 保险：容量初始化也可能越界
        cur = []
    cur_s, best, best_s = score(cur), list(cur), score(cur)
    alpha = (t_end / t0) ** (1.0 / max(1, iters))

    for it in range(iters):
        temp = t0 * (alpha ** it)
        trial = list(cur)
        r = rng.random()
        if r < 0.4 or not trial:
            trial.append(rng.choice(elems))
        elif r < 0.7:
            trial.pop(rng.randrange(len(trial)))
        else:
            trial[rng.randrange(len(trial))] = rng.choice(elems)
        if cost_of(trial) > problem.budget + 1e-9:
            continue
        s = score(trial)
        if s >= cur_s or rng.random() < math.exp((s - cur_s) / max(temp, 1e-9)):
            cur, cur_s = trial, s
            if s > best_s:
                best, best_s = list(trial), s
    return best, {"solver": "sa_caftca", "iters": iters, "lam": lam,
                  "ignores_theta": True,
                  "theta_violations": sum(1 for c in best if not problem.is_eligible(c))}
