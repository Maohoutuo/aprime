# -*- coding: utf-8 -*-
"""六个基线规划器（附录 A）。

RN / TVPG / TCPG / MSA / MSAGI 复现自文献 [41] 的思路，GraphDP 复现自
文献 [18]；两者原始代码均未公开，这里按其文字描述实现，并记录简化点：

* 路径在网格上按曼哈顿最短路构造，时间窗内的剩余步数用"停留"补齐；
* 预算 = 所有被招募路径的时间步数之和（报酬 1.0/步，论文 4.1.1）；
* "任务激励成本"用插入该路点带来的绕行步数近似（原文未给出精确定义）。

所有规划器签名一致：plan(instance, cfg, rng, **kw) -> Solution
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import EnvConfig
from ..env.model import (Instance, Solution, Worker, build_path, manhattan,
                         pad_to_window, path_to_solution, shortest_path, steps_for)
from ..env.objective import Coverage

Plan = Dict[str, List[Tuple[int, int]]]   # wid -> 路点列表


# --------------------------------------------------------------------------- 公共工具
def plan_to_solution(instance: Instance, plan: Plan) -> Solution:
    sol: Solution = {}
    for wid, wps in plan.items():
        w = instance.worker(wid)
        cells = build_path(w, wps)
        if cells is None:
            continue
        cells = pad_to_window(cells, w, prefer=wps[-1] if wps else None)
        sol[wid] = path_to_solution(w, cells)
    return sol


def plan_cost(instance: Instance, plan: Plan) -> float:
    return float(sum(instance.worker(wid).window for wid in plan))


def evaluate_plan(instance: Instance, plan: Plan, cfg: EnvConfig) -> Tuple[float, Coverage]:
    sol = plan_to_solution(instance, plan)
    cov = Coverage(instance)
    for path in sol.values():
        cov.add_path(path)
    return cov.objective(cfg), cov


def _reachable(w: Worker, cell: Tuple[int, int]) -> bool:
    return steps_for(w.origin, cell, w.speed) + steps_for(cell, w.dest, w.speed) <= w.window


def _reach_set(instance: Instance, w: Worker) -> List[Tuple[int, int]]:
    """工作者在时间窗内可达的细胞集合（按 speed 计算的曼哈顿椭圆），按实例缓存。"""
    cache = instance.meta.setdefault("_reach_cache", {})
    if w.wid in cache:
        return cache[w.wid]
    cells: List[Tuple[int, int]] = []
    for x in range(instance.nx):
        for y in range(instance.ny):
            if steps_for(w.origin, (x, y), w.speed) + steps_for((x, y), w.dest, w.speed) <= w.window - 1:
                cells.append((x, y))
    cache[w.wid] = cells
    return cells


def _candidate_cells(cov: Coverage, w: Worker, rng: random.Random, k: int = 15) -> List[Tuple[int, int]]:
    """在工作者可达区域内采样候选路点，偏向覆盖不足的区域。

    注意：早期版本在整张网格上均匀采样，在 32×32 的大规模档会导致
    绝大多数候选点不可达，令贪心类基线退化为随机基线（见 docs/calibration.md）。
    """
    inst = cov.instance
    reach = _reach_set(inst, w)
    if not reach:
        return []
    weights = []
    for c in reach:
        weights.append(1.0 / (1.0 + float(cov.counts[c[0], c[1], :].sum())))
    k = min(k, len(reach))
    picked = rng.choices(reach, weights=weights, k=k)
    # 去重但保持采样顺序
    seen = set()
    out = []
    for c in picked:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _try_insert(instance: Instance, cov: Coverage, w: Worker, wps: Sequence[Tuple[int, int]],
                cand: Tuple[int, int], cfg: EnvConfig) -> Tuple[float, List[Tuple[int, int]], int]:
    """返回 (目标函数增量, 新路点列表, 绕行步数)；不可行时增量为 -inf。"""
    base = build_path(w, wps)
    if base is None:
        return -math.inf, list(wps), 0
    new_wps = list(wps) + [cand]
    cells = build_path(w, new_wps)
    if cells is None:
        return -math.inf, list(wps), 0
    detour = len(cells) - len(base)
    base_obj = cov.objective(cfg)
    cells_padded = pad_to_window(cells, w, prefer=cand)
    path = path_to_solution(w, cells_padded)
    cov.add_path(path)
    new_obj = cov.objective(cfg)
    cov.add_path(path, sign=-1)
    return new_obj - base_obj, new_wps, detour


def _best_action(instance: Instance, cov: Coverage, plan: Plan, cost: float, cfg: EnvConfig,
                 rng: random.Random, mode: str, k: int, worker_batch: int,
                 allow_new: bool) -> Optional[Tuple[float, str, List[Tuple[int, int]], float]]:
    """在所有候选动作中挑选最优：返回 (score, wid, 新路点, 成本增量)。

    mode="tvpg"：按边际覆盖增益排序（论文中的任务价值优先）；
    mode="tcpg"：先按激励成本（绕行步数）排序，成本相同再比增益。
    """
    best = None
    best_key = None

    def _key(gain: float, detour: int, kind: str) -> Tuple[float, float]:
        if mode == "tcpg" and kind == "task":
            return (-float(detour), gain)
        return (gain, -float(detour))

    recruited = list(plan.keys())
    if recruited:
        batch = recruited if len(recruited) <= worker_batch else rng.sample(recruited, worker_batch)
        for wid in batch:
            w = instance.worker(wid)
            for cand in _candidate_cells(cov, w, rng, k):
                gain, new_wps, detour = _try_insert(instance, cov, w, plan[wid], cand, cfg)
                if gain <= 1e-12:
                    continue
                key = _key(gain, detour, "task")
                if best_key is None or key > best_key:
                    best_key, best = key, (key[0], wid, new_wps, 0.0)

    if allow_new:
        free = [w for w in instance.workers if w.wid not in plan and cost + w.window <= instance.budget]
        if len(free) > worker_batch:
            free = rng.sample(free, worker_batch)
        for w in free:
            base_cells = build_path(w, [])
            if base_cells is None:
                continue
            base_path = path_to_solution(w, pad_to_window(base_cells, w))
            cov.add_path(base_path)
            base_obj = cov.objective(cfg)
            local_best = None
            local_key = None
            for cand in _candidate_cells(cov, w, rng, k):
                gain, new_wps, detour = _try_insert(instance, cov, w, [], cand, cfg)
                if gain <= 1e-12:
                    continue
                key = _key(gain, detour, "task")
                if local_key is None or key > local_key:
                    local_key, local_best = key, new_wps
            cov.add_path(base_path, sign=-1)
            if local_best is None:
                cov.add_path(base_path)
                gain0 = cov.objective(cfg) - base_obj
                cov.add_path(base_path, sign=-1)
                if gain0 <= 1e-12:
                    continue
                local_key, local_best = (gain0, 0.0), []
            if best_key is None or local_key > best_key:
                best_key = local_key
                best = (local_key[0], w.wid, local_best, w.window)
    return best


def _apply(instance: Instance, cov: Coverage, plan: Plan, wid: str,
           wps: List[Tuple[int, int]]) -> float:
    w = instance.worker(wid)
    cells = build_path(w, wps)
    cells = pad_to_window(cells, w, prefer=wps[-1] if wps else None)
    path = path_to_solution(w, cells)
    cov.add_path(path)
    plan[wid] = list(wps)
    return float(len(path))


# --------------------------------------------------------------------------- 基线
def rn_plan(instance: Instance, cfg: EnvConfig, rng: random.Random) -> Solution:
    """随机：随机招募工作者，沿路径随机插入可行任务。"""
    order = [w for w in instance.workers]
    rng.shuffle(order)
    plan: Plan = {}
    cost = 0.0
    for w in order:
        if cost + w.window > instance.budget:
            continue
        wps: List[Tuple[int, int]] = []
        for _ in range(rng.randint(0, 3)):
            cand = (rng.randrange(instance.nx), rng.randrange(instance.ny))
            if _reachable(w, cand) and build_path(w, wps + [cand]) is not None:
                wps.append(cand)
        plan[w.wid] = wps
        cost += w.window
        if instance.budget - cost < min(x.window for x in instance.workers):
            break
    return plan_to_solution(instance, plan)


def _greedy(instance: Instance, cfg: EnvConfig, rng: random.Random, mode: str,
            k: int = 15, worker_batch: int = 8) -> Solution:
    cov = Coverage(instance)
    plan: Plan = {}
    cost = 0.0
    while True:
        action = _best_action(instance, cov, plan, cost, cfg, rng, mode, k, worker_batch, allow_new=True)
        if action is None:
            break
        _, wid, wps, extra = action
        if extra and cost + extra > instance.budget:
            break
        cost += extra
        _apply(instance, cov, plan, wid, wps)
    return plan_to_solution(instance, plan)


def tvpg_plan(instance: Instance, cfg: EnvConfig, rng: random.Random) -> Solution:
    """任务价值优先贪心：优先插入边际覆盖增益最大的任务。"""
    return _greedy(instance, cfg, rng, mode="tvpg")


def tcpg_plan(instance: Instance, cfg: EnvConfig, rng: random.Random) -> Solution:
    """任务成本优先贪心：优先插入激励成本（绕行步数）最低的任务。"""
    return _greedy(instance, cfg, rng, mode="tcpg")


def _solution_to_plan(instance: Instance, solution: Solution) -> Plan:
    """从"路径"反推路点（这里只保留路径中偏离最短路的关键点，简化为等长采样）。"""
    plan: Plan = {}
    for wid, path in solution.items():
        w = instance.worker(wid)
        cells = [(int(p[0]), int(p[1])) for p in path]
        # 提取偏离 origin->dest 最短路最多的点作为路点
        base = set(shortest_path(w.origin, w.dest))
        wps: List[Tuple[int, int]] = []
        for c in cells:
            if c not in base and c != w.origin and c != w.dest and (not wps or wps[-1] != c):
                wps.append(c)
        plan[wid] = wps
    return plan


def _sa(instance: Instance, cfg: EnvConfig, rng: random.Random, init: Plan,
        iters: int = 2500, t0: float = 0.05, t1: float = 0.001) -> Solution:
    cur = {k: list(v) for k, v in init.items()}
    cur_obj, cur_cov = evaluate_plan(instance, cur, cfg)
    best, best_obj = {k: list(v) for k, v in cur.items()}, cur_obj
    all_ids = [w.wid for w in instance.workers]
    recruited = set(cur.keys())

    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / max(1, iters - 1))
        cand = {k: list(v) for k, v in cur.items()}
        op = rng.random()
        if op < 0.30 and recruited:                     # 移动一个路点
            wid = rng.choice(sorted(recruited))
            w = instance.worker(wid)
            if cand[wid]:
                i = rng.randrange(len(cand[wid]))
                c = (rng.randrange(instance.nx), rng.randrange(instance.ny))
                if _reachable(w, c):
                    cand[wid][i] = c
        elif op < 0.50 and recruited:                   # 删除路点
            wid = rng.choice(sorted(recruited))
            if cand[wid]:
                cand[wid].pop(rng.randrange(len(cand[wid])))
        elif op < 0.72:                                 # 新增路点
            target = sorted(recruited) or []
            wid = rng.choice(target) if target and rng.random() < 0.7 else rng.choice(all_ids)
            if wid not in cand:
                w = instance.worker(wid)
                if plan_cost(instance, cand) + w.window > instance.budget:
                    continue
                cand[wid] = []
            w = instance.worker(wid)
            c = (rng.randrange(instance.nx), rng.randrange(instance.ny))
            if _reachable(w, c) and build_path(w, cand[wid] + [c]) is not None:
                cand[wid].append(c)
        elif op < 0.88 and len(recruited) >= 2:         # 交换两个工作者的路点
            a, b = rng.sample(sorted(recruited), 2)
            if cand[a] and cand[b]:
                i, j = rng.randrange(len(cand[a])), rng.randrange(len(cand[b]))
                cand[a][i], cand[b][j] = cand[b][j], cand[a][i]
        else:                                           # 招募 / 淘汰工作者
            if recruited and rng.random() < 0.5:
                wid = rng.choice(sorted(recruited))
                if len(recruited) > 1:
                    cand.pop(wid)
            else:
                free = [wid for wid in all_ids if wid not in recruited]
                if free:
                    w = instance.worker(rng.choice(free))
                    if plan_cost(instance, cand) + w.window <= instance.budget:
                        cand[w.wid] = []
        if plan_cost(instance, cand) > instance.budget + 1e-9:
            continue
        obj, _ = evaluate_plan(instance, cand, cfg)
        if obj >= cur_obj or rng.random() < math.exp((obj - cur_obj) / max(temp, 1e-6)):
            cur, cur_obj = cand, obj
            recruited = set(cur.keys())
            if obj > best_obj:
                best, best_obj = {k: list(v) for k, v in cand.items()}, obj
    return plan_to_solution(instance, best)


def msa_plan(instance: Instance, cfg: EnvConfig, rng: random.Random) -> Solution:
    """多起点模拟退火：从随机解出发做邻域搜索。"""
    init = _solution_to_plan(instance, rn_plan(instance, cfg, rng))
    best_sol, best_obj = None, -math.inf
    for _ in range(3):
        sol = _sa(instance, cfg, rng, init)
        obj = evaluate_plan(instance, _solution_to_plan(instance, sol), cfg)[0]
        if obj > best_obj:
            best_sol, best_obj = sol, obj
    return best_sol


def msagi_plan(instance: Instance, cfg: EnvConfig, rng: random.Random) -> Solution:
    """带贪心初始化的多起点模拟退火。"""
    init = _solution_to_plan(instance, tvpg_plan(instance, cfg, rng))
    best_sol, best_obj = None, -math.inf
    for _ in range(3):
        sol = _sa(instance, cfg, rng, init)
        obj = evaluate_plan(instance, _solution_to_plan(instance, sol), cfg)[0]
        if obj > best_obj:
            best_sol, best_obj = sol, obj
    return best_sol


# --------------------------------------------------------------------------- GraphDP
def _dp_moves(speed: int) -> List[Tuple[int, int, int]]:
    """DP 的合法移动集合：(dx, dy, backpointer_code)。

    speed=1 时正好是原始的"四邻域"，且走法顺序与编码与旧实现一致，
    保证 cells_per_step=1 的档位结果逐位不变；
    speed>1 时再加上沿单轴最多 speed 格、以及 1 格对角。
    """
    moves: List[Tuple[int, int, int]] = []
    for k in range(1, speed + 1):
        moves.append((k, 0, k))
    for k in range(1, speed + 1):
        moves.append((-k, 0, speed + k))
    for k in range(1, speed + 1):
        moves.append((0, k, 2 * speed + k))
    for k in range(1, speed + 1):
        moves.append((0, -k, 3 * speed + k))
    if speed > 1:
        base = 4 * speed
        moves.append((1, 1, base + 1))
        moves.append((1, -1, base + 2))
        moves.append((-1, 1, base + 3))
        moves.append((-1, -1, base + 4))
    return moves


def _dp_path(instance: Instance, w: Worker, weight: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """在时间分层网格上做 DP，最大化沿路径的权重和（含起终点约束）。

    移动集由 ``w.speed`` 决定，与 ``shortest_path`` 保持一致——否则速度快的
    参与者会被 DP 判为不可达，GraphDP 在大规模档会凭空退化。
    """
    nx, ny = instance.nx, instance.ny
    T = w.window
    speed = max(1, int(w.speed))
    moves = _dp_moves(speed)
    neg = -1e9
    prev = np.full((nx, ny), neg, dtype=np.float64)
    prev[w.origin] = float(weight[w.t_start][w.origin])
    back = np.zeros((T, nx, ny), dtype=np.int8)   # 0 = stay，其余见 _dp_moves

    for ti in range(1, T):
        cur = prev.copy()
        best_src = np.zeros((nx, ny), dtype=np.int8)
        for dx, dy, code in moves:
            xs0, xs1 = max(0, dx), nx + min(0, dx)
            ys0, ys1 = max(0, dy), ny + min(0, dy)
            if xs0 >= xs1 or ys0 >= ys1:
                continue
            sx0, sy0 = -min(0, dx), -min(0, dy)
            sx1 = sx0 + (xs1 - xs0)
            sy1 = sy0 + (ys1 - ys0)
            src = prev[sx0:sx1, sy0:sy1]
            view = cur[xs0:xs1, ys0:ys1]
            m = src > view
            view[m] = src[m]
            best_src[xs0:xs1, ys0:ys1][m] = code
        cur = cur + weight[w.t_start + ti]
        prev = cur
        back[ti] = best_src

    x, y = w.dest
    if prev[x, y] <= neg / 2:
        return None
    cells: List[Tuple[int, int]] = []
    for ti in range(T - 1, -1, -1):
        cells.append((x, y))
        if ti == 0:
            break
        d = int(back[ti][x, y])
        if d == 0:
            pass
        elif d <= speed:                       # +x
            x -= d
        elif d <= 2 * speed:                   # -x
            x += d - speed
        elif d <= 3 * speed:                   # +y
            y -= d - 2 * speed
        elif d <= 4 * speed:                   # -y
            y += d - 3 * speed
        else:                                  # 对角
            k = d - 4 * speed
            x += -1 if k in (1, 2) else 1
            y += -1 if k in (1, 3) else 1
    cells.reverse()
    if cells[0] != w.origin:
        return None
    return cells


def _weights_from(instance, cov: Coverage) -> np.ndarray:
    """把覆盖计数转成 (T, nx, ny) 的路径权重：越少被覆盖越值钱；封锁区重罚。"""
    w = 1.0 / np.power(1.0 + cov.counts.astype(np.float64), 2.0)
    blocked = instance.meta.get("blocked_cells", set())
    if blocked:
        for (x, y) in blocked:
            if 0 <= x < instance.nx and 0 <= y < instance.ny:
                w[x, y, :] = -1e6
    return np.transpose(w, (2, 0, 1))


def graphdp_plan(instance: Instance, cfg: EnvConfig, rng: random.Random) -> Solution:
    """基于图的动态规划 + 工作者替换（文献 [18] 的思路）。"""
    cov = Coverage(instance)
    sol: Solution = {}
    cost = 0.0
    order = list(instance.workers)
    rng.shuffle(order)
    for w in order:
        if cost + w.window > instance.budget:
            continue
        cells = _dp_path(instance, w, _weights_from(instance, cov))
        if cells is None:
            continue
        path = path_to_solution(w, cells)
        cov.add_path(path)
        sol[w.wid] = path
        cost += w.window
        if instance.budget - cost < min(x.window for x in instance.workers):
            break

    # 工作者替换：把贡献最低的已招募者换成未招募者，若 J 提升则接受
    for _ in range(6):
        if not sol:
            break
        cur_obj = cov.objective(cfg)
        # 逐一计算"移除该 worker 后的目标值"，找贡献最低者
        worst_wid, worst_after = None, -math.inf
        for wid in sol:
            cov.add_path(sol[wid], sign=-1)
            after = cov.objective(cfg)
            cov.add_path(sol[wid])
            if after > worst_after:
                worst_after, worst_wid = after, wid
        if worst_wid is None:
            break
        freed_budget = instance.worker(worst_wid).window
        used_cost = sum(instance.worker(k).window for k in sol) - freed_budget
        free = [w for w in instance.workers if w.wid not in sol and used_cost + w.window <= instance.budget]
        if not free:
            break
        w_best = None
        base = {k: v for k, v in sol.items() if k != worst_wid}
        cov_base = Coverage(instance)
        for p in base.values():
            cov_base.add_path(p)
        base_obj = cov_base.objective(cfg)
        for w in free:
            cells = _dp_path(instance, w, _weights_from(instance, cov_base))
            if cells is None:
                continue
            path = path_to_solution(w, cells)
            cov_base.add_path(path)
            obj = cov_base.objective(cfg)
            cov_base.add_path(path, sign=-1)
            if obj > base_obj + 1e-9 and (w_best is None or obj > w_best[1]):
                w_best = (w, obj, path)
        if w_best is None:
            break
        sol.pop(worst_wid)
        cov = cov_base
        sol[w_best[0].wid] = w_best[2]
        cov.add_path(w_best[2])
    return sol


PLANNERS = {
    "RN": rn_plan,
    "TVPG": tvpg_plan,
    "TCPG": tcpg_plan,
    "MSA": msa_plan,
    "MSAGI": msagi_plan,
    "GraphDP": graphdp_plan,
}


def run_planner(name: str, instance: Instance, cfg: EnvConfig, seed: int = 0) -> Solution:
    return PLANNERS[name](instance, cfg, random.Random(seed))
