# -*- coding: utf-8 -*-
"""实例、工作者与解的数据结构，以及路径可行性校验。

解的表示与论文 Fig.2 / 附录 B 一致：
    Solution = {"worker_id": [[x, y, t], [x, y, t], ...], ...}
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

Cell = Tuple[int, int]
Task = Tuple[int, int, int]


@dataclass(frozen=True)
class Worker:
    wid: str#生成一个工作ID
    origin: Cell#生成一个起点
    dest: Cell#生成一个终点
    t_start: int#生成一个开始时间
    t_end: int#生成一个结束时间                # 闭区间
    speed: int = 1#生成一个速度            # 每时间步最多可移动的格数（1 格/步 = 四邻域或停留）

    @property
    def window(self) -> int:
        return self.t_end - self.t_start + 1#返回一个窗口

    @property
    def min_steps(self) -> int:
        """到达终点所需的最少时间步数（按每步最多 speed 格移动）。"""
        return steps_for(self.origin, self.dest, self.speed)#返回一个最小步数


def steps_for(a: Cell, b: Cell, speed: int) -> int:
    """从 a 到 b 需要的最少时间步数：每步最多走 ``speed`` 格（曼哈顿距离）。"""
    if speed < 1:
        raise ValueError("cells_per_step must be >= 1")
    return (manhattan(a, b) + speed - 1) // speed


@dataclass
class Instance:
    setting_name: str#生成一个设置名称
    index: int#生成一个索引
    grid: Cell#生成一个网格
    n_slots: int#生成一个槽位
    budget: float#生成一个预算
    workers: List[Worker]#生成一个工作者列表
    seed: int = 0#生成一个种子
    meta: dict = field(default_factory=dict)

    @property
    def nx(self) -> int:
        return self.grid[0]#返回一个x轴长度

    @property
    def ny(self) -> int:
        return self.grid[1]#返回一个y轴长度

    def worker(self, wid: str) -> Worker:
        for w in self.workers:#遍历工作者列表
            if w.wid == wid:
                return w
        raise KeyError(wid)#生成一个错误

    @property
    def n_tasks(self) -> int:
        return self.nx * self.ny * self.n_slots#返回一个任务数量


Solution = Dict[str, List[List[int]]]#生成一个解决方案


# --------------------------------------------------------------------------- 路径工具
def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])#返回一个曼哈顿距离


def shortest_path(a: Cell, b: Cell, speed: int = 1) -> List[Cell]:
    """网格上的最短（曼哈顿）路径，含起点与终点。

    ``speed`` > 1 时每个时间步沿单一轴方向最多前进 ``speed`` 格，
    因此返回的细胞数（= 时间步数）为 ``ceil(|dx|/speed) + ceil(|dy|/speed) + 1``。
    """
    if speed < 1:
        raise ValueError("cells_per_step must be >= 1")
    x, y = a
    path = [(x, y)]#生成一个路径
    while x != b[0]:
        step = min(speed, abs(b[0] - x))
        x += step if b[0] > x else -step
        path.append((x, y))#添加一个点
    while y != b[1]:
        step = min(speed, abs(b[1] - y))
        y += step if b[1] > y else -step
        path.append((x, y))#添加一个点
    return path


def build_path(worker: Worker, waypoints: Sequence[Cell]) -> List[Cell] | None:
    """把 [origin] + waypoints + [dest] 串成一条细胞序列。

    每步最多移动 ``worker.speed`` 格；若所需时间步数超过时间窗则返回 None。
    """
    pts = [worker.origin] + [w for w in waypoints] + [worker.dest]
    cells: List[Cell] = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        seg = shortest_path(a, b, worker.speed)[1:]
        cells.extend(seg)
    return cells if len(cells) <= worker.window else None


def pad_to_window(cells: Sequence[Cell], worker: Worker, prefer: Cell | None = None) -> List[Cell]:
    """用"停留"把细胞序列补齐到时间窗长度。

    停留优先放置在最不冗余的位置：默认在路径中部均匀插入，
    若给定 prefer 则优先在该细胞上停留（用于提升覆盖多样性）。
    """
    cells = list(cells)
    pad = worker.window - len(cells)
    if pad <= 0:
        return cells
    if prefer is not None and prefer in cells:
        out: List[Cell] = []
        extra = pad
        for c in cells:
            out.append(c)
            if c == prefer and extra > 0:
                out.extend([c] * extra)
                extra = 0
        while len(out) < worker.window:
            out.append(out[-1])
        return out
    # 均匀插入停留
    out = []
    n = len(cells)
    for i, c in enumerate(cells):
        out.append(c)
        if i < n - 1 and pad > 0:
            take = pad // (n - 1 - i) if (n - 1 - i) else 0
            out.extend([c] * take)
            pad -= take
    while len(out) < worker.window:
        out.append(out[-1])
    return out


def path_to_solution(worker: Worker, cells: Sequence[Cell]) -> List[List[int]]:
    return [[c[0], c[1], worker.t_start + i] for i, c in enumerate(cells)]


def solution_to_tasks(path: Sequence[Sequence[int]]) -> List[Task]:
    return [(int(x), int(y), int(t)) for x, y, t in path]


# --------------------------------------------------------------------------- 可行性
@dataclass
class ValidationResult:#ValidationResult类  
    feasible: bool
    cost: float#生成一个成本
    violations: List[str]#生成一个违反

    def to_dict(self) -> dict:
        return {"feasible": self.feasible, "cost": self.cost, "violations": self.violations}#返回一个字典
        return {"feasible": self.feasible, "cost": self.cost, "violations": self.violations}


def validate_solution(instance: Instance, solution: Solution, check_coverage: bool = False) -> ValidationResult:
    violations: List[str] = []
    cost = 0.0
    coords: Dict[str, Cell] = {}
    blocked = instance.meta.get("blocked_cells", set())

    for wid, path in solution.items():
        try:
            w = instance.worker(wid)
        except KeyError:
            violations.append(f"unknown_worker:{wid}")
            continue
        if not path:
            violations.append(f"empty_path:{wid}")
            continue
        coords[wid] = (w.origin[0], w.origin[1])
        # 1) 时间戳必须严格从 t_start 连续递增
        for i, step in enumerate(path):
            if len(step) != 3:
                violations.append(f"bad_step:{wid}@{i}")
                continue
            t_expect = w.t_start + i
            if int(step[2]) != t_expect:
                violations.append(f"bad_time:{wid}@{i}")
            x, y = int(step[0]), int(step[1])
            if not (0 <= x < instance.nx and 0 <= y < instance.ny):
                violations.append(f"out_of_grid:{wid}@{i}")
        # 2) 起点/终点
        if (int(path[0][0]), int(path[0][1])) != w.origin:
            violations.append(f"bad_origin:{wid}")
        if (int(path[-1][0]), int(path[-1][1])) != w.dest:
            violations.append(f"bad_dest:{wid}")
        if len(path) > w.window:
            violations.append(f"window_overflow:{wid}")
        # 3) 每步位移不超过该工作者的速度上限（speed=1 即四邻域或停留）
        for i in range(1, len(path)):
            d = manhattan((int(path[i - 1][0]), int(path[i - 1][1])), (int(path[i][0]), int(path[i][1])))
            if d > w.speed:
                violations.append(f"illegal_move:{wid}@{i}")
        # 4) 不得进入封锁区
        if blocked:
            for step in path:
                if (int(step[0]), int(step[1])) in blocked:
                    violations.append(f"blocked_cell:{wid}@{step[2]}")
                    break
        cost += len(path)

    if cost > instance.budget + 1e-9:
        violations.append(f"budget_exceeded:{cost:.2f}>{instance.budget}")
    return ValidationResult(feasible=not violations, cost=cost, violations=violations)
