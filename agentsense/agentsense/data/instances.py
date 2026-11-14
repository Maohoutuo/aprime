# -*- coding: utf-8 -*-
"""数据层：合成实例生成 + T-Drive 轨迹加载（可选）。

论文使用 T-Drive 与 Grab-Posisi 两个真实移动数据集。数据本身不在仓库中，
且 Grab-Posisi 需要 Kaggle 授权，因此这里提供两条路径：

1. `synthetic`（默认）：按表 2 的规模生成与真实数据同构的工作者
   （起点 / 终点 / 可用时间窗 / 速度），保证整条流水线可跑、可复现；
2. `tdrive`：若本地存在 T-Drive 样本（见 docs/assumptions.md 的格式说明），
   则从真实轨迹中采样工作者。

两条路径输出同一种 `Instance`，因此基线与智能体代码无需改动。
"""

from __future__ import annotations

import functools
import gzip
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from ..config import SETTINGS, Setting
from ..env.model import Cell, Instance, Worker, steps_for


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def instance_seed(setting: Setting, index: int) -> int:
    """固定 seed，保证任何机器上重建同一批实例。"""
    base = {"tdrive": 1_000_000, "grab": 2_000_000}[setting.dataset]
    scale = {"small": 0, "medium": 100, "large": 200}[setting.scale]
    return base + scale + index * 7919


def sample_endpoints(nx: int, ny: int, window: int, cps: int, rng: random.Random,
                     max_steps_ratio: float = 1.0,
                     min_steps_ratio: float = 1.0 / 3.0,
                     tries: int = 256) -> Tuple[Cell, Cell]:
    """在 nx×ny 网格上抽一对"时间窗内可行"的起终点（步数按 ``cps`` 折算）。

    可行性 = ``steps_for(origin, dest, cps) <= max_steps_ratio * (window - 1)``，
    因此 ``1 - steps/(window-1)`` 就是到达终点后的剩余余量（slack），与真实轨迹
    的审计口径一致。这里用**拒绝采样**而不是"先抽距离再夹进网格"：后者在
    速度较大的档位会把终点全挤到网格边界，让路线分布退化。
    """
    span = max(1, window - 1)
    cap = max(1, int(math.floor(max_steps_ratio * span)))
    lo = max(1, int(math.floor(min_steps_ratio * cap)))
    if nx * ny <= 1:
        return (0, 0), (0, 0)
    for _ in range(tries):
        origin = (rng.randrange(nx), rng.randrange(ny))
        dest = (rng.randrange(nx), rng.randrange(ny))
        steps = steps_for(origin, dest, cps)
        if lo <= steps <= cap:
            return origin, dest
    origin = (rng.randrange(nx), rng.randrange(ny))
    dest = (min(nx - 1, origin[0] + min(nx - 1, cap * cps)), origin[1])
    return origin, dest


# --------------------------------------------------------------------------- 合成工作者
def synthetic_workers(setting: Setting, rng: random.Random) -> List[Worker]:
    nx, ny, T = setting.grid_x, setting.grid_y, setting.n_slots
    # 速度上限用实测值（config.Setting.cells_per_step，见 experiments/measure_speed.py）
    cps = max(1, int(setting.cells_per_step))
    workers: List[Worker] = []
    for i in range(setting.n_workers):
        # 时间窗长度：至少覆盖到对角线行程，最多占满整个时域
        min_win = max(3, min(T, int(round(0.55 * T))))
        win = rng.randint(min_win, T)
        t_start = rng.randint(0, T - win)
        # 起终点：时间窗内可行（步数按 cps 折算），并保留一部分剩余余量
        origin, dest = sample_endpoints(nx, ny, win, cps, rng)
        workers.append(
            Worker(
                wid=f"w{i:03d}",
                origin=origin,
                dest=dest,
                t_start=t_start,
                t_end=t_start + win - 1,
                speed=cps,
            )
        )
    return workers


# --------------------------------------------------------------------------- T-Drive
def _load_tdrive_traces(
    path: str,
    bbox: Tuple[float, float, float, float] = (116.20, 39.80, 116.55, 40.05),
    grid: Tuple[int, int] = (8, 8),
    slot_min: int = 15,
    limit_taxis: Optional[int] = None,
) -> Dict[str, List[Tuple[int, Cell]]]:
    """读取 T-Drive 样本，返回 {taxi_id: [(slot_index, (x, y)), ...]}。

    期望的原始格式（微软 T-Drive 样本，逗号分隔）：
        taxi_id,2008-02-02 15:36:08,116.38545,39.90776
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    opener = gzip.open if path.endswith(".gz") else open
    lon0, lat0, lon1, lat1 = bbox
    nx, ny = grid
    traces: Dict[str, List[Tuple[int, Cell]]] = defaultdict(list)
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            tid, ts, lon, lat = parts[0], parts[1], parts[2], parts[3]
            try:
                lon_v, lat_v = float(lon), float(lat)
            except ValueError:
                continue
            if not (lon0 <= lon_v <= lon1 and lat0 <= lat_v <= lat1):
                continue
            try:
                dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            x = int((lon_v - lon0) / (lon1 - lon0) * nx)
            y = int((lat_v - lat0) / (lat1 - lat0) * ny)
            slot = (dt.hour * 60 + dt.minute) // slot_min
            traces[tid].append((slot, (min(x, nx - 1), min(y, ny - 1))))
            if limit_taxis and len(traces) > limit_taxis:
                break
    for tid in list(traces):
        traces[tid].sort()
        # 只保留连续时间步的片段
        traces[tid] = [(t, c) for t, c in traces[tid]]
    return dict(traces)


@functools.lru_cache(maxsize=8)
def _cached_traces(path: str, bbox: Tuple[float, float, float, float],
                   grid: Tuple[int, int], slot_min: int,
                   limit_taxis: Optional[int]):
    return _load_tdrive_traces(path, bbox=bbox, grid=grid, slot_min=slot_min,
                               limit_taxis=limit_taxis)


def load_tdrive_traces(
    path: str,
    bbox: Tuple[float, float, float, float] = (116.20, 39.80, 116.55, 40.05),
    grid: Tuple[int, int] = (8, 8),
    slot_min: int = 15,
    limit_taxis: Optional[int] = None,
) -> Dict[str, List[Tuple[int, Cell]]]:
    """带进程内缓存的入口。

    清洗后的 T-Drive 有 214 万行，解析一次约 13 秒；而每次构造实例都会调用
    这里，不缓存的话 50 天的实验会被重复解析拖垮。
    """
    return _cached_traces(path, tuple(bbox), tuple(grid), int(slot_min), limit_taxis)


def _contiguous_runs(pts: List[Tuple[int, Cell]], min_len: int) -> List[List[Tuple[int, Cell]]]:
    """把稀疏轨迹切成时间上连续的片段（相邻观测必须落在相邻 slot）。"""
    runs: List[List[Tuple[int, Cell]]] = []
    cur: List[Tuple[int, Cell]] = []
    for slot, cell in pts:
        if cur and slot - cur[-1][0] == 1:
            cur.append((slot, cell))
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = [(slot, cell)]
    if len(cur) >= min_len:
        runs.append(cur)
    return runs


def _trips(run: List[Tuple[int, Cell]]) -> List[Tuple[Cell, Cell, int]]:
    """把连续轨迹切成"移动段"，返回 [(origin, dest, duration_slots), ...]。

    静止段定义为同一格连续 >= 2 个 slot。这对应真实语义下的一次出行：
    从 origin 出发、连续移动 duration 个 slot 后停下（到达 dest）。
    """
    trips: List[Tuple[Cell, Cell, int]] = []
    n = len(run)
    i = 0
    while i < n - 1:
        j = i + 1
        while j < n and run[j][1] != run[j - 1][1]:
            j += 1
        if j - i >= 2:
            trips.append((run[i][1], run[j - 1][1], j - 1 - i))
        i = j if j > i else i + 1
    return trips


def tdrive_workers(
    setting: Setting,
    rng: random.Random,
    path: Optional[str] = None,
    bbox: Tuple[float, float, float, float] = (116.20, 39.80, 116.55, 40.05),
    theta: float = 0.55,
    min_slack: Optional[float] = None,
) -> List[Worker]:
    """从真实轨迹中截取参与者，时间预算由显式参数 ``theta`` 决定。

    关键约定（写进论文的方法节）：

        W = ceil(d_trip * (1 + theta)) + 1        # 时间窗长度（slot）
        slack = 1 - steps_for(origin, dest, cps) / (W - 1)

    其中 ``d_trip`` 是真实轨迹测得的移动时长，``cps = setting.cells_per_step``
    是实测的每步可移动格数（``steps_for`` = ceil(曼哈顿距离 / cps)）。**theta 是
    参与者的申报弹性，不是从数据里观测到的量**——被动轨迹数据（T-Drive / 订单
    记录）里根本没有"我可以多花 30% 时间"这个字段。把 theta 显式化，才能让
    alpha_min(theta) 有物理含义，也才能诚实地报告结论对 theta 的敏感性。

    ``min_slack`` 若给定，则额外要求 slack 不低于该值（默认 None，仅要求
    ``steps_for(origin, dest, cps) <= W - 1`` 即可行）。
    """
    if not path:
        raise FileNotFoundError("T-Drive 数据路径未提供")
    cps = max(1, int(setting.cells_per_step))
    traces = load_tdrive_traces(
        path, bbox=bbox, grid=(setting.grid_x, setting.grid_y), slot_min=setting.slot_min
    )
    T = setting.n_slots

    cands: List[Tuple[Cell, Cell, int]] = []
    for _tid, pts in traces.items():
        seg = [(s, c) for s, c in pts if c is not None]
        for run in _contiguous_runs(seg, min_len=3):
            cands.extend(_trips(run[:T]))
    if len(cands) < setting.n_workers:
        raise ValueError(f"可用真实出行段不足：{len(cands)} < {setting.n_workers}")
    rng.shuffle(cands)

    floor = min_slack if min_slack is not None else 0.0
    workers: List[Worker] = []
    for i, (origin, dest, d_trip) in enumerate(cands):
        if len(workers) >= setting.n_workers:
            break
        win = int(math.ceil(d_trip * (1.0 + theta))) + 1
        if win > T:
            continue
        span = win - 1
        if span <= 0:
            continue
        slack = 1.0 - steps_for(origin, dest, cps) / span
        if slack < floor:
            continue
        t_start = rng.randint(0, T - win)
        workers.append(Worker(wid=f"t{i:03d}", origin=origin, dest=dest,
                              t_start=t_start, t_end=t_start + win - 1, speed=cps))
    if len(workers) < setting.n_workers:
        raise ValueError("采样到的工作者数量不足")
    return workers


# ----------------------------------- 生成实验输入数据的入口函数
"""
配置名 + 编号 + 数据来源
            ↓
查找配置，设置随机种子
            ↓
生成工作者列表
            ↓
把配置和工作者打包成 Instance 返回
"""
def make_instance(
    setting_name: str,
    index: int,
    source: str = "synthetic",
    tdrive_path: Optional[str] = None,
) -> Instance:
    dataset, scale = setting_name.split("_")
    setting = SETTINGS[(dataset, scale)]
    seed = instance_seed(setting, index)
    rng = _rng(seed)
    used_source = source
    workers: List[Worker]
    if source == "tdrive":
        try:
            workers = tdrive_workers(setting, rng, path=tdrive_path)
        except Exception:
            workers = synthetic_workers(setting, rng)
            used_source = "synthetic(fallback)"
    else:
        workers = synthetic_workers(setting, rng)
    return Instance(
        setting_name=setting.name,
        index=index,
        grid=(setting.grid_x, setting.grid_y),
        n_slots=setting.n_slots,
        budget=float(setting.total_budget),
        workers=workers,
        seed=seed,
        meta={
            "dataset": setting.dataset,
            "scale": setting.scale,
            "source": used_source,
            "slot_minutes": setting.slot_min,
            "horizon_minutes": setting.horizon_min,
            # 实测速度（单一真源）：已写进每个 Worker.speed，同时供 mapus 的
            # 校验/路由直接读取。来源见 experiments/measure_speed.py。
            "cells_per_step": int(setting.cells_per_step),
            "cells_per_step_source": ("tdrive_measured" if setting.dataset == "tdrive"
                                      else "proxy(tdrive @same grid/slot)"),
        },
    )
