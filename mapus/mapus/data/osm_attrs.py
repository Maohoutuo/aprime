# -*- coding: utf-8 -*-
"""读取用 OpenStreetMap 重建的道路/兴趣点属性层。

产出物由 ``data/osm_attrs/build_osm_attrs.py`` 生成，本模块只负责读与整形，
不重新查询网络。它替代的是 Grab-Posisi 那 13 项**从未公开发布**的属性，
因此这里的数值是**可复现的公开数据重建**，论文里必须这么写，
不能宣称"使用了 Grab-Posisi 的属性表"。

用法::

    from mapus.data.osm_attrs import available, land_use_grid, road_features

    if available():
        lu = land_use_grid(16)            # (16, 16, 6)，逐格和为 1
        rf = road_features(16)            # 道路长度/交叉口/等级构成等
"""

from __future__ import annotations

import csv
import functools
import gzip
import os
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import LAND_USE

#: 与 ``mapus.data.instances.GEOLIFE_PATH`` 同风格：模块级缺省 + 调用期读环境变量，
#: 这样 ``$env:OSM_ATTRS_PATH`` 在 import 之后仍能生效。
OSM_ATTRS_PATH = os.environ.get(
    "OSM_ATTRS_PATH", r"D:\composition\data\osm_attrs\attrs.csv.gz")

N_CAT = len(LAND_USE)


def _path() -> Optional[str]:
    path = os.environ.get("OSM_ATTRS_PATH", OSM_ATTRS_PATH)
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    return None


def available() -> bool:
    """属性层是否已构建。判据是**产出物**在位，不是"装了 osmnx"。"""
    return _path() is not None


@functools.lru_cache(maxsize=4)
def _load(path: str) -> Dict[Tuple[int, int, int], dict]:
    table: Dict[Tuple[int, int, int], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            table[(int(row["n"]), int(row["x"]), int(row["y"]))] = row
    return table


def load_attrs(path: Optional[str] = None) -> Dict[Tuple[int, int, int], dict]:
    """``{(分辨率, x, y): 属性行}``。带进程内缓存（构建一次要 15 万次采样）。"""
    path = path or _path()
    if path is None:
        raise FileNotFoundError(
            "属性层产出物不存在；先跑 data/osm_attrs/build_osm_attrs.py，"
            "或设 OSM_ATTRS_PATH 指向已构建的 attrs.csv.gz")
    return _load(path)


def land_use_grid(n: int, path: Optional[str] = None) -> np.ndarray:
    """``(n, n, 6)`` 的逐格土地利用分布，每格和为 1，类别顺序同 ``config.LAND_USE``。

    **缺格填均匀分布**而不是填 0：0 会让 ``route_land_use`` 的加权平均除以 0，
    且"该格没有数据"与"该格六类都没有"是两回事，填均匀是更保守的表示。
    返回数组的 ``np.isnan`` 不会出现，缺格情况由 ``coverage(n)`` 单独报告。
    """
    table = load_attrs(path)
    out = np.full((n, n, N_CAT), 1.0 / N_CAT, dtype=float)
    for (rn, x, y), row in table.items():
        if rn != n:
            continue
        shares = np.array([float(row["lu_share_" + c]) for c in LAND_USE])
        s = shares.sum()
        out[x, y] = shares / s if s > 1e-12 else 1.0 / N_CAT
    return out


def coverage(n: int, path: Optional[str] = None) -> float:
    """该分辨率下**有数据**的格子占比（0–1）。报告时必须报这个数。"""
    table = load_attrs(path)
    have = sum(1 for (rn, _, _) in table if rn == n)
    return have / float(n * n) if n else 0.0


#: 道路类属性 → 输出字段。用于 ``road_features``。
_ROAD_FIELDS = ("n_road_seg", "road_len_m", "n_intersection", "n_building", "n_poi")


def road_features(n: int, path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """``(n, n)`` 的道路/建筑/兴趣点属性数组。

    ``lanes_mean`` 与 ``maxspeed_mean`` 在原始数据里是空串（该道路未标注），
    这里填 ``nan`` 而不是 0——0 会被误读成"车道数为零"。
    """
    table = load_attrs(path)
    out: Dict[str, np.ndarray] = {k: np.zeros((n, n), dtype=float) for k in _ROAD_FIELDS}
    for k in ("lanes_mean", "maxspeed_mean"):
        out[k] = np.full((n, n), np.nan, dtype=float)
    groups = [k[len("road_share_"):] for k in next(iter(table.values())) if k.startswith("road_share_")]
    for g in groups:
        out["road_share_" + g] = np.zeros((n, n), dtype=float)
    for (rn, x, y), row in table.items():
        if rn != n:
            continue
        for k in _ROAD_FIELDS:
            out[k][x, y] = float(row[k])
        for k in ("lanes_mean", "maxspeed_mean"):
            v = row[k]
            out[k][x, y] = float(v) if v not in ("", None) else np.nan
        for g in groups:
            out["road_share_" + g][x, y] = float(row["road_share_" + g])
    return out
