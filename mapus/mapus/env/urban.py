# -*- coding: utf-8 -*-
"""城市属性层：6 类土地利用 + 犯罪分布（论文 3.3 式 (3)(4)、附录 B.1）。

论文用 POI 聚合出 6 类土地利用，T-Drive 区域另加犯罪统计（Zhang et al. 2025），
并剔除记录数 < 10 的格子。原始 POI / 犯罪数据不在论文仓库里，因此本工程用
**结构化的合成城市**替代：在归一化坐标 [0,1]² 上定义一套"城市形态"（商业中心、
居住环、工业区、文教区、医院、绿地与水系），再按网格分辨率采样。

这样做的两个好处：
* 小/中/大三档网格采样的是**同一个城市**的粗/细粒度版本，跨规模可比；
* 土地利用有强空间结构，路线的 ``h(r_u)`` 才会因路线不同而明显不同，
  PSS 才有区分度（否则所有路线都近似均匀分布，cos 相似度趋于常数）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import LAND_USE, LAND_USE_GLYPH

N_CAT = len(LAND_USE)


# --------------------------------------------------------------------------- 城市形态
# (cx, cy, sigma_x, sigma_y, 权重, 类别)；坐标为 [0,1]² 归一化平面
_TDRIVE_LAYOUT: Tuple[Tuple[float, ...], ...] = (
    # 商业核心
    (0.52, 0.50, 0.13, 0.13, 1.20, 5),
    (0.28, 0.66, 0.10, 0.09, 0.70, 5),
    (0.74, 0.26, 0.10, 0.09, 0.60, 5),
    (0.55, 0.80, 0.07, 0.06, 0.40, 5),
    # 居住环（北京式的成片居住区）
    (0.30, 0.32, 0.15, 0.13, 0.55, 4),
    (0.68, 0.68, 0.15, 0.14, 0.55, 4),
    (0.20, 0.55, 0.12, 0.11, 0.45, 4),
    (0.60, 0.90, 0.12, 0.09, 0.40, 4),
    (0.88, 0.55, 0.10, 0.12, 0.40, 4),
    (0.45, 0.18, 0.13, 0.09, 0.35, 4),
    # 工业区
    (0.14, 0.16, 0.14, 0.13, 1.00, 1),
    (0.86, 0.30, 0.11, 0.10, 0.70, 1),
    (0.50, 0.95, 0.11, 0.08, 0.50, 1),
    # 文教区（大学城）
    (0.24, 0.84, 0.12, 0.10, 0.90, 2),
    (0.76, 0.10, 0.09, 0.08, 0.55, 2),
    # 医疗
    (0.48, 0.58, 0.09, 0.09, 1.40, 3),
    (0.33, 0.46, 0.08, 0.08, 1.10, 3),
    (0.66, 0.42, 0.08, 0.08, 1.10, 3),
    (0.16, 0.42, 0.08, 0.08, 0.90, 3),
    # 绿地与水系（沿对角线的一条绿带 + 几处公园）
    (0.50, 0.30, 0.07, 0.30, 0.90, 0),   # 斜向绿带
    (0.66, 0.80, 0.10, 0.09, 1.00, 0),
    (0.12, 0.66, 0.09, 0.08, 0.80, 0),
    (0.88, 0.82, 0.10, 0.09, 0.70, 0),
    (0.40, 0.06, 0.08, 0.07, 0.60, 0),
    (0.05, 0.05, 0.09, 0.09, 0.60, 0),
)

_GRAB_LAYOUT: Tuple[Tuple[float, ...], ...] = (
    # 东南亚城市：高密度混合，商业沿主轴分布
    (0.45, 0.50, 0.20, 0.22, 1.30, 5),
    (0.80, 0.45, 0.14, 0.18, 0.90, 5),
    (0.20, 0.58, 0.13, 0.16, 0.80, 5),
    (0.62, 0.85, 0.10, 0.10, 0.50, 5),
    (0.35, 0.25, 0.18, 0.14, 0.70, 4),
    (0.70, 0.72, 0.16, 0.14, 0.65, 4),
    (0.15, 0.25, 0.12, 0.12, 0.55, 4),
    (0.88, 0.80, 0.10, 0.11, 0.45, 4),
    (0.88, 0.20, 0.09, 0.10, 0.60, 1),
    (0.12, 0.85, 0.09, 0.10, 0.60, 1),
    (0.55, 0.15, 0.09, 0.08, 0.60, 2),
    (0.30, 0.88, 0.09, 0.08, 0.60, 2),
    (0.50, 0.62, 0.09, 0.09, 1.20, 3),
    (0.76, 0.55, 0.08, 0.08, 1.00, 3),
    (0.38, 0.48, 0.09, 0.10, 0.85, 0),
    (0.68, 0.30, 0.07, 0.13, 0.75, 0),
    (0.18, 0.45, 0.06, 0.08, 0.60, 0),
)


def _field(nx: int, ny: int, layout, grain: float = 0.35) -> np.ndarray:
    """在 nx×ny 网格上采样高斯混合，返回 (nx, ny, 6) 的逐格得分场（未归一化）。

    ``grain`` 把城市结构缩到"1–2 格"的尺度：论文的参与者路线只有 5–15 步，
    若地块是 10 格宽的大片区，任何路线都只会落在一类土地里，h(r_u) 退化成
    one-hot，PSS 也就失去意义。真实城市在 1 km 网格上确实是混合用地，因此这里
    用细粒度地块（相邻格可能不同类），保证路线能跨越多类土地。
    """
    xs = (np.arange(nx) + 0.5) / nx
    ys = (np.arange(ny) + 0.5) / ny
    lx, ly = np.meshgrid(xs, ys, indexing="ij")
    out = np.zeros((nx, ny, N_CAT), dtype=np.float64)
    for cx, cy, sx, sy, w, cat in layout:
        gx = np.exp(-((lx - cx) ** 2) / (2 * (sx * grain) ** 2 + 1e-6))
        gy = np.exp(-((ly - cy) ** 2) / (2 * (sy * grain) ** 2 + 1e-6))
        out[:, :, cat] += w * gx * gy
    # 背景：所有格子都有一点居住与商业成分，避免出现"纯空地"
    out[:, :, 4] += 0.08
    out[:, :, 5] += 0.06
    for cat in range(N_CAT):
        out[:, :, cat] /= max(1e-9, out[:, :, cat].max())
    # 一格相关长度，让同类土地仍有轻微成片
    k = np.array([0.25, 0.5, 0.25])
    for cat in range(N_CAT):
        sm = out[:, :, cat]
        sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, sm)
        sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, sm)
        out[:, :, cat] = sm
    return out


# 六类的目标占比（vegetation, industrial, institutional, medical, residential, commercial）
TARGET_SHARE = (0.20, 0.15, 0.13, 0.07, 0.25, 0.20)


def _assign_categories(score: np.ndarray, seed: int = 7) -> np.ndarray:
    """配额约束下的逐格抽样：既保证六类占比，又让相邻格混合（细粒度混合用地）。

    若按全局得分排序做硬分配，小网格上会退化成"大块 Voronoi 片区"，5–15 步的
    路线只会落在一类土地里；因此改为**按局部混合权重抽样 + 剩余配额约束**。
    """
    nx, ny, nc = score.shape
    total = nx * ny
    quota = np.maximum(1, np.floor(np.array(TARGET_SHARE) * total).astype(int))
    while quota.sum() > total:
        quota[int(np.argmax(quota))] -= 1
    while quota.sum() < total:
        quota[int(np.argmax(np.array(TARGET_SHARE) * total - quota))] += 1

    rng = np.random.default_rng(seed)
    prob = score / np.maximum(1e-9, score.sum(axis=2, keepdims=True))
    flat = prob.reshape(total, nc).copy()
    left = quota.copy()
    cat = np.zeros(total, dtype=int)
    order = rng.permutation(total)
    for cell in order:
        w = flat[cell] * (left > 0)
        if w.sum() <= 1e-12:
            w = (left > 0).astype(float)
        w = w / w.sum()
        c = int(rng.choice(nc, p=w))
        cat[cell] = c
        left[c] -= 1
    return cat.reshape(nx, ny)


def _crime_field(nx: int, ny: int, land_use: np.ndarray, seed: int, exclude_below: float = 10.0):
    """犯罪强度：商业/工业密度与环境噪声的混合，并剔除记录数 < 10 的格子（附录 B.1）。"""
    rng = np.random.default_rng(seed)
    density = 2.4 * land_use[:, :, 5] + 1.6 * land_use[:, :, 1] + 0.4 * land_use[:, :, 4]
    noise = rng.gamma(shape=2.0, scale=0.5, size=(nx, ny))
    hotspots = np.zeros((nx, ny))
    for _ in range(max(2, (nx * ny) // 40)):
        cx, cy = rng.integers(0, nx), rng.integers(0, ny)
        hotspots[cx, cy] += rng.uniform(4.0, 12.0)
    # 热点做一次平滑，形成片区而非孤立点。
    # 注意：``np.convolve(..., mode="same")`` 在**卷积核比序列还长**时返回
    # ``len(k)`` 长度而不是 ``len(v)``，小网格（nx 或 ny < 3）上会直接抛
    # 广播错误。真实实验网格都 ≥8，但测试与边界情形不该因此崩掉。
    k = np.array([0.25, 0.5, 0.25])
    for axis in (0, 1):
        size = hotspots.shape[axis]
        if size < len(k):
            continue  # 网格太小，平滑没有意义，保留原热点
        hotspots = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, hotspots)
    raw = 40.0 * density * noise + hotspots
    raw = np.maximum(raw, 0.0)
    masked = np.where(raw >= exclude_below, raw, 0.0)
    return masked


@dataclass
class UrbanGrid:
    """网格级城市属性（论文 3.3 与附录 B.1）。"""

    nx: int
    ny: int
    dataset: str
    land_use: np.ndarray          # (nx, ny, 6)，逐格分布，和为 1
    category: np.ndarray          # (nx, ny)，κ(g) = argmax_c land_use
    crime: np.ndarray             # (nx, ny)，犯罪记录数（0 = 被剔除或不适用）
    crime_norm: np.ndarray        # (nx, ny) ∈ [0,1]，risk(·) 用的归一化强度
    #: 该分辨率下**有实测数据**的格子占比。合成布局恒为 1.0；
    #: ``source="osm"`` 时是真实覆盖率，报告里必须带这个数。
    coverage: float = 1.0

    # ---- 式 (3)：沿路线的经验土地利用分布
    def route_land_use(self, cells, prior: float = 0.0) -> np.ndarray:
        """h(r_u)；``prior`` 为 Laplace 平滑强度（见 PathSatisfaction 的说明）。"""
        h = np.zeros(N_CAT, dtype=np.float64)
        n = 0
        for x, y in cells:
            if 0 <= x < self.nx and 0 <= y < self.ny:
                h[self.category[x, y]] += 1.0
                n += 1
        if n == 0:
            return h
        h /= n
        if prior > 0:
            q = self.global_share()
            h = (n * h + prior * q) / (n + prior)
        return h

    def global_share(self) -> np.ndarray:
        q = np.bincount(self.category.ravel(), minlength=N_CAT).astype(np.float64)
        return q / max(1.0, q.sum())

    # ---- 式 (2) 的 risk(r_u)：沿线不良暴露的均值
    def route_risk(self, cells) -> float:
        vals = [self.crime_norm[x, y] for x, y in cells if 0 <= x < self.nx and 0 <= y < self.ny]
        return float(np.mean(vals)) if vals else 0.0

    def risk(self, x: int, y: int) -> float:
        return float(self.crime_norm[x, y])

    # ---- 提示词用的紧凑表示
    def land_use_text(self) -> str:
        legend = "  ".join(f"{LAND_USE_GLYPH[i]}={LAND_USE[i]}" for i in range(N_CAT))
        rows = []
        for y in range(self.ny):
            rows.append("".join(LAND_USE_GLYPH[self.category[x, y]] for x in range(self.nx)))
        return f"legend: {legend}\ngrid (row = y, col = x, origin at top-left):\n" + "\n".join(rows)

    def crime_text(self) -> str:
        rows = []
        peak = float(self.crime_norm.max()) or 1.0
        for y in range(self.ny):
            row = []
            for x in range(self.nx):
                v = self.crime[x, y]
                row.append("." if v <= 0 else str(min(9, max(1, int(round(9 * v / peak))))))
            rows.append("".join(row))
        return ("crime level (0-9, '.' = no/few records; only cells with >=10 incidents are kept):\n"
                + "\n".join(rows))

    def summary(self, x: int, y: int) -> str:
        dist = self.land_use[x, y]
        top = sorted(range(N_CAT), key=lambda i: -dist[i])[:3]
        lu = ", ".join(f"{LAND_USE[i]} {dist[i]:.2f}" for i in top)
        risk = self.crime_norm[x, y]
        level = "low" if risk < 0.2 else ("moderate" if risk < 0.5 else "high")
        return f"cell ({x}, {y}): dominant land use = {LAND_USE[self.category[x, y]]} ({lu}); crime risk = {level} ({risk:.2f})"

    def to_dict(self) -> dict:
        return {"nx": self.nx, "ny": self.ny, "dataset": self.dataset,
                "land_use": self.land_use.round(6).tolist(),
                "category": self.category.tolist(),
                "crime": self.crime.round(3).tolist(),
                "crime_norm": self.crime_norm.round(6).tolist(),
                "coverage": round(float(self.coverage), 4)}


_URBAN_CACHE: Dict[Tuple[str, int, int, str], UrbanGrid] = {}


def make_urban_grid(dataset: str, nx: int, ny: int, source: str = "synthetic") -> UrbanGrid:
    """按数据集构造（并缓存）城市属性。dataset ∈ {tdrive, grab}。

    ``source="osm"`` 时土地利用改为**真实 OSM 测量**（``data/osm_attrs`` 产出物），
    而不是手绘的高斯团块；``source="synthetic"`` 保持原行为不变（默认）。

    **犯罪层仍是合成的**：北京没有公开的细粒度犯罪数据，故即便土地利用是真的，
    ``crime`` 也只是土地利用的函数 + 噪声。论文里必须分开写这两件事，
    不能因为土地利用变真了就顺势说"城市属性都是真实的"。
    """
    key = (dataset, nx, ny, source)
    if key in _URBAN_CACHE:
        return _URBAN_CACHE[key]

    if source == "osm":
        from ..data.osm_attrs import available, land_use_grid, coverage

        if not available():
            raise FileNotFoundError(
                "source='osm' 但属性层产出物不存在；先跑 "
                "data/osm_attrs/build_osm_attrs.py，或改用 source='synthetic'")
        if nx != ny:
            raise ValueError(
                "source='osm' 目前只支持方形网格（属性层按 n×n 构建），"
                "收到 %dx%d；grab 布局的 %dx%d 这类非方网格请用 source='synthetic'"
                % (nx, ny, nx, ny))
        land_use = land_use_grid(nx)
        # 真实数据本身就是混合用地，直接用 argmax，不再套 TARGET_SHARE 的配额，
        # 否则会把实测占比改写成设计占比。
        category = np.argmax(land_use, axis=2).astype(int)
        crime = _crime_field(nx, ny, land_use, seed=9001 + nx * 31 + ny)
        peak = float(crime.max()) or 1.0
        grid = UrbanGrid(nx=nx, ny=ny, dataset=dataset, land_use=land_use,
                         category=category, crime=crime, crime_norm=crime / peak)
        grid.coverage = coverage(nx)
        _URBAN_CACHE[key] = grid
        return grid

    layout = _TDRIVE_LAYOUT if dataset == "tdrive" else _GRAB_LAYOUT
    score = _field(nx, ny, layout)
    category = _assign_categories(score, seed=(1000 if dataset == "tdrive" else 2000) + nx * 31 + ny)
    land_use = score / np.maximum(1e-9, score.sum(axis=2, keepdims=True))
    if dataset == "tdrive":
        crime = _crime_field(nx, ny, land_use, seed=9001 + nx * 31 + ny)
    else:  # 附录 B.1：Grab-Posisi 区域没有可靠犯罪数据，只有土地利用
        crime = np.zeros((nx, ny))
    peak = float(crime.max()) or 1.0
    crime_norm = crime / peak
    grid = UrbanGrid(nx=nx, ny=ny, dataset=dataset, land_use=land_use, category=category,
                     crime=crime, crime_norm=crime_norm)
    _URBAN_CACHE[key] = grid
    return grid
