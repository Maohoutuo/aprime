# -*- coding: utf-8 -*-
"""密度加压实例：让**更多参与者竞争同一片栅格**。

Day-1 实测发现：20–60 人 / 预算 40–60 步的规模下覆盖几乎等于"填满预算"
（`coverage ≈ budget`），路线空间重叠地板只有 0.02–0.09，于是
"个性化 ↑ → 冗余 ↑ → α ↓ → 近似比 ↓" 这条链条**没有肉**。

方案 E1.2 的做法是把 workers 从 500 扩到 5,000（密度天然升高）。在那套批量管线就绪之前，
这里用同一张 `SETTINGS` 栅格（默认 8×8）+ 更大参与者数来**人为造出同样的密度**：
网格不变、时间窗不变、速度口径不变，只把 `n_workers` 放大。

与研究口径的关系：这只是"压力测试实例"，**不要**把它当作 T-Drive/gMission 的规模复现；
结果文件会带 ``source="dense(...)"`` 标签。
"""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

from agentsense.config import SETTINGS
from agentsense.data.instances import tdrive_workers
from agentsense.env.model import Instance

from mapus.data.instances import (
    MapusInstance,
    REAL_TRAJECTORY_SOURCES,
    mapus_workers,
)
from mapus.env.urban import make_urban_grid, urban_note
from mapus.profiles import build_profiles


def make_dense_instance(dataset: str = "tdrive", scale: str = "small",
                        n_workers: int = 160, budget: Optional[float] = None,
                        slack: float = 0.55, seed: int = 0, index: int = 0,
                        profile_seed: int = 20260325,
                        source: str = "synthetic",
                        urban_source: str = "synthetic",
                        crime_source: str = "synthetic",
                        pref_source: str = "archetype") -> MapusInstance:
    """同栅格、同时间窗、同速度口径，只放大参与者数量。

    ``source="tdrive"`` 时参与者改为**真实 T-Drive 出行段**（``t000`` 起编号），
    栅格/时槽/速度口径不变。这能消掉"参与者是随机采样"这一条最弱的假设；
    城市属性可用 OSM 实测（``urban_source="osm"``），偏好可用显示性暴露
    （``pref_source="revealed"``，起终点走廊 × 实测用地）。``meta["source"]``、
    ``urban_note`` 与 ``pref_note`` 如实标注实际使用的口径；**无公开偏好调查数据，
    不编造偏好向量**。

    ``source="geolife"`` 同理，改用 GeoLife 1.3 归一化后的真实轨迹
    （见 ``mapus.data.instances.GEOLIFE_PATH``）——用于 Grab-Posisi 拿不到时的
    跨数据集验证替代，**不要**与 T-Drive 的结果混在一张表里当同一数据源。

    ``urban_source="osm"`` 把土地利用从手绘高斯团块换成 **OSM 实测**
    （见 ``mapus.data.osm_attrs``）。``crime_source="zhang2025"`` 再把犯罪层换成
    Zhang et al. 2025 北京犯罪栅格（``data/crime/crime_grid_<n>_beijing.csv.gz``，
    已按附录 B.1 剔除 <10 条格子），仅支持 tdrive 的 8/16/32 方网格。
    两者独立，可混合：source="osm" + crime_source="zhang2025" 即**土地利用与犯罪均为实测**。
    非方网格（如 grab 的 8x4）的犯罪层保持合成，会报错。
    """
    base_setting = SETTINGS[(dataset, scale)]
    setting = dataclasses.replace(base_setting, n_workers=int(n_workers))
    rng = random.Random(seed * 7919 + index)
    if source in REAL_TRAJECTORY_SOURCES:
        path = REAL_TRAJECTORY_SOURCES[source]()
        if path is None:
            raise FileNotFoundError(
                "source=%r 但对应的真实轨迹文件不存在；请设环境变量或改用 "
                "source='synthetic'" % source)
        workers = tdrive_workers(setting, rng, path=path)
        worker_src = "%s(real)" % source
    else:
        workers = mapus_workers(setting, rng, slack=slack)
        worker_src = f"synthetic(slack={slack})"

    pref_note = ("pref=archetype(手写)" if pref_source == "archetype"
                 else "pref=revealed(起终点走廊×OSM)")
    base = Instance(
        setting_name=f"{dataset}_{scale}_dense{n_workers}",
        index=index,
        grid=(setting.grid_x, setting.grid_y),
        n_slots=setting.n_slots,
        budget=float(base_setting.total_budget if budget is None else budget),
        workers=workers,
        seed=seed,
        meta={
            "dataset": dataset,
            "scale": scale,
            "source": f"dense(n_workers={n_workers}, workers={worker_src}, "
                      f"grid={setting.grid_x}x{setting.grid_y}, slots={setting.n_slots})",
            "worker_source": worker_src,
            "slot_minutes": setting.slot_min,
            "horizon_minutes": setting.horizon_min,
            "cells_per_step": int(setting.cells_per_step),
            "cells_per_step_source": ("tdrive_measured" if dataset == "tdrive"
                                      else "proxy(tdrive @same grid/slot)"),
            "density_workers_per_cell": round(n_workers / (setting.grid_x * setting.grid_y), 3),
            "urban_source": urban_source,
            "crime_source": crime_source,
            "pref_source": pref_source,
            "urban_note": urban_note(urban_source, crime_source),
            "pref_note": pref_note,
        },
    )
    urban = make_urban_grid(dataset, setting.grid_x, setting.grid_y,
                            source=urban_source, crime_source=crime_source)
    profiles = build_profiles(base, seed=profile_seed + index, urban=urban,
                              pref_source=pref_source)
    return MapusInstance(base=base, urban=urban, profiles=profiles,
                         setting=setting, task_index=index)
