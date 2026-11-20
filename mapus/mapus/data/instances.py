# -*- coding: utf-8 -*-
"""数据层：在 agentsense 的实例生成之上，附加城市属性与参与者画像。

* 实例（工作者、预算、网格、时域）与 agentsense 共用同一套 ``SETTINGS``
  （== MAPUS 表 1），以保证 6 个经典基线与已验证的标定口径可比；
* 每个实例额外挂一个 ``UrbanGrid``（土地利用 + 犯罪）与一组成员画像。
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from agentsense.config import SETTINGS, Setting
from agentsense.data.instances import instance_seed, make_instance, sample_endpoints, synthetic_workers
from agentsense.env.model import Instance, Worker

from ..env.urban import UrbanGrid, make_urban_grid, urban_note
from ..profiles import ParticipantProfile, build_profiles

#: 真实 T-Drive 清洗文件（由 data/tdrive/prep_tdrive.py 生成）。
#: 图片 1(b) 那类"时间窗内留有大量余量"的轨迹必须来自真实数据，
#: 否则 θ 权衡曲线失去物理含义。可用环境变量覆盖。
TDRIVE_PATH = os.environ.get(
    "TDRIVE_PATH", r"D:\composition\data\tdrive\tdrive_clean.txt.gz")

#: 替代轨迹源：GeoLife 1.3（MSRA），已由 ``data/geolife/normalize_geolife.py``
#: 归一化成与 T-Drive 相同的四列格式 ``tid,ts,lon,lat``。
#:
#: **用途**：Grab-Posisi 拿不到（需邮件授权、属性层未公开）时的真实稠密轨迹替代，
#: 供 CROSS 跨数据集验证与 E0.1 的第二份真实轨迹使用。
#:
#: **口径**：归一化时已按 15 分钟槽重采样到每槽一点，且 ``tid = <用户>_<日期>``。
#: 后者是为了避免 ``docs/e0_1_data_access_plan.md`` §3.2 的 T5「跨天塌陷」——
#: 下游 ``slot`` 不含日分量，GeoLife 跨 2007-04…2012-08 五年，按用户 ID 分组会把
#: 不同天的同一时刻压进同一个 slot。
GEOLIFE_PATH = os.environ.get(
    "GEOLIFE_PATH", r"D:\composition\data\geolife\geolife_norm.txt.gz")


#: Grab-Posisi 官方数据（轨迹 + POI）**尚未获得**（截至 2026-11-10）：轨迹需 Kaggle 授权，
#: POI 论文未公开。因此 ``dataset="grab"`` 的实例全部来自"合成同分布移动性 + 合成城市"，
#: 结果只能作为机制层面的跨数据集验证。拿到官方数据并接入后，把这里置 True，
#: 让结果文件里的 data_source 标签同步改变。
GRAB_OFFICIAL_DATA = False


def _tdrive_source():
    return TDRIVE_PATH if TDRIVE_PATH and os.path.exists(TDRIVE_PATH) else None


def _geolife_source():
    # 运行期读取环境变量，这样 ``$env:GEOLIFE_PATH`` 在 import 之后仍能生效
    # （模块级的 ``GEOLIFE_PATH`` 只是缺省值，不是唯一入口）。
    path = os.environ.get("GEOLIFE_PATH", GEOLIFE_PATH)
    return path if path and os.path.exists(path) else None


#: 轨迹源名 → 解析函数。``make_dense_instance(source=...)`` 据此选真实轨迹文件。
REAL_TRAJECTORY_SOURCES = {
    "tdrive": _tdrive_source,
    "geolife": _geolife_source,
}


def data_source_label(dataset: str) -> str:
    """结果文件里记录的数据来源标签，避免把合成数据误写成官方数据集复现。"""
    if dataset == "grab":
        return "grab_official" if GRAB_OFFICIAL_DATA else "synthetic(grab_layout)"
    return "tdrive(real)" if _tdrive_source() else "synthetic(tdrive_layout)"


def _cps_source(dataset: str) -> str:
    """``cells_per_step`` 的来源标签（实测 / 代理），写进 meta 便于事后审计。"""
    return "tdrive_measured" if dataset == "tdrive" else "proxy(tdrive @same grid/slot)"


@dataclass
class MapusInstance:
    """agentsense 的 Instance + 城市属性 + 画像。所有属性访问代理到 base。"""

    base: Instance
    urban: UrbanGrid
    profiles: Dict[str, ParticipantProfile]
    setting: Setting
    task_index: int = 0

    def __getattr__(self, item):
        return getattr(self.base, item)

    @property
    def prefs(self) -> Dict[str, np.ndarray]:
        return {wid: p.pref for wid, p in self.profiles.items()}


def mapus_workers(setting: Setting, rng: random.Random, slack: float = 0.55) -> List[Worker]:
    """参与者抽样：与 agentsense 同分布，但**保证有剩余步数**。

    论文图 1(b) 的参与者都在时间窗内留有大段余量（例如 16:13 出发、17:54 到达），
    这正是"用 residual steps 插入偏好路点"的前提。若无余量，任何偏好感知的绕行
    都不可行，PRG 也无从发挥作用。因此这里要求"按实测速度折算的步数"不超过时间窗的
    ``slack`` 比例（默认 55%）——注意判据是 ``steps_for(·, cps)`` 而不是曼哈顿距离，
    否则速度 > 1 时会误判为没有余量。
    """
    nx, ny, T = setting.grid_x, setting.grid_y, setting.n_slots
    cps = max(1, int(setting.cells_per_step))
    workers: List[Worker] = []
    for i in range(setting.n_workers):
        min_win = max(3, min(T, int(round(0.6 * T))))
        win = rng.randint(min_win, T)
        t_start = rng.randint(0, T - win)
        origin, dest = sample_endpoints(nx, ny, win, cps, rng,
                                        max_steps_ratio=slack, min_steps_ratio=0.5)
        workers.append(Worker(
            wid=f"w{i:03d}", origin=origin, dest=dest,
            t_start=t_start, t_end=t_start + win - 1, speed=cps))
    return workers


def make_mapus_instance(setting_name: str, index: int = 0, source: str = "synthetic",
                        profile_seed: int = 20260325,
                        urban_source: str = "synthetic",
                        crime_source: str = "synthetic",
                        pref_source: str = "archetype") -> MapusInstance:
    dataset, scale = setting_name.split("_")
    setting = SETTINGS[(dataset, scale)]
    seed = instance_seed(setting, index)
    rng = random.Random(seed)
    used_source = source
    if source == "tdrive":
        base = None
        path = _tdrive_source()
        if path is None:
            print("[instances] T-Drive 清洗文件缺失，回退合成数据", file=sys.stderr)
        else:
            try:
                base = make_instance(setting_name, index, source="tdrive",
                                     tdrive_path=path)
            except Exception as exc:
                print("[instances] T-Drive 加载失败（%s），回退合成数据" % exc,
                      file=sys.stderr)
    else:
        base = None
    if base is None:
        workers = mapus_workers(setting, rng)
        base = Instance(setting_name=setting.name, index=index,
                        grid=(setting.grid_x, setting.grid_y), n_slots=setting.n_slots,
                        budget=float(setting.total_budget), workers=workers, seed=seed,
                        meta={"dataset": dataset, "scale": scale, "source": "synthetic(slack)",
                              "slot_minutes": setting.slot_min,
                              "horizon_minutes": setting.horizon_min,
                              "cells_per_step": int(setting.cells_per_step),
                              "cells_per_step_source": _cps_source(dataset)})
        used_source = "synthetic(slack)"
    urban = make_urban_grid(
        dataset, base.nx, base.ny, source=urban_source, crime_source=crime_source)
    if urban_source != "synthetic" or crime_source != "synthetic":
        base.meta["urban_source"] = urban_source
        base.meta["urban_note"] = urban_note(urban_source, crime_source)
    pref_note = ("pref=archetype(手写)" if pref_source == "archetype"
                 else "pref=revealed(起终点走廊×OSM)")
    base.meta["pref_source"] = pref_source
    base.meta["pref_note"] = pref_note
    profiles = build_profiles(base, seed=profile_seed, urban=urban,
                              pref_source=pref_source)
    return MapusInstance(base=base, urban=urban, profiles=profiles,
                         setting=setting, task_index=index)


def make_pool_instance(dataset: str = "tdrive", scale: str = "medium", pool_size: int = 30,
                       task_index: int = 0, seed: int = 31415,
                       urban_source: str = "synthetic",
                       crime_source: str = "synthetic",
                       pref_source: str = "archetype") -> MapusInstance:
    """公平性实验（图 4）：固定 30 人的人才池 + 60 个不同的传感任务。

    每个"任务"用一张平滑随机的**兴趣区域权重**来体现（``cell_weights`` 参与
    Q 与熵的计算），因此同一批工作者在不同任务上的边际覆盖增益不同，
    选择策略之间的差异才会显现。
    """
    setting = SETTINGS[(dataset, scale)]
    rng = random.Random(seed)
    workers: List[Worker] = mapus_workers(setting, rng, slack=0.55)[:pool_size]
    for i, w in enumerate(workers):
        workers[i] = Worker(wid=f"w{i:03d}", origin=w.origin, dest=w.dest,
                            t_start=w.t_start, t_end=w.t_end, speed=w.speed)
    urban = make_urban_grid(
        dataset, setting.grid_x, setting.grid_y,
        source=urban_source, crime_source=crime_source)
    base = Instance(
        setting_name=setting.name, index=task_index, grid=(setting.grid_x, setting.grid_y),
        n_slots=setting.n_slots, budget=float(setting.total_budget), workers=workers,
        seed=seed, meta={"dataset": dataset, "scale": scale, "source": "synthetic_pool",
                         "slot_minutes": setting.slot_min, "horizon_minutes": setting.horizon_min,
                         "cells_per_step": int(setting.cells_per_step),
                         "cells_per_step_source": _cps_source(dataset)})
    if urban_source != "synthetic" or crime_source != "synthetic":
        base.meta["urban_source"] = urban_source
        base.meta["urban_note"] = urban_note(urban_source, crime_source)
    pref_note = ("pref=archetype(手写)" if pref_source == "archetype"
                 else "pref=revealed(起终点走廊×OSM)")
    base.meta["pref_source"] = pref_source
    base.meta["pref_note"] = pref_note
    # 任务兴趣区域：2~3 个高斯斑块，值域 [0.2, 1.0]
    trng = np.random.default_rng(seed + 7919 * task_index)
    xs = (np.arange(setting.grid_x) + 0.5) / setting.grid_x
    ys = (np.arange(setting.grid_y) + 0.5) / setting.grid_y
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    field = np.zeros_like(gx)
    for _ in range(int(trng.integers(2, 4))):
        cx, cy = trng.uniform(0.15, 0.85), trng.uniform(0.15, 0.85)
        sx, sy = trng.uniform(0.12, 0.28), trng.uniform(0.12, 0.28)
        field += trng.uniform(0.5, 1.0) * np.exp(-(((gx - cx) ** 2) / (2 * sx ** 2)
                                                  + ((gy - cy) ** 2) / (2 * sy ** 2)))
    field = field / (field.max() or 1.0)
    base.meta["cell_weights"] = 0.2 + 0.8 * field
    profiles = build_profiles(base, seed=20260325 + task_index, urban=urban,
                              pref_source=pref_source)
    return MapusInstance(base=base, urban=urban, profiles=profiles, setting=setting,
                         task_index=task_index)
