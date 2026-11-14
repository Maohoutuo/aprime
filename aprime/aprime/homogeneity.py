# -*- coding: utf-8 -*-
"""偏好同质性 φ：密度加压失败后找到的**真正**的冗余来源。

Day-2 实测教训（`results/density_stress_*.md`）：把参与人数从 20 拉到 320、网格与预算不变，
合法候选路线两两重叠 `ρ` 稳定在 0.06 左右、覆盖照样贴满预算。原因很直接——
`mapus_workers` 在全栅格上均匀撒 O–D，人一多候选池就更**多样**，贪心反而更容易挑出
不重叠的 6 条。所以"人多"≠"冗余多"。

真正决定 `ρ(θ)` 的是**偏好是否同质**：θ 越高，路线必须越贴合本人偏好；如果人群偏好分散，
高 θ 仍会把不同人推向不同区域（`ρ` 不升）；如果偏好同质（例如都是生态爱好者 / 都是通勤族），
高 θ 就把所有人推向同一片区域，`ρ` 才会上来、`α` 才会掉。

φ 的定义（凸组合，连续可调）：

    pref_u(φ) = (1 − φ) · pref_u + φ · anchor

φ = 0 为原画像（异质），φ = 1 为完全同质。
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, Optional

import numpy as np

from mapus.profiles import ARCHETYPES, ParticipantProfile


def anchor_pref(archetype: str = "eco_enthusiast", jitter: float = 0.1,
                seed: int = 0) -> np.ndarray:
    """锚点偏好：某个 archetype 的向量 + 小幅抖动后归一化。"""
    rng = random.Random(seed)
    base = np.array(ARCHETYPES[archetype]["pref"], dtype=np.float64)
    noisy = np.clip(base + np.array([rng.uniform(-jitter, jitter) for _ in base]), 1e-3, None)
    return noisy / noisy.sum()


def blend_profiles(profiles: Dict[str, ParticipantProfile], phi: float,
                   anchor: Optional[np.ndarray] = None,
                   archetype: str = "eco_enthusiast",
                   seed: int = 0) -> Dict[str, ParticipantProfile]:
    """把每个参与者的偏好向锚点做凸组合：φ=0 不变，φ=1 完全同质。"""
    phi = float(min(1.0, max(0.0, phi)))
    if phi <= 0.0:
        return dict(profiles)
    anchor = anchor_pref(archetype, seed=seed) if anchor is None else np.asarray(anchor, float)
    out: Dict[str, ParticipantProfile] = {}
    for wid, p in profiles.items():
        pref = (1.0 - phi) * p.pref.astype(np.float64) + phi * anchor
        pref = pref / pref.sum()
        out[wid] = dataclasses.replace(p, pref=pref)
    return out


def with_homogeneity(instance, phi: float, archetype: str = "eco_enthusiast",
                     seed: int = 0):
    """返回一个**新**的实例视图，其画像偏好已按 φ 混合。

    注意不能用 ``copy.copy``：``MapusInstance.__getattr__`` 会把复制过程中的属性访问
    代理到 ``base``，在 `base` 尚未绑定时无限递归。这里显式重建。
    """
    from mapus.data.instances import MapusInstance

    profiles = blend_profiles(instance.profiles, phi, archetype=archetype, seed=seed)
    instance.base.meta["pref_homogeneity"] = float(phi)
    instance.base.meta["pref_anchor"] = archetype
    return MapusInstance(base=instance.base, urban=instance.urban, profiles=profiles,
                         setting=instance.setting,
                         task_index=getattr(instance, "task_index", 0))
