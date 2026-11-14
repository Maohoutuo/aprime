# -*- coding: utf-8 -*-
r"""φ 的可测代理：人群偏好同质性到底怎么从数据里算出来。

定理 2 的 θ\*(φ) 版里 φ 不能是个"实验室旋钮"，必须在真实数据上可估计。这里给三个
候选统计量（都可以从画像 / POI 签到 / 问卷得到）：

| 统计量 | 定义 | 方向 |
|---|---|---|
| ``phi_cos`` | 参与者两两偏好向量的**平均余弦** | 越大越同质（φ=1 时为 1） |
| ``top1_share`` | 每人最强偏好类别的份额的平均 | 越大越集中在单一类别 |
| ``entropy_norm`` | 个体偏好熵的均值 / log(C) | 越小越集中 |

**推荐用 ``phi_cos``**：它对"大家偏好是否指向同一方向"最直接，且与 `homogeneity.py`
的凸组合旋钮是**单调对应**的（`tests/test_aprime.py::test_phi_proxy_monotone_in_knob` 守着这条）。

注意：`phi_cos` 与 φ 的**绝对值**不必相等（凸组合的几何平均效应），理论里只用它的
**序**（谁更同质）以及"φ̂ 越大 → ρ(θ) 的斜率越陡"这个方向性陈述，不要写 `φ = phi_cos`。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def pref_concentration(profiles: Dict[str, object]) -> dict:
    """从画像字典算 φ 的三个代理量。"""
    prefs = np.array([np.asarray(p.pref, dtype=np.float64) for p in profiles.values()])
    n = len(prefs)
    if n < 2:
        return {"phi_cos": 1.0, "top1_share": 1.0, "entropy_norm": 0.0, "n": n}
    prefs = prefs / prefs.sum(axis=1, keepdims=True)   # 概率单纯形（和为 1）
    unit = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)   # L2 归一化后才能算余弦
    gram = unit @ unit.T
    iu = np.triu_indices(n, 1)
    phi_cos = float(gram[iu].mean())
    top1 = float(prefs.max(axis=1).mean())
    ent = -(prefs * np.log(prefs + 1e-12)).sum(axis=1).mean()
    return {
        "phi_cos": round(phi_cos, 4),
        "top1_share": round(top1, 4),
        "entropy_norm": round(float(ent / np.log(prefs.shape[1])), 4),
        "n": n,
    }
