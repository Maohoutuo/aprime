# -*- coding: utf-8 -*-
"""参与者画像（论文附录 B.2，图 10 的模板）。

五个 archetype 各自带一个土地利用偏好向量与人群先验；个体的偏好 = archetype
向量 + 小幅扰动后归一化。画像同时生成自然语言 description，用于提示词。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import LAND_USE
from agentsense.env.model import Instance


# 偏好向量顺序 = LAND_USE = (vegetation, industrial, institutional, medical, residential, commercial)
ARCHETYPES: Dict[str, dict] = {
    "eco_enthusiast": {
        "pref": (0.46, 0.02, 0.07, 0.03, 0.27, 0.15),
        "age_group": {"young": 0.4, "middle_aged": 0.45, "senior": 0.15},
        "economic": {"poor": 0.15, "low": 0.25, "middle": 0.42, "high": 0.13, "wealthy": 0.05},
        "hobbies": ["hiking", "gardening", "birdwatching", "cycling", "photography"],
        "text": ("prefers green spaces and quiet neighbourhoods; avoids congested and polluted areas; "
                 "often takes detours through parks if the time budget allows"),
    },
    "city_dweller": {
        "pref": (0.07, 0.05, 0.05, 0.04, 0.39, 0.40),
        "age_group": {"young": 0.5, "middle_aged": 0.4, "senior": 0.1},
        "economic": {"poor": 0.08, "low": 0.18, "middle": 0.40, "high": 0.24, "wealthy": 0.10},
        "hobbies": ["shopping", "dining out", "live music", "fitness", "networking"],
        "text": ("favours dense residential and commercial districts; likes lively streets and shops; "
                 "is time-conscious and dislikes congestion"),
    },
    "industrial_worker": {
        "pref": (0.06, 0.48, 0.10, 0.04, 0.24, 0.08),
        "age_group": {"young": 0.25, "middle_aged": 0.55, "senior": 0.20},
        "economic": {"poor": 0.28, "low": 0.38, "middle": 0.27, "high": 0.06, "wealthy": 0.01},
        "hobbies": ["repairing machines", "fishing", "watching sports", "cooking", "card games"],
        "text": ("frequently operates near industrial zones; cares about pay for the sensing shift and "
                 "prefers familiar working areas"),
    },
    "community_helper": {
        "pref": (0.12, 0.04, 0.24, 0.22, 0.28, 0.10),
        "age_group": {"young": 0.15, "middle_aged": 0.45, "senior": 0.40},
        "economic": {"poor": 0.18, "low": 0.30, "middle": 0.38, "high": 0.11, "wealthy": 0.03},
        "hobbies": ["volunteering", "visiting clinics", "walking", "reading", "community events"],
        "text": ("prefers regions near hospitals, schools and community facilities; willing to help but "
                 "needs the route to stay close to family obligations"),
    },
    "explorer": {
        "pref": (0.17, 0.17, 0.16, 0.17, 0.17, 0.16),
        "age_group": {"young": 0.55, "middle_aged": 0.35, "senior": 0.10},
        "economic": {"poor": 0.12, "low": 0.22, "middle": 0.38, "high": 0.20, "wealthy": 0.08},
        "hobbies": ["geocaching", "urban sketching", "podcast recording", "travelling", "gaming"],
        "text": ("exhibits balanced preferences across different regions; enjoys novelty and is willing "
                 "to explore unfamiliar streets"),
    },
}

ECONOMIC_STATUS = ("poor", "low", "middle", "high", "wealthy")

#: 人群构成先验：各 archetype 在人群中的份额。
#:
#: **诚实声明**：以下所有 ``source`` 都标为 ``unvalidated``——本工程**没有任何公开
#: 偏好调查数据**支撑这些权重，它们是设计选择而非观测值。``uniform`` 是原实现的
#: 行为（``rng.choice`` 等概率），保留为默认值以保证历史结果可复现。
#:
#: 要把它变成**有依据**的先验，需要的是（按优先级）：
#:   1. 北京/新加坡的**居民出行或活动调查**（给出通勤/购物/休闲的份额）→ 映射到
#:      ``city_dweller`` / ``community_helper`` / ``explorer``；
#:   2. **职业构成普查**（第二产业就业占比）→ 映射到 ``industrial_worker``；
#:   3. **环境态度分段调查**（latent class 的各类占比）→ 映射到 ``eco_enthusiast``。
#: 在此之前，任何声称"符合真实人群"的表述都不成立；能主张的只有
#: ``experiments/run_pref_mixture_sensitivity.py`` 给出的**敏感性界**。
PREF_MIXTURES: Dict[str, dict] = {
    "uniform": {
        "weights": {"eco_enthusiast": 0.20, "city_dweller": 0.20, "industrial_worker": 0.20,
                    "community_helper": 0.20, "explorer": 0.20},
        "source": "unvalidated(uniform) — 原实现，等概率抽样，非调查数据",
    },
    # 下面四个是"方向上可能更像真实城市"的替代先验，**同样未经调查验证**，
    # 仅用于敏感性分析：若结论在这些先验下都不变，则缺失的调查数据不构成威胁。
    "eco_heavy": {
        "weights": {"eco_enthusiast": 0.40, "city_dweller": 0.15, "industrial_worker": 0.15,
                    "community_helper": 0.15, "explorer": 0.15},
        "source": "unvalidated(eco_heavy) — 假设环境态度者占 40%，无调查依据",
    },
    "city_heavy": {
        "weights": {"eco_enthusiast": 0.15, "city_dweller": 0.40, "industrial_worker": 0.15,
                    "community_helper": 0.15, "explorer": 0.15},
        "source": "unvalidated(city_heavy) — 假设商业/居住偏好者占 40%，无调查依据",
    },
    "industrial_heavy": {
        "weights": {"eco_enthusiast": 0.15, "city_dweller": 0.15, "industrial_worker": 0.40,
                    "community_helper": 0.15, "explorer": 0.15},
        "source": "unvalidated(industrial_heavy) — 假设第二产业就业者占 40%，无调查依据",
    },
    "community_heavy": {
        "weights": {"eco_enthusiast": 0.15, "city_dweller": 0.15, "industrial_worker": 0.15,
                    "community_helper": 0.40, "explorer": 0.15},
        "source": "unvalidated(community_heavy) — 假设照护/社区型占 40%，无调查依据",
    },
    # ---- 半标定：唯一一个有权重来源的先验（2026-11-12）----
    # 见 ``data/context/README.md`` 与 ``mapus/docs/calibration.md`` §4。
    "calibrated_bj2022": {
        "weights": {"eco_enthusiast": 0.2090, "city_dweller": 0.2090,
                    "industrial_worker": 0.1640, "community_helper": 0.2090,
                    "explorer": 0.2090},
        "source": ("半标定(bj2022)：industrial_worker=0.164 来自《北京统计年鉴 2023》"
                   "表 3-9「常住就业人口规模及构成」2022 年第二产业构成 16.4%"
                   "（1132.1 万常住就业人口中 186 万）；其余 4 项等分 0.836，"
                   "**仍无调查依据**。仅完成 1/5 个边缘约束，联合与独立性假设未解除。"),
    },
    "calibrated_bj2022_firm": {
        "weights": {"eco_enthusiast": 0.2115, "city_dweller": 0.2115,
                    "industrial_worker": 0.1540, "community_helper": 0.2115,
                    "explorer": 0.2115},
        "source": ("半标定(bj2022, 法人单位口径)：industrial_worker=0.154 来自表 3-14"
                   "「全市法人单位从业人员」2022 年第二产业合计 159.3 万 / 1052.6 万"
                   "（采矿业 2.6 + 制造业 76.1 + 电力热力燃气水 10.3 + 建筑业 70.3）。"
                   "与 calibrated_bj2022 的 16.4% 差在口径，**两者不能混用**；"
                   "同时给出是为了量化口径选择本身的影响。"),
    },
}


@dataclass
class ParticipantProfile:
    wid: str
    gender: str
    age: int
    age_group: str
    economic_status: str
    hobbies: List[str]
    archetype: str
    description: str
    pref: np.ndarray          # (6,) 归一化土地利用偏好 p_u
    speed: float = 1.0
    # 历史被选中次数 h_u（式 6 的公平分项）
    hired: int = 0

    def to_text(self) -> str:
        prefs = ", ".join(f"{LAND_USE[i]} {self.pref[i]:.2f}" for i in range(len(LAND_USE)))
        return (
            "{\n"
            f"  worker_id: {self.wid},\n"
            f"  gender: {self.gender},\n"
            f"  age: {self.age}, age_group: {self.age_group},\n"
            f"  economic_status: {self.economic_status},\n"
            f"  hobbies: [{', '.join(self.hobbies)}],\n"
            f"  type: {self.archetype},\n"
            f"  description: {self.description},\n"
            f"  preferences: {{{prefs}}},\n"
            f"  speed: {self.speed}\n"
            "}"
        )

    def to_dict(self) -> dict:
        return {"worker_id": self.wid, "gender": self.gender, "age": self.age,
                "age_group": self.age_group, "economic_status": self.economic_status,
                "hobbies": self.hobbies, "type": self.archetype,
                "description": self.description,
                "preferences": {LAND_USE[i]: round(float(self.pref[i]), 4)
                                for i in range(len(LAND_USE))},
                "history_selected": self.hired}


def _pick(rng: random.Random, dist: Dict[str, float]) -> str:
    keys = list(dist)
    return rng.choices(keys, weights=[dist[k] for k in keys], k=1)[0]


def _age_for(rng: random.Random, group: str) -> int:
    return {"young": rng.randint(18, 30),
            "middle_aged": rng.randint(31, 55),
            "senior": rng.randint(56, 75)}[group]


def build_profiles(instance: Instance, seed: int = 20260325, jitter: float = 0.35,
                   mix_uniform: float = 0.05,
                   only: List[str] | None = None,
                   mixture: "str | dict" = "uniform") -> Dict[str, ParticipantProfile]:
    """为实例中的每个 worker 生成画像（确定性：同一 seed 结果一致）。

    ``mix_uniform`` 把 archetype 向量与均匀分布混合：真实人群的偏好通常不会极端
    到"只走绿化带"，适度混合同时让 PSS 落在论文的量级区间（见 docs/calibration.md）。

    ``mixture`` 选 :data:`PREF_MIXTURES` 里的一个人群构成先验，或直接传一个
    形如 ``{"weights": {...}, "source": ...}`` 的字典。默认 ``"uniform"``
    即原实现的等概率抽样；**所有先验都未经调查数据验证**，见该字典的说明。
    """
    rng = random.Random(seed + instance.index * 977 + len(instance.workers))
    names = only or list(ARCHETYPES)
    out: Dict[str, ParticipantProfile] = {}
    uniform = np.full(len(LAND_USE), 1.0 / len(LAND_USE))
    mix = PREF_MIXTURES[mixture] if isinstance(mixture, str) else mixture
    weights = [float(mix["weights"].get(n, 0.0)) for n in names]
    if sum(weights) <= 0:
        raise ValueError("mixture %r 在 %r 上的权重全为 0" % (mixture, names))
    # 等权重时必须走 ``rng.choice``：``rng.choices(weights=...)`` 消耗随机数的
    # 方式不同，会让所有历史结果（含已发表的表）无法逐位复现。
    flat = len(set(weights)) == 1
    for w in instance.workers:
        arche = rng.choice(names) if flat else rng.choices(names, weights=weights, k=1)[0]
        spec = ARCHETYPES[arche]
        group = _pick(rng, spec["age_group"])
        econ = _pick(rng, spec["economic"])
        base = np.array(spec["pref"], dtype=np.float64)
        noisy = np.clip(base + np.array([rng.uniform(-jitter, jitter) for _ in range(len(base))]), 1e-3, None)
        pref = noisy / noisy.sum()
        pref = (1.0 - mix_uniform) * pref + mix_uniform * uniform
        pref = pref / pref.sum()
        out[w.wid] = ParticipantProfile(
            wid=w.wid,
            gender=rng.choice(["male", "female", "non-binary"]),
            age=_age_for(rng, group),
            age_group=group,
            economic_status=econ,
            hobbies=rng.sample(spec["hobbies"], k=rng.randint(2, min(4, len(spec["hobbies"])))),
            archetype=arche,
            description=spec["text"],
            pref=pref,
            speed=float(instance.meta.get("cells_per_step", 1)),
        )
    return out
