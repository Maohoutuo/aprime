# -*- coding: utf-8 -*-
"""方案缺的一环：**基数预算需要一个“每个人要价”的激励层**。

现状：`APrimeConfig.budget_kind = "count"`（方案原文 `|S(Ω)| ≤ B`）本身代码里有了，
可 `B` 取多少没有依据——`tdrive_small` 的预算是 40 步，拿 40 作**人数**预算，对 20 人的
实例等于不限；把它当"名额"又凭空捏造，不是论文该干的。

本模块给 `count` 口径提供一个**可解释、可扩展、不把合成数据当真**的推导：

    pay_u = base + per_cell·dist(origin,dest) + per_step·t_window       [dist_cost]
    flat:   pay_u = mean_u(dist_cost)                                    [everyone same price]
    supply: pay_u = dist_cost_u · mult_u(路口附近商业/机构 POI 密度)    [补齐均值]

三个模型的**总购买力被锁定相同**（flat 和 supply 做均值重归一化），所以在同一张
网格上跑完，差出的全部是"选人均匀 vs 按距离 vs 按地段需求"这条结构性差异——
背包因子是多大，就量在 `count_budget()` 里，不掩盖。

没有任何一个模型声称是观测到的计价法：T-Drive 没有支付字段，gMission/EverySender
的坐标是归一化平面、无法和这里的栅格直接对照。这套层的出身是**模型假设层**，
结果文件里会如实写进 `provenance.models`。
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mapus.config import LAND_USE
from agentsense.env.model import Instance, manhattan

try:  # 兼容两种包路径：直接执行 vs 作为 aprime.synth 导入
    from aprime.provenance import KIND_SYNTHETIC
except ImportError:  # pragma: no cover - 兜底
    KIND_SYNTHETIC = "synthetic"

#: 土地利用下标（mapus/config.py：vegetation, industrial, institutional, medical,
#: residential, commercial）——需求密度用商业与机构占比，两者都意味着“这一格有人需要传感”
COMMERCIAL = 5
INSTITUTIONAL = 2

VALID_MODELS = ("flat", "dist_cost", "supply_demand")

#: 环境放行开关（与 aprime.provenance 独立；严格更小的权限范围）。
_ALLOW_ENV = "APRIME_ALLOW_SYNTHETIC_IN_REAL"


@dataclass(frozen=True)
class IncentiveParams:
    """每种模型共享的参数；同一组参数下三模型总购买力一致。"""

    base: float = 2.0            # 起步价：每人出现即支付，与"选几个人"呈线性
    per_cell: float = 0.45       # 按曼哈顿距离计价（0–14 格左右）
    per_step: float = 0.10       # 按时间窗长度计价（人被占用时长）
    demand_gamma: float = 0.6    # supply_demand 中地段溢价强度（0 = 回退 dist_cost）
    demand_floor: float = 0.05   # 分母平滑，避免偏远格被放大到无穷
    min_mult: float = 0.5        # 溢价系数下限
    max_mult: float = 2.0        # 溢价系数上限
    rate: float = 1.0            # 每步产出（名额口径下为 1；步数口径下与步数挂钩）
    jitter_sigma: float = 0.0    # 乘性异质性（log-normal），>0 后非确定性


@dataclass
class IncentiveLayer:
    model: str
    params: IncentiveParams
    #: 逐人要价（货币单位，与 instance.budget 同量纲——按“按步数预算当钱花”口径）
    pay: Dict[str, float]
    #: 逐人产出（每步）；与 pay 不绑定，供未来扩展的“质量加权覆盖”口径用
    rate: Dict[str, float]
    #: 每人至多选一条（A′ 的路线级前提）
    capacity: Dict[str, int]
    #: 实验中真实使用的钱预算（默认 = instance.budget，见 count_budget）
    money_budget: float = 0.0
    #: 供需溢价中用到的地段需求（供 audit / 论文附录）
    demand: Dict[str, float] = field(default_factory=dict)
    #: 随机种子（jitter_sigma > 0 时有意义）
    seed: int = 0

    def is_uniform(self) -> bool:
        """flat 模型：人人同价，背包退化成纯基数。"""
        return self.model == "flat"

    def describe(self) -> str:
        return "%s(base=%.2f, per_cell=%.2f, per_step=%.2f, gamma=%.2f, seed=%d)" % (
            self.model, self.params.base, self.params.per_cell,
            self.params.per_step, self.params.demand_gamma, self.seed)


def _worker_instance(instance) -> Instance:
    """MapusInstance（有 urban）或裸的 Instance（无 urban，需求密度按均值补）。"""
    base = getattr(instance, "base", instance)
    if not isinstance(base, Instance):
        raise TypeError("期望带 `.base` 的 MapusInstance / Instance，收到 %r" % type(instance))
    return base


def _urban_of(instance) -> Optional[object]:
    """MapusInstance 上可读到 urban；裸 Instance 则没有，返回 None。"""
    return getattr(instance, "urban", None)


def _demand_density(instance) -> np.ndarray:
    """逐格需求密度 = 商业 + 机构土地利用占比，归一化到均值 1。

    若传入裸 Instance（无城市属性），退化为全 1：等于不给供需溢价任何空间信息，
    与 supply_demand 的设计一致——要地段差异，就必须把 OSM 属性层塞进去。
    """
    urban = getattr(instance, "urban", None)
    if urban is None:
        return np.ones((1, 1), dtype=np.float64)
    lu = urban.land_use  # (nx, ny, 6)
    d = lu[:, :, COMMERCIAL] + lu[:, :, INSTITUTIONAL]
    mean = float(d.mean())
    return d / max(mean, 1e-9)


def make_incentive_layer(instance, model: str = "dist_cost",
                         params: Optional[IncentiveParams] = None,
                         seed: int = 0,
                         money_budget: Optional[float] = None) -> IncentiveLayer:
    """为实例里每个参与者算一张要价表。

    参数
    ----
    instance : MapusInstance / Instance
    model : "flat" | "dist_cost" | "supply_demand"
    params : 见 IncentiveParams（None 用默认；不同模型共享同一组，保证总购买力可比）
    seed : jitter 种子（jitter_sigma > 0 时使用）
    money_budget : 货币预算；None 时按 ``instance.budget``（把步数预算当钱花）
    """
    if model not in VALID_MODELS:
        raise ValueError("未知的激励模型 %r；可用 %s" % (model, VALID_MODELS))
    inst = _worker_instance(instance)
    p = params or IncentiveParams()
    rng = random.Random(seed)

    pay: Dict[str, float] = {}
    rate: Dict[str, float] = {}
    capacity: Dict[str, int] = {}
    demand: Dict[str, float] = {}
    demand_field = _demand_density(instance)

    for w in inst.workers:
        dist = manhattan(w.origin, w.dest)
        win = w.window
        raw = p.base + p.per_cell * dist + p.per_step * win
        if p.jitter_sigma > 0:
            raw *= math.exp(rng.gauss(0.0, p.jitter_sigma))
        pay[w.wid] = raw
        rate[w.wid] = p.rate
        capacity[w.wid] = 1
        gx, gy = w.origin
        demand[w.wid] = float(demand_field[gx, gy]) if 0 <= gx < demand_field.shape[0] \
            and 0 <= gy < demand_field.shape[1] else 1.0

    if model == "supply_demand" and len(pay) > 0:
        mult: Dict[str, float] = {}
        values = list(demand.values())
        dmean = float(np.mean(values)) if values else 1.0
        for wid, d in demand.items():
            m = 1.0 + p.demand_gamma * (d - dmean) / (dmean + p.demand_floor)
            m = min(max(m, p.min_mult), p.max_mult)
            mult[wid] = m
        # 均值重归一化：把乘数的系统性平移收回来，使三模型总购买力完全一致
        scale = float(np.mean(list(pay.values()))) / max(
            float(np.mean([pay[w] * mult[w] for w in pay])), 1e-9)
        for wid in pay:
            pay[wid] *= scale * mult[wid]
    elif model == "flat":
        mean_pay = float(np.mean(list(pay.values())))
        for wid in pay:
            pay[wid] = mean_pay

    return IncentiveLayer(
        model=model, params=p, pay=pay, rate=rate, capacity=capacity,
        money_budget=float(money_budget if money_budget is not None else inst.budget),
        demand=demand, seed=seed,
    )


# --------------------------------------------------------------------- 预算换算：名额 ≠ 步数
@dataclass
class PurchasePower:
    """给定钱预算下，能买下多少名额的上下界——差距本身就是**背包因子**。"""

    k_min: int
    k_max: int
    mean_pay: float
    max_pay: float
    money_budget: float
    backpack_factor: float  # k_max / k_min；=1 说明纯基数（flat）
    model: str

    def to_dict(self) -> Dict[str, float]:
        return {
            "k_min": self.k_min, "k_max": self.k_max, "mean_pay": self.mean_pay,
            "max_pay": self.max_pay, "money_budget": self.money_budget,
            "backpack_factor": round(self.backpack_factor, 3),
            "model": self.model,
        }


def count_budget(layer: IncentiveLayer) -> PurchasePower:
    """钱预算下能买的名额区间 [k_min, k_max]。

    * ``k_max``：最乐观——每次买当下最便宜的人；
    * ``k_min``：最坏——任意 k 个人的总价都不超支，即 ``k * max_pay <= M``；
    * 两者相等（flat 模型）时退化为纯基数预算，定理 2(a) 的 ``½`` 可直接用；
    * 两者不等时，``k_max / k_min`` 就是“把基数预算换成带权预算”带来的松弛比例，
      论文里该数是报价的一部分，不能省。
    """
    M = layer.money_budget
    pays = sorted(layer.pay.values())
    k_max = 0
    acc = 0.0
    for p in pays:
        if acc + p <= M + 1e-9:
            acc += p
            k_max += 1
        else:
            break
    k_min = int(math.floor(M / max(pays))) if pays else 0
    mean_pay = float(np.mean(pays)) if pays else 0.0
    max_pay = float(max(pays)) if pays else 0.0
    factor = (k_max / k_min) if k_min > 0 else float("inf")
    return PurchasePower(k_min=k_min, k_max=k_max, mean_pay=mean_pay,
                         max_pay=max_pay, money_budget=M,
                         backpack_factor=factor, model=layer.model)


def recommended_k(layer: IncentiveLayer, fraction: float = 0.5) -> int:
    """在 [k_min, k_max] 里选一个保守估计（默认取最坏情况，即 k_min）。"""
    pp = count_budget(layer)
    return max(1, pp.k_min if pp.k_min > 0 else pp.k_max // 2)


# ------------------------------------------------- 与步数预算对齐（消掉自由尺度）
def quota_equivalent(budget_steps: float, mean_route_steps: float,
                     min_k: int = 1) -> int:
    """步数预算 ``B`` 在“平均一条路线占 ``mean_route_steps`` 步”下等价于几个名额。

    这是把两套预算口径接起来的**唯一有依据**的换算：既有的 40/60/100 步预算
    实际选出 4–6 人（README 已记录），除以平均路线长度正好落回这个量级。
    不做这一步而随手写 ``budget=40`` 去当名额，就是给 20 人的实例设了个不限。
    """
    if mean_route_steps <= 0:
        raise ValueError("mean_route_steps 必须为正")
    return max(int(min_k), int(round(float(budget_steps) / float(mean_route_steps))))


def match_money_to_quota(layer: IncentiveLayer, k: int, mode: str = "worst") -> float:
    """反解出一个钱预算 M，使它能买下 ``k`` 个名额。

    * ``mode="worst"``（默认）：``M = k · max_pay``，保证**任意** k 个人都买得起，
      于是 ``count_budget(layer).k_min == k``，口径最保守、与步数预算可比；
    * ``mode="mean"``：``M = k · mean_pay``，期望意义下买得起 k 个。

    两种都只影响“钱预算取多大”，不改变谁贵谁便宜的**结构**；
    结构才是本模块要提供的量（见模块文档）。
    """
    if mode not in ("worst", "mean"):
        raise ValueError("mode 必须是 'worst' 或 'mean'")
    pays = list(layer.pay.values())
    if not pays:
        return 0.0
    k = max(0, min(int(k), len(pays)))
    unit = max(pays) if mode == "worst" else float(np.mean(pays))
    return float(k) * float(unit)


# --------------------------------------------------------------------- 可读摘要
def model_summary(layers: Sequence[IncentiveLayer]) -> str:
    lines = ["激励层（同一组参数下总购买力相等，差异为结构性，非尺度性）:"]
    for L in layers:
        pp = count_budget(L)
        lines.append("  %-13s pay∈[%.2f, %.2f]  mean=%.2f  k∈[%d, %d]  "
                     "backpack=%.2f  uniform=%s" % (
                         L.describe(), min(L.pay.values()), max(L.pay.values()),
                         pp.mean_pay, pp.k_min, pp.k_max, pp.backpack_factor,
                         L.is_uniform()))
    return "\n".join(lines)


def provenance_block(instance, models: Optional[Mapping[str, Any]] = None) -> dict:
    """产出 `aprime.provenance.stamp` 可直接吞下的 models/assumed 块。"""
    from aprime.provenance import provenance_of

    prov = provenance_of(instance)
    return {
        "models": dict(models or {}),
        "assumed": {
            "incentive_layer_is_assumed": True,
            "pay_source": "no public payment field in T-Drive; modeled by IncentiveParams",
            "urban_source": prov.urban_label,
            "profiles_source": prov.profiles_label,
        },
    }
