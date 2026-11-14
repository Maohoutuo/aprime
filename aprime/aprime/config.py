# -*- coding: utf-8 -*-
"""A′ 的自由度集中地。

与 ``mapus.MAPUSConfig`` 的关系：满意度函数 S(·) 完全复用 MAPUS 的 PSS 口径
（``mu / pss_prior / pss_scale``，见 ``mapus.env.metrics_obj.PathSatisfaction``），
本配置只新增 A′ 特有的三个量：**个性化强度 θ**、**预算口径**、**候选池大小 k**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mapus.config import DEFAULT_MAPUS


@dataclass
class APrimeConfig:
    #: 个性化强度 θ_u（方案 §2.1 的硬约束下界）。标量为默认值，可为每个参与者
    #: 覆盖（``per_worker_theta``），以支持 E3.1 里"低能力子群 θ 更严"的构造。
    theta: float = 0.0
    per_worker_theta: Optional[dict] = None

    #: θ 的**尺度口径**。这是跨数据集比较的前提，别忽略：
    #:
    #: * ``"absolute"``（历史默认）—— 合法条件是 ``S_u(r) ≥ θ``，θ 是**绝对**满意度下限。
    #:   满意度尺度一旦因数据源不同而整体漂移，同一个 θ 的严格程度就完全不同。
    #:   实测：换成 OSM 实测用地后满意度水平下移约 0.17–0.26，同样 θ=0.6
    #:   在合成数据上剩 59/60 人合法、在实测数据上只剩 36/60——**两者不可比**。
    #: * ``"relative"`` —— 合法条件改为 ``S_u(r) ≥ θ · max_r S_u(r)``，即
    #:   "达到本人最佳满意度的 θ 倍"。θ 变成无量纲比例，对每条路线满意度的
    #:   整体平移/缩放不变，因此**跨数据源可比**。
    #:
    #: 方向保持一致（θ 越大越严），故两口径可互换使用而不改实验脚本的 θ 网格。
    #: 由于 ``θ_u = θ·best_u ≤ best_u``（θ ≤ 1），relative 口径下**每个候选池非空的
    #: 工作者在 θ ≤ 1 时必然仍有可行解**——这是 absolute 口径没有的性质。
    theta_mode: str = "absolute"

    #: 预算口径。方案 §2.1 写 ``|S(Ω)| ≤ B``（人数），但论文表 1 的 B 是**步数**预算
    #: （40/60/100 对应 20/40/60 位参与者，选中 4–6 人）。两种都实现：
    #:   "steps" —— Σ|r_u| ≤ B，与 agentsense/MAPUS 的表 1 口径一致（默认）
    #:   "count" —— |Ω| ≤ B，即方案原文的写法
    budget_kind: str = "steps"
    budget: Optional[float] = None  # None → 用 instance.budget

    #: 候选池大小 k = |C_u|（方案 §2.4）。**候选池与 θ 无关**，θ 扫 10 档复用同一份，
    #: 这是方案 §7.4「成本砍 60%」的前提，也是 E1.1 曲线可比的前提。
    k_cand: int = 12

    #: PSS 口径（与 mapus 的标定一致，见 mapus/docs/calibration.md）
    mu: float = DEFAULT_MAPUS.mu
    pss_prior: float = DEFAULT_MAPUS.pss_prior
    pss_scale: float = DEFAULT_MAPUS.pss_scale

    def theta_of(self, wid: str) -> float:
        if self.per_worker_theta and wid in self.per_worker_theta:
            return float(self.per_worker_theta[wid])
        return float(self.theta)

    def validate(self) -> None:
        """口径拼错时要早失败，而不是静默退回绝对口径给出错误结论。"""
        if self.theta_mode not in ("absolute", "relative"):
            raise ValueError("unknown theta_mode %r; expected 'absolute' or 'relative'"
                             % (self.theta_mode,))

    def budget_of(self, instance) -> float:
        return float(self.budget if self.budget is not None else instance.budget)


#: E1.1 的 θ 扫描档位（方案 §4.1：θ ∈ [0,1] 取 10 档）
THETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DEFAULT = APrimeConfig()
