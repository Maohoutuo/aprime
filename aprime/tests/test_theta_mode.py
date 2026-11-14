# -*- coding: utf-8 -*-
"""θ 的两种尺度口径：``absolute`` 与 ``relative``。

守的核心性质：**relative 口径对满意度整体平移/缩放不变**。
这是跨数据集比较成立的前提——实测换数据源会让满意度水平整体下移
（合成 → OSM 实测下移约 0.17–0.26），absolute 口径下同一个 θ 的严格程度
随之剧变，导致"θ=0.6 在 A 数据上剩 59 人、在 B 数据上剩 36 人"这种假差异。
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.config import APrimeConfig  # noqa: E402
from aprime.model import Candidate, SelectionProblem  # noqa: E402


def _c(wid: str, sat: float, tag: str = "") -> Candidate:
    path = ((0, 0, 0),)
    return Candidate(wid=wid, path=path, cells=frozenset({(0, 0)}), sat=sat, tag=tag,
                     space_time=frozenset({(0, 0, 0)}))


def _problem(cands, theta: float, mode: str = "absolute") -> SelectionProblem:
    cfg = APrimeConfig(theta=theta, budget_kind="count", budget=99.0, theta_mode=mode)
    return SelectionProblem(instance=types.SimpleNamespace(budget=99.0),
                            candidates=list(cands), cfg=cfg)


def test_relative_threshold_is_theta_times_own_best():
    """每个人用自己的最佳满意度做基准，不是全局统一的下限。"""
    cands = [_c("u1", 0.90), _c("u1", 0.50), _c("u2", 0.40), _c("u2", 0.10)]
    p = _problem(cands, theta=0.5, mode="relative")
    assert p.theta_of("u1") == pytest.approx(0.45)   # 0.5 × 0.90
    assert p.theta_of("u2") == pytest.approx(0.20)   # 0.5 × 0.40
    # u2 的最佳 0.40 ≥ 0.20 合法、次优 0.10 < 0.20 不合法；
    # u1 两条都合法。注意 u2 的绝对满意度低于 u1，但在这里**不受歧视**——
    # 这正是相对口径的要点，也是 absolute 口径做不到的。
    assert [c.sat for c in p.eligible] == [0.90, 0.50, 0.40]


def test_relative_is_invariant_to_uniform_shift():
    """全部满意度平移后，合法集合必须完全不变 —— 这是可比性的定义。"""
    base = [_c("u1", 0.90), _c("u1", 0.60), _c("u1", 0.30)]
    shifted = [_c("u1", 0.70), _c("u1", 0.40), _c("u1", 0.10)]

    p_base = _problem(base, theta=0.5, mode="relative")
    p_shift = _problem(shifted, theta=0.5, mode="relative")
    assert len(p_base.eligible) == len(p_shift.eligible) == 2

    # 对照组：absolute 口径在同样的平移下结论会变（2 → 1），说明问题真实存在
    a_base = _problem(base, theta=0.5, mode="absolute")
    a_shift = _problem(shifted, theta=0.5, mode="absolute")
    assert len(a_base.eligible) == 2
    assert len(a_shift.eligible) == 1


def test_relative_keeps_every_pool_non_empty_up_to_theta_one():
    """θ·best ≤ best（θ≤1），故每个人的最优路线永远合法。

    absolute 口径没有这个性质：满意度整体偏低时，一个绝对值 θ 会把整池清空。
    """
    cands = [_c("u1", 0.20), _c("u1", 0.05), _c("u2", 0.15), _c("u2", 0.02)]
    # 绝对口径 θ=0.5 会清空两人
    assert _problem(cands, theta=0.5, mode="absolute").eligible_workers() == []
    # 相对口径 θ=1.0 仍保住每人的最优路线
    p = _problem(cands, theta=1.0, mode="relative")
    assert p.eligible_workers() == ["u1", "u2"]
    assert sorted(c.sat for c in p.eligible) == [0.15, 0.20]


def test_relative_thresholds_are_one_on_normalized_scale():
    """相对口径下可行域在 θ∈[0,1] 内必然非空，故两个闭式阈值都落在 1。"""
    p = _problem([_c("u1", 0.3), _c("u2", 0.8)], theta=0.5, mode="relative")
    th = p.theta_thresholds()
    assert th["theta_all"] == 1.0
    assert th["theta_any"] == 1.0
    assert th["n_workers"] == 2


def test_absolute_is_still_the_default():
    """不能因为加了相对口径就悄悄改掉历史行为。"""
    assert APrimeConfig().theta_mode == "absolute"
    p = _problem([_c("u1", 0.9), _c("u1", 0.5)], theta=0.6)
    assert p.theta_of("u1") == pytest.approx(0.6)  # 逐人阈值回退到标量 θ


def test_unknown_theta_mode_fails_fast():
    """口径拼错必须报错，而不是静默按绝对口径算出一份错误的跨数据集结论。"""
    with pytest.raises(ValueError, match="theta_mode"):
        _problem([_c("u1", 0.9)], theta=0.5, mode="relativ")


def test_relative_respects_per_worker_override_when_no_candidates():
    """没有候选的工作者不进入相对阈值表，回退到 cfg 的常规解析。"""
    cfg = APrimeConfig(theta=0.5, theta_mode="relative", budget_kind="count", budget=9.0,
                       per_worker_theta={"ghost": 0.77})
    p = SelectionProblem(instance=types.SimpleNamespace(budget=9.0),
                         candidates=[_c("u1", 0.9)], cfg=cfg)
    assert p.theta_of("u1") == pytest.approx(0.45)
    assert p.theta_of("ghost") == pytest.approx(0.77)
