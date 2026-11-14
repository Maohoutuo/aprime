# -*- coding: utf-8 -*-
"""不分桶的表示度：`ks_capability` / `rank_shift` / `bootstrap_ci`。

这三个是为了替代 `gqc_gap` 而加的：`gqc_gap` 依赖能力分桶，分组塌成一档时
恒为 0、最小组过小时份额不可信（见 docs §8.7–8.9）。这里验证它们在
能力轴上直接比较分布，不经过任何桶。
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.analysis.fairness import (bootstrap_ci, capability, ks_capability,  # noqa: E402
                                      rank_shift, representation_metrics)
from aprime.config import APrimeConfig  # noqa: E402
from aprime.model import Candidate, SelectionProblem  # noqa: E402


def _c(wid: str, sat: float, tag: str) -> Candidate:
    path = ((0, 0, 0),)
    return Candidate(wid=wid, path=path, cells=frozenset({(0, 0)}), sat=sat, tag=tag,
                     space_time=frozenset({(0, 0, 0)}))


def _problem(cands, theta=0.5) -> SelectionProblem:
    cfg = APrimeConfig(theta=theta, budget_kind="count", budget=99.0)
    return SelectionProblem(instance=types.SimpleNamespace(budget=99.0),
                            candidates=list(cands), cfg=cfg)


def _skewed():
    """能力分层：w1 有 4 条合法、w2 有 2 条、w3 只有 1 条。θ=0.5。"""
    cands = [_c("w1", 0.9, "a"), _c("w1", 0.8, "b"), _c("w1", 0.7, "c"), _c("w1", 0.6, "d"),
             _c("w2", 0.9, "a"), _c("w2", 0.6, "b"), _c("w2", 0.3, "c"),
             _c("w3", 0.9, "a"), _c("w3", 0.2, "b")]
    return _problem(cands)


def test_capability_counts_eligible_routes():
    p = _skewed()
    cap = capability(p)["n_eligible_routes"]
    assert cap == {"w1": 4, "w2": 2, "w3": 1}


def test_rank_shift_sign_says_who_gets_picked():
    """正 = 偏向高能力者，负 = 偏向低能力者。符号必须能直接读。"""
    p = _skewed()
    high = [c for c in p.candidates if c.wid == "w1" and c.tag == "a"]
    low = [c for c in p.candidates if c.wid == "w3" and c.tag == "a"]
    assert rank_shift(p, high) > 0
    assert rank_shift(p, low) < 0
    # 中位选取应接近 0（w2 能力居中）
    mid = [c for c in p.candidates if c.wid == "w2" and c.tag == "a"]
    assert abs(rank_shift(p, mid)) < 0.2


def test_ks_is_zero_when_selection_matches_pool():
    """选中者能力分布与池一致时 KS 必须为 0——否则度量有系统性偏置。"""
    p = _skewed()
    one_each = [_c("w1", 0.9, "a"), _c("w2", 0.9, "a"), _c("w3", 0.9, "a")]
    assert ks_capability(p, one_each) == pytest.approx(0.0, abs=1e-9)


def test_ks_detects_picking_only_top():
    p = _skewed()
    top = [c for c in p.candidates if c.wid == "w1" and c.tag == "a"]
    # 池能力 = {1,2,4}，只选能力 4 → F 差在 x=2 处达到 2/3
    assert ks_capability(p, top) == pytest.approx(2 / 3, abs=1e-6)


def test_representation_metrics_survives_single_bucket_grouping():
    """能力全相同 → `gqc_gap` 恒为 0（无信息），但这两个度量仍如实报 0。

    这里验证的是"不经过分桶"这一性质：结果由能力分布直接决定，
    不依赖任何组的划分，因此不会因为分组退化而突然变成 0 或失效。
    """
    cands = [_c(f"u{i}", 0.9, "a") for i in range(6)]
    p = _problem(cands, theta=0.5)
    sel = p.candidates[:2]
    out = representation_metrics(p, sel)
    # 所有人能力相同，选中者与池分布一致 → 正确的 0，而不是"退化"
    assert out["ks_capability"] == pytest.approx(0.0)
    assert out["rank_shift"] == pytest.approx(0.0)


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    vals = [0.10, 0.14, 0.09, 0.20, 0.11, 0.13, 0.07, 0.16]
    lo, hi = bootstrap_ci(vals, seed=1)
    m = sum(vals) / len(vals)
    assert lo <= m <= hi
    assert (lo, hi) == bootstrap_ci(vals, seed=1)  # 同种子必须可复现
    assert lo < hi


def test_bootstrap_ci_handles_degenerate_input():
    lo, hi = bootstrap_ci([0.3])
    assert lo != lo and hi != hi  # 单点无法估区间 → NaN，而不是编一个出来
