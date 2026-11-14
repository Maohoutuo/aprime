# -*- coding: utf-8 -*-
"""MRG floor、上限与记账不变量。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aprime.solvers.gqc import GQCParams, _floors_for, _mrg_blocked


def test_mrg_cap_blocks_extreme_small_group():
    params = GQCParams(min_rep=1, mrg_cap=2.0)
    share = {"tight": 0.025, "two": 0.075, "ample": 0.9}

    assert _floors_for(params, share, 7, 1.0) == {
        "tight": 0, "two": 1, "ample": 6,
    }
    assert _mrg_blocked(params, share, 7) == ["tight"]


def test_mrg_cap_zero_protects_every_nonempty_group_when_feasible():
    params = GQCParams(min_rep=1, mrg_cap=0.0)
    share = {"a": 0.099, "b": 0.099, "c": 0.802}

    assert _floors_for(params, share, 10, 1.0) == {"a": 1, "b": 1, "c": 8}
    assert _mrg_blocked(params, share, 10) == []


def test_mrg_reverts_when_every_group_floor_is_not_feasible():
    params = GQCParams(min_rep=1, mrg_cap=0.0)
    share = {"a": 0.025, "b": 0.075, "c": 0.9}

    # 抬到 {1,1,6} 总和 8 > k=7，故整体退回纯比例
    assert _floors_for(params, share, 7, 1.0) == {"a": 0, "b": 0, "c": 6}


def test_default_params_preserve_proportional_floors():
    params = GQCParams()
    share = {"tight": 0.025, "ample": 0.975}

    assert _floors_for(params, share, 7, 1.0) == {"tight": 0, "ample": 6}
