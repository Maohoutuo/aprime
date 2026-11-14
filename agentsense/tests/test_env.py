# -*- coding: utf-8 -*-
"""环境与基线的单元测试。

运行： python -m pytest tests -q
"""

import math
import random

import numpy as np
import pytest

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV, SETTINGS
from agentsense.data.instances import make_instance, instance_seed
from agentsense.env.model import (Worker, build_path, manhattan, pad_to_window,
                                 path_to_solution, validate_solution)
from agentsense.env.objective import Coverage, evaluate


@pytest.fixture(scope="module")
def small_inst():
    return make_instance("tdrive_small", 0)


def test_table2_consistency():
    """表 2 的配置必须自洽：网格 × 时域 = 任务数，且时域能被时间槽整除。"""
    for (dataset, scale), s in SETTINGS.items():
        assert s.horizon_min % s.slot_min == 0, (dataset, scale)
        assert s.n_slots >= 4
        assert s.n_tasks == s.n_regions * s.n_slots


def test_instance_deterministic():
    a = make_instance("tdrive_small", 3)
    b = make_instance("tdrive_small", 3)
    assert instance_seed(SETTINGS[("tdrive", "small")], 3) == a.seed
    assert [(w.wid, w.origin, w.dest, w.t_start, w.t_end) for w in a.workers] == \
           [(w.wid, w.origin, w.dest, w.t_start, w.t_end) for w in b.workers]


def test_workers_are_reachable(small_inst):
    for w in small_inst.workers:
        assert w.min_steps <= w.window - 1, f"{w.wid} 不可达"
        assert build_path(w, []) is not None


def test_build_path_and_padding():
    w = Worker("w", (0, 0), (2, 0), 0, 4)
    cells = build_path(w, [])
    assert cells == [(0, 0), (1, 0), (2, 0)]
    padded = pad_to_window(cells, w)
    assert len(padded) == w.window
    assert padded[0] == w.origin and padded[-1] == w.dest


def test_validate_rejects_bad_solutions(small_inst):
    w = small_inst.workers[0]
    good = path_to_solution(w, pad_to_window(build_path(w, []), w))
    assert validate_solution(small_inst, {w.wid: good}).feasible

    # 起点错误
    bad = [row[:] for row in good]
    bad[0][0] = (bad[0][0] + 1) % small_inst.nx
    assert not validate_solution(small_inst, {w.wid: bad}).feasible

    # 跳跃移动
    jump = [row[:] for row in good]
    if len(jump) > 2:
        jump[1][0] = (jump[1][0] + 3) % small_inst.nx
        assert not validate_solution(small_inst, {w.wid: jump}).feasible


def test_budget_enforced(small_inst):
    sol = {}
    cost = 0
    for w in small_inst.workers:
        if cost + w.window > small_inst.budget:
            break
        sol[w.wid] = path_to_solution(w, pad_to_window(build_path(w, []), w))
        cost += w.window
    assert validate_solution(small_inst, sol).feasible
    # 再加一个工作者必然超预算
    extra = next(w for w in small_inst.workers if w.wid not in sol)
    sol[extra.wid] = path_to_solution(extra, pad_to_window(build_path(extra, []), extra))
    res = validate_solution(small_inst, sol)
    assert not res.feasible and any("budget" in v for v in res.violations)


def test_objective_monotone(small_inst):
    empty = Coverage(small_inst)
    assert empty.objective(DEFAULT_ENV) == 0.0

    w = small_inst.workers[0]
    path = path_to_solution(w, pad_to_window(build_path(w, []), w))
    one = Coverage(small_inst)
    one.add_path(path)
    assert one.objective(DEFAULT_ENV) > 0.0

    # 覆盖越多，目标函数不减
    sol2 = {w.wid: path}
    w2 = small_inst.workers[1]
    sol2[w2.wid] = path_to_solution(w2, pad_to_window(build_path(w2, []), w2))
    two = Coverage(small_inst)
    for p in sol2.values():
        two.add_path(p)
    assert two.objective(DEFAULT_ENV) >= one.objective(DEFAULT_ENV)


def test_cost_equals_steps(small_inst):
    w = small_inst.workers[0]
    sol = {w.wid: path_to_solution(w, pad_to_window(build_path(w, []), w))}
    ev = evaluate(small_inst, sol, DEFAULT_ENV)
    assert ev["cost"] == float(len(sol[w.wid])) == float(w.window)


@pytest.mark.parametrize("name", list(PLANNERS))
def test_planners_feasible(name, small_inst):
    sol = PLANNERS[name](small_inst, DEFAULT_ENV, random.Random(7))
    res = validate_solution(small_inst, sol)
    assert res.feasible, res.violations[:3]
    assert res.cost <= small_inst.budget + 1e-9
    assert sol, "规划器不应返回空解"


def test_entropy_bounds(small_inst):
    w = small_inst.workers[0]
    cov = Coverage(small_inst)
    cov.add_path(path_to_solution(w, pad_to_window(build_path(w, []), w)))
    h_norm = cov.entropy("norm", e_scale=1.0)
    h_bits = cov.entropy("equal")
    assert 0.0 <= h_norm <= 1.0
    assert h_bits > 0.0
    assert math.isfinite(cov.objective(DEFAULT_ENV))
