# -*- coding: utf-8 -*-
"""单元测试：环境层（城市属性 / PSS / 路线可行性）。

    py -3.14 -m pytest tests -q
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.config import DEFAULT_ENV
from mapus.config import DEFAULT_MAPUS
from mapus.data.instances import make_mapus_instance
from mapus.env.metrics_obj import PathSatisfaction, path_cells, pss_of, route_overlap
from mapus.env.urban import LAND_USE, N_CAT, make_urban_grid
from mapus.routing import (baseline_route, pad_pref, repair_from_llm, route_cells,
                           validate_route)


@pytest.fixture(scope="module")
def inst():
    return make_mapus_instance("tdrive_small", 0)


def test_land_use_covers_all_categories():
    for nx, ny in ((8, 8), (16, 16), (32, 32), (8, 4), (16, 8), (32, 16)):
        urban = make_urban_grid("tdrive", nx, ny)
        assert urban.category.shape == (nx, ny)
        present = set(np.unique(urban.category).tolist())
        assert present == set(range(N_CAT)), f"{nx}x{ny} missing {set(range(N_CAT)) - present}"
        # 每格的土地利用分布归一
        assert np.allclose(urban.land_use.sum(axis=2), 1.0, atol=1e-6)


def test_crime_only_for_tdrive():
    t = make_urban_grid("tdrive", 8, 8)
    g = make_urban_grid("grab", 8, 4)
    assert t.crime.max() > 0
    assert float(g.crime.max()) == 0.0
    # 附录 B.1：记录数 < 10 的格子被剔除
    assert set(np.unique(t.crime[t.crime > 0]).round(1)) == set() or t.crime[t.crime > 0].min() >= 10


def test_route_land_use_is_distribution(inst):
    cells = [(0, 0), (1, 0), (2, 0)]
    h = inst.urban.route_land_use(cells)
    assert abs(float(h.sum()) - 1.0) < 1e-9
    assert (h >= 0).all()


def test_pss_prefers_matching_land_use(inst):
    urban = inst.urban
    veg = np.zeros(N_CAT); veg[0] = 1.0
    veg_cells = [tuple(np.argwhere(urban.category == 0)[0]) for _ in range(6)]
    ind_cells = [tuple(np.argwhere(urban.category == 1)[0]) for _ in range(6)]
    sat = PathSatisfaction(urban, mu=0.0, prior=0.0, scale=1.0)
    assert sat.score(veg, veg_cells) > sat.score(veg, ind_cells)


def test_route_overlap_bounds():
    a = [(0, 0), (1, 0), (2, 0)]
    assert route_overlap(a, a) == pytest.approx(1.0)
    assert route_overlap(a, [(5, 5)]) == pytest.approx(0.0)
    assert 0 < route_overlap(a, [(2, 0), (3, 0)]) < 1


def test_baseline_route_is_feasible(inst):
    for w in inst.workers[:6]:
        prof = inst.profiles[w.wid]
        route = baseline_route(inst, w, prof.pref, inst.urban, DEFAULT_MAPUS, DEFAULT_ENV,
                               random.Random(0))
        ok, viol = validate_route(w, route, inst.nx, inst.ny)
        assert ok, (w.wid, viol)
        assert len(route) == w.window
        assert route[0][:2] == list(w.origin)
        assert route[-1][:2] == list(w.dest)


def test_pad_pref_uses_full_window_and_hits_preference(inst):
    w = inst.workers[0]
    prof = inst.profiles[w.wid]
    cells = route_cells(w, [])
    padded = pad_pref(cells, w, prof.pref, inst.urban)
    assert len(padded) == w.window
    top_cat = int(np.argmax(prof.pref))
    share = sum(1 for c in padded if inst.urban.category[c] == top_cat) / len(padded)
    base_share = sum(1 for c in cells if inst.urban.category[c] == top_cat) / len(cells)
    assert share >= base_share


def test_repair_projects_infeasible_path(inst):
    w = inst.workers[3]
    prof = inst.profiles[w.wid]
    bad = [[0, 0, 0], [5, 5, 1], [inst.nx + 3, 0, 2]]   # 越界 + 跳步
    ok, _ = validate_route(w, bad, inst.nx, inst.ny)
    assert not ok
    fixed = repair_from_llm(w, bad, inst.nx, inst.ny, pref=prof.pref, urban=inst.urban)
    if fixed is not None:
        assert validate_route(w, fixed, inst.nx, inst.ny)[0]


def test_pss_of_matches_config(inst):
    prof = inst.profiles[inst.workers[0].wid]
    cells = path_cells([(0, 0, 0), (0, 0, 1), (1, 0, 2), (1, 0, 3)])
    a = pss_of(inst.urban, DEFAULT_MAPUS).score(prof.pref, cells)
    b = PathSatisfaction(inst.urban, DEFAULT_MAPUS.mu, DEFAULT_MAPUS.pss_prior,
                         DEFAULT_MAPUS.pss_scale).score(prof.pref, cells)
    assert a == pytest.approx(b)


def test_vectorized_coverage_q_matches_reference(inst):
    """向量化后的 Coverage.q 必须与 agentsense 的三重循环实现逐位一致。"""
    from agentsense.env.objective import Coverage
    from mapus.env.fast_coverage import _orig_q, patch_coverage
    cov = Coverage(inst)
    for w in inst.workers:
        cov.add_path([[w.origin[0], w.origin[1], w.t_start + i] for i in range(w.window)])
    for mode in ("distinct", "visits"):
        ref = _orig_q(cov, mode)
        patch_coverage()
        assert cov.q(mode) == pytest.approx(ref, rel=1e-12, abs=1e-9)
