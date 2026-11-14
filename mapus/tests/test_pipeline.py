# -*- coding: utf-8 -*-
"""单元测试：解析器与三阶段流程（离线代理模型）。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.config import DEFAULT_ENV
from agentsense.env.model import validate_solution
from mapus.config import MAPUSConfig
from mapus.data.instances import make_mapus_instance
from mapus.env.metrics_obj import evaluate_mapus
from mapus.llm_backend import SurrogateLLM
from mapus.parse import extract_json, parse_accept, parse_offer, parse_refine, parse_single
from mapus.parse import parse_tiebreak
from mapus.pipeline import run_mapus
from mapus.routing import validate_route


# --------------------------------------------------------------------------- 解析
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blah {"a": 1} trailing') == {"a": 1}
    assert extract_json("no json here") is None


def test_parse_refine_handles_tuples():
    path, expl = parse_refine('{"final_path": [(0, 0, 1), (1, 0, 2)], "explanation": "x"}')
    assert path == [[0, 0, 1], [1, 0, 2]]
    assert expl == "x"


def test_parse_accept_fails_open():
    assert parse_accept('{"accept": true}')[0] is True
    assert parse_accept('{"accept": false}')[0] is False
    # 没有 JSON 时默认接受（格式错误不应被当成拒绝）
    assert parse_accept("Sure, I will take the task.")[0] is True


def test_parse_tiebreak_and_single():
    assert parse_tiebreak('{"selected": "w002"}', ["w001", "w002"]) == "w002"
    out = parse_single('{"selected": [{"worker_id": "w001", "path": [[0,0,0],[1,0,1]]}]}')
    assert out["w001"] == [[0, 0, 0], [1, 0, 1]]


def test_parse_offer():
    txt = ('{"refined_routes": {"w001": [[0,0,0]], "w002": [[1,1,0]]}, '
           '"incentives": {"w001": "hi"}}')
    routes, inc = parse_offer(txt, "w001", "w002")
    assert set(routes) == {"w001", "w002"}
    assert inc["w001"] == "hi"


# --------------------------------------------------------------------------- 流程
@pytest.fixture(scope="module")
def inst():
    return make_mapus_instance("grab_small", 0)


def test_mapus_runs_and_respects_budget(inst):
    cfg = MAPUSConfig()
    run = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, SurrogateLLM(),
                    seed=1000, parallel=1)
    assert run.solution
    cost = sum(len(p) for p in run.solution.values())
    assert cost <= inst.budget + 1e-9
    for wid, path in run.solution.items():
        w = inst.worker(wid)
        ok, viol = validate_route(w, path, inst.nx, inst.ny)
        assert ok, (wid, viol)
    ev = evaluate_mapus(inst, run.solution, inst.urban, inst.prefs, DEFAULT_ENV)
    assert ev["coverage"] > 0 and -1.0 < ev["pss"] < 2.0


def test_mapus_is_deterministic_with_surrogate(inst):
    cfg = MAPUSConfig()
    a = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, SurrogateLLM(), seed=1000)
    b = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, SurrogateLLM(), seed=1000)
    assert a.meta["selected"] == b.meta["selected"]
    assert a.stats["coverage"] == pytest.approx(b.stats["coverage"])


def test_ablation_switches_change_result(inst):
    base = run_mapus(inst, inst.urban, inst.profiles, MAPUSConfig(), DEFAULT_ENV,
                     SurrogateLLM(), seed=1000)
    no_prg = run_mapus(inst, inst.urban, inst.profiles, MAPUSConfig(use_prg=False), DEFAULT_ENV,
                       SurrogateLLM(), seed=1000)
    assert no_prg.stats["final_pss"] != pytest.approx(base.stats["final_pss"])
    no_nrr = run_mapus(inst, inst.urban, inst.profiles, MAPUSConfig(use_nrr=False), DEFAULT_ENV,
                       SurrogateLLM(), seed=1000)
    assert no_nrr.stats["neg_attempts"] == 0


def test_fairness_prefers_unselected(inst):
    from mapus.agents.coordinator import CoordinatorAgent
    cfg = MAPUSConfig(tie_break="rule")
    ca = CoordinatorAgent(inst, inst.urban, cfg, DEFAULT_ENV, SurrogateLLM())
    routes = {}
    for w in inst.workers[:6]:
        routes[w.wid] = [[w.origin[0], w.origin[1], w.t_start + i]
                         for i in range(w.min_steps + 1)]
    hist = {wid: (0 if i == 5 else 3) for i, wid in enumerate(routes)}
    picked, _ = ca.select(routes, inst.profiles, histories=hist, budget=10 ** 6)
    assert list(picked)[0] in routes
