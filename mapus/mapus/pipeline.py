# -*- coding: utf-8 -*-
"""MAPUS 主流程（论文图 2 的三个阶段）与基线适配。"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agentsense.baselines.planners import PLANNERS
from agentsense.config import EnvConfig
from agentsense.env.model import Instance, Solution

from .agents.coordinator import CoordinatorAgent, NegotiationStat, SelectionStat
from .agents.participant import ParticipantAgent, RouteGenStat
from .agents.prompts import single_llm_prompt
from .config import MAPUSConfig
from .env.metrics_obj import evaluate_mapus, path_cells, route_overlap
from .env.urban import UrbanGrid
from .llm_backend import BaseLLM
from .parse import parse_single
from .profiles import ParticipantProfile
from .routing import baseline_route, repair_from_llm, validate_route


@dataclass
class MapusRun:
    solution: Solution                     # 被选中的参与者 -> 路线
    accepted_routes: Dict[str, list]       # 阶段 1 结束时所有接单参与者的路线
    stats: dict = field(default_factory=dict)
    runtimes: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- 阶段 1
def _stage1(instance: Instance, workers, profiles: Dict[str, ParticipantProfile], urban: UrbanGrid,
            cfg: MAPUSConfig, env_cfg: EnvConfig, llm: BaseLLM, seed: int,
            parallel: int) -> Tuple[Dict[str, list], Dict[str, RouteGenStat], int, int]:
    agents = {w.wid: ParticipantAgent(instance, w, profiles[w.wid], urban, cfg, env_cfg, llm,
                                      seed=seed + 17 * i)
              for i, w in enumerate(workers)}

    def _one(wid: str):
        pa = agents[wid]
        accept, reason = pa.decide_accept(instance.budget)
        if not accept:
            return wid, None, None, reason
        out = pa.generate_route()
        return wid, out, reason, ""

    accepted: Dict[str, list] = {}
    stats: Dict[str, RouteGenStat] = {}
    n_refuse = 0
    reasons = []
    if parallel > 1 and len(workers) > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(_one, [w.wid for w in workers]))
    else:
        results = [_one(w.wid) for w in workers]
    for wid, out, reason, _ in results:
        if out is None:
            n_refuse += 1
            reasons.append(f"{wid}: {reason}")
            continue
        accepted[wid] = out["route"]
        stats[wid] = out["stat"]
    return accepted, stats, n_refuse, reasons


# --------------------------------------------------------------------------- 主流程
def run_mapus(instance: Instance, urban: UrbanGrid, profiles: Dict[str, ParticipantProfile],
              cfg: MAPUSConfig, env_cfg: EnvConfig, llm: BaseLLM, seed: int = 0,
              parallel: int = 8, histories: Optional[Dict[str, int]] = None,
              participants: Optional[Sequence] = None) -> MapusRun:
    workers = list(participants if participants is not None else instance.workers)
    t0 = time.time()
    accepted, gen_stats, n_refuse, reasons = _stage1(
        instance, workers, profiles, urban, cfg, env_cfg, llm, seed, parallel)
    t1 = time.time()

    ca = CoordinatorAgent(instance, urban, cfg, env_cfg, llm)
    selected, sel_stat = ca.select(accepted, profiles, histories=histories)
    t2 = time.time()

    agents = {w.wid: ParticipantAgent(instance, w, profiles[w.wid], urban, cfg, env_cfg, llm)
              for w in workers}
    refined, neg_stat = ca.negotiate(selected, profiles, agents)
    t3 = time.time()

    prefs = {k: p.pref for k, p in profiles.items()}
    ev_before = evaluate_mapus(instance, selected, urban, prefs, env_cfg, mu=cfg.mu)
    ev = evaluate_mapus(instance, refined, urban, prefs, env_cfg, mu=cfg.mu)
    stats = {
        "n_candidates": len(workers),
        "n_accepted": len(accepted),
        "n_refused": n_refuse,
        "refuse_reasons": reasons[:10],
        "n_selected": len(refined),
        "llm_calls_gen": sum(s.calls for s in gen_stats.values()),
        "llm_invalid": sum(s.invalid for s in gen_stats.values()),
        "llm_repaired": sum(s.repaired for s in gen_stats.values()),
        "refines_accepted": sum(s.accepted_refines for s in gen_stats.values()),
        "base_pss": _mean([s.base_pss for s in gen_stats.values()]),
        "final_pss": _mean([s.final_pss for s in gen_stats.values()]),
        "base_cover": _mean([s.base_cover for s in gen_stats.values()]),
        "final_cover": _mean([s.final_cover for s in gen_stats.values()]),
        "sel_llm_ties": sel_stat.llm_ties,
        "neg_attempts": neg_stat.attempts,
        "neg_rounds": neg_stat.rounds,
        "neg_accepted": neg_stat.accepted,
        "neg_rejected": neg_stat.rejected,
        "neg_failed_pairs": neg_stat.failed_pairs,
        "neg_overlap_before": _mean(neg_stat.overlap_before),
        "neg_overlap_after": _mean(neg_stat.overlap_after),
        "coverage": ev["coverage"],
        "pss": ev["pss"],
        "cost": ev["cost"],
        "entropy": ev["entropy"],
        "count": ev["count"],
        "avg_overlap": ev["avg_overlap"],
        "max_overlap": ev["max_overlap"],
        "pre_neg_coverage": ev_before["coverage"],
        "pre_neg_pss": ev_before["pss"],
        "pre_neg_entropy": ev_before["entropy"],
        "pre_neg_count": ev_before["count"],
        "pre_neg_overlap": ev_before["avg_overlap"],
    }
    return MapusRun(solution=refined, accepted_routes=accepted, stats=stats,
                    runtimes={"stage1": t1 - t0, "stage2": t2 - t1, "stage3": t3 - t2,
                              "total": t3 - t0},
                    meta={"selected": list(selected.keys()),
                          "accepted": list(accepted.keys())})


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


# --------------------------------------------------------------------------- 经典基线
def run_classical(name: str, instance: Instance, env_cfg: EnvConfig, seed: int = 0) -> Solution:
    return PLANNERS[name](instance, env_cfg, random.Random(seed))


# --------------------------------------------------------------------------- 扰动（附录 B.6 / 图 12）
def run_disturbance(instance: Instance, urban: UrbanGrid, profiles: Dict[str, ParticipantProfile],
                    solution: Solution, cfg: MAPUSConfig, env_cfg: EnvConfig, llm: BaseLLM,
                    event: str, blocked: Sequence[tuple] = (), targets: Sequence[tuple] = (),
                    seed: int = 0, parallel: int = 8) -> Dict[str, object]:
    """协调者广播动态事件，参与者各自判断是否受影响的闭环。"""
    prefs = {k: p.pref for k, p in profiles.items()}

    def _evaluate(sol: Solution, weights=None) -> dict:
        saved = instance.meta.get("cell_weights")
        if weights is not None:
            instance.meta["cell_weights"] = weights
        try:
            return evaluate_mapus(instance, sol, urban, prefs, env_cfg,
                                  mu=cfg.mu, prior=cfg.pss_prior, scale=cfg.pss_scale)
        finally:
            if weights is not None:
                instance.meta["cell_weights"] = saved

    weights = None
    if targets:
        weights = np.ones((instance.nx, instance.ny))
        for (x, y) in targets:
            weights[max(0, x - 1):x + 2, max(0, y - 1):y + 2] = 4.0
    before = _evaluate(solution, weights)

    agents = {w.wid: ParticipantAgent(instance, w, profiles[w.wid], urban, cfg, env_cfg, llm)
              for w in instance.workers}

    def _one(wid: str):
        pa = agents[wid]
        pa.current_route = list(solution[wid])
        out = pa.react_to_disturbance(event, blocked, targets)
        return wid, out

    if parallel > 1 and len(solution) > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(_one, list(solution.keys())))
    else:
        results = [_one(wid) for wid in solution]

    refined: Solution = {}
    decisions = []
    for wid, out in results:
        old = solution[wid]
        new = old
        changed = False
        if out["adjust"] and out["path"]:
            w = instance.worker(wid)
            ok, viol = validate_route(w, out["path"], instance.nx, instance.ny,
                                      tuple(blocked), instance.meta.get("cells_per_step", 1))
            if not ok:
                fixed = repair_from_llm(w, out["path"], instance.nx, instance.ny, tuple(blocked),
                                        pref=profiles[wid].pref, urban=urban)
                if fixed is not None and not (set(map(tuple, blocked)) &
                                              {(int(p[0]), int(p[1])) for p in fixed}):
                    new = fixed
                    changed = True
            else:
                new = out["path"]
                changed = True
        refined[wid] = new
        decisions.append({"worker": wid, "adjust": bool(out["adjust"]), "changed": changed,
                          "explanation": out["explanation"]})
    after = _evaluate(refined, weights)
    return {"before": before, "after": after, "decisions": decisions,
            "coverage_before": before["coverage"], "coverage_after": after["coverage"],
            "pss_before": before["pss"], "pss_after": after["pss"],
            "n_changed": sum(1 for d in decisions if d["changed"])}


# --------------------------------------------------------------------------- 单 LLM 基线
def run_single_llm(instance: Instance, urban: UrbanGrid, profiles: Dict[str, ParticipantProfile],
                   cfg: MAPUSConfig, env_cfg: EnvConfig, llm: BaseLLM,
                   seed: int = 0) -> Tuple[Solution, dict]:
    t0 = time.time()
    candidates: Dict[str, list] = {}
    for w in instance.workers:
        route = baseline_route(instance, w, profiles[w.wid].pref, urban, cfg, env_cfg,
                               random.Random(seed + 31 * len(candidates)), lam=1.0)
        if route:
            candidates[w.wid] = route
    system, user = single_llm_prompt(instance, profiles, urban, candidates)
    text = llm.complete(system, user, json_mode=True, meta={
        "kind": "single", "instance": instance, "urban": urban, "cfg": cfg,
        "env_cfg": env_cfg, "candidates": candidates,
    })
    plan = parse_single(text)
    solution: Solution = {}
    cost = 0.0
    n_invalid = 0
    for wid, path in plan.items():
        if wid not in candidates:
            continue
        w = instance.worker(wid)
        if cost + w.window > instance.budget + 1e-9:
            continue
        ok, _ = validate_route(w, path, instance.nx, instance.ny,
                               tuple(instance.meta.get("blocked_cells", ()) or ()),
                               instance.meta.get("cells_per_step", 1))
        if not ok:
            fixed = repair_from_llm(w, path, instance.nx, instance.ny,
                                    tuple(instance.meta.get("blocked_cells", ()) or ()))
            if fixed is None:
                n_invalid += 1
                continue
            path = fixed
            n_invalid += 1
        solution[wid] = path
        cost += w.window
    if not solution:  # LLM 完全没给出可用解时退化为随机（论文中不会出现）
        solution = run_classical("RN", instance, env_cfg, seed)
    ev = evaluate_mapus(instance, solution, urban, {k: p.pref for k, p in profiles.items()},
                        env_cfg, mu=cfg.mu)
    info = dict(ev)
    info.update({"runtime": time.time() - t0, "n_invalid": n_invalid,
                 "n_selected": len(solution)})
    return solution, info
