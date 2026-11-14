# -*- coding: utf-8 -*-
"""Participant Agent（PA）：接单决策 + 算法 1 的偏好感知路线生成。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agentsense.config import EnvConfig
from agentsense.env.model import Instance, Worker

from ..config import MAPUSConfig
from ..env.metrics_obj import PathSatisfaction, path_cells, pss_of, route_coverage_utility
from ..env.urban import UrbanGrid
from ..llm_backend import BaseLLM
from ..parse import extract_json, parse_accept, parse_refine, parse_tool_calls
from ..profiles import ParticipantProfile
from ..routing import baseline_route, repair_from_llm, route_utility, validate_route
from .prompts import accept_prompt, disturbance_prompt, refine_prompt
from .tools import run_tool_calls


@dataclass
class RouteGenStat:
    calls: int = 0
    invalid: int = 0
    repaired: int = 0
    accepted_refines: int = 0
    base_utility: float = 0.0
    final_utility: float = 0.0
    base_pss: float = 0.0
    final_pss: float = 0.0
    base_cover: float = 0.0
    final_cover: float = 0.0
    explanation: str = ""


class ParticipantAgent:
    """一个参与者 = 一个自治 agent（图 2(a)）。"""

    def __init__(self, instance: Instance, worker: Worker, profile: ParticipantProfile,
                 urban: UrbanGrid, cfg: MAPUSConfig, env_cfg: EnvConfig, llm: BaseLLM,
                 seed: int = 0):
        self.instance = instance
        self.worker = worker
        self.profile = profile
        self.urban = urban
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.llm = llm
        self.rng = random.Random(seed)
        self.blocked = tuple(instance.meta.get("blocked_cells", ()) or ())
        self.current_route: List[List[int]] = []

    # ------------------------------------------------------------------ 接单
    def decide_accept(self, budget_left: float) -> Tuple[bool, str]:
        # 路线长度至少为曼哈顿距离 + 1（含起点），且不能超过时间窗
        if self.worker.window < self.worker.min_steps + 1:
            return False, "the trip cannot be completed within my time window"
        if not self.cfg.llm_accept:
            return True, "rule-based acceptance: the trip is feasible"
        system, user = accept_prompt(self.instance, self.worker, self.profile, budget_left, self.urban)
        text = self.llm.complete(system, user, json_mode=True,
                                 meta={"kind": "accept", "worker": self.worker,
                                       "profile": self.profile, "instance": self.instance,
                                       "urban": self.urban, "cfg": self.cfg, "env_cfg": self.env_cfg})
        accept, reason = parse_accept(text)
        return accept, reason

    # ------------------------------------------------------------------ 算法 1
    def generate_route(self, instructions: str = "") -> Optional[Dict[str, Any]]:
        cfg, env_cfg = self.cfg, self.env_cfg
        w = self.worker
        # 算法 1 第 1 行：经典规划器只优化覆盖（λ=1），偏好由 LLM 精修引入
        r0 = baseline_route(self.instance, w, self.profile.pref, self.urban, cfg, env_cfg,
                            random.Random(self.rng.random()), blocked=self.blocked, lam=1.0)
        if not r0:
            return None
        stat = RouteGenStat(
            base_utility=route_utility(self.instance, r0, self.profile.pref, self.urban, cfg, env_cfg),
            base_pss=pss_of(self.urban, cfg).score(self.profile.pref, path_cells(r0)),
            base_cover=route_coverage_utility(self.instance, r0, env_cfg),
        )
        best_route, best_util = r0, stat.base_utility
        if not cfg.use_prg:
            # 消融 w/o PRG：直接用经典规划器的路线（论文用 TVPG 基线替代）
            stat.final_utility, stat.final_pss = stat.base_utility, stat.base_pss
            stat.final_cover = stat.base_cover
            return {"route": r0, "baseline": r0, "stat": stat}

        for _ in range(cfg.max_refine):
            system, user = refine_prompt(self.instance, w, self.profile, best_route, self.urban,
                                         instructions=instructions, blocked=self.blocked)
            text = self._call(system, user, best_route)
            stat.calls += 1
            tool_calls = parse_tool_calls(text)
            if tool_calls and cfg.max_tool_rounds > 0:
                for _round in range(cfg.max_tool_rounds):
                    results = run_tool_calls(self.urban, tool_calls)
                    user2 = user + f"\n\nTool results:\n{results}\n\nNow output the final JSON object only."
                    text = self._call(system, user2, best_route)
                    stat.calls += 1
                    tool_calls = parse_tool_calls(text)
                    if not tool_calls:
                        break
            path, explanation = parse_refine(text)
            if not path:
                stat.invalid += 1
                continue
            ok, _viol = validate_route(w, path, self.instance.nx, self.instance.ny,
                                       self.blocked, self.instance.meta.get("cells_per_step", 1))
            if not ok:
                stat.invalid += 1
                if cfg.infeasible_policy != "repair":
                    continue
                repaired = repair_from_llm(w, path, self.instance.nx, self.instance.ny, self.blocked,
                                           pref=self.profile.pref, urban=self.urban)
                if repaired is None:
                    continue
                path = repaired
                stat.repaired += 1
            util = route_utility(self.instance, path, self.profile.pref, self.urban, cfg, env_cfg)
            if util > best_util + 1e-9:
                best_route, best_util = path, util
                stat.accepted_refines += 1
                stat.explanation = explanation

        stat.final_utility = best_util
        stat.final_pss = pss_of(self.urban, cfg).score(self.profile.pref, path_cells(best_route))
        stat.final_cover = route_coverage_utility(self.instance, best_route, env_cfg)
        return {"route": best_route, "baseline": r0, "stat": stat}

    # ------------------------------------------------------------------ 扰动（图 12）
    def react_to_disturbance(self, event: str, blocked_cells: Sequence[tuple] = (),
                             targets: Sequence[tuple] = ()) -> Dict[str, Any]:
        system, user = disturbance_prompt(self.instance, self.worker, self.profile,
                                         self.current_route, event, self.urban)
        text = self.llm.complete(system, user, json_mode=True, meta={
            "kind": "disturbance", "instance": self.instance, "worker": self.worker,
            "profile": self.profile, "urban": self.urban, "cfg": self.cfg, "env_cfg": self.env_cfg,
            "current_path": self.current_route, "event": event,
            "blocked": list(blocked_cells), "targets": list(targets),
        })
        obj = extract_json(text) or {}
        adjust = bool(obj.get("adjust", obj.get("adjustment", False)))
        path, explanation = parse_refine(text)
        return {"adjust": adjust, "path": path, "explanation": explanation}

    # ------------------------------------------------------------------ 工具
    def _call(self, system: str, user: str, current_path: Optional[list] = None) -> str:
        return self.llm.complete(system, user, json_mode=True, meta={
            "kind": "refine", "instance": self.instance, "worker": self.worker,
            "profile": self.profile, "urban": self.urban, "cfg": self.cfg, "env_cfg": self.env_cfg,
            "current_path": current_path,
        })
