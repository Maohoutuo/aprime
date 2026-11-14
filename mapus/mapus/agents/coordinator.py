# -*- coding: utf-8 -*-
"""Coordinator Agent（CA）：算法 2 的公平感知选择 + 算法 3 的协商式路线精修。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agentsense.config import EnvConfig
from agentsense.env.model import Instance
from agentsense.env.objective import Coverage

from ..config import MAPUSConfig
from ..env.metrics_obj import pairwise_overlap, path_cells, pss_of, route_overlap
from ..env.urban import UrbanGrid
from ..llm_backend import BaseLLM
from ..parse import parse_offer, parse_tiebreak
from ..profiles import ParticipantProfile
from ..routing import repair_from_llm, validate_route
from .prompts import negotiation_feedback_prompt, negotiation_offer_prompt, tiebreak_prompt


@dataclass
class SelectionStat:
    steps: int = 0
    llm_ties: int = 0
    score_trace: List[float] = field(default_factory=list)


@dataclass
class NegotiationStat:
    attempts: int = 0
    rounds: int = 0
    accepted: int = 0
    rejected: int = 0
    failed_pairs: int = 0
    overlap_before: List[float] = field(default_factory=list)
    overlap_after: List[float] = field(default_factory=list)
    calls: int = 0


class CoordinatorAgent:
    """系统级的协调者（图 2(b)(c)）。"""

    def __init__(self, instance: Instance, urban: UrbanGrid, cfg: MAPUSConfig,
                 env_cfg: EnvConfig, llm: BaseLLM):
        self.instance = instance
        self.urban = urban
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.llm = llm

    # ------------------------------------------------------------------ 辅助
    def _objective(self, routes: Dict[str, list]) -> float:
        cov = Coverage(self.instance)
        for path in routes.values():
            cov.add_path(path)
        return cov.objective(self.env_cfg)

    @staticmethod
    def _minmax(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        lo, hi = min(values.values()), max(values.values())
        if hi - lo < 1e-12:
            return {k: 0.0 for k in values}
        return {k: (v - lo) / (hi - lo) for k, v in values.items()}

    # ------------------------------------------------------------------ 算法 2
    def select(self, routes: Dict[str, list], profiles: Dict[str, ParticipantProfile],
               histories: Optional[Dict[str, int]] = None, budget: Optional[float] = None,
               beta: Optional[float] = None) -> Tuple[Dict[str, list], SelectionStat]:
        """式 (6) 的贪心选择。

        收益项除了边际覆盖增益 Δφ(u)，还并入式 (1) 里的参与者满意项 η·S(r_u)
        —— 否则选择阶段只看覆盖，会把高满意度的工作者筛掉，最终方案 PSS 反而下降。
        """
        cfg = self.cfg
        histories = histories or {}
        beta = cfg.beta if beta is None else beta
        budget = self.instance.budget if budget is None else budget
        stat = SelectionStat()
        sat = pss_of(self.urban, cfg)

        selected: Dict[str, list] = {}
        remaining = dict(routes)
        budget_left = float(budget)
        while budget_left > 0 and remaining:
            feasible = {wid: p for wid, p in remaining.items()
                        if self.instance.worker(wid).window <= budget_left + 1e-9}
            if not feasible:
                break
            base_obj = self._objective(selected)
            gains: Dict[str, float] = {}
            satis: Dict[str, float] = {}
            for wid, path in feasible.items():
                selected[wid] = path
                gains[wid] = self._objective(selected) - base_obj
                selected.pop(wid)
                if wid in profiles:
                    satis[wid] = sat.score(profiles[wid].pref, path_cells(path))
            fair = {wid: 1.0 / (1.0 + histories.get(wid, 0)) for wid in feasible}
            f_gain = self._minmax(gains)
            f_fair = self._minmax(fair)
            if cfg.use_fps:
                # 式 (1) 的收益项：边际覆盖增益 Δφ(u) 与路线满意 S(r_u) 各自归一化后平均
                f_sat = self._minmax(satis)
                benefit = {wid: 0.5 * f_gain[wid] + 0.5 * f_sat[wid] for wid in feasible}
            else:
                benefit = f_gain
            scores = {wid: beta * benefit[wid] + (1.0 - beta) * f_fair[wid] for wid in feasible}
            if not cfg.use_fps:
                # 消融 w/o FPS：纯覆盖贪心，不做公平项、也不做 LLM 平局裁决
                scores = f_gain
            best_score = max(scores.values())
            tied = [wid for wid, s in scores.items() if s >= best_score - cfg.tie_eps]
            if len(tied) == 1 or not cfg.use_fps:
                pick = tied[0]
            elif cfg.tie_break == "llm":
                pick = self._tiebreak(tied, profiles, scores)
                stat.llm_ties += 1
            else:
                pick = min(tied, key=lambda w: (histories.get(w, 0), -gains[w], w))
            selected[pick] = remaining.pop(pick)
            budget_left -= self.instance.worker(pick).window
            stat.steps += 1
            stat.score_trace.append(best_score)
        return selected, stat

    def _tiebreak(self, tied: Sequence[str], profiles: Dict[str, ParticipantProfile],
                  scores: Dict[str, float]) -> str:
        cands = [(wid, profiles[wid]) for wid in tied]
        system, user = tiebreak_prompt(self.instance, cands, scores)
        text = self.llm.complete(system, user, json_mode=True, meta={
            "kind": "tiebreak", "candidates": [profiles[w] for w in tied],
            "scores": scores, "instance": self.instance, "urban": self.urban,
            "cfg": self.cfg, "env_cfg": self.env_cfg,
        })
        picked = parse_tiebreak(text, tied)
        return picked if picked in tied else tied[0]

    # ------------------------------------------------------------------ 算法 3
    def negotiate(self, routes: Dict[str, list], profiles: Dict[str, ParticipantProfile],
                  agents: Dict[str, object]) -> Tuple[Dict[str, list], NegotiationStat]:
        cfg = self.cfg
        stat = NegotiationStat()
        current = {k: list(v) for k, v in routes.items()}
        failed: set = set()
        if not cfg.use_nrr or len(current) < 2:
            return current, stat

        for _attempt in range(cfg.neg_pairs):
            pairs = pairwise_overlap(current)
            candidates = {p: o for p, o in pairs.items() if p not in failed and o >= cfg.overlap_tau}
            if not candidates:
                break
            (u, v), overlap = max(candidates.items(), key=lambda kv: kv[1])
            stat.attempts += 1
            stat.overlap_before.append(overlap)
            feedback = {u: "", v: ""}
            success = False
            for _round in range(cfg.neg_rounds):
                stat.rounds += 1
                system, user = negotiation_offer_prompt(
                    self.instance, self.urban, current[u], current[v],
                    profiles[u], profiles[v], overlap, feedback[u], feedback[v])
                text = self.llm.complete(system, user, json_mode=True, meta={
                    "kind": "offer", "instance": self.instance, "urban": self.urban,
                    "cfg": cfg, "env_cfg": self.env_cfg, "u": self.instance.worker(u),
                    "v": self.instance.worker(v), "profile_u": profiles[u],
                    "profile_v": profiles[v], "route_u": current[u], "route_v": current[v],
                })
                stat.calls += 1
                proposed, incentives = parse_offer(text, u, v)
                if not proposed:
                    feedback = {u: "no valid proposal", v: "no valid proposal"}
                    continue
                accept = {}
                for wid in (u, v):
                    path = proposed.get(wid)
                    if not path:
                        accept[wid] = False
                        feedback[wid] = "no proposal for me"
                        stat.rejected += 1
                        continue
                    w = self.instance.worker(wid)
                    blocked = tuple(self.instance.meta.get("blocked_cells", ()) or ())
                    ok, viol = validate_route(w, path, self.instance.nx, self.instance.ny,
                                              blocked, self.instance.meta.get("cells_per_step", 1))
                    if not ok:
                        # 协调者的提案没满足对方的时间窗/端点约束时，先按路点投影回可行空间，
                        # 再交给参与者评估是否接受（保证协商讨论的是"可行"的候选）。
                        fixed = repair_from_llm(w, path, self.instance.nx, self.instance.ny, blocked,
                                                pref=profiles[wid].pref, urban=self.urban)
                        if fixed is None:
                            accept[wid] = False
                            feedback[wid] = f"proposed route is infeasible ({', '.join(viol[:2])})"
                            stat.rejected += 1
                            continue
                        path = fixed
                        proposed[wid] = fixed
                    agree, reason = self._feedback(wid, profiles[wid], path, current[wid],
                                                   incentives.get(wid, ""), feedback.get(wid, ""))
                    accept[wid] = agree
                    feedback[wid] = reason
                    stat.calls += 1
                    if not agree:
                        stat.rejected += 1
                if accept.get(u) and accept.get(v):
                    current[u], current[v] = proposed[u], proposed[v]
                    stat.accepted += 1
                    stat.overlap_after.append(route_overlap(path_cells(current[u]), path_cells(current[v])))
                    success = True
                    break
            if not success:
                failed.add((u, v))
                stat.failed_pairs += 1
        return current, stat

    def _feedback(self, wid: str, profile: ParticipantProfile, proposed, original,
                  incentive: str, memory: str) -> Tuple[bool, str]:
        worker = self.instance.worker(wid)
        system, user = negotiation_feedback_prompt(self.instance, worker, profile, proposed,
                                                  original, incentive, memory)
        text = self.llm.complete(system, user, json_mode=True, meta={
            "kind": "feedback", "instance": self.instance, "worker": worker, "profile": profile,
            "urban": self.urban, "cfg": self.cfg, "env_cfg": self.env_cfg,
            "proposed": proposed, "original": original,
        })
        from ..parse import parse_accept
        agree, reason = parse_accept(text)
        return agree, reason
