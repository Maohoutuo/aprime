# -*- coding: utf-8 -*-
"""提示词：严格对应论文图 5–图 8 的模板。

每个提示词末尾附一个 ``STATE: {...}`` 机器可读块。它是给**离线替代模型**
（``SurrogateLLM``）解析用的；真实 LLM 被明确告知忽略该块。这样同一套 prompt
既能驱动真实模型，也能在无网络/省额度时用确定性代理跑通全流程。
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import LAND_USE
from ..env.urban import UrbanGrid
from ..profiles import ParticipantProfile
from agentsense.env.model import Instance, Worker

MAX_GRID_CHARS = 4096


def _state(kind: str, **kw) -> str:
    payload = {"kind": kind}
    payload.update(kw)
    return "\n\nSTATE: " + json.dumps(payload, ensure_ascii=False)


def _path_str(path: Sequence[Sequence[int]], limit: int = 64) -> str:
    shown = [[int(p[0]), int(p[1]), int(p[2])] for p in path]
    if len(shown) > limit:
        head, tail = shown[: limit // 2], shown[-(limit // 2):]
        return json.dumps(head)[:-1] + ", ... , " + json.dumps(tail)[1:]
    return json.dumps(shown)


def _grid_block(urban: UrbanGrid) -> str:
    lu = urban.land_use_text()
    crime = urban.crime_text() if float(urban.crime.max()) > 0 else "(no crime data for this region)"
    if len(lu) + len(crime) > MAX_GRID_CHARS:
        return ("(grid overview omitted: map too large; use the retrieval tools below for individual cells)")
    return f"Land-use map:\n{lu}\n\nCrime map:\n{crime}"


# --------------------------------------------------------------------------- 图 5 路线生成
def refine_prompt(instance: Instance, worker: Worker, profile: ParticipantProfile,
                  current_path: Sequence[Sequence[int]], urban: UrbanGrid,
                  instructions: str = "", blocked: Sequence[Tuple[int, int]] = ()) -> Tuple[str, str]:
    system = (
        f"You are a worker agent participating in an urban sensing task on a 2D grid of size "
        f"({instance.nx}, {instance.ny}). Time steps range from 0 to {instance.n_slots - 1}. "
        f"At each time step, you may move up to {worker.speed} grid unit(s) in total distance "
        f"(Manhattan metric), choosing among up, down, left, right, or staying in place.\n\n"
        "Your goal is to plan a feasible and realistic route from the origin to the destination, "
        "balancing efficiency, personal preferences, and overall convenience.\n\n"
        "When necessary, you may query grid-level environmental attributes to inform your planning "
        "decisions, including land-use characteristics and area safety levels, using the following tools:\n"
        "- retrieve_grid_geographic_attributes(x, y)\n"
        "- retrieve_grid_crime_info(x, y)\n\n"
        "If you want to call a tool, reply with a JSON object "
        '{"tool_calls": [{"name": "retrieve_grid_geographic_attributes", "x": 3, "y": 4}]} '
        "and nothing else. You will then receive the results."
    )
    residual = worker.window - len(current_path)
    blocked_txt = (f"\nBlocked cells to avoid: {sorted(blocked)}" if blocked else "")
    user = (
        f"Please help optimize the path for the worker traveling from {worker.origin} to {worker.dest} "
        f"(x, y, t), with a speed limit of {worker.speed} grid unit(s) per unit time.\n"
        f"The worker must start at t={worker.t_start} and finish at t={worker.t_end} "
        f"(the route must contain exactly {worker.window} steps).{blocked_txt}\n\n"
        f"Here is an initial path for the worker to begin with:\n{_path_str(current_path)}\n\n"
        f"The worker has {residual} residual step(s) remaining after completing the initial path.\n\n"
        f"The worker's profile and preferences are as follows:\n{profile.to_text()}\n\n"
        "Here are additional instructions from the coordinator agent regarding path planning:\n"
        f"Instruction: {instructions or 'None. Maximise my own route utility.'}\n"
        "Please carefully take the coordinator agent's instructions into account when generating the final path.\n\n"
        "The land-use categories are exactly: vegetation, industrial, institutional, medical, "
        "residential, commercial. Your route utility is λ·(sensing coverage utility of the route) + "
        "(1−λ)·(path satisfaction score), where the path satisfaction score is the cosine similarity "
        "between your preference distribution and the empirical land-use distribution along your route, "
        "minus a penalty for passing through high-crime cells.\n\n"
        f"{_grid_block(urban)}\n\n"
        "Please generate an optimized path following these steps:\n"
        f"1. Use the residual {residual} step(s) to add new waypoints along the initial path. These waypoints "
        "should prioritize areas that align with the worker's preferences and the attributes of the region.\n"
        "2. Refine the path to better accommodate the worker's preferred areas and time constraints.\n\n"
        "Output format:\n"
        "{\n"
        '  "final_path": [[x, y, t], ...],\n'
        '  "explanation": "reasoning process"\n'
        "}\n"
        "The path must start at the origin at t=" + str(worker.t_start) + ", end at the destination at t="
        + str(worker.t_end) + ", contain one entry per time step, and never move more than "
        f"{worker.speed} grid unit(s) per step. "
        "Do NOT include any comments or non-coordinate text inside final_path. "
        "Do NOT include any text outside the JSON object. Ignore the machine-readable STATE block."
    )
    user += _state("refine", worker=worker.wid)
    return system, user


# --------------------------------------------------------------------------- 接单决策
def accept_prompt(instance: Instance, worker: Worker, profile: ParticipantProfile,
                  budget_left: float, urban: UrbanGrid) -> Tuple[str, str]:
    system = (
        "You are an autonomous participant agent in a participatory urban sensing platform. "
        "You decide whether to accept a sensing task, based on your own itinerary, your personal "
        "preferences and your economic situation. You are not forced to accept."
    )
    user = (
        f"A sensing task is broadcast. Grid size ({instance.nx}, {instance.ny}), time steps 0.."
        f"{instance.n_slots - 1}. Your sensing shift would run from t={worker.t_start} to "
        f"t={worker.t_end} and you would travel from {worker.origin} to {worker.dest}, at "
        f"{worker.window} steps would be your sensing shift. Only "
        f"{budget_left:.0f} units of budget remain for the whole platform, so you may not be selected even "
        "if you accept; accepting simply means you are willing to take part.\n\n"
        f"Your profile:\n{profile.to_text()}\n\n"
        "Decide whether you are willing to accept the task. Consider: is the trip feasible for you; does it "
        "fit your preferences; is the reward worth your time given your economic situation? Decline only if "
        "the task is clearly unattractive or infeasible for you.\n\n"
        "Output format: {\"accept\": true/false, \"reason\": \"one short sentence\"}\n"
        "Do NOT include any text outside the JSON object. Ignore the machine-readable STATE block."
    )
    user += _state("accept", worker=worker.wid)
    return system, user


# --------------------------------------------------------------------------- 图 6 平局裁决
def tiebreak_prompt(instance: Instance, candidates: Sequence[Tuple[str, ParticipantProfile]],
                    scores: Dict[str, float]) -> Tuple[str, str]:
    system = (
        "You are a fairness-aware assistant for participant selection in an urban sensing task.\n"
        "You are given a set of candidate participants who are tied according to quantitative selection "
        "criteria and all satisfy budget and feasibility constraints. For each candidate, you are provided "
        "with a short profile containing background and historical participation information.\n"
        "Your task is to select one participant by considering high-level fairness and equity principles "
        "that are not captured by numerical scores, such as prior participation opportunities or "
        "socioeconomic context."
    )
    lines = [
        "The following candidate participants are tied under the current selection criteria. All candidates "
        "satisfy feasibility and budget constraints. For each candidate, a profile is provided including "
        "background information and historical participation records. Select one participant from the list "
        "based on fairness considerations.",
    ]
    for wid, prof in candidates:
        lines.append(f"- candidate {wid} (score {scores.get(wid, 0.0):.4f}, "
                     f"selected {prof.hired} time(s) before): {prof.to_text()}")
    lines.append("Output format: {\"selected\": \"<worker_id>\", \"reason\": \"one short sentence\"}")
    lines.append("Do NOT include any text outside the JSON object. Ignore the machine-readable STATE block.")
    user = "\n".join(lines) + _state("tiebreak", candidates=[w for w, _ in candidates])
    return system, user


# --------------------------------------------------------------------------- 图 7 协商（协调者）
def negotiation_offer_prompt(instance: Instance, urban: UrbanGrid,
                             route_u: Sequence[Sequence[int]], route_v: Sequence[Sequence[int]],
                             prof_u: ParticipantProfile, prof_v: ParticipantProfile,
                             overlap: float, feedback_u: str = "", feedback_v: str = "") -> Tuple[str, str]:
    system = (
        "You are a coordinator agent responsible for refining sensing routes through negotiation in an "
        "urban sensing task. You act as a mediator between participant agents and cannot enforce route "
        "changes; all refinements must be accepted by the participants.\n"
        "Your objective is to propose refined routes that:\n"
        "- reduce spatial overlap between participants;\n"
        "- remain feasible under task setting and constraints;\n"
        "- respect participant preferences as much as possible.\n"
        "You should iteratively improve your proposals based on feedback from previous negotiation rounds.\n\n"
        f"Task setting: grid size ({instance.nx}, {instance.ny}), time steps 0..{instance.n_slots - 1}, "
        "each participant starts at their own origin at their own first time step.\n\n"
        "When needed, you may query grid-level attributes to assess route feasibility and safety with "
        "following tools:\n- retrieve_grid_geographic_attributes(x, y)\n- retrieve_grid_crime_info(x, y)\n"
        "Use the retrieved information to guide route adjustments."
    )
    user = (
        f"Two participants have been identified with highly overlapping routes (overlap = {overlap:.3f}).\n\n"
        f"Participant u: {{ID: {prof_u.wid}, Route: {_path_str(route_u)}, Profile: {prof_u.to_text()}, "
        f"Feedback: {feedback_u or 'none'}}}\n\n"
        f"Participant v: {{ID: {prof_v.wid}, Route: {_path_str(route_v)}, Profile: {prof_v.to_text()}, "
        f"Feedback: {feedback_v or 'none'}}}\n\n"
        "Your task is to propose refined routes for both participants that reduce overlap while keeping "
        "each participant's origin and destination unchanged. Adjustments should focus on overlapping or "
        "middle segments and preserve route feasibility and realism.\n"
        "You may optionally provide short incentive messages explaining the rationale behind the proposed "
        "refinements to each participant.\n\n"
        f"{_grid_block(urban)}\n\n"
        "Both refined routes must keep the same number of steps as the original route of that participant, "
        "start at that participant's origin, end at that participant's destination, and never move more than "
        "one grid unit per step.\n\n"
        "Output format:\n"
        "{\n"
        '  "refined_routes": {"<u_id>": [[x, y, t], ...], "<v_id>": [[x, y, t], ...]},\n'
        '  "incentives": {"<u_id>": "incentive to accept refined route", "<v_id>": "..."}\n'
        "}\n"
        "Do NOT include any text outside the JSON object. Ignore the machine-readable STATE block."
    )
    user += _state("offer", u=prof_u.wid, v=prof_v.wid)
    return system, user


# --------------------------------------------------------------------------- 图 8 协商（参与者）
def negotiation_feedback_prompt(instance: Instance, worker: Worker, prof: ParticipantProfile,
                                proposed: Sequence[Sequence[int]], original: Sequence[Sequence[int]],
                                incentive: str = "", memory: str = "") -> Tuple[str, str]:
    system = (
        "You are a participant agent in an urban sensing task. Your role is to evaluate route refinements "
        "proposed by the coordinator agent and decide whether to accept them.\n"
        "Your decision should consider the following information:\n"
        f"- Task setting: grid size ({instance.nx}, {instance.ny}), time steps 0..{instance.n_slots - 1}\n"
        f"- Your schedule: origin {worker.origin}, destination {worker.dest}, "
        f"t={worker.t_start}..{worker.t_end}, speed {worker.speed}\n"
        f"- Your profile and preferences: {prof.to_text()}\n\n"
        "When making decision, consider the following criteria:\n"
        "- feasibility considering your schedule and task constraints;\n"
        "- alignment with your personal preferences; and\n"
        "- the reasonableness of the coordinator's explanation.\n"
        "You may tolerate minor inconvenience if the proposal is well justified. If you reject a proposal, "
        "clearly state the primary reason for rejection."
    )
    user = (
        "A refined route has been proposed for you.\n"
        f"- Proposed route: {_path_str(proposed)}\n"
        f"- Original route: {_path_str(original)}\n"
        f"- Coordinator message (if any): {incentive or 'none'}\n"
        f"- Feedback from previous negotiation rounds (if any): {memory or 'none'}\n\n"
        "Your task is to determine whether to accept the proposed route. Reject it if it is infeasible for "
        "you (wrong origin/destination, wrong time steps, impossible moves) or if it makes your route clearly "
        "worse for you.\n\n"
        "Output format:\n"
        "{\n"
        '  "agreement": true/false,\n'
        '  "feedback": "brief explanation of your decision. If rejected, clearly state the main reason."\n'
        "}\n"
        "Do NOT include any text outside the JSON object. Ignore the machine-readable STATE block."
    )
    user += _state("feedback", worker=worker.wid)
    return system, user


# --------------------------------------------------------------------------- 扰动广播（图 12）
def disturbance_prompt(instance: Instance, worker: Worker, profile: ParticipantProfile,
                       current_path: Sequence[Sequence[int]], event: str,
                       urban: UrbanGrid) -> Tuple[str, str]:
    system = (
        "You are a participant agent in an ongoing urban sensing task. The coordinator agent has just "
        "broadcast a real-time event. You must decide whether your already planned sensing route is "
        "affected by the event and, if necessary, adjust your route.\n"
        "Keep your origin and destination unchanged and keep the same time steps as before. If the event "
        "does not affect you, or if it affects you but cannot be avoided (for example your destination is "
        "inside the affected area), keep your original route."
    )
    user = (
        f"Broadcast from the coordinator agent: {event}\n\n"
        f"Your current planned route: {_path_str(current_path)}\n"
        f"Your schedule: origin {worker.origin}, destination {worker.dest}, "
        f"t={worker.t_start}..{worker.t_end}, speed {worker.speed}.\n"
        f"Your profile:\n{profile.to_text()}\n\n"
        f"{_grid_block(urban)}\n\n"
        "Output format:\n"
        "{\n"
        '  "adjust": true/false,\n'
        '  "final_path": [[x, y, t], ...],\n'
        '  "explanation": "why you keep or change your route"\n'
        "}\n"
        "If adjust is false, repeat your current route in final_path. The route must start at your origin at "
        f"t={worker.t_start} and end at your destination at t={worker.t_end}, one entry per time step, at most "
        f"{worker.speed} grid unit(s) per step. Do NOT include any text outside the JSON object. "
        "Ignore the machine-readable STATE block."
    )
    user += _state("disturbance", worker=worker.wid)
    return system, user


# --------------------------------------------------------------------------- 单 LLM 基线
def single_llm_prompt(instance: Instance, profiles: Dict[str, ParticipantProfile],
                      urban: UrbanGrid, candidates: Dict[str, Sequence[Sequence[int]]]) -> Tuple[str, str]:
    system = (
        "You are a single large-language-model planner for a participatory urban sensing platform. "
        "You receive the entire problem at once: the task setting, the budget, every candidate participant "
        "with their mobility constraints, personal profile and a feasible baseline route. You must directly "
        "produce the final sensing plan without any task decomposition or multi-agent coordination."
    )
    lines = [
        f"Grid size ({instance.nx}, {instance.ny}); time steps 0..{instance.n_slots - 1}; "
        f"total recruitment budget B = {instance.budget:.0f}. Each selected participant costs the number of "
        "time steps of their route (reward 1.0 per step), and the sum of costs must not exceed B.\n",
        "Objective: maximise the spatio-temporal sensing coverage of the union of the selected routes while "
        "keeping each participant's route close to their personal land-use preferences "
        "(cosine similarity between preference distribution and the empirical land-use distribution along "
        "the route, minus a penalty for high-crime cells).\n",
        "Candidates:",
    ]
    for wid, prof in profiles.items():
        w = instance.worker(wid)
        lines.append(
            f"- {wid}: origin {w.origin}, destination {w.dest}, t={w.t_start}..{w.t_end} "
            f"(cost {w.window}), speed {w.speed}; baseline route {_path_str(candidates.get(wid, []))}; "
            f"profile {prof.to_text()}"
        )
    lines.append("")
    lines.append(_grid_block(urban))
    lines.append("")
    lines.append(
        "Output format:\n{\n"
        '  "selected": [{"worker_id": "w001", "path": [[x, y, t], ...]}, ...],\n'
        '  "explanation": "reasoning process"\n}\n'
        "Total cost of the selected routes must be <= B. Each path must start at that participant's origin at "
        "their first time step, end at their destination at their last time step, contain exactly one entry "
        "per time step, and never move more than one grid unit per step. "
        "Do NOT include any text outside the JSON object. Ignore the machine-readable STATE block."
    )
    user = "\n".join(lines) + _state("single", workers=list(profiles.keys()))
    return system, user
