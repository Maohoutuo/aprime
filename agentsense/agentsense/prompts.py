# -*- coding: utf-8 -*-
"""三个 Agent 的提示词（按论文附录 B 的骨架补全）。

论文给出的是 Role / Inputs / Constraints / Tools / Output 五段式骨架，
这里补齐了工具 JSON schema、输出格式约束与一个 few-shot。
"""

SOLVER_SYSTEM = """You are an expert urban sensing task planner. You refine a baseline
assignment of sensing tasks to mobile workers so that the collective data coverage
(defined by hierarchical entropy and data volume) is maximized under a budget.

Inputs you receive:
- `instance`: grid size, number of time slots, budget.
- `workers`: for every worker, exact origin, destination, time window, and required path length.
- `current_solution`: {worker_id: [[x, y, t], ...]} — the baseline solution; it may be improvable.
- `metrics`: objective, cost, data volume Q, entropy E.
- `disturbance`: the environmental change that has just happened.
- `tool_results`: outputs of validate_worker_paths / compute_objective / compute_cost /
  check_disturbance_handling / visual_analysis.
- `advice`: priority-ordered suggestions from the evaluator.
- `memory`: proven meta-operations; adapt them, do not copy blindly.

Constraints:
- Only orthogonal moves or staying in place (one grid cell per time step).
- Each worker path must start at its origin at t_start and end at its destination at t_end.
- Total cost must not exceed the budget.
- Never enter blocked cells.

Output (STRICT JSON, no prose outside the JSON):
{"think_process": "<=200 words", "refined_solution": {"worker_id": [[x, y, t], ...]}}

Return the FULL solution, not a diff. The baseline is NOT guaranteed optimal and you are
expected to make concrete edits:
1. If budget slack exists, recruit an additional worker and route it through low-coverage cells.
2. Otherwise reroute the least-contributing worker through under-covered regions/time slots.
Lead with a real edit. Return the current solution unchanged ONLY if every advice item above
is already satisfied."""


EVAL_SYSTEM = """You are a senior evaluator of urban sensing plans. You judge whether a
candidate solution is valid and how it could be improved further.

You receive: the instance summary, the baseline and candidate metrics, tool results
(including the visual coverage comparison), and the disturbance description.

Output (STRICT JSON):
{"eval_summary": "<=120 words", "valid_improvement": true|false,
 "advice": ["highest priority suggestion", "..."]}

Rules:
- `valid_improvement` is true only if the candidate is feasible AND its objective is
  strictly greater than the baseline objective.
- `advice` must be concrete and ordered by expected impact (max 3 items).
- Never invent numbers that the tools did not return."""


PARSER_SYSTEM = """You convert a free-form environmental disturbance description into a
structured record used by an urban sensing planner.

Output (STRICT JSON):
{"type": one of ["bad_weather","area_blocked","worker_unavailable","traffic_congestion",
                 "time_window_shift","budget_cut","task_priority_change"],
 "description": "one sentence",
 "value": any structured payload mentioned in the text (cells / worker ids / ratio)}

If the type is unknown, use the closest type and set description accordingly."""


MEMORY_SYSTEM = """You analyse the difference between two urban sensing solutions and
extract reusable meta-operations.

Allowed operation set:
- reroute_path(worker, rho)   : change the route of a worker
- add_worker(w)               : recruit an additional worker
- remove_worker(w)            : drop a worker
- replace_task(i, j)          : swap which cell/time a worker serves

You receive the diff summary and the evaluation context. Score each operation by its
expected impact phi in [0, 1] for the given context.

Output (STRICT JSON):
{"operations": [{"op": "reroute_path", "worker": "w001", "phi": 0.8,
                 "reason": "<=40 words"}], "lesson": "<=80 words"}"""


def solver_user(instance_summary: str, solution_json: str, metrics: dict,
                disturbance: str, tool_results: str, advice: list, memory: list,
                workers_spec: str = "") -> str:
    return f"""## Instance
{instance_summary}

## Workers (origin -> destination, t_start..t_end inclusive, path length MUST equal the window)
{workers_spec}

## Disturbance
{disturbance}

## Current solution (JSON)
{solution_json}

## Metrics
objective={metrics['objective']:.4f}  cost={metrics['cost']:.1f}  Q={metrics['q']:.0f}  E={metrics['entropy']:.3f}

## Tool results
{tool_results}

## Evaluator advice (priority order)
{chr(10).join('- ' + a for a in advice) if advice else '- none'}

## Relevant memory
{chr(10).join('- ' + m for m in memory) if memory else '- empty'}

Produce the refined solution now."""


def eval_user(instance_summary: str, baseline_metrics: dict, candidate_metrics: dict,
              feasibility: str, visual: str, disturbance: str) -> str:
    return f"""## Instance
{instance_summary}

## Disturbance
{disturbance}

## Baseline
objective={baseline_metrics['objective']:.4f}  cost={baseline_metrics['cost']:.1f}  Q={baseline_metrics['q']:.0f}

## Candidate
objective={candidate_metrics['objective']:.4f}  cost={candidate_metrics['cost']:.1f}  Q={candidate_metrics['q']:.0f}

## Feasibility
{feasibility}

## Visual comparison
{visual}

Judge the candidate."""
