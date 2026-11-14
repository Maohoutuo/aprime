import json
import random

from agentsense.agents import SolverAgent
from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.disturbances import apply_disturbance
from agentsense.loop import refine


class StubClient:
    """返回空解，用于验证扰动后含失效 worker 的基线不会崩溃。"""

    provider = "openai"

    def complete(self, system, user, json_mode=True):
        return json.dumps({"think_process": "stub", "refined_solution": {}})


def test_worker_unavailable_baseline_is_sanitized():
    inst = make_instance("tdrive_small", 0)
    base = PLANNERS["RN"](inst, DEFAULT_ENV, random.Random(3))
    rng = random.Random(7)
    dist_inst = apply_disturbance(inst, "worker_unavailable", rng)
    removed = {w.wid for w in inst.workers} - {w.wid for w in dist_inst.workers}
    assert removed & set(base), "test premise: baseline references removed workers"

    # 不应抛出 KeyError
    res = refine(dist_inst, base, {"type": "worker_unavailable", "description": "x"},
                 client=StubClient(), max_iter=1)
    assert res is not None


def test_solver_prompt_tolerates_unknown_worker_ids():
    inst = make_instance("tdrive_small", 0)
    solver = SolverAgent(StubClient(), DEFAULT_ENV)
    bad_solution = {"w999": [[0, 0, 0]]}
    cand, _ = solver.propose(inst, bad_solution, {"type": "none", "description": "n"},
                             [], [], random.Random(0))
    assert isinstance(cand, dict)
