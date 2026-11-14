import json
import random

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.loop import refine_single


class CountingClient:
    provider = "openai"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, json_mode=True):
        self.calls += 1
        return json.dumps({"think_process": "unchanged", "refined_solution": {}})


def test_single_agent_baseline_uses_exactly_one_llm_call():
    inst = make_instance("tdrive_small", 0)
    baseline = PLANNERS["RN"](inst, DEFAULT_ENV, random.Random(42))
    client = CountingClient()
    result = refine_single(inst, baseline, {"type": "none", "description": "none"}, client)
    assert client.calls == 1  # no evaluator / no retry / no memory
    assert result.history[0]["objective"] == 0.0  # empty model output → zero coverage
    assert result.history[0]["valid_improvement"] is False
