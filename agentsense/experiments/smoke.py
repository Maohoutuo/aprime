# -*- coding: utf-8 -*-
"""冒烟测试：单实例跑通所有基线，打印目标值/成本/覆盖。"""

import random
import time

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.env.model import validate_solution
from agentsense.env.objective import evaluate

SETTING = "tdrive_small"
IDX = 0

inst = make_instance(SETTING, IDX)
print(f"{inst.setting_name} #{IDX} workers={len(inst.workers)} grid={inst.grid} "
      f"slots={inst.n_slots} budget={inst.budget}")
print("windows:", sorted({w.window for w in inst.workers}))
print("distances:", sorted({abs(w.origin[0]-w.dest[0])+abs(w.origin[1]-w.dest[1]) for w in inst.workers}))

for name, fn in PLANNERS.items():
    t0 = time.time()
    sol = fn(inst, DEFAULT_ENV, random.Random(12345))
    dt = time.time() - t0
    vr = validate_solution(inst, sol)
    ev = evaluate(inst, sol, DEFAULT_ENV)
    print(f"{name:8s} J={ev['objective']:.3f} cost={ev['cost']:5.1f}/{inst.budget:.0f} "
          f"Q={ev['q']:5.1f} H={ev['entropy']:.2f} workers={ev['n_workers']:2d} "
          f"feasible={vr.feasible} {dt*1000:.0f}ms")
    if not vr.feasible:
        print("   violations:", vr.violations[:4])
