# -*- coding: utf-8 -*-
"""快速看三档规模上基线目标值的量级（对应论文表 4 的 Base Obj.）。"""

import random
import time

from agentsense.baselines.planners import PLANNERS
from agentsense.config import DEFAULT_ENV
from agentsense.data.instances import make_instance
from agentsense.env.model import validate_solution
from agentsense.env.objective import evaluate

PAPER = {
    ("tdrive", "small"): {"RN": 4.505, "TVPG": 4.565, "TCPG": 4.588, "MSA": 4.662, "MSAGI": 4.702, "GraphDP": 4.780},
    ("tdrive", "medium"): {"RN": 4.706, "TVPG": 4.761, "TCPG": 5.118, "MSA": 5.086, "MSAGI": 5.090, "GraphDP": 5.205},
    ("tdrive", "large"): {"RN": 5.029, "TVPG": 5.602, "TCPG": 5.544, "MSA": 5.631, "MSAGI": 5.682, "GraphDP": 5.701},
}

for name in ("TVPG", "MSAGI", "GraphDP", "RN"):
    print(f"--- {name} ---")
    for scale in ("small", "medium", "large"):
        inst = make_instance(f"tdrive_{scale}", 0)
        t0 = time.time()
        sol = PLANNERS[name](inst, DEFAULT_ENV, random.Random(12345))
        dt = time.time() - t0
        vr = validate_solution(inst, sol)
        ev = evaluate(inst, sol, DEFAULT_ENV)
        ref = PAPER[("tdrive", scale)][name]
        print(f"  {scale:6s} ours={ev['objective']:.3f} paper={ref:.3f} "
              f"cost={ev['cost']:5.1f}/{inst.budget:.0f} Q={ev['q']:5.0f} H={ev['entropy']:.2f} "
              f"feas={vr.feasible} {dt:.1f}s")
