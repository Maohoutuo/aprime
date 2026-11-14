# -*- coding: utf-8 -*-
"""临时诊断：不同偏好平坦度下的 PSS 分布（不进入最终交付）。"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agentsense.config import DEFAULT_ENV
from mapus.data.instances import MapusInstance, make_mapus_instance
from mapus.env.metrics_obj import PathSatisfaction, path_cells, evaluate_mapus
from mapus.pipeline import run_classical
from mapus.profiles import build_profiles
from mapus.routing import baseline_route
from mapus.llm_backend import SurrogateLLM
from mapus.config import MAPUSConfig

for mix in (0.2, 0.4, 0.55, 0.7):
    inst = make_mapus_instance("tdrive_small", 0)
    profiles = build_profiles(inst.base, seed=20260325, mix_uniform=mix)
    sat0 = PathSatisfaction(inst.urban, mu=0.0)
    cos_rn, cos_pref = [], []
    cfg = MAPUSConfig()
    llm = SurrogateLLM()
    for w in inst.workers[:12]:
        sol = run_classical("RN", inst, DEFAULT_ENV, seed=1)
        prof = profiles[w.wid]
        # 纯 PSS 最优路线（λ=0）：估计 PSS 的理论可达上限
        best = baseline_route(inst, w, prof.pref, inst.urban, cfg, DEFAULT_ENV, random.Random(0), lam=0.0)
        cos_pref.append(sat0.score(prof.pref, path_cells(best)))
    for name in ("RN", "TVPG", "GraphDP"):
        sol = run_classical(name, inst, DEFAULT_ENV, seed=1)
        cos_rn.append(np.mean([sat0.score(profiles[w].pref, path_cells(p)) for w, p in sol.items()]))
    print(f"mix={mix:.2f}  base cos RN/TVPG/GraphDP = "
          f"{cos_rn[0]:.3f}/{cos_rn[1]:.3f}/{cos_rn[2]:.3f}   PSS-optimal route cos = {np.mean(cos_pref):.3f}")
