# -*- coding: utf-8 -*-
"""临时诊断：h(r) 的 Laplace 平滑强度 κ 对 PSS 区间的影响（不进入最终交付）。"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agentsense.config import DEFAULT_ENV
from mapus.data.instances import make_mapus_instance
from mapus.env.metrics_obj import PathSatisfaction, path_cells
from mapus.pipeline import run_classical
from mapus.routing import baseline_route
from mapus.config import MAPUSConfig

inst = make_mapus_instance("tdrive_small", 0)
u = inst.urban
q = np.bincount(u.category.ravel(), minlength=6) / u.category.size


def h_smooth(cells, kappa):
    h = u.route_land_use(cells)
    return (len(cells) * h + kappa * q) / (len(cells) + kappa)


def cos(p, cells, kappa):
    h = h_smooth(cells, kappa)
    return float(p @ h / (np.linalg.norm(p) * np.linalg.norm(h) + 1e-12))


cfg = MAPUSConfig()
for kappa in (0.0, 2.0, 4.0, 8.0):
    row = []
    for name in ("RN", "TVPG", "GraphDP"):
        sol = run_classical(name, inst, DEFAULT_ENV, seed=1)
        row.append(np.mean([cos(inst.profiles[w].pref, path_cells(p), kappa) for w, p in sol.items()]))
    best = []
    for w in inst.workers[:12]:
        p = inst.profiles[w.wid]
        r = baseline_route(inst, w, p.pref, u, cfg, DEFAULT_ENV, random.Random(0), lam=0.0)
        best.append(cos(p.pref, path_cells(r), kappa))
    print(f"kappa={kappa:4.1f}  RN/TVPG/GraphDP = {row[0]:.3f}/{row[1]:.3f}/{row[2]:.3f}   "
          f"optimal = {np.mean(best):.3f}   spread = {np.mean(best)-row[0]:.3f}")
