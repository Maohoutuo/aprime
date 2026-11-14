# -*- coding: utf-8 -*-
r"""环境自检：城市属性 / 画像 / PSS / 三阶段流程（用代理模型，不消耗额度）。

    py -3.14 experiments\smoke.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapus.config import DEFAULT_ENV, DEFAULT_MAPUS
from mapus.data.instances import make_mapus_instance
from mapus.env.metrics_obj import PathSatisfaction, evaluate_mapus
from mapus.llm_backend import SurrogateLLM
from mapus.pipeline import run_classical, run_mapus, run_single_llm
from mapus.routing import baseline_route, validate_route
import random


def main() -> None:
    for setting in ("tdrive_small", "grab_small", "tdrive_medium"):
        inst = make_mapus_instance(setting, 0)
        urban = inst.urban
        print(f"\n=== {setting}: grid {inst.nx}x{inst.ny}, T={inst.n_slots}, "
              f"budget={inst.budget:.0f}, workers={len(inst.workers)}")
        cats = [0] * 6
        for x in range(inst.nx):
            for y in range(inst.ny):
                cats[urban.category[x, y]] += 1
        print("   land-use cells:", dict(zip(("veg", "ind", "inst", "med", "res", "com"), cats)))
        print("   crime cells kept:", int((urban.crime > 0).sum()))

        w = inst.workers[0]
        prof = inst.profiles[w.wid]
        t0 = time.time()
        route = baseline_route(inst, w, prof.pref, urban, DEFAULT_MAPUS, DEFAULT_ENV, random.Random(0))
        ok, viol = validate_route(w, route, inst.nx, inst.ny)
        pss = PathSatisfaction(urban, DEFAULT_MAPUS.mu).score(prof.pref, [(p[0], p[1]) for p in route])
        print(f"   baseline route len={len(route)} feasible={ok} {viol[:2]} pss={pss:.3f} "
              f"({time.time()-t0:.2f}s)")

        for name in ("RN", "TVPG", "GraphDP"):
            sol = run_classical(name, inst, DEFAULT_ENV)
            ev = evaluate_mapus(inst, sol, urban, inst.prefs, DEFAULT_ENV, DEFAULT_MAPUS.mu)
            print(f"   {name:8s} cov={ev['coverage']:.3f} pss={ev['pss']:.3f} "
                  f"n={ev['n_selected']} cost={ev['cost']:.0f} overlap={ev['avg_overlap']:.3f}")

        llm = SurrogateLLM()
        t0 = time.time()
        run = run_mapus(inst, urban, inst.profiles, DEFAULT_MAPUS, DEFAULT_ENV, llm, seed=0, parallel=1)
        s = run.stats
        print(f"   MAPUS    cov={s['coverage']:.3f} pss={s['pss']:.3f} n={s['n_selected']} "
              f"accepted={s['n_accepted']} cost={s['cost']:.0f} overlap={s['avg_overlap']:.3f} "
              f"({time.time()-t0:.1f}s, calls={llm.usage['calls']})")
        print(f"            base->final pss {s['base_pss']:.3f}->{s['final_pss']:.3f}, "
              f"cover {s['base_cover']:.3f}->{s['final_cover']:.3f}, "
              f"neg {s['pre_neg_coverage']:.3f}->{s['coverage']:.3f}")

        sol, info = run_single_llm(inst, urban, inst.profiles, DEFAULT_MAPUS, DEFAULT_ENV, SurrogateLLM())
        print(f"   single   cov={info['coverage']:.3f} pss={info['pss']:.3f} n={info['n_selected']}")

    print("\nOK")


if __name__ == "__main__":
    main()
