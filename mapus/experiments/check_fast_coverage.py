# -*- coding: utf-8 -*-
"""验证向量化 Coverage.q 与原实现逐位一致。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.env.objective import Coverage
from mapus.env.fast_coverage import _orig_q, patch_coverage
from mapus.data.instances import make_mapus_instance


def main() -> None:
    inst = make_mapus_instance("tdrive_medium", 0)
    cov = Coverage(inst)
    for w in inst.workers:
        cov.add_path([[w.origin[0], w.origin[1], w.t_start + i] for i in range(w.window)])
    for qm in ("distinct", "visits"):
        a = _orig_q(cov, qm)
        patch_coverage()
        b = cov.q(qm)
        assert abs(a - b) < 1e-9, (qm, a, b)

    pool = make_mapus_instance("tdrive_medium", 1)
    pool.base.meta["cell_weights"] = np.random.default_rng(0).uniform(
        0.2, 1.0, size=(pool.nx, pool.ny))
    c = Coverage(pool)
    for w in pool.workers:
        c.add_path([[w.origin[0], w.origin[1], w.t_start + i] for i in range(w.window)])
    for qm in ("distinct", "visits"):
        a = _orig_q(c, qm)
        patch_coverage()
        b = c.q(qm)
        assert abs(a - b) < 1e-9, (qm, a, b)
    print("fast_coverage: 与原实现一致")


if __name__ == "__main__":
    main()
