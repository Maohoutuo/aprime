# -*- coding: utf-8 -*-
r"""标定 PSS 的两个不可恢复常数（a, μ）。

论文式 (2)：S(r_u) = sim(p_u, h(r_u)) − μ·risk(r_u)。其中
* h(r_u) 需要原始 POI 数据（论文未公开），
* μ 与 PSS 的绝对尺度都没有给出。
因此这里沿用 E(D) 的 e_scale 标定思路：**只用论文表 2 中 T-Drive/Small 的
6 个基线 PSS 作为锚点**，最小二乘拟合
      PSS ≈ a · cos(p_u, h(r_u)) − μ · risk(r_u)
Medium / Large 与 Grab-Posisi 全部留出不参与拟合。

    py -3.14 experiments\calibrate.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.config import DEFAULT_ENV
from mapus.config import DEFAULT_MAPUS
from mapus.data.instances import make_mapus_instance
from mapus.env.metrics_obj import PathSatisfaction, path_cells
from mapus.pipeline import run_classical

# 论文表 2：T-Drive / Small 的 PSS（锚点）
ANCHORS = {"RN": 0.632, "TVPG": 0.634, "TCPG": 0.631, "MSA": 0.630, "MSAGI": 0.633, "GraphDP": 0.659}
ORDER = list(ANCHORS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--prior", type=float, default=6.0, help="h(r) 的 Laplace 平滑强度 κ")
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    cos_map, risk_map = {}, {}
    for name in ORDER:
        cos_v, risk_v = [], []
        for i in range(args.n):
            inst = make_mapus_instance("tdrive_small", i)
            sat = PathSatisfaction(inst.urban, mu=0.0, prior=args.prior, scale=1.0)
            sol = run_classical(name, inst, DEFAULT_ENV, seed=1000 + i)
            for wid, path in sol.items():
                cells = path_cells(path)
                cos_v.append(sat.land_use(cells) @ inst.profiles[wid].pref /
                             (np.linalg.norm(sat.land_use(cells)) *
                              np.linalg.norm(inst.profiles[wid].pref) + 1e-12))
                risk_v.append(sat.risk(cells))
        cos_map[name], risk_map[name] = float(np.mean(cos_v)), float(np.mean(risk_v))
        print(f"{name:8s} cos={cos_map[name]:.4f} risk={risk_map[name]:.4f} anchor={ANCHORS[name]:.3f}")

    # μ 固定（论文未给出，取 0.5 这一稳健值）；只标定尺度 a，使其对齐 6 个锚点的均值。
    # 说明：锚点彼此只差 0.03（0.630–0.659），而本实现的 cos 在不同基线间差异更大，
    # 因此 (a, μ) 联合最小二乘会退化到 μ→0（见 docs/calibration.md 的讨论），
    # 这里不追求拟合锚点差异，只对齐**水平**，方法间的差异留作复现结果报告。
    mu = float(DEFAULT_MAPUS.mu)
    target = float(np.mean([ANCHORS[k] for k in ORDER]))
    mean_cos = float(np.mean([cos_map[k] for k in ORDER]))
    mean_risk = float(np.mean([risk_map[k] for k in ORDER]))
    a = (target + mu * mean_risk) / mean_cos
    print(f"\nfitted: scale a={a:.4f}  (mu fixed at {mu}, target mean anchor {target:.4f})")
    for k in ORDER:
        ours = a * cos_map[k] - mu * risk_map[k]
        print(f"  {k:8s} ours={ours:.4f}  anchor={ANCHORS[k]:.3f}  diff={ours-ANCHORS[k]:+.3f}")
    rmse = float(np.sqrt(np.mean([(a * cos_map[k] - mu * risk_map[k] - ANCHORS[k]) ** 2 for k in ORDER])))

    payload = {"prior": args.prior, "scale": a, "mu": mu, "rmse": rmse,
               "anchors": ANCHORS, "cosine": {k: round(v, 4) for k, v in cos_map.items()},
               "risk": {k: round(v, 4) for k, v in risk_map.items()},
               "fitted_pss": {k: round(a * cos_map[k] - mu * risk_map[k], 4) for k in ORDER},
               "n": args.n}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
