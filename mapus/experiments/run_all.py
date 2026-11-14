# -*- coding: utf-8 -*-
r"""一键跑完整套离线实验（代理模型，不消耗 API 额度）。

    py -3.14 experiments\run_all.py                 # 全跑
    py -3.14 experiments\run_all.py --only main     # 只跑主结果

产物：results/*.json, results/*.md, results/figures/*.png, report.html
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(name: str, args: list[str]) -> None:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    proc = subprocess.run([PY, os.path.join(HERE, args[0])] + args[1:], cwd=os.path.dirname(HERE))
    print(f"--- {name}: exit={proc.returncode}, {time.time()-t0:.1f}s", flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed")


STEPS = {
    "main": [
        ("main-grab", ["run_main.py", "--dataset", "grab", "--n", "20", "--parallel", "8",
                       "--backend", "surrogate"]),
        ("main-tdrive", ["run_main.py", "--dataset", "tdrive", "--n", "20", "--parallel", "8",
                         "--backend", "surrogate"]),
    ],
    "ablation": [("ablation", ["run_ablation.py", "--dataset", "tdrive", "--n", "20",
                               "--parallel", "8", "--backend", "surrogate"])],
    "negotiation": [("negotiation", ["run_negotiation.py", "--n", "20", "--parallel", "6",
                                     "--backend", "surrogate"])],
    "fairness": [("fairness", ["run_fairness.py", "--n-tasks", "60", "--backend", "surrogate"])],
    "runtime": [("runtime", ["run_runtime.py", "--n", "5", "--backend", "surrogate"])],
    "case": [("case-study", ["run_case_study.py", "--backend", "surrogate"])],
    "report": [("figures", ["make_figures.py"]), ("report", ["build_report.py"])],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔：main,ablation,negotiation,fairness,"
                                               "runtime,case,report")
    args = ap.parse_args()
    keys = [k for k in args.only.split(",") if k] or list(STEPS)
    for k in keys:
        for name, cmd in STEPS[k]:
            run(name, cmd)


if __name__ == "__main__":
    main()
