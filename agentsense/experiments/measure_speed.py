#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实测每个设置下"每时间步可移动格数"（cells_per_step），写进 SETTINGS 的口径来源。

为什么要测：Worker.speed 是预算/可行性/覆盖三处成本口径的唯一真源（``steps_for``）。
若它恒为 1，参与者就被迫"每 15 min 只走 1 格"，与真实轨迹差一个数量级：
- 真实的参与者一步能跨好几格，时间窗内的"剩余步数"因此远多于假设；
- 剩余步数正是偏好插入（PRG）、协商绕行（NRR）的物理基础。

口径（唯一一种，避免歧义）：

    cells_per_step(setting) = median{ manhattan(c_t, c_{t+1}) : 相邻时槽且位移 > 0 }

* 只取**相邻时槽**（gap = 1），避免用多槽位移除以槽数而低估速度；
* 只取**非零位移**：``cells_per_step`` 是"速度上限"，停车/等客的零位移对
  衡量速度没有意义（T-Drive 中约 10–15% 的相邻槽对位移为 0）。

来源说明：

* ``tdrive_*``：真实 T-Drive 清洗数据（``data/tdrive/tdrive_clean.txt.gz``）；
* ``grab_*``：**官方 Grab-Posisi 数据尚未获得**，用 T-Drive 在"同一 bbox、同一
  网格分辨率、同一 5 min 时槽"下实测的位移作代理——它等价于"把北京出租车的
  物理速度换算到 grab 的格子尺寸上"。拿到官方数据后必须重跑本脚本替换。

用法：

    py -3.14 experiments\\measure_speed.py
    py -3.14 experiments\\measure_speed.py --json speed_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.config import SETTINGS, Setting
from agentsense.data.instances import load_tdrive_traces

TDRIVE_PATH = os.environ.get(
    "TDRIVE_PATH", os.path.join("D:\\composition", "data", "tdrive", "tdrive_clean.txt.gz"))
BBOX = (116.20, 39.80, 116.55, 40.05)


def displacements(setting: Setting) -> List[int]:
    traces = load_tdrive_traces(TDRIVE_PATH, bbox=BBOX,
                                grid=(setting.grid_x, setting.grid_y),
                                slot_min=setting.slot_min)
    out: List[int] = []
    for pts in traces.values():
        pts = sorted(pts)
        for i in range(1, len(pts)):
            if pts[i][0] - pts[i - 1][0] != 1:
                continue
            d = abs(pts[i][1][0] - pts[i - 1][1][0]) + abs(pts[i][1][1] - pts[i - 1][1][1])
            if d > 0:
                out.append(d)
    return out


def measure(verbose: bool = True) -> Tuple[Dict[str, int], Dict[str, dict]]:
    cps: Dict[str, int] = {}
    detail: Dict[str, dict] = {}
    if verbose:
        print("| setting | grid | slot_min | pairs | p50 | p75 | cells_per_step |")
        print("|---|---|---|---|---|---|---|")
    for key in [("tdrive", "small"), ("tdrive", "medium"), ("tdrive", "large"),
                ("grab", "small"), ("grab", "medium"), ("grab", "large")]:
        setting = SETTINGS[key]
        d = displacements(setting)
        if not d:
            raise SystemExit(f"{setting.name}: 没有可用的相邻时槽样本，检查 T-DRIVE 数据")
        p50 = float(statistics.median(d))
        p75 = float(statistics.quantiles(d, n=4)[2]) if len(d) > 3 else p50
        value = max(1, int(round(p50)))
        cps[setting.name] = value
        detail[setting.name] = {
            "grid": [setting.grid_x, setting.grid_y], "slot_min": setting.slot_min,
            "pairs": len(d), "p50": p50, "p75": p75, "cells_per_step": value,
            "source": ("tdrive_real" if setting.dataset == "tdrive"
                       else "proxy(tdrive @same grid/slot)"),
        }
        if verbose:
            print("| %s | %d×%d | %d | %d | %.1f | %.1f | **%d** |"
                  % (setting.name, setting.grid_x, setting.grid_y, setting.slot_min,
                     len(d), p50, p75, value))
    return cps, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="把明细写到该路径（默认只打印）")
    args = ap.parse_args()

    if not os.path.exists(TDRIVE_PATH):
        print("T-Drive 清洗文件缺失：", TDRIVE_PATH, file=sys.stderr)
        return 1
    cps, detail = measure()
    print("\n-- 建议写入 SETTINGS 的 cells_per_step --")
    for name, v in cps.items():
        print(f"   {name:14s} = {v}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"cells_per_step": cps, "detail": detail}, fh, ensure_ascii=False, indent=2)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
