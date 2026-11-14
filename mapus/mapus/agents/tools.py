# -*- coding: utf-8 -*-
"""附录 A.2/A.4 提到的两个检索工具。"""

from __future__ import annotations

import json
from typing import Iterable, List

from ..env.urban import UrbanGrid


def retrieve_grid_geographic_attributes(urban: UrbanGrid, x: int, y: int) -> str:
    if not (0 <= x < urban.nx and 0 <= y < urban.ny):
        return f"error: cell ({x}, {y}) is outside the grid"
    return urban.summary(x, y)


def retrieve_grid_crime_info(urban: UrbanGrid, x: int, y: int) -> str:
    if not (0 <= x < urban.nx and 0 <= y < urban.ny):
        return f"error: cell ({x}, {y}) is outside the grid"
    if float(urban.crime.max()) <= 0:
        return f"cell ({x}, {y}): no crime data available for this region"
    return (f"cell ({x}, {y}): {urban.crime[x, y]:.1f} recorded incidents, "
            f"normalised risk {urban.crime_norm[x, y]:.2f}")


TOOL_REGISTRY = {
    "retrieve_grid_geographic_attributes": retrieve_grid_geographic_attributes,
    "retrieve_grid_crime_info": retrieve_grid_crime_info,
}


def run_tool_calls(urban: UrbanGrid, calls: Iterable[dict]) -> str:
    """执行 LLM 的工具请求，返回一段可直接拼回对话的文本。"""
    out: List[str] = []
    for call in calls or []:
        name = str(call.get("name", ""))
        fn = TOOL_REGISTRY.get(name)
        try:
            x, y = int(call.get("x")), int(call.get("y"))
        except (TypeError, ValueError):
            out.append(f"{name}: error: coordinates must be integers")
            continue
        out.append(f"{name}({x}, {y}) -> " + (fn(urban, x, y) if fn else f"unknown tool {name}"))
    return "\n".join(out) if out else "no tool calls"
