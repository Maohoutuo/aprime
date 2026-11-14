# -*- coding: utf-8 -*-
"""A′：个性化满意度硬约束下的路线级参与人选择。

本包只依赖同级的 ``mapus`` 与 ``agentsense``（复用其网格、真实轨迹实例与 PSS 口径），
不重复实现数据层。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _dep in ("agentsense", "mapus"):
    _p = os.path.join(_ROOT, _dep)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["config", "model", "candidates", "solvers", "analysis"]
