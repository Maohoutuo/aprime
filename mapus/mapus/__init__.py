# -*- coding: utf-8 -*-
"""MAPUS: Language-Grounded Multi-Agent Planning for Personalized and Fair
Participatory Urban Sensing (arXiv:2603.24014v1) —— 复现工程根包。

本包只新增论文中 MAPUS 特有的部分（城市属性、参与者画像、PSS 指标、三个
LLM Agent 与协调流程）。PUS 环境、覆盖目标函数与 6 个经典基线直接复用同目录
下的 ``agentsense`` 工程（同一课题组的前置工作，指标口径与 MAPUS 表 2/3 一致）。
"""

from .config import MAPUSConfig, LLMConfig, DEFAULT_MAPUS, DEFAULT_LLM

__all__ = ["MAPUSConfig", "LLMConfig", "DEFAULT_MAPUS", "DEFAULT_LLM"]
__version__ = "0.1.0"
