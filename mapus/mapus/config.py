# -*- coding: utf-8 -*-
"""MAPUS 的超参数与实验配置。

论文给出数值的部分直接落地（表 1 的 6 个配置、式 1/2 的形式、式 5/6 的
β 与 F(u)），论文未给出的自由度集中在本文件，便于二次标定与复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 复用 agentsense 的表 2 配置（与 MAPUS 的表 1 完全一致）
from agentsense.config import SETTINGS, Setting  # noqa: F401  (re-export)
from agentsense.config import DEFAULT_ENV, MAX_ITERATIONS, N_TRIALS, EnvConfig

# --------------------------------------------------------------------------- 土地利用
# 式 (4)：六个土地利用类别
LAND_USE = ("vegetation", "industrial", "institutional", "medical", "residential", "commercial")
LAND_USE_CODE = {name: i for i, name in enumerate(LAND_USE)}
LAND_USE_GLYPH = ("V", "I", "E", "M", "R", "C")  # 提示词里的紧凑网格


@dataclass
class MAPUSConfig:
    """MAPUS 的超参数（论文式 1、2、6 与算法 1–3）。"""

    # ---- 式 (11)：系统效用与参与者满意的权衡
    eta: float = 0.5
    # ---- 算法 1：路径局部效用 U = λ·φ_route + (1-λ)·S(r)
    lam: float = 0.5
    # ---- 式 (2)：PSS = a·cos(p_u, h(r_u)) − μ·risk(r_u)
    # 三个常数由论文表 2 的 6 个基线锚点拟合（见 docs/calibration.md），
    # 与 E(D) 的 e_scale 标定同一思路：只拟合不可恢复的尺度，不拟合方法差异。
    mu: float = 0.5
    pss_prior: float = 6.0
    # 余弦相似度的尺度 a。cells_per_step 改为实测速度后重新标定为 1.186
    # （原 cps=1 时为 1.276 → 1.30），取整为 1.20，见 docs/calibration.md。
    pss_scale: float = 1.20
    # ---- 算法 1：LLM 精修轮数 Imax
    max_refine: int = 2
    # ---- 式 (6)：覆盖增益与公平的权衡 β
    beta: float = 0.5
    # ---- 算法 2：平局阈值 ε（归一化后）
    tie_eps: float = 0.02
    # ---- 算法 3：重叠阈值 τ、单对最大轮数 K、最大尝试对数 N
    # 2026-11-10 定档：实测速度下路线重叠地板降到 0.015–0.03，τ=0.10 会让 NRR 在
    # T-Drive 几乎不触发，故取 docs/calibration.md 建议的 0.05（敏感性见该文件 τ 节）。
    overlap_tau: float = 0.05
    neg_rounds: int = 3
    neg_pairs: int = 5
    # ---- 参与者是否真的调用 LLM 决定"接单"（False 时按可行性规则接单）
    llm_accept: bool = True
    # ---- 平局裁决：llm / rule
    tie_break: str = "llm"
    # ---- 是否把紧凑网格概览注入提示词（等价于工具的批量查询）
    inject_grid_overview: bool = True
    # ---- 工具调用最多几轮补充查询
    max_tool_rounds: int = 1
    # ---- LLM 建议路径不可行时的处理：repair（投影回可行空间，保留语义）或 reject（丢弃）
    infeasible_policy: str = "repair"

    # ---- 消融开关（图 3）
    use_prg: bool = True   # Preference-aware Route Generation
    use_fps: bool = True   # Fairness-aware Participant Selection
    use_nrr: bool = True   # Negotiation-based Route Refinement


DEFAULT_MAPUS = MAPUSConfig()


@dataclass
class LLMConfig:
    """LLM 后端配置。mock 用于离线自测；siliconflow/openai 走 OpenAI 兼容接口。"""

    provider: str = "mock"          # mock | siliconflow | openai | deepseek | anthropic
    model: str = "mock"
    temperature: float = 0.5
    max_tokens: int = 4096
    retries: int = 4
    timeout: float = 180.0
    # 并发线程数（参与者之间互相独立）
    workers: int = 8
    # 磁盘缓存：同一 (model, prompt) 只真正调用一次
    cache: str = "results/llm_cache"
    # 论文默认基座 gpt-4.1-mini-2025-04-14（本机通过 OpenAI 兼容网关使用同等规模模型）
    dataset: str = "tdrive"


DEFAULT_LLM = LLMConfig()
