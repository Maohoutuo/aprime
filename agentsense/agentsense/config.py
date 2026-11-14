# -*- coding: utf-8 -*-
"""论文中的固定实验配置（表 2）。

AgentSense: LLMs Empower Generalizable and Explainable
Web-Based Participatory Urban Sensing (arXiv:2510.19661v2)

约定
----
* 空间网格离散为 nx × ny，时间轴按 slot_minutes 切分。
* 一个"感知任务" = 网格中的一个 (x, y, t) 单元（location–time cell）。
* 每个时间步的归一化报酬为 1.0（论文 4.1.1），因此预算 B 的上限等价于
  "所有被招募工作者的路径时间步之和"。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Setting:
    name: str
    dataset: str
    scale: str
    n_workers: int
    grid_x: int
    grid_y: int
    total_budget: float
    horizon_min: int
    slot_min: int
    n_instances: int = 20
    #: 每个时间步可移动的格数（速度上限）。**实测值**，口径唯一：
    #: 相邻时槽、位移非零的曼哈顿位移中位数（``experiments/measure_speed.py``）。
    #:   tdrive: 8×8/15min → 3，16×16/15min → 7，32×32/15min → 13
    #:   grab  : 8×4/5min → 2，16×8/5min → 4，32×16/5min → 8
    #: grab 官方轨迹尚未获得，其值用 T-Drive 在"同一 bbox、同一网格、同一时槽"
    #: 下实测的位移作代理（等价于把物理速度换算到 grab 的格子尺寸上），
    #: 拿到官方数据后需重跑 measure_speed.py 替换，见 docs/assumptions.md。
    #: 该字段是成本口径的唯一真源：生成器写进 meta["cells_per_step"]，
    #: 并直接作为 ``Worker.speed``；所有步数一律走 ``steps_for()``（= ceil(d/cps)）。
    cells_per_step: int = 1

    @property
    def n_slots(self) -> int:
        return self.horizon_min // self.slot_min

    @property
    def n_regions(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def n_tasks(self) -> int:
        return self.n_regions * self.n_slots


# 表 2：Details of datasets
# 注意：cells_per_step 是实测值（experiments/measure_speed.py），不是假设值。
SETTINGS = {
    ("tdrive", "small"): Setting("tdrive_small", "tdrive", "small", 20, 8, 8, 40, 120, 15,
                                 cells_per_step=3),
    ("tdrive", "medium"): Setting("tdrive_medium", "tdrive", "medium", 40, 16, 16, 60, 240, 15,
                                  cells_per_step=7),
    ("tdrive", "large"): Setting("tdrive_large", "tdrive", "large", 60, 32, 32, 100, 360, 15,
                                 cells_per_step=13),
    ("grab", "small"): Setting("grab_small", "grab", "small", 15, 8, 4, 40, 40, 5,
                               cells_per_step=2),
    ("grab", "medium"): Setting("grab_medium", "grab", "medium", 30, 16, 8, 60, 80, 5,
                                cells_per_step=4),
    ("grab", "large"): Setting("grab_large", "grab", "large", 45, 32, 16, 100, 160, 5,
                               cells_per_step=8),
}

# 论文默认参数
ALPHA = 0.5              # J(D) = alpha * E(D) + (1 - alpha) * log2 Q(D)
REWARD_PER_STEP = 1.0    # 每个时间步的归一化报酬
MAX_ITERATIONS = 10      # 改进迭代上限（论文 4.1.2 的 SR 定义）
N_TRIALS = 20            # 每个设置 20 个任务实例


@dataclass
class EnvConfig:
    """目标函数与成本口径。论文未给出的自由度集中在这里，便于标定。"""

    alpha: float = ALPHA
    # 数据量 Q 的口径：distinct = 不同的 (x,y,t) 单元数；visits = 访问总次数
    q_mode: str = "distinct"
    # 层次化熵的口径：
    #   relative: H / log2(min(#bins, 观测到的样本数)) 的均值 × e_scale   ← 默认
    #   norm    : H / log2(#bins) 的均值 × e_scale
    #   equal   : (H_time + H_space + H_space_time) / 3   （bit 均值）
    #   sum     : H_time + H_space + H_space_time          （bit 求和）
    #   weighted: 按 e_weights 加权
    e_mode: str = "relative"
    e_weights: tuple = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    # 归一化熵的整体尺度（relative / norm 时生效）。
    e_scale: float = 4.2
    # 成本口径：每条被招募路径的时间步数之和（报酬 1.0/步）
    reward_per_step: float = REWARD_PER_STEP
    # 注意：真正的速度口径在 Setting.cells_per_step（实测值，按数据集/规模不同），
    # 生成器会把它写进每个 Worker.speed；这里的字段只是历史占位，不要用它做计算。
    cells_per_step: int = 1


DEFAULT_ENV = EnvConfig()
