from .urban import UrbanGrid, make_urban_grid, N_CAT
from .metrics_obj import PathSatisfaction, coverage_utility, route_pss, evaluate_mapus
from .fast_coverage import patch_coverage

# 导入即生效：让 agentsense 的 Coverage.q 走向量化实现（语义一致，快 ~100x）
patch_coverage()

__all__ = ["UrbanGrid", "make_urban_grid", "N_CAT",
           "PathSatisfaction", "coverage_utility", "route_pss", "evaluate_mapus"]
