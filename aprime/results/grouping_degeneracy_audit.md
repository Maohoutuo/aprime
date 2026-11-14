# 分组退化审计：`gqc_gap` 的论证是否建立在退化基线上

*`experiments/audit_grouping_degeneracy.py` 生成；3 种子/格。*

判据：组数 < 2 → **退化**（`gqc_gap ≡ 0`）；最小组份额 < 15% → **不可判读**（E5.2 门槛）；否则 **可用**。

## 总览

| 实验 | 方案 | 它论证的因素 | 格数 | 退化 | 不可判读 | 可用 | 结论 |
|---|---|---|---|---|---|---|---|
| E5.15 偏好混合敏感性 | value | 公平差距是否随偏好先验移动 | 3 | 0 | 3 | 0 | 无退化但有不可判读格，需按格剔除 |
| 公平-覆盖权衡 | value | 公平差距是否随覆盖要求移动（用的是 GQCParams 默认 value 桶） | 20 | 0 | 20 | 0 | 无退化但有不可判读格，需按格剔除 |
| GQC-MRG 主实验 | quantile_tied | 公平差距是否随 MRG 开关/同质化移动 | 40 | 0 | 12 | 28 | 无退化但有不可判读格，需按格剔除 |
| GQC-MRG 敏感性 | quantile_tied | 公平差距对参数的敏感性 | 40 | 0 | 12 | 28 | 无退化但有不可判读格，需按格剔除 |
| GQC-tied 变体 | quantile_tied | tied 桶相对 value 桶的改进 | 20 | 0 | 4 | 16 | 无退化但有不可判读格，需按格剔除 |
| GQC-tied 变体 | value | （对照组：value 桶本身） | 20 | 5 | 15 | 0 | **有退化格，结论不可直接用于跨条件比较** |
| 预算标定 | quantile_tied | 名额下界随预算档位移动 | 8 | 0 | 2 | 6 | 无退化但有不可判读格，需按格剔除 |

## 问题格明细

| 实验 | 方案 | 尺度 | φ | θ | 组数 | 最小组份额 | 判定 |
|---|---|---|---|---|---|---|---|
| E5.15 偏好混合敏感性 | value | small | — | 0.0 | 2.0 | 0.050 | **不可判读** |
| E5.15 偏好混合敏感性 | value | small | — | 0.3 | 2.0 | 0.100 | **不可判读** |
| E5.15 偏好混合敏感性 | value | small | — | 0.6 | 3.0 | 0.100 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.0 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.1 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.2 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.3 | 2.0 | 0.025 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.4 | 2.0 | 0.025 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.5 | 3.0 | 0.025 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.6 | 3.0 | 0.025 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.7 | 3.0 | 0.083 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.8 | 3.0 | 0.129 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.00 | 0.9 | 3.0 | 0.136 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.0 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.1 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.2 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.3 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.4 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.5 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.6 | 2.0 | 0.050 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.7 | 2.0 | 0.025 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.8 | 3.0 | 0.025 | **不可判读** |
| 公平-覆盖权衡 | value | small | 0.95 | 0.9 | 3.0 | 0.054 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | small | 0.95 | 0.9 | 2.0 | 0.054 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.00 | 0.0 | 2.0 | 0.100 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.00 | 0.1 | 2.0 | 0.100 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.00 | 0.2 | 2.0 | 0.125 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.00 | 0.9 | 2.0 | 0.128 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.0 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.1 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.2 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.3 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.4 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.5 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 主实验 | quantile_tied | medium | 0.95 | 0.6 | 2.0 | 0.075 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | small | 0.95 | 0.9 | 2.0 | 0.054 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.00 | 0.0 | 2.0 | 0.100 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.00 | 0.1 | 2.0 | 0.100 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.00 | 0.2 | 2.0 | 0.125 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.00 | 0.9 | 2.0 | 0.128 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.0 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.1 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.2 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.3 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.4 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.5 | 2.0 | 0.025 | **不可判读** |
| GQC-MRG 敏感性 | quantile_tied | medium | 0.95 | 0.6 | 2.0 | 0.075 | **不可判读** |
| GQC-tied 变体 | quantile_tied | medium | — | 0.0 | 2.0 | 0.100 | **不可判读** |
| GQC-tied 变体 | quantile_tied | medium | — | 0.1 | 2.0 | 0.100 | **不可判读** |
| GQC-tied 变体 | quantile_tied | medium | — | 0.2 | 2.0 | 0.125 | **不可判读** |
| GQC-tied 变体 | quantile_tied | medium | — | 0.9 | 2.0 | 0.128 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.0 | 2.0 | 0.050 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.1 | 2.0 | 0.050 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.2 | 2.0 | 0.050 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.3 | 2.0 | 0.025 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.4 | 2.0 | 0.025 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.5 | 3.0 | 0.025 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.6 | 3.0 | 0.025 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.7 | 3.0 | 0.083 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.8 | 3.0 | 0.129 | **不可判读** |
| GQC-tied 变体 | value | small | — | 0.9 | 3.0 | 0.136 | **不可判读** |
| GQC-tied 变体 | value | medium | — | 0.0 | 1.0 | 1.000 | **退化** |
| GQC-tied 变体 | value | medium | — | 0.1 | 1.0 | 1.000 | **退化** |
| GQC-tied 变体 | value | medium | — | 0.2 | 1.0 | 1.000 | **退化** |
| GQC-tied 变体 | value | medium | — | 0.3 | 1.0 | 1.000 | **退化** |
| GQC-tied 变体 | value | medium | — | 0.4 | 1.0 | 1.000 | **退化** |
| GQC-tied 变体 | value | medium | — | 0.5 | 2.0 | 0.025 | **不可判读** |
| GQC-tied 变体 | value | medium | — | 0.6 | 2.0 | 0.075 | **不可判读** |
| GQC-tied 变体 | value | medium | — | 0.7 | 2.0 | 0.025 | **不可判读** |
| GQC-tied 变体 | value | medium | — | 0.8 | 3.0 | 0.026 | **不可判读** |
| GQC-tied 变体 | value | medium | — | 0.9 | 3.0 | 0.128 | **不可判读** |
| 预算标定 | quantile_tied | medium | — | 0.0 | 2.0 | 0.100 | **不可判读** |
| 预算标定 | quantile_tied | medium | — | 0.9 | 2.0 | 0.128 | **不可判读** |
