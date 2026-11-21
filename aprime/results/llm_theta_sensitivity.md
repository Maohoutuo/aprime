# LLM 候选生成：θ 敏感性（E4.2 的 θ × 模型 × 规模维度）

命中率 ＝ OPT 选中的参与者里其最优路线落入 $C_u$ 的比例；A′/OPT ＝ A′ 在 $C_u$ 上的覆盖 ÷ 全池 ILP 最优。

## 真实 LLM：命中率随 θ 的变化（跨模型 × 规模）

| 模型 | 规模 | θ=0.00 | θ=0.30 | θ=0.60 | θ=0.90 | 单调性（命中率） |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | small | — | 0.313 | 0.464 | 0.691 | **单调不减（随 θ 上升）** |
| `deepseek-ai/DeepSeek-V3.2` | small | — | 0.291 | 0.496 | 0.693 | **单调不减（随 θ 上升）** |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.349 | 0.329 | 0.520 | 0.715 | **非单调** |

**读法**：若各模型（各规模）的「单调性」列一致为「单调不减」，则“命中率随 θ 上升”是跨模型/跨规模成立的性质，而非单个模型的特例。

### 各模型 A′/OPT 随 θ

| 模型 | 规模 | θ=0.00 | θ=0.30 | θ=0.60 | θ=0.90 |
|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | small | — | 0.919 | 0.914 | 0.934 |
| `deepseek-ai/DeepSeek-V3.2` | small | — | 0.923 | 0.925 | 0.945 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.920 | 0.925 | 0.930 | 0.952 |

## 全模式明细

| 模型 | 规模 | θ | 生成器 | ε | 命中率 | A′/OPT | n |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | small | 0.30 | `full` | 0.00 | 1.000 | 0.989 | 10 |
| `Qwen/Qwen3.8-27B` | small | 0.30 | `llm` | 0.00 | 0.313 | 0.919 | 10 |
| `Qwen/Qwen3.8-27B` | small | 0.60 | `full` | 0.00 | 1.000 | 0.978 | 10 |
| `Qwen/Qwen3.8-27B` | small | 0.60 | `llm` | 0.00 | 0.464 | 0.914 | 10 |
| `Qwen/Qwen3.8-27B` | small | 0.90 | `full` | 0.00 | 1.000 | 0.986 | 10 |
| `Qwen/Qwen3.8-27B` | small | 0.90 | `llm` | 0.00 | 0.691 | 0.934 | 10 |
| `deepseek-ai/DeepSeek-V3.2` | small | 0.30 | `full` | 0.00 | 1.000 | 0.989 | 10 |
| `deepseek-ai/DeepSeek-V3.2` | small | 0.30 | `llm` | 0.00 | 0.291 | 0.923 | 10 |
| `deepseek-ai/DeepSeek-V3.2` | small | 0.60 | `full` | 0.00 | 1.000 | 0.978 | 10 |
| `deepseek-ai/DeepSeek-V3.2` | small | 0.60 | `llm` | 0.00 | 0.496 | 0.925 | 10 |
| `deepseek-ai/DeepSeek-V3.2` | small | 0.90 | `full` | 0.00 | 1.000 | 0.986 | 10 |
| `deepseek-ai/DeepSeek-V3.2` | small | 0.90 | `llm` | 0.00 | 0.693 | 0.945 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.00 | `full` | 0.00 | 1.000 | 0.989 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.00 | `llm` | 0.00 | 0.349 | 0.920 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.30 | `full` | 0.00 | 1.000 | 0.989 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.30 | `llm` | 0.00 | 0.329 | 0.925 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.60 | `full` | 0.00 | 1.000 | 0.978 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.60 | `llm` | 0.00 | 0.520 | 0.930 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.90 | `full` | 0.00 | 1.000 | 0.986 | 10 |
| `deepseek-ai/DeepSeek-V4-Flash` | small | 0.90 | `llm` | 0.00 | 0.715 | 0.952 | 10 |
| `sim` | medium | 0.00 | `full` | 0.00 | 1.000 | 0.978 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.00 | 0.243 | 0.824 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.10 | 0.292 | 0.821 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.20 | 0.358 | 0.838 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.30 | 0.424 | 0.851 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.40 | 0.526 | 0.868 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.50 | 0.625 | 0.896 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.60 | 0.707 | 0.908 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.70 | 0.740 | 0.923 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.80 | 0.815 | 0.944 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 0.90 | 0.875 | 0.965 | 10 |
| `sim` | medium | 0.00 | `oracle_eps` | 1.00 | 1.000 | 0.999 | 10 |
| `sim` | medium | 0.00 | `random` | 0.00 | 0.227 | 0.843 | 10 |
| `sim` | medium | 0.00 | `topk_cov` | 0.00 | 0.849 | 0.969 | 10 |
| `sim` | medium | 0.00 | `topk_sat` | 0.00 | 0.376 | 0.887 | 10 |
| `sim` | medium | 0.30 | `full` | 0.00 | 1.000 | 0.977 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.00 | 0.238 | 0.822 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.10 | 0.292 | 0.828 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.20 | 0.353 | 0.841 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.30 | 0.419 | 0.854 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.40 | 0.526 | 0.875 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.50 | 0.614 | 0.898 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.60 | 0.701 | 0.911 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.70 | 0.723 | 0.926 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.80 | 0.799 | 0.946 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 0.90 | 0.864 | 0.964 | 10 |
| `sim` | medium | 0.30 | `oracle_eps` | 1.00 | 1.000 | 0.998 | 10 |
| `sim` | medium | 0.30 | `random` | 0.00 | 0.261 | 0.843 | 10 |
| `sim` | medium | 0.30 | `topk_cov` | 0.00 | 0.876 | 0.969 | 10 |
| `sim` | medium | 0.30 | `topk_sat` | 0.00 | 0.393 | 0.887 | 10 |
| `sim` | medium | 0.60 | `full` | 0.00 | 1.000 | 0.982 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.00 | 0.201 | 0.817 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.10 | 0.255 | 0.823 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.20 | 0.320 | 0.844 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.30 | 0.386 | 0.847 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.40 | 0.499 | 0.865 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.50 | 0.598 | 0.896 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.60 | 0.696 | 0.909 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.70 | 0.728 | 0.924 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.80 | 0.793 | 0.941 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 0.90 | 0.853 | 0.959 | 10 |
| `sim` | medium | 0.60 | `oracle_eps` | 1.00 | 1.000 | 0.999 | 10 |
| `sim` | medium | 0.60 | `random` | 0.00 | 0.244 | 0.832 | 10 |
| `sim` | medium | 0.60 | `topk_cov` | 0.00 | 0.832 | 0.968 | 10 |
| `sim` | medium | 0.60 | `topk_sat` | 0.00 | 0.392 | 0.894 | 10 |
| `sim` | medium | 0.90 | `full` | 0.00 | 1.000 | 0.984 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.00 | 0.233 | 0.775 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.10 | 0.314 | 0.800 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.20 | 0.430 | 0.813 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.30 | 0.501 | 0.831 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.40 | 0.620 | 0.865 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.50 | 0.643 | 0.881 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.60 | 0.713 | 0.898 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.70 | 0.768 | 0.928 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.80 | 0.810 | 0.941 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 0.90 | 0.886 | 0.962 | 10 |
| `sim` | medium | 0.90 | `oracle_eps` | 1.00 | 1.000 | 1.000 | 10 |
| `sim` | medium | 0.90 | `random` | 0.00 | 0.315 | 0.809 | 10 |
| `sim` | medium | 0.90 | `topk_cov` | 0.00 | 0.854 | 0.969 | 10 |
| `sim` | medium | 0.90 | `topk_sat` | 0.00 | 0.539 | 0.920 | 10 |
| `sim` | small | 0.00 | `full` | 0.00 | 1.000 | 0.989 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.00 | 0.326 | 0.919 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.10 | 0.392 | 0.925 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.20 | 0.440 | 0.936 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.30 | 0.505 | 0.945 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.40 | 0.602 | 0.948 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.50 | 0.681 | 0.948 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.60 | 0.728 | 0.962 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.70 | 0.795 | 0.969 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.80 | 0.844 | 0.977 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 0.90 | 0.905 | 0.980 | 10 |
| `sim` | small | 0.00 | `oracle_eps` | 1.00 | 1.000 | 0.995 | 10 |
| `sim` | small | 0.00 | `random` | 0.00 | 0.250 | 0.922 | 10 |
| `sim` | small | 0.00 | `topk_cov` | 0.00 | 0.443 | 0.937 | 10 |
| `sim` | small | 0.00 | `topk_sat` | 0.00 | 0.231 | 0.898 | 10 |
| `sim` | small | 0.30 | `full` | 0.00 | 1.000 | 0.989 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.00 | 0.329 | 0.915 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.10 | 0.367 | 0.925 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.20 | 0.432 | 0.926 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.30 | 0.521 | 0.956 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.40 | 0.544 | 0.951 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.50 | 0.638 | 0.955 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.60 | 0.690 | 0.953 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.70 | 0.785 | 0.962 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.80 | 0.831 | 0.970 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 0.90 | 0.919 | 0.975 | 10 |
| `sim` | small | 0.30 | `oracle_eps` | 1.00 | 1.000 | 0.997 | 10 |
| `sim` | small | 0.30 | `random` | 0.00 | 0.253 | 0.920 | 10 |
| `sim` | small | 0.30 | `topk_cov` | 0.00 | 0.428 | 0.937 | 10 |
| `sim` | small | 0.30 | `topk_sat` | 0.00 | 0.262 | 0.898 | 10 |
| `sim` | small | 0.60 | `full` | 0.00 | 1.000 | 0.978 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.00 | 0.270 | 0.879 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.10 | 0.338 | 0.886 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.20 | 0.405 | 0.904 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.30 | 0.418 | 0.914 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.40 | 0.551 | 0.937 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.50 | 0.699 | 0.948 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.60 | 0.733 | 0.962 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.70 | 0.800 | 0.958 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.80 | 0.895 | 0.977 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 0.90 | 0.919 | 0.980 | 10 |
| `sim` | small | 0.60 | `oracle_eps` | 1.00 | 1.000 | 0.995 | 10 |
| `sim` | small | 0.60 | `random` | 0.00 | 0.267 | 0.895 | 10 |
| `sim` | small | 0.60 | `topk_cov` | 0.00 | 0.528 | 0.931 | 10 |
| `sim` | small | 0.60 | `topk_sat` | 0.00 | 0.359 | 0.897 | 10 |
| `sim` | small | 0.90 | `full` | 0.00 | 1.000 | 0.986 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.00 | 0.307 | 0.819 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.10 | 0.342 | 0.822 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.20 | 0.452 | 0.851 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.30 | 0.544 | 0.888 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.40 | 0.601 | 0.898 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.50 | 0.638 | 0.892 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.60 | 0.664 | 0.907 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.70 | 0.763 | 0.940 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.80 | 0.852 | 0.960 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 0.90 | 0.916 | 0.972 | 10 |
| `sim` | small | 0.90 | `oracle_eps` | 1.00 | 1.000 | 0.998 | 10 |
| `sim` | small | 0.90 | `random` | 0.00 | 0.282 | 0.805 | 10 |
| `sim` | small | 0.90 | `topk_cov` | 0.00 | 0.608 | 0.923 | 10 |
| `sim` | small | 0.90 | `topk_sat` | 0.00 | 0.608 | 0.916 | 10 |

