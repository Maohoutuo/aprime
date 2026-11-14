# 实现与论文的逐项对照

```
论文组件                         代码位置
──────────────────────────────── ────────────────────────────────────────────────
式 (4) 六类土地利用              mapus/env/urban.py: LAND_USE, _field, _assign_categories
式 (3) h(r_u)                    mapus/env/urban.py: UrbanGrid.route_land_use
式 (2) PSS                       mapus/env/metrics_obj.py: PathSatisfaction
式 (12)(13) 覆盖效用 φ(A)        agentsense/env/objective.py: Coverage（复用，已标定）
式 (7) 路线重叠                  mapus/env/metrics_obj.py: route_overlap

图 10 画像模板                   mapus/profiles.py: ParticipantProfile.to_text
附录 B.2 五个 archetype          mapus/profiles.py: ARCHETYPES
附录 B.1 土地利用 + 犯罪          mapus/env/urban.py: make_urban_grid

图 5 / 算法 1 提示词             mapus/agents/prompts.py: refine_prompt
算法 1 控制流                    mapus/agents/participant.py: generate_route
TSPTW 基线 R0                    mapus/routing.py: baseline_route
VALID(·)                         mapus/routing.py: validate_route
不可行解投影                     mapus/routing.py: repair_from_llm

图 6 / 算法 2 平局裁决           mapus/agents/prompts.py: tiebreak_prompt
算法 2 控制流                    mapus/agents/coordinator.py: select
式 (5)(6) Δφ / F / J             mapus/agents/coordinator.py: select

图 7 协商（协调者）               mapus/agents/prompts.py: negotiation_offer_prompt
图 8 协商（参与者）               mapus/agents/prompts.py: negotiation_feedback_prompt
算法 3 控制流                    mapus/agents/coordinator.py: negotiate

附录 A.2 工具                    mapus/agents/tools.py
图 2 三阶段总装                  mapus/pipeline.py: run_mapus
表 1 配置                        agentsense/config.py: SETTINGS（复用）
六个经典基线                      agentsense/baselines/planners.py（复用）
单 LLM 基线                      mapus/pipeline.py: run_single_llm
附录 B.6 扰动案例                mapus/pipeline.py: run_disturbance
```

## 数据流

```
make_mapus_instance(setting, index)
  ├── Worker 列表（起点/终点/时间窗/速度）        agentsense.data.instances
  ├── Instance（网格、时域、预算）                 agentsense.env.model
  ├── UrbanGrid（土地利用 6 类 + 犯罪风险）        mapus.env.urban
  └── ParticipantProfile（画像 + 偏好向量 p_u）    mapus.profiles

run_mapus
  ├── 阶段 1（PA，可并行）
  │     decide_accept → generate_route（算法 1）
  │       R0 = baseline_route(λ=1, 只优化覆盖)
  │       for i in 1..Imax: LLM 精修 → VALID → 投影 → U(R) 比较，取更优
  ├── 阶段 2（CA，算法 2）
  │     Δφ(u) / F(u) → 归一化 → J(u) → 平局时 LLM 裁决
  └── 阶段 3（CA + PA，算法 3）
        找最大重叠对 → CA 提案 → PA 逐个投票 → 双方都接受才采纳
```

## 三个"必须这样实现，否则机制失效"的点

1. **经典基线只优化覆盖**（`baseline_route(lam=1.0)`）。如果经典规划器已经按偏好选点，
   LLM 精修就没有提升空间（实测 `final_pss == base_pss`），PRG 的消融也失去意义。
2. **停留补齐要偏好感知**（`routing.pad_pref`）。论文允许 "staying in place"，而
   `h(r_u)` 按**时间步**统计，所以把剩余步数停在偏好最契合的格子是提升 PSS 的直接机制。
   基线用均匀补齐（中性），精修用偏好补齐。
3. **协商提案先投影再投票**。协调者直接输出的 `(x,y,t)` 序列几乎总会在某个参与者
   的时间窗上越界；不投影的话协商接受率为 0，NRR 形同虚设（实测 0/30）。

## 并行与可复现

* 阶段 1 的参与者彼此独立 → `ThreadPoolExecutor`；
* 真实 LLM 调用有**磁盘缓存**（`results/llm_cache/<hash>.json`），同一 prompt 只付费一次，
  因此实验可以中断续跑、也可以完全复现；
* 离线代理 `SurrogateLLM` 完全确定性（seed 由 worker id 的 CRC32 派生），
  用于单元测试与不消耗额度的全量扫描；
* `mapus/env/fast_coverage.py` 把 agentsense 的 `Coverage.q` 换成 NumPy 向量化实现
  （语义逐位一致，有单元测试保证），否则 SA 在 Large 档的 Python 三重循环会慢两个数量级。
