# -*- coding: utf-8 -*-
"""候选路线池 C_u：**与 θ 无关**，一次生成、θ 扫 10 档全程复用（方案 §7.4）。

生成策略（对应方案 §2.4 的「经典规划器保证 C_u 非空且可行」）：

1. ``lam=1`` —— 纯覆盖最优的经典路线（保证池里一定有一条"覆盖好"的）；
2. ``lam∈{0.75,0.5,0.25,0}`` —— 覆盖/满意度不同权重下的路线（覆盖"个性化谱系"）；
3. 随机路点采样 —— 增加几何多样性，避免池子全是同一条路线的微调。

之后按**多样性**挑 k 条：先放覆盖最优与满意度最优各一条，再反复加入"与已选路线
最小 Jaccard 距离最大"的那条。这样 θ 提高时被筛掉的才是真正"低满意度"的路线，
而不是被生成顺序偏置的路线。
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from agentsense.config import DEFAULT_ENV
from agentsense.env.model import Worker, build_path, steps_for

from mapus.config import DEFAULT_MAPUS, MAPUSConfig
from mapus.env.metrics_obj import PathSatisfaction, path_cells
from mapus.routing import baseline_route, cells_to_path

from .model import Candidate

CellXY = Tuple[int, int]


def _reachable(instance, w: Worker) -> List[CellXY]:
    cache = instance.meta.setdefault("_reach_cache", {})
    if w.wid in cache:
        return cache[w.wid]
    out: List[CellXY] = []
    for x in range(instance.nx):
        for y in range(instance.ny):
            if steps_for(w.origin, (x, y), w.speed) + steps_for((x, y), w.dest, w.speed) <= w.window - 1:
                out.append((x, y))
    cache[w.wid] = out
    return out


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _routes_for_worker(instance, w: Worker, pref, urban, cfg: MAPUSConfig, env_cfg,
                       rng: random.Random, n_random: int = 4,
                       lams: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0)) -> List[Tuple[List, str]]:
    routes: List[Tuple[List, str]] = []
    seen: Set[frozenset] = set()

    def _add(path, tag):
        if not path:
            return
        key = frozenset(path_cells(path))
        if key in seen:
            return
        seen.add(key)
        routes.append((path, tag))

    for lam in lams:
        path = baseline_route(instance, w, pref, urban, cfg, env_cfg, rng, lam=lam)
        _add(path, f"lam={lam}")

    pool = _reachable(instance, w)
    if pool:
        for _ in range(n_random):
            m = rng.randint(1, 3)
            wps = rng.sample(pool, min(m, len(pool)))
            cells = build_path(w, wps)
            if cells is None:
                continue
            _add(cells_to_path(w, cells), "random")
    return routes


def build_candidates(instance, k: int = 12, seed: int = 0,
                     cfg: Optional[MAPUSConfig] = None, env_cfg=DEFAULT_ENV,
                     workers: Optional[Iterable[str]] = None,
                     lams: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
                     n_random: int = 4) -> List[Candidate]:
    """为一个实例生成与 θ 无关的候选池（每个参与者至多 k 条）。"""
    cfg = cfg or DEFAULT_MAPUS
    sat = PathSatisfaction(instance.urban, mu=cfg.mu, prior=cfg.pss_prior, scale=cfg.pss_scale)
    out: List[Candidate] = []
    for i, w in enumerate(instance.workers):
        if workers is not None and w.wid not in set(workers):
            continue
        rng = random.Random(seed * 100003 + i)
        pref = instance.profiles[w.wid].pref
        raw = _routes_for_worker(instance, w, pref, instance.urban, cfg, env_cfg, rng,
                                 n_random=n_random, lams=lams)
        cands: List[Candidate] = []
        for path, tag in raw:
            cells_xy = path_cells(path)
            cands.append(Candidate(
                wid=w.wid,
                path=tuple((int(s[0]), int(s[1]), int(s[2])) for s in path),
                cells=frozenset(cells_xy),
                sat=float(sat.score(pref, cells_xy)),
                tag=tag,
                space_time=frozenset((int(s[0]), int(s[1]), int(s[2])) for s in path),
            ))
        out.extend(_diverse(cands, k))
    return out


def _diverse(cands: Sequence[Candidate], k: int) -> List[Candidate]:
    """按"覆盖最优 + 满意度最优 + 最大最小 Jaccard 距离"挑 k 条。"""
    if len(cands) <= k:
        return list(cands)
    chosen: List[Candidate] = []
    by_cov = max(cands, key=lambda c: (len(c.cells), c.sat))
    chosen.append(by_cov)
    by_sat = max(cands, key=lambda c: (c.sat, len(c.cells)))
    if by_sat not in chosen:
        chosen.append(by_sat)
    while len(chosen) < k:
        best, best_key = None, None
        for c in cands:
            if c in chosen:
                continue
            d = min(_jaccard(set(c.cells), set(o.cells)) for o in chosen)
            key = (d, c.sat)  # 距离优先，其次满意度
            if best is None or key > best_key:
                best, best_key = c, key
        if best is None:
            break
        chosen.append(best)
    return chosen
