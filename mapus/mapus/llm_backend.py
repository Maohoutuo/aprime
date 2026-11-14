# -*- coding: utf-8 -*-
"""LLM 后端：远程（OpenAI 兼容 / 硅基流动）+ 磁盘缓存 + 确定性代理。

设计目的
--------
* ``RemoteLLM``：真实模型调用，带重试与 **磁盘缓存**（同一 prompt 只付费一次），
  便于实验可重跑、可复现、可分批。
* ``SurrogateLLM``：无网络时的确定性替身。它直接执行提示词所描述的优化目标
  （λ·覆盖 + (1−λ)·PSS 的局部搜索），用来在**不消耗额度**的情况下跑通全流程与
  单元测试。报告中一律标注为 ``surrogate``，不与真实 LLM 结果混为一谈。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agentsense.config import EnvConfig
from agentsense.env.model import Instance, Worker, shortest_path
from agentsense.llm import LLMClient, load_dotenv

from .env.metrics_obj import PathSatisfaction, path_cells
from .env.urban import UrbanGrid
from .routing import _reachable_cells, route_cells, route_utility, cells_to_path
from .config import MAPUSConfig


# --------------------------------------------------------------------------- 基类
class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str, json_mode: bool = True,
                 meta: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    @property
    def usage(self) -> dict:
        return {"calls": 0, "cached": 0}


# --------------------------------------------------------------------------- 远程 + 缓存
class RemoteLLM(BaseLLM):
    def __init__(self, provider: str, model: str, temperature: float = 0.5,
                 max_tokens: int = 4096, retries: int = 4, timeout: float = 180.0,
                 cache_dir: str = "results/llm_cache", use_cache: bool = True,
                 verbose: bool = False):
        self.name = f"{provider}:{model}"
        self.client = LLMClient(provider=provider, model=model, temperature=temperature,
                                max_tokens=max_tokens, retries=retries, timeout=timeout,
                                verbose=verbose)
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.verbose = verbose
        self._lock = threading.Lock()
        self._calls = 0
        self._cached = 0
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, user: str, json_mode: bool) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(b"|json" if json_mode else b"|text")
        h.update(system.encode())
        h.update(user.encode())
        return h.hexdigest()

    def complete(self, system: str, user: str, json_mode: bool = True,
                 meta: Optional[Dict[str, Any]] = None) -> str:
        key = self._key(system, user, json_mode)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.use_cache and path.exists():
            with self._lock:
                self._cached += 1
            try:
                return json.loads(path.read_text(encoding="utf-8"))["response"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        last = None
        for attempt in range(3):
            try:
                text = self.client.complete(system, user, json_mode=json_mode)
                break
            except RuntimeError as exc:      # 含 HTTP 429 / 5xx，客户端内部已重试过
                last = exc
                # TPM/并发限流是按分钟计的，短暂重试没用，必须等一段时间
                wait = 20.0 * (2 ** attempt)
                if self.verbose:
                    print(f"[llm] retry in {wait:.0f}s: {str(exc)[:120]}", flush=True)
                time.sleep(wait)
        else:
            raise last  # type: ignore[misc]
        with self._lock:
            self._calls += 1
        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": self.name, "response": text}, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, path)
        return text

    @property
    def usage(self) -> dict:
        base = self.client.usage
        base.update({"cached": self._cached, "remote_calls": self._calls})
        return base


def _deviations(worker: Worker, path) -> list:
    """取出路径中"偏离起点→终点最短路"的格子，作为路点（去重保序）。"""
    if not path:
        return []
    base = set(shortest_path(worker.origin, worker.dest))
    out = []
    for step in path:
        cell = (int(step[0]), int(step[1]))
        if cell in base or cell == worker.origin or cell == worker.dest:
            continue
        if not out or out[-1] != cell:
            out.append(cell)
    return out


def make_llm(provider: str = "mock", model: str = "mock", **kw) -> BaseLLM:
    if provider in ("mock", "surrogate", "offline"):
        return SurrogateLLM()
    load_dotenv()
    return RemoteLLM(provider=provider, model=model, **kw)


# --------------------------------------------------------------------------- 代理
class SurrogateLLM(BaseLLM):
    """按提示词语义执行确定性优化，作为真实 LLM 的离线替身。"""

    name = "surrogate"

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = 0

    # ---------------------------------------------------------------- 入口
    def complete(self, system: str, user: str, json_mode: bool = True,
                 meta: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            self._calls += 1
        meta = meta or {}
        kind = meta.get("kind")
        if kind == "refine":
            return self._refine(meta)
        if kind == "accept":
            return json.dumps({"accept": True, "reason": "feasible and the reward fits my schedule"})
        if kind == "tiebreak":
            return self._tiebreak(meta)
        if kind == "offer":
            return self._offer(meta)
        if kind == "feedback":
            return self._feedback(meta)
        if kind == "single":
            return self._single(meta)
        if kind == "disturbance":
            return self._disturbance(meta)
        return json.dumps({"error": f"surrogate has no policy for kind={kind}"})

    @property
    def usage(self) -> dict:
        return {"calls": self._calls, "cached": 0}

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def _rng(*keys) -> random.Random:
        # 不能用内置 hash()：字符串哈希在每个进程里都被随机化，会破坏可复现性
        raw = "|".join(str(k) for k in keys).encode("utf-8")
        return random.Random(zlib.crc32(raw))

    # ---------------------------------------------------------------- 局部搜索
    def _plan(self, instance: Instance, worker: Worker, pref: np.ndarray, urban: UrbanGrid,
              cfg: MAPUSConfig, env_cfg: EnvConfig, wait: Optional[list] = None,
              rng: random.Random | None = None,
              init_waypoints: Optional[list] = None) -> list:
        """局部搜索：插入/删除路点，最大化 λ·φ_route + (1−λ)·PSS − ρ·(与待避让路线的重合)。

        候选池按"偏好类别的格子优先"排序，模拟 LLM 依据画像与土地利用做定向选点；
        每轮先尝试插入，若无法改进再尝试删除，避免只增不减。
        """
        avoid = set(wait or ())
        rng = rng or random.Random(0)
        waypoints: list = list(init_waypoints or [])
        best_cells = route_cells(worker, waypoints) or route_cells(worker, [])
        if best_cells is None:
            return []
        best = self._score(instance, worker, pref, urban, cfg, env_cfg, best_cells, avoid)

        reach = _reachable_cells(instance, worker)
        if not reach:
            return cells_to_path(worker, best_cells, pref=pref, urban=urban, pref_pad=True)
        ranked = sorted(reach, key=lambda c: -float(pref[urban.category[c]]))
        pool = ranked[: max(8, len(reach) // 6)]
        extra = [c for c in rng.sample(reach, min(len(reach), 24)) if c not in pool]
        pool = pool + extra

        for _round in range(8):
            improved = False
            for cand in pool:
                cells = route_cells(worker, waypoints + [cand])
                if cells is None:
                    continue
                val = self._score(instance, worker, pref, urban, cfg, env_cfg, cells, avoid)
                if val > best + 1e-9:
                    best, best_cells, best_wp, improved = val, cells, waypoints + [cand], True
            if improved:
                waypoints = best_wp
                continue
            for i in range(len(waypoints)):
                trial = waypoints[:i] + waypoints[i + 1:]
                cells = route_cells(worker, trial)
                if cells is None:
                    continue
                val = self._score(instance, worker, pref, urban, cfg, env_cfg, cells, avoid)
                if val > best + 1e-9:
                    best, best_cells, waypoints, improved = val, cells, trial, True
                    break
            if not improved:
                break
        return cells_to_path(worker, best_cells, pref=pref, urban=urban, pref_pad=True)

    def _score(self, instance, worker, pref, urban, cfg, env_cfg, cells, avoid) -> float:
        path = cells_to_path(worker, cells, pref=pref, urban=urban, pref_pad=True)
        util = route_utility(instance, path, pref, urban, cfg, env_cfg)
        if avoid:
            overlap = len(set(cells) & avoid) / max(1, len(set(cells)))
            util -= 1.2 * overlap
        return util

    # ---------------------------------------------------------------- 各类策略
    def _refine(self, meta) -> str:
        worker = meta["worker"]
        init = meta.get("current_path") or []
        waypoints = _deviations(worker, init)
        path = self._plan(meta["instance"], worker, meta["profile"].pref, meta["urban"],
                          meta["cfg"], meta["env_cfg"], rng=self._rng("refine", worker.wid),
                          init_waypoints=waypoints)
        return json.dumps({"final_path": path,
                           "explanation": "added waypoints that raise route utility under the time window"})

    def _tiebreak(self, meta) -> str:
        cands = meta.get("candidates") or []
        if not cands:
            return json.dumps({"selected": "", "reason": "no candidates"})
        best = min(cands, key=lambda p: (p.hired, p.wid))
        return json.dumps({"selected": best.wid,
                           "reason": f"{best.wid} has been selected least often so far"})

    def _offer(self, meta) -> str:
        instance, urban, cfg, env_cfg = meta["instance"], meta["urban"], meta["cfg"], meta["env_cfg"]
        u, v = meta["u"], meta["v"]
        ru = [tuple(c) for c in path_cells(meta["route_u"])]
        rv = [tuple(c) for c in path_cells(meta["route_v"])]
        nu = self._plan(instance, u, meta["profile_u"].pref, urban, cfg, env_cfg,
                        wait=rv, rng=self._rng("offer-u", u.wid, v.wid))
        nv = self._plan(instance, v, meta["profile_v"].pref, urban, cfg, env_cfg,
                        wait=[tuple(c) for c in path_cells(nu)],
                        rng=self._rng("offer-v", u.wid, v.wid))
        return json.dumps({
            "refined_routes": {u.wid: nu, v.wid: nv},
            "incentives": {u.wid: "this detour avoids the overlapping corridor and keeps your preference "
                                   "areas reachable",
                           v.wid: "this variant removes the duplicated segment between you and your partner"},
        })

    def _feedback(self, meta) -> str:
        instance, worker, cfg, env_cfg = meta["instance"], meta["worker"], meta["cfg"], meta["env_cfg"]
        old = meta["original"]
        new = meta["proposed"]
        pref = meta["profile"].pref
        urban = meta["urban"]
        u_old = route_utility(instance, old, pref, urban, cfg, env_cfg)
        u_new = route_utility(instance, new, pref, urban, cfg, env_cfg)
        agree = u_new >= u_old - 0.05
        reason = ("the refined route keeps my preferences and is feasible" if agree
                  else "the refined route makes my own trip clearly worse")
        return json.dumps({"agreement": agree, "feedback": reason})

    def _disturbance(self, meta) -> str:
        """扰动广播：被封锁的格子绕开；高价值区域在可行时纳入。
        目的地落在受影响区域内时保持原路线（与论文图 12 的 edge case 一致）。"""
        instance, worker, cfg, env_cfg = meta["instance"], meta["worker"], meta["cfg"], meta["env_cfg"]
        urban, prof = meta["urban"], meta["profile"]
        current = meta.get("current_path") or []
        blocked = {tuple(c) for c in meta.get("blocked", ())}
        targets = {tuple(c) for c in meta.get("targets", ())}
        cells = [(int(p[0]), int(p[1])) for p in current]
        if blocked:
            if worker.dest in blocked:
                return json.dumps({"adjust": False, "final_path": current,
                                   "explanation": "my destination lies inside the affected area, "
                                                  "so no feasible detour exists"})
            if not (blocked & set(cells)):
                return json.dumps({"adjust": False, "final_path": current,
                                   "explanation": "the affected area is not on my route"})
            instance.meta["blocked_cells"] = blocked
            wait = set(blocked)
        else:
            wait = set()
        if targets:
            # 优先把高价值格子排进路线：缩短到目标的绕行
            wait = set()
        waypoints = _deviations(worker, current)
        saved_w = None
        if targets:
            target = min(targets, key=lambda c: abs(c[0] - worker.origin[0]) + abs(c[1] - worker.origin[1]))
            waypoints = waypoints + [target]
            # 让"覆盖高价值区域"进入参与者的路线效用（否则代理模型没有动机绕行）
            saved_w = instance.meta.get("cell_weights")
            w = np.ones((instance.nx, instance.ny))
            for (x, y) in targets:
                w[max(0, x - 1):x + 2, max(0, y - 1):y + 2] = 4.0
            instance.meta["cell_weights"] = w
        path = self._plan(instance, worker, prof.pref, urban, cfg, env_cfg, wait=wait,
                          rng=self._rng("dist", worker.wid), init_waypoints=waypoints)
        if blocked:
            instance.meta.pop("blocked_cells", None)
        if saved_w is not None:
            instance.meta["cell_weights"] = saved_w
        elif targets:
            instance.meta.pop("cell_weights", None)
        return json.dumps({"adjust": True, "final_path": path,
                           "explanation": "replanned my route around the broadcast information"})

    def _single(self, meta) -> str:
        instance, env_cfg, cfg = meta["instance"], meta["env_cfg"], meta["cfg"]
        candidates: Dict[str, list] = meta["candidates"]
        chosen: Dict[str, list] = {}
        used = 0.0
        # 覆盖贪心 + 预算约束（单模型的"一次性"解）
        from agentsense.env.objective import Coverage
        cov = Coverage(instance)
        remaining = dict(candidates)
        while remaining:
            best = None
            for wid, path in remaining.items():
                w = instance.worker(wid)
                if used + w.window > instance.budget + 1e-9:
                    continue
                cov.add_path(path)
                gain = cov.objective(env_cfg)
                cov.add_path(path, sign=-1)
                if best is None or gain > best[0]:
                    best = (gain, wid, path)
            if best is None:
                break
            _, wid, path = best
            cov.add_path(path)
            chosen[wid] = path
            used += instance.worker(wid).window
            remaining.pop(wid)
        return json.dumps({"selected": [{"worker_id": k, "path": v} for k, v in chosen.items()],
                           "explanation": "one-shot greedy plan"})
