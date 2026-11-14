# -*- coding: utf-8 -*-
"""实验公共工具：方法注册、并行执行、结果落盘。"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.config import DEFAULT_ENV
from mapus.config import DEFAULT_MAPUS, LLMConfig
from mapus.data.instances import make_mapus_instance
from mapus.env.metrics_obj import evaluate_mapus
from mapus.llm_backend import make_llm
from mapus.pipeline import run_classical, run_mapus, run_single_llm

CLASSICAL = ["RN", "TVPG", "TCPG", "MSA", "MSAGI", "GraphDP"]

# 论文表 2 / 表 3 的锚点（Coverage, PSS）
PAPER = {
    "tdrive": {
        "small": {"RN": (4.557, 0.632), "TVPG": (4.563, 0.634), "TCPG": (4.594, 0.631),
                  "MSA": (4.662, 0.630), "MSAGI": (4.661, 0.633), "GraphDP": (4.761, 0.659),
                  "GPT-4.1-mini": (4.576, 0.742), "MAPUS": (4.677, 0.875)},
        "medium": {"RN": (4.626, 0.605), "TVPG": (4.753, 0.610), "TCPG": (5.084, 0.621),
                   "MSA": (5.065, 0.622), "MSAGI": (5.052, 0.616), "GraphDP": (5.102, 0.651),
                   "GPT-4.1-mini": (4.872, 0.706), "MAPUS": (5.115, 0.875)},
        "large": {"RN": (4.974, 0.594), "TVPG": (5.067, 0.590), "TCPG": (5.043, 0.602),
                  "MSA": (5.131, 0.598), "MSAGI": (5.128, 0.590), "GraphDP": (5.214, 0.628),
                  "GPT-4.1-mini": (4.936, 0.691), "MAPUS": (5.251, 0.898)},
    },
    "grab": {
        "small": {"RN": (4.236, 0.727), "TVPG": (4.342, 0.714), "TCPG": (4.232, 0.730),
                  "MSA": (4.372, 0.744), "MSAGI": (4.392, 0.752), "GraphDP": (4.369, 0.749),
                  "GPT-4.1-mini": (4.293, 0.782), "MAPUS": (4.386, 0.854)},
        "medium": {"RN": (4.710, 0.696), "TVPG": (4.737, 0.695), "TCPG": (4.806, 0.703),
                   "MSA": (4.900, 0.699), "MSAGI": (4.887, 0.714), "GraphDP": (4.913, 0.719),
                   "GPT-4.1-mini": (4.755, 0.742), "MAPUS": (4.890, 0.834)},
        "large": {"RN": (5.399, 0.648), "TVPG": (5.406, 0.640), "TCPG": (5.587, 0.658),
                  "MSA": (5.586, 0.655), "MSAGI": (5.567, 0.653), "GraphDP": (5.611, 0.700),
                  "GPT-4.1-mini": (5.413, 0.729), "MAPUS": (5.701, 0.831)},
    },
}


def build_llm(backend: str = "surrogate", model: Optional[str] = None,
              workers: int = 8, cache: str = "results/llm_cache", no_cache: bool = False):
    if backend in ("surrogate", "mock"):
        return make_llm("mock")
    kw = LLMConfig()
    return make_llm(backend, model or kw.model, temperature=kw.temperature,
                    max_tokens=kw.max_tokens, retries=kw.retries, timeout=kw.timeout,
                    cache_dir=cache, use_cache=not no_cache)


def run_method(name: str, inst, cfg, llm, seed: int = 0) -> Dict[str, float]:
    t0 = time.time()
    if name in CLASSICAL:
        sol = run_classical(name, inst, DEFAULT_ENV, seed=seed)
        ev = evaluate_mapus(inst, sol, inst.urban, inst.prefs, DEFAULT_ENV,
                            mu=cfg.mu, prior=cfg.pss_prior, scale=cfg.pss_scale)
        ev["runtime"] = time.time() - t0
        return ev
    if name in ("GPT-4.1-mini", "single-llm", "single"):
        sol, info = run_single_llm(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm, seed=seed)
        return info
    if name == "MAPUS":
        run = run_mapus(inst, inst.urban, inst.profiles, cfg, DEFAULT_ENV, llm, seed=seed,
                        parallel=getattr(llm, "_parallel", 8))
        ev = dict(run.stats)
        ev["runtime"] = run.runtimes["total"]
        ev["_run"] = run
        return ev
    raise ValueError(name)


def aggregate(rows: Sequence[dict], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if not vals:
            continue
        out[k] = {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0,
                  "n": len(vals)}
    return out


def run_instances(fn: Callable, indices: Sequence[int], parallel: int = 1) -> List:
    if parallel <= 1:
        return [fn(i) for i in indices]
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        return list(ex.map(fn, indices))


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_default)


def _default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def fmt(stat: Dict[str, float], digits: int = 3) -> str:
    if not stat:
        return "-"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"
