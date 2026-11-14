# -*- coding: utf-8 -*-
r"""估算「真实 LLM 版实验」的调用量、token 与费用（本身不产生 API 花费）。

方法
----
* **调用次数**：直接取已跑完的真实 LLM 主表（``results/main_*_qwen72b_n3.json``）里的计数
  （``n_accepted``/``n_refused``/``llm_calls_gen``/``sel_llm_ties``/``neg_attempts``），
  消融变体按开关去掉对应调用（w/o PRG 无精修、w/o FPS 无平局裁决、w/o NRR 无协商）。
* **每次调用的提示词长度**：用「记录型代理」在本地跑同样的实例，它不联网但走完全相同的
  提示词构造路径，因此能精确量出每类调用（accept/refine/tiebreak/offer/feedback）的字符数。
* **每次调用的回复长度**：把这些提示词拿到磁盘缓存里反查真实 Qwen2.5-72B 的回复，取其长度。
* **换算**：token ≈ 字符 / ratio（默认 3.6 入 / 3.5 出，Qwen BPE 对英文提示词的经验值）；
  单价取 SiliconFlow 站点 Qwen2.5-72B-Instruct 的 \$0.59/M ≈ ¥4.13/M（输入输出同价）。

    py -3.14 experiments\estimate_llm_cost.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsense.config import DEFAULT_ENV
from mapus.config import MAPUSConfig
from mapus.data.instances import make_pool_instance, make_mapus_instance
from mapus.llm_backend import SurrogateLLM
from mapus.pipeline import run_mapus

CACHE = "results/llm_cache"
MODEL = "siliconflow:Qwen/Qwen2.5-72B-Instruct"
SCALES = ("small", "medium", "large")
DATASETS = ("tdrive", "grab")
VARIANTS = {"Full": (), "w/o PRG": ("refine",), "w/o FPS": ("tiebreak",),
            "w/o NRR": ("offer", "feedback")}


# --------------------------------------------------------------------------- 缓存
def load_cache() -> Dict[str, int]:
    index: Dict[str, int] = {}
    for root, _dirs, files in os.walk(CACHE):
        for fn in files:
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(root, fn), encoding="utf-8") as fh:
                        index[fn[:-5]] = len(json.load(fh).get("response", ""))
                except (OSError, json.JSONDecodeError):
                    continue
    return index


def prompt_key(system: str, user: str, json_mode: bool) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"|json" if json_mode else b"|text")
    h.update(system.encode())
    h.update(user.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- 记录型代理
class RecordingSurrogate(SurrogateLLM):
    def __init__(self, cache: Dict[str, int]):
        super().__init__()
        self.cache = cache
        self.records: List[Tuple[str, int, int]] = []      # (kind, in_chars, out_chars|0)

    def complete(self, system: str, user: str, json_mode: bool = True, meta=None) -> str:
        kind = (meta or {}).get("kind") or "?"
        out = self.cache.get(prompt_key(system, user, json_mode), 0)
        self.records.append((kind, len(system) + len(user), out))
        return super().complete(system, user, json_mode=json_mode, meta=meta)


def record_profile(setting: str, cache: Dict[str, int], instances=(0, 1)) -> Dict[str, dict]:
    """返回 {kind: {"calls", "in_chars"(每调用均值), "out_chars"(每调用均值)}}。"""
    buckets: Dict[str, List[int]] = defaultdict(list)
    outs: Dict[str, List[int]] = defaultdict(list)
    for i in instances:
        llm = RecordingSurrogate(cache)
        inst = make_mapus_instance(setting, i)
        run_mapus(inst, inst.urban, inst.profiles, MAPUSConfig(), DEFAULT_ENV, llm,
                  seed=1000 + i, parallel=1)
        for kind, in_chars, out_chars in llm.records:
            buckets[kind].append(in_chars)
            if out_chars:
                outs[kind].append(out_chars)
    all_out = [v for vs in outs.values() for v in vs] or [200]
    prof = {}
    for kind, vals in buckets.items():
        prof[kind] = {
            "calls": len(vals),
            "in_chars": statistics.mean(vals),
            "out_chars": statistics.mean(outs[kind]) if outs.get(kind) else statistics.mean(all_out),
            "hit": len(outs.get(kind, [])) / max(1, len(vals)),
        }
    return prof


def fairness_profile(cache: Dict[str, int]) -> Dict[str, dict]:
    """公平性实验的单任务（30 人各生成一条路线；UR/LS/CG/FPS 的选择不调 LLM）。"""
    from mapus.agents.participant import ParticipantAgent
    llm = RecordingSurrogate(cache)
    pool = make_pool_instance("tdrive", "medium", pool_size=30, task_index=0, seed=31415)
    for w in pool.workers:
        pa = ParticipantAgent(pool, w, pool.profiles[w.wid], pool.urban, MAPUSConfig(),
                              DEFAULT_ENV, llm, seed=0 * 31 + int(w.wid[1:]))
        pa.generate_route()
    buckets: Dict[str, List[int]] = defaultdict(list)
    outs: Dict[str, List[int]] = defaultdict(list)
    for kind, in_chars, out_chars in llm.records:
        buckets[kind].append(in_chars)
        if out_chars:
            outs[kind].append(out_chars)
    all_out = [v for vs in outs.values() for v in vs] or [200]
    return {kind: {"calls": len(vals), "in_chars": statistics.mean(vals),
                   "out_chars": statistics.mean(outs[kind]) if outs.get(kind) else statistics.mean(all_out)}
            for kind, vals in buckets.items()}


# --------------------------------------------------------------------------- 真实调用次数
MEASURED_SEC = {("tdrive", "medium"): 153.0, ("tdrive", "large"): 287.0,
                ("grab", "small"): 112.3, ("grab", "medium"): 97.8, ("grab", "large"): 289.1}
DEFAULT_RATE = 1.6      # calls/s，与实测未命中缓存的档位同量级


def real_calls(dataset: str, scale: str) -> dict:
    path = f"results/main_{dataset}_qwen72b_n3.json"
    with open(path, encoding="utf-8") as fh:
        det = json.load(fh)["scales"][scale]["MAPUS"]["detail"]
    mean = lambda k: float(det.get(k, {"mean": 0.0})["mean"])  # noqa: E731
    return {
        "accept": mean("n_accepted") + mean("n_refused"),
        "refine": mean("llm_calls_gen"),
        "tiebreak": mean("sel_llm_ties"),
        "offer": mean("neg_attempts"),
        "feedback": 2.0 * mean("neg_attempts"),
    }


def real_rate(dataset: str, scale: str, calls: dict) -> float:
    sec = MEASURED_SEC.get((dataset, scale))
    return sum(calls.values()) / sec if sec else DEFAULT_RATE


def active_calls(calls: dict, variant: str) -> dict:
    drop = VARIANTS[variant]
    return {k: (0.0 if k in drop else v) for k, v in calls.items()}


def tokens(calls: dict, prof: Dict[str, dict], ri: float, ro: float) -> Tuple[float, float]:
    tin = tout = 0.0
    for kind, n in calls.items():
        p = prof.get(kind)
        if not p or n <= 0:
            continue
        tin += n * p["in_chars"] / ri
        tout += n * p["out_chars"] / ro
    return tin, tout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-in", type=float, default=4.13)
    ap.add_argument("--price-out", type=float, default=4.13)
    ap.add_argument("--ratio-in", type=float, default=3.6)
    ap.add_argument("--ratio-out", type=float, default=3.5)
    args = ap.parse_args()

    cache = load_cache()
    print(f"缓存条目 {len(cache)}，平均回复 {statistics.mean(cache.values()):.0f} 字符\n")

    profiles = {}
    for dataset in DATASETS:
        for scale in SCALES:
            setting = f"{dataset}_{scale}"
            profiles[setting] = record_profile(setting, cache)
            p = profiles[setting]
            print(f"[profile] {setting:14s} " + "  ".join(
                f"{k}:{v['calls']}×{v['in_chars']:.0f}c(hit{v['hit']*100:.0f}%)"
                for k, v in sorted(p.items())), flush=True)

    fair = fairness_profile(cache)
    print("[profile] fairness/task  " + "  ".join(
        f"{k}:{v['calls']}×{v['in_chars']:.0f}c" for k, v in sorted(fair.items())) + "\n")

    def price(tin, tout):
        return tin / 1e6 * args.price_in + tout / 1e6 * args.price_out

    out = {"price_in": args.price_in, "price_out": args.price_out, "rows": [], "summary": {}}
    print(f"{'setting':14s} {'variant':8s} {'calls':>6s} {'tok_in':>8s} {'tok_out':>8s} {'¥':>7s} {'秒':>7s}")
    for dataset in DATASETS:
        for scale in SCALES:
            setting = f"{dataset}_{scale}"
            calls = real_calls(dataset, scale)
            rate = real_rate(dataset, scale, calls)
            for variant in VARIANTS:
                c = active_calls(calls, variant)
                tin, tout = tokens(c, profiles[setting], args.ratio_in, args.ratio_out)
                row = {"setting": setting, "variant": variant, "calls": sum(c.values()),
                       "tok_in": tin, "tok_out": tout, "cny": price(tin, tout),
                       "sec": sum(c.values()) / rate}
                out["rows"].append(row)
                print(f"{setting:14s} {variant:8s} {row['calls']:6.0f} {tin:8.0f} {tout:8.0f} "
                      f"{row['cny']:7.2f} {row['sec']:7.0f}")

    f_calls = {k: v["calls"] for k, v in fair.items()}
    # 公平性实验只调 generate_route（UR/LS/CG/FPS 的选择是规则的）：按真实模型的人均精修次数校准
    gate = (real_calls("tdrive", "medium")["refine"] / 40.0) / (fair["refine"]["calls"] / 30.0)
    f_calls["refine"] *= gate
    f_tin, f_tout = tokens(f_calls, fair, args.ratio_in, args.ratio_out)
    f_cny = price(f_tin, f_tout)
    f_rate = real_rate("tdrive", "medium", real_calls("tdrive", "medium"))
    f_sec = sum(f_calls.values()) / f_rate
    print(f"{'fairness/task':14s} {'—':8s} {sum(f_calls.values()):6.0f} {f_tin:8.0f} {f_tout:8.0f} "
          f"{f_cny:7.2f} {f_sec:7.0f}")
    out["fairness"] = {"calls": sum(f_calls.values()), "tok_in": f_tin, "tok_out": f_tout,
                       "cny": f_cny, "sec": f_sec}

    print("\n=== 汇总 ===")
    def total(dataset, scales, variants):
        return [r for r in out["rows"] if r["setting"].startswith(dataset + "_")
                and r["setting"].split("_")[1] in scales and r["variant"] in variants]

    plan = {
        "消融 (T-Drive+Grab, 3 档, 4 变体, n)": [r for r in out["rows"] if r["variant"] in VARIANTS],
        "协商 (2 数据集 × medium, n)": [r for r in out["rows"]
                                       if r["variant"] == "Full" and r["setting"].endswith("medium")],
    }
    for label, rows in plan.items():
        per1 = sum(r["cny"] for r in rows)
        sec1 = sum(r["sec"] for r in rows)
        line = "  ".join(f"n={n}: ¥{per1*n:6.2f}/{sec1*n/3600:5.1f}h" for n in (1, 2, 3, 5, 20))
        print(f"{label}\n    {line}")
        out["summary"][label] = {"per_n1_cny": per1, "per_n1_hours": sec1 / 3600}

    line = "  ".join(f"{n} 任务: ¥{f_cny*n:6.2f}/{f_sec*n/3600:5.1f}h" for n in (10, 30, 60))
    print(f"公平性 (T-Drive, 30 人池)\n    {line}")
    out["summary"]["公平性"] = {"per_task_cny": f_cny, "per_task_hours": f_sec / 3600}

    with open("results/cost_estimate.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nwrote results/cost_estimate.json")


if __name__ == "__main__":
    main()
