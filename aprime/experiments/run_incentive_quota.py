# -*- coding: utf-8 -*-
"""基数预算口径的“名额 = 多少”推导（模块：aprime.synth.incentive）。

核心问题：`budget_kind="count"` 的 `B` 取多少？直接拿步数 40/60/100 当名额，
对 20 人的实例等于不限；随手捏一个数字又没依据。

本脚本把步数预算锚定为“等价名额”，再给三种激励模型分别算出钱预算下的
可购名额 [k_min, k_max]，并把 `k_max / k_min` 当背包因子如实报告——不掩盖，
不伪造。所有结果文件都带 `provenance` 块（aprime.provenance）。

    py -3.14 experiments/run_incentive_quota.py --settings tdrive_small,tdrive_medium
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.model import make_problem  # noqa: E402
from aprime.provenance import stamp  # noqa: E402
from aprime.solvers.exact import exact  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402
from aprime.synth.incentive import (count_budget, match_money_to_quota,  # noqa: E402
                                    make_incentive_layer, quota_equivalent)
from mapus.data.instances import make_mapus_instance  # noqa: E402


def _route_steps(prob) -> float:
    """当前实例候选路线平均长度：用来把步数预算换成等价名额。"""
    elems = prob.elements(eligible_only=False)
    return mean([e.steps for e in elems]) if elems else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="tdrive_small", help="逗号分隔，如 tdrive_small,tdrive_medium")
    ap.add_argument("--k-cand", type=int, default=12)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for setting in [s.strip() for s in args.settings.split(",") if s.strip()]:
        for i in range(3):
            inst = make_mapus_instance(setting, i, source="tdrive", urban_source="osm")
            cands = build_candidates(inst, k=args.k_cand, seed=args.seed + i)
            steps_mean = _route_steps(make_problem(inst, cands, APrimeConfig()))

            rec = {"setting": setting, "index": i,
                   "budget_steps": float(inst.budget),
                   "mean_route_steps": round(steps_mean, 3),
                   "quota_equivalent": quota_equivalent(float(inst.budget), steps_mean)}

            # ── baseline：按原样的步数预算跑一次 A′
            prob_s = make_problem(inst, cands,
                                  APrimeConfig(theta=args.theta, budget_kind="steps"))
            sel_s, _ = solve(prob_s, with_ls=True, rng=rng)
            rec["steps_coverage"] = prob_s.coverage(sel_s)
            rec["steps_selected"] = len(sel_s)
            rec["steps_cost"] = prob_s.cost(sel_s)

            # ── 三种激励模型：钱预算锚定到等价名额 k*
            k_star = rec["quota_equivalent"]
            if k_star <= 0:
                k_star = max(1, len(cands) // 4)
            for model in ("flat", "dist_cost", "supply_demand"):
                layer = make_incentive_layer(inst, model=model, seed=args.seed + i,
                                             money_budget=match_money_to_quota(
                                                 make_incentive_layer(inst, model=model,
                                                                    seed=args.seed + i),
                                                 k_star, mode="worst"))
                pp = count_budget(layer)
                k = pp.k_min  # 保守：任意 k 个人都买得起

                prob_c = make_problem(inst, cands,
                                      APrimeConfig(theta=args.theta, budget_kind="count",
                                                   budget=float(k)))
                sel_c, _ = solve(prob_c, with_ls=True, rng=rng)
                prefix = "count_%s" % model
                rec[prefix + "_k"] = k
                rec[prefix + "_backpack"] = pp.backpack_factor
                rec[prefix + "_coverage"] = prob_c.coverage(sel_c)
                rec[prefix + "_selected"] = len(sel_c)
                rec[prefix + "_cost"] = prob_c.cost(sel_c)

                # 参与者来源与激励层都盖章：绝不能被当成“天然的名额预算”
                prob_c.instance.base.meta.setdefault("_provenance_audit", {})
                prob_c.instance.base.meta["_provenance_audit"][model] = {
                    "pay_range": [round(min(layer.pay.values()), 3),
                                  round(max(layer.pay.values()), 3)],
                    "backpack_factor": round(pp.backpack_factor, 3),
                    "money_budget": round(layer.money_budget, 3),
                }

            rec["model_note"] = ("flat 情况下 backpack=1（纯基数，定理 2½ 直接适用）；"
                                 "dist_cost / supply_demand backpack>1，与½(1−1/e) 对照看")
            rows.append(rec)
            print("[%s idx=%d] quota*=%.0f | steps: cov=%.1f sel=%d cost=%.1f | "
                  "flat: k=%d cov=%.1f | dist: k=%d cov=%.1f bp=%.2f | "
                  "supply: k=%d cov=%.1f bp=%.2f" % (
                      setting, i, k_star,
                      rec["steps_coverage"], rec["steps_selected"], rec["steps_cost"],
                      rec["count_flat_k"], rec["count_flat_coverage"],
                      rec["count_dist_cost_k"], rec["count_dist_cost_coverage"],
                      rec["count_dist_cost_backpack"],
                      rec["count_supply_demand_k"], rec["count_supply_demand_coverage"],
                      rec["count_supply_demand_backpack"]))

    stem = os.path.join(args.out_dir, "incentive_quota")
    payload = {
        "settings": args.settings,
        "seed": args.seed,
        "note": ("count 口径的 B = k* = floor(步数预算 / 平均路线步数)，再按最坏情况锚定钱预算；"
                 "flat/d_cost/supply 同一购买力，差异纯结构性；provenance 标注为合成激励层"),
        "rows": rows,
    }
    stamp(payload, models={"incentive": True},
          assumed={"pay": "no public payment field in T-Drive; modeled as base + per_cell·dist + per_step·window",
                   "supply_demand": "demand from commercial+institutional land-use share (OSM)",
                   "budget_anchor": "k* = budget_steps / mean_route_steps; money = k* · max_pay (worst case)"})
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    head = ("| setting | idx | 步预算 | 平均步长 | quota* | 步口径cov | 步口径sel | flat k | flat cov | dist k | dist cov | bp | supply k | supply cov | bp |")
    lines = [f"# 基数预算口径的名额推导（{args.settings}，seed={args.seed}）",
             "", "注：所有激励层均为**合成假设层**（T-Drive 无支付字段），已写入 provenance；",
             "      同一实例内三模型总购买力相等，差异为结构性；bp = k_max/k_min（backpack factor）。",
             "", head, "|" + "---|" * 16]
    for r in rows:
        lines.append("| %s | %d | %s | %s | %d | %.1f | %d | %d | %.1f | %d | %.1f | %.2f | %d | %.1f | %.2f |" % (
            r["setting"], r["index"], r["budget_steps"], r["mean_route_steps"],
            r["quota_equivalent"], r["steps_coverage"], r["steps_selected"],
            r["count_flat_k"], r["count_flat_coverage"],
            r["count_dist_cost_k"], r["count_dist_cost_coverage"],
            r["count_dist_cost_backpack"],
            r["count_supply_demand_k"], r["count_supply_demand_coverage"],
            r["count_supply_demand_backpack"]))
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", stem + ".json")
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()
