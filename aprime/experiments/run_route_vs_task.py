# -*- coding: utf-8 -*-
r"""E5.3：**路线级 vs 任务级**消融（创新点 1 的直接证据）。

    py -3.14 experiments/run_route_vs_task.py --seeds 5

方案 §2.1 的核心主张是「决策变量从**任务**改为**路线**」。本实验在同一实例、
同一预算口径、同一 θ 下跑两个模型：

* **路线级（本文 A′）**：元素 = 一条完整路线 `(u, r)`，覆盖 = 该路线经过的栅格集合，
  代价 = `steps(r)`，每人至多一条。
* **任务级（消融基线）**：元素 = 一个**单个栅格任务** `(u, c)`，代价 = `u` 往返该格的步数
  `steps(origin→c) + steps(c→dest)`，满意度 = 「访问过该格的候选路线里最好的满意度」，
  每人至多一个任务（与路线级保持同样的"每人一条"耦合，只放开"空间耦合"）。

两者预算、θ、实例完全一致，**唯一差别是元素是否沿路径耦合**。因此覆盖差异就是
"路线级建模"本身贡献的价值。

任务级的元素规模会受到控制（只保留在路线候选里出现过的格子），否则 ILP/贪心都跑不动；
这一点在结果文件里如实记录。

输出 `results/route_vs_task_<setting>_n<n>.json` / `.md`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from agentsense.env.model import steps_for  # noqa: E402
from mapus.env.metrics_obj import PathSatisfaction  # noqa: E402

from aprime.analysis.fairness import fairness_report  # noqa: E402
from aprime.availability import require_available  # noqa: E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.model import Candidate, SelectionProblem, make_problem  # noqa: E402
from aprime.solvers.exact import exact  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


def _problem(inst, cands, theta: float, budget_steps: float) -> SelectionProblem:
    return make_problem(inst, cands,
                        APrimeConfig(theta=theta, budget=budget_steps,
                                     budget_kind="steps"))


def build_task_candidates(inst, route_cands: Sequence[Candidate],
                          cfg: APrimeConfig) -> List[Candidate]:
    """把「路线候选」降解成「单格任务候选」。

    对每个参与者 `u`，取它所有路线候选覆盖过的格子集合 `C_u`；对每个格子 `c ∈ C_u`
    造一个任务元素：

    * ``cells = {c}``
    * ``steps = 往返步数``（origin→c→dest）
    * ``sat = 访问过 c 的路线候选里最大的满意度``（对任务级最有利的口径，避免低估基线）

    这样任务级模型在"信息"上不吃亏：它知道每个格子的最佳可达满意度，只是丢掉了
    「一条路线必须沿路径连续走完」这个约束。
    """
    sat = PathSatisfaction(inst.urban, mu=cfg.mu, prior=cfg.pss_prior, scale=cfg.pss_scale)
    by_w_route: Dict[str, List[Candidate]] = {}
    for c in route_cands:
        by_w_route.setdefault(c.wid, []).append(c)

    workers = {w.wid: w for w in inst.workers}
    out: List[Candidate] = []
    for wid, rcands in by_w_route.items():
        w = workers.get(wid)
        if w is None:
            continue
        # 该参与者所有路线覆盖过的格子
        cells_u: set = set()
        for c in rcands:
            cells_u |= set(c.cells)
        # 每格的最佳满意度
        best_sat: Dict[Tuple[int, int], float] = {}
        for c in rcands:
            for cell in c.cells:
                best_sat[cell] = max(best_sat.get(cell, 0.0), c.sat)

        pref = inst.profiles[wid].pref
        for (x, y) in sorted(cells_u):
            trip = steps_for(w.origin, (x, y), w.speed) + steps_for((x, y), w.dest, w.speed)
            if trip <= 0:
                continue
            s = best_sat.get((x, y))
            if s is None:
                s = float(sat.score(pref, [(x, y)]))
            # path 只需长度等于 trip（Candidate.steps = len(path)）
            path = tuple((int(x), int(y), t) for t in range(int(trip)))
            out.append(Candidate(wid=wid, path=path, cells=frozenset({(int(x), int(y))}),
                                 sat=float(s), tag="task"))
    return out


def _run_arm(inst, cands, theta, budget, with_ls=True):
    prob = _problem(inst, cands, theta, budget)
    sel, _ = solve(prob, with_ls=with_ls)
    fr = fairness_report(prob, sel, grid=(inst.base.grid if hasattr(inst, "base")
                                         else (inst.nx, inst.ny)))
    return prob, sel, prob.coverage(sel), fr


class _CappedProblem(SelectionProblem):
    """在 SelectionProblem 上额外限制「每人至多 cap 个元素」。

    路线级天然是 cap=1（每人一条路线）；任务级在容量对齐时需要 cap>1，
    否则"1 格/人 vs 5 格/人"的差异会盖过建模本身的差异。
    """

    def __init__(self, *a, cap: int = 1, **kw):
        super().__init__(*a, **kw)
        self.cap = int(cap)

    def feasible(self, sel) -> bool:
        if not super().feasible(sel):
            return False
        cnt: Dict[str, int] = {}
        for c in sel:
            cnt[c.wid] = cnt.get(c.wid, 0) + 1
            if cnt[c.wid] > self.cap:
                return False
        return True


def _greedy_capped(prob: "_CappedProblem"):
    """按边际覆盖密度贪心，允许每人至多 ``prob.cap`` 个元素。"""
    elems = prob.eligible
    used: Dict[str, int] = {}
    sel: List[Candidate] = []
    covered: set = set()
    cost = 0.0
    while True:
        best, best_key = None, None
        for e in elems:
            if used.get(e.wid, 0) >= prob.cap:
                continue
            c = float(e.steps)
            if cost + c > prob.budget + 1e-9:
                continue
            gain = len(e.cells - covered)
            if gain <= 0:
                continue
            key = gain / c if c > 0 else float(gain)
            if best is None or key > best_key:
                best, best_key = e, key
        if best is None:
            break
        sel.append(best)
        used[best.wid] = used.get(best.wid, 0) + 1
        covered |= best.cells
        cost += float(best.steps)
    return sel


def _run_task_capped(inst, cands, theta, budget, cap):
    prob = _CappedProblem(instance=inst, candidates=list(cands),
                          cfg=APrimeConfig(theta=theta, budget=budget,
                                           budget_kind="steps"),
                          cap=cap)
    sel = _greedy_capped(prob)
    fr = fairness_report(prob, sel, grid=(inst.base.grid if hasattr(inst, "base")
                                         else (inst.nx, inst.ny)))
    return prob, sel, prob.coverage(sel), fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=20)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--budget-steps", type=float, default=120.0)
    ap.add_argument("--thetas", default="0.0,0.3,0.6,0.9")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    require_available("SEC6")  # 真实数据实验搁置中：docs/e0_1_data_access_plan.md §8

    thetas = [float(x) for x in args.thetas.split(",") if x]
    rows: List[dict] = []

    for seed in range(args.seeds):
        inst = make_dense_instance(args.dataset, args.scale,
                                   n_workers=args.n_workers, seed=seed, index=0)
        rcands = build_candidates(inst, k=args.k, seed=seed,
                                  lams=(1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0),
                                  n_random=16)
        cfg = APrimeConfig(theta=0.0, budget=args.budget_steps, budget_kind="steps")
        tcands = build_task_candidates(inst, rcands, cfg)
        # 容量对齐：路线级平均覆盖多少格，任务级就给每人多少格的名额
        avg_cells = mean(len(c.cells) for c in rcands)
        cap_task = max(1, int(round(avg_cells)))
        print(f"[seed {seed}] 路线元素 {len(rcands)}，任务元素 {len(tcands)}，"
              f"路线平均覆盖 {avg_cells:.2f} 格 → 任务级每人名额 cap={cap_task}")

        for theta in thetas:
            # 路线级（每人一条）
            prb_r, sel_r, cov_r, fr_r = _run_arm(inst, rcands, theta, args.budget_steps)
            # 任务级：cap=1（严格每人一个任务）
            prb_t1, sel_t1, cov_t1, fr_t1 = _run_task_capped(
                inst, tcands, theta, args.budget_steps, cap=1)
            # 任务级：cap=avg_cells（容量对齐，可领多格但仍按往返步数计费）
            prb_t, sel_t, cov_t, fr_t = _run_task_capped(
                inst, tcands, theta, args.budget_steps, cap=cap_task)
            # 无约束最优（θ=0 的 ILP）作公共参照
            sel_o, info_o = exact(_problem(inst, rcands, 0.0, args.budget_steps), time_limit=60)
            opt = _problem(inst, rcands, 0.0, args.budget_steps).coverage(sel_o)
            rows.append({
                "seed": seed, "theta": theta, "cap_task": cap_task,
                "cov_route": cov_r, "cov_task": cov_t, "cov_task_cap1": cov_t1,
                "opt_theta0": opt,
                "sel_route": len(sel_r), "sel_task": len(sel_t),
                "ratio_route": round(cov_r / opt, 4) if opt else None,
                "ratio_task": round(cov_t / opt, 4) if opt else None,
                "ratio_task_cap1": round(cov_t1 / opt, 4) if opt else None,
                "n_elem_route": len(rcands), "n_elem_task": len(tcands),
                "feasible_rate_route": fr_r["feasible_rate"],
                "feasible_rate_task": fr_t["feasible_rate"],
                "excl_route": fr_r["exclusion_gap"],
                "excl_task": fr_t["exclusion_gap"],
                "gqc_route": fr_r["gqc_gap"],
                "gqc_task": fr_t["gqc_gap"],
            })
            print(f"  θ={theta:.1f}  路线级={cov_r}  任务级cap1={cov_t1}  "
                  f"任务级对齐={cov_t}  OPT(θ=0)={opt}")

    def agg(key, theta, fld):
        v = [r[fld] for r in rows if r["theta"] == theta and r[fld] is not None]
        return round(mean(v), 4) if v else None

    print("\n===== 路线级 vs 任务级 =====")
    print(f"{'theta':>5s} {'cov_route':>9s} {'cov_task':>9s} {'Δ':>6s} {'r_route':>8s} {'r_task':>8s}")
    for theta in thetas:
        cr, ct = agg(None, theta, "cov_route"), agg(None, theta, "cov_task")
        rr, rt = agg(None, theta, "ratio_route"), agg(None, theta, "ratio_task")
        print(f"{theta:5.1f} {cr:9.2f} {ct:9.2f} {cr-ct:6.2f} {rr:8.3f} {rt:8.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir,
                        f"route_vs_task_{args.dataset}_{args.scale}_n{args.seeds}")
    payload = {
        "dataset": args.dataset, "scale": args.scale, "n_workers": args.n_workers,
        "k": args.k, "budget_steps": args.budget_steps, "thetas": thetas,
        "seeds": args.seeds, "rows": rows,
        "task_level_definition": (
            "元素=单格任务(u,c)；代价=往返步数；满意度=访问过 c 的路线候选中的最大值；"
            "每人至多一个任务。只保留在路线候选中出现过的格子。"),
    }
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = [f"# 路线级 vs 任务级（E5.3，{args.dataset}/{args.scale}）", "",
             f"参与者 {args.n_workers}、k={args.k}、步数预算 {args.budget_steps:.0f}、"
             f"{args.seeds} 个种子；`OPT(θ=0)` 为路线池上的无约束 ILP 最优。", "",
             "**任务级口径**：元素=单格任务 `(u,c)`，代价=往返步数，满意度=访问过该格的最佳路线满意度，"
             "每人至多一个任务。两臂预算、θ、实例完全一致，唯一差别是元素是否沿路径耦合。", "",
             "| θ | 路线级覆盖 | 任务级覆盖 | Δ（路线−任务） | 路线级/OPT | 任务级/OPT |",
             "|---|---|---|---|---|---|"]
    for theta in thetas:
        cr, ct = agg(None, theta, "cov_route"), agg(None, theta, "cov_task")
        rr, rt = agg(None, theta, "ratio_route"), agg(None, theta, "ratio_task")
        lines.append(f"| {theta:.1f} | {cr:.2f} | {ct:.2f} | **{cr-ct:+.2f}** | "
                     f"{rr:.3f} | {rt:.3f} |")
    lines += ["", "## 公平性对照（同 θ）", "",
              "| θ | 路线级 excl | 任务级 excl | 路线级 gqc | 任务级 gqc |",
              "|---|---|---|---|---|"]
    for theta in thetas:
        e1, e2 = agg(None, theta, "excl_route"), agg(None, theta, "excl_task")
        g1, g2 = agg(None, theta, "gqc_route"), agg(None, theta, "gqc_task")
        lines.append(f"| {theta:.1f} | {e1:+.4f} | {e2:+.4f} | {g1:.4f} | {g2:.4f} |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {stem}.json\nwrote {stem}.md")


if __name__ == "__main__":
    main()
