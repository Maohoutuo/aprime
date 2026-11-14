# -*- coding: utf-8 -*-
r"""E4.2 / E4.3：候选生成器质量 → 近似比（定理 4 的一致性 + 鲁棒性）。

    py -3.14 experiments/run_llm_candidates.py --n 5

**关于「LLM」的诚实声明**：本机无外网、无 LLM API，因此这里**不调用任何 LLM**。
定理 4 的数学对象是「候选池 C_u 是否包含最优路线」以及「C_u 被污染的比例 η」，
这与生成器是 LLM、启发式还是随机采样**无关**。本实验因此实现的是**候选生成器的
可控仿真**，验证的是**定理 4 的界结构**（一致性 (1−ε)·OPT 与鲁棒性 (1−ε)(1−η′)），
而不是某个具体 LLM 的实际质量。论文中必须按此口径表述。

实验设计
--------
1. **全候选池**：每人 k_full 条路线，用 ILP 求 `OPT_full`（真最优，作分母）。
2. **候选生成器**（`C_u ⊆ R_u`，`|C_u| = k`）：
   * `full`       —— 全池（上界参照）
   * `topk_sat`   —— 按满意度取 top-k
   * `topk_cov`   —— 按覆盖取 top-k
   * `random`     —— 均匀随机 k 条
   * `oracle_eps` —— **命中率可控**：以概率 ε 放入该参与者的最优路线，其余随机补足
3. **命中率**（E4.2 的自变量）：`OPT_full` 选中的参与者中，其最优路线落入 `C_u` 的比例。
4. **污染**（E4.3 的自变量 η）：把比例 η 的候选替换成「**通过 θ 但覆盖极差**」的诱饵路线。
   若诱饵不满足 θ，算法会直接过滤掉它，污染就没有意义，所以诱饵必须在**可行域内**。
   两种污染模式：`random`（随机替换）与 `adversarial`（专门顶掉每人最好的那条）。

输出 `results/llm_candidates_<setting>_n<n>.json` 与同名 `.md`。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from statistics import mean
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.model import Candidate, SelectionProblem, make_problem  # noqa: E402
from aprime.solvers.exact import exact  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


# --------------------------------------------------------------------- 工具
def _by_worker(cands: Sequence[Candidate]) -> Dict[str, List[Candidate]]:
    out: Dict[str, List[Candidate]] = {}
    for c in cands:
        out.setdefault(c.wid, []).append(c)
    return out


def _problem(inst, cands, theta: float, budget_steps: float, k: int) -> SelectionProblem:
    return make_problem(inst, cands,
                        APrimeConfig(theta=theta, k_cand=k, budget=budget_steps,
                                     budget_kind="steps"))


def _opt(inst, cands, theta, budget_steps, k, time_limit=60.0):
    """在给定候选池上求精确最优（ILP）。"""
    prob = _problem(inst, cands, theta, budget_steps, k)
    sel, info = exact(prob, time_limit=time_limit)
    return sel, prob.coverage(sel), info.get("status", "")


def _select(inst, cands, theta, budget_steps, k):
    """A′ 在给定候选池上的解与覆盖。"""
    prob = _problem(inst, cands, theta, budget_steps, k)
    sel, _ = solve(prob, with_ls=True)
    return prob.coverage(sel), len(sel)


# ------------------------------------------------------- 候选生成器（可控）
def gen_pool(by_w: Dict[str, List[Candidate]], k: int, mode: str,
             rng: random.Random, opt_routes: Optional[Dict[str, Candidate]] = None,
             eps: float = 0.0) -> List[Candidate]:
    """按指定策略为每个参与者生成 |C_u| ≤ k 的候选池。"""
    out: List[Candidate] = []
    for wid, cands in by_w.items():
        pool = list(cands)
        if mode == "full" or len(pool) <= k:
            out.extend(pool)
            continue
        if mode == "topk_sat":
            chosen = sorted(pool, key=lambda c: (-c.sat, -len(c.cells)))[:k]
        elif mode == "topk_cov":
            chosen = sorted(pool, key=lambda c: (-len(c.cells), -c.sat))[:k]
        elif mode == "random":
            chosen = rng.sample(pool, k)
        elif mode == "oracle_eps":
            chosen = []
            tgt = (opt_routes or {}).get(wid)
            if tgt is not None and rng.random() < eps:
                chosen.append(tgt)
            rest = [c for c in pool if c not in chosen]
            chosen.extend(rng.sample(rest, min(k - len(chosen), len(rest))))
            chosen = chosen[:k]
        else:
            raise ValueError(f"unknown mode {mode}")
        out.extend(chosen)
    return out


def poison(pool: List[Candidate], by_w: Dict[str, List[Candidate]], eta: float,
           mode: str, rng: random.Random) -> List[Candidate]:
    """把比例 η 的候选替换成「通过 θ 但覆盖极差」的诱饵。

    诱饵从同一参与者的候选里挑**覆盖最小**的那条（它仍然是一条真实路线，
    因此满足同一 θ 约束）——这样污染不会被 θ 过滤器直接消掉。
    """
    if eta <= 0:
        return list(pool)
    n_pollute = int(round(len(pool) * eta))
    if n_pollute <= 0:
        return list(pool)
    out = list(pool)
    if mode == "random":
        idxs = rng.sample(range(len(out)), n_pollute)
        for i in idxs:
            wid = out[i].wid
            decoy = min(by_w[wid], key=lambda c: (len(c.cells), c.sat))
            out[i] = decoy
        return out
    # adversarial：优先顶掉每位参与者覆盖最好的那条
    order = sorted(range(len(out)), key=lambda i: -len(out[i].cells))
    for i in order[:n_pollute]:
        wid = out[i].wid
        decoy = min(by_w[wid], key=lambda c: (len(c.cells), c.sat))
        out[i] = decoy
    return out


# --------------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=20,
                    help="参与者数（小到 ILP 可在限时内求 OPT）")
    ap.add_argument("--k-full", type=int, default=8, help="全候选池每人条数")
    ap.add_argument("--k", type=int, default=4, help="生成器输出的 |C_u|")
    ap.add_argument("--budget-steps", type=float, default=40.0)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--eta-grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    eps_grid = [float(x) for x in args.eps_grid.split(",") if x]
    eta_grid = [float(x) for x in args.eta_grid.split(",") if x]

    # 每次 (seed) 一个实例，缓存全池与 OPT
    instances = []
    for seed in range(args.seeds):
        inst = make_dense_instance(args.dataset, args.scale, n_workers=args.n_workers,
                                   seed=seed, index=0)
        cands = build_candidates(inst, k=args.k_full, seed=seed,
                                 lams=(1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0),
                                 n_random=16)
        sel_opt, opt_cov, status = _opt(inst, cands, args.theta, args.budget_steps, args.k_full)
        if opt_cov <= 0:
            print(f"[seed {seed}] OPT=0 (status={status})，跳过")
            continue
        opt_routes = {c.wid: c for c in sel_opt}
        ppw = _by_worker(cands)
        sizes = [len(v) for v in ppw.values()]
        if sizes and min(sizes) <= args.k:
            raise SystemExit(
                f"候选池太小：每人最少只有 {min(sizes)} 条 ≤ k={args.k}，"
                f"生成器无法真正截断。请增大 --k-full 或 raw 生成量。")
        instances.append({
            "seed": seed, "inst": inst, "cands": cands,
            "by_w": ppw, "opt_cov": opt_cov,
            "opt_routes": opt_routes, "status": status,
        })
        print(f"[seed {seed}] OPT_full={opt_cov} ({status}) 选中 {len(sel_opt)} 人；"
              f"每人候选 {min(sizes)}–{max(sizes)} 条（k={args.k}）")

    if not instances:
        print("没有可用实例，退出。")
        return

    rows: List[dict] = []

    # ---------------- E4.2：命中率 → 近似比 ----------------
    for mode in ("full", "topk_sat", "topk_cov", "random", "oracle_eps"):
        eps_list = eps_grid if mode == "oracle_eps" else [0.0]
        for eps in eps_list:
            for rec in instances:
                rng = random.Random(1000 * rec["seed"] + 17)
                pool = gen_pool(rec["by_w"], args.k, mode, rng,
                                opt_routes=rec["opt_routes"], eps=eps)
                if not pool:
                    continue
                cov, n_sel = _select(rec["inst"], pool, args.theta,
                                     args.budget_steps, args.k)
                # 命中率：OPT 选中的参与者里，其最优路线是否落进 C_u
                pool_set = {(c.wid, id(c)) for c in pool}
                by_w_pool = _by_worker(pool)
                hits, tot = 0, 0
                for wid, tgt in rec["opt_routes"].items():
                    tot += 1
                    if any(abs(c.sat - tgt.sat) < 1e-12 and c.cells == tgt.cells
                           for c in by_w_pool.get(wid, [])):
                        hits += 1
                rows.append({
                    "exp": "E4.2", "mode": mode, "eps": eps, "eta": 0.0,
                    "seed": rec["seed"], "k": args.k,
                    "cov": cov, "opt": rec["opt_cov"],
                    "ratio": round(cov / rec["opt_cov"], 4),
                    "hit_rate": round(hits / tot, 4) if tot else 0.0,
                    "n_selected": n_sel,
                })

    # ---------------- E4.3：污染 η → 近似比 ----------------
    # 基线池统一用 topk_sat（"一个还不错的生成器"），再施加污染
    for pmode in ("random", "adversarial"):
        for eta in eta_grid:
            for rec in instances:
                rng = random.Random(2000 * rec["seed"] + 23)
                base = gen_pool(rec["by_w"], args.k, "topk_sat", rng,
                                opt_routes=rec["opt_routes"])
                pool = poison(base, rec["by_w"], eta, pmode, rng)
                cov, n_sel = _select(rec["inst"], pool, args.theta,
                                     args.budget_steps, args.k)
                rows.append({
                    "exp": "E4.3", "mode": pmode, "eps": 0.0, "eta": eta,
                    "seed": rec["seed"], "k": args.k,
                    "cov": cov, "opt": rec["opt_cov"],
                    "ratio": round(cov / rec["opt_cov"], 4),
                    "hit_rate": None,
                    "n_selected": n_sel,
                })

    # ------------------------------------------------------------------ 汇总
    def agg(exp, mode, key, val):
        v = [r["ratio"] for r in rows
             if r["exp"] == exp and r["mode"] == mode and r[key] == val]
        return round(mean(v), 4) if v else None

    print("\n===== E4.2 命中率 → 近似比（|C_u|=%d, 全池 k=%d）====="
          % (args.k, args.k_full))
    print(f"{'mode':12s} {'eps':>5s} {'hit_rate':>9s} {'ratio':>7s}")
    for mode in ("full", "topk_sat", "topk_cov", "random", "oracle_eps"):
        eps_list = eps_grid if mode == "oracle_eps" else [0.0]
        for eps in eps_list:
            hr = [r["hit_rate"] for r in rows
                  if r["exp"] == "E4.2" and r["mode"] == mode and r["eps"] == eps
                  and r["hit_rate"] is not None]
            rt = agg("E4.2", mode, "eps", eps)
            if rt is None:
                continue
            print(f"{mode:12s} {eps:5.1f} {mean(hr):9.3f} {rt:7.3f}")

    print("\n===== E4.3 污染 η → 近似比（基线 topk_sat）=====")
    print(f"{'pollute':12s} {'eta':>5s} {'ratio':>7s}")
    for pmode in ("random", "adversarial"):
        for eta in eta_grid:
            rt = agg("E4.3", pmode, "eta", eta)
            if rt is None:
                continue
            print(f"{pmode:12s} {eta:5.2f} {rt:7.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"llm_candidates_{args.dataset}_{args.scale}_n{len(instances)}")
    payload = {
        "note": ("候选生成器的可控仿真：本机无 LLM API，验证的是定理 4 的界结构"
                 "（命中率→近似比、污染→退化），不是某个具体 LLM 的质量。"),
        "dataset": args.dataset, "scale": args.scale,
        "n_workers": args.n_workers, "k_full": args.k_full, "k": args.k,
        "budget_steps": args.budget_steps, "theta": args.theta,
        "seeds": len(instances), "rows": rows,
    }
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # markdown
    lines = [f"# 候选生成器质量 → 近似比（E4.2/E4.3，{args.dataset}/{args.scale}）", "",
             f"参与者 {args.n_workers}、全池 k_full={args.k_full}、生成器 |C_u|={args.k}、"
             f"步数预算 {args.budget_steps:.0f}、θ={args.theta}、{len(instances)} 个种子。", "",
             "> **口径声明**：本机无 LLM API，本文验证的是定理 4 的**界结构**"
             "（候选命中率 → 近似比；污染比例 → 退化），不是具体 LLM 的质量。", "",
             "## E4.2 候选命中率 → 近似比（一致性）", "",
             "| 生成器 | ε | 命中率 | A′/OPT |", "|---|---|---|---|"]
    for mode in ("full", "topk_sat", "topk_cov", "random", "oracle_eps"):
        eps_list = eps_grid if mode == "oracle_eps" else [0.0]
        for eps in eps_list:
            hr = [r["hit_rate"] for r in rows
                  if r["exp"] == "E4.2" and r["mode"] == mode and r["eps"] == eps
                  and r["hit_rate"] is not None]
            rt = agg("E4.2", mode, "eps", eps)
            if rt is None:
                continue
            tag = "-" if eps == 0.0 and mode != "oracle_eps" else f"{eps:.1f}"
            lines.append(f"| `{mode}` | {tag} | {mean(hr):.3f} | **{rt:.3f}** |")
    lines += ["", "## E4.3 污染 η → 近似比（鲁棒性，基线 topk_sat）", "",
              "| 污染模式 | η | A′/OPT |", "|---|---|---|"]
    for pmode in ("random", "adversarial"):
        for eta in eta_grid:
            rt = agg("E4.3", pmode, "eta", eta)
            if rt is None:
                continue
            lines.append(f"| {pmode} | {eta:.2f} | **{rt:.3f}** |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nwrote {stem}.json\nwrote {stem}.md")


if __name__ == "__main__":
    main()
