# -*- coding: utf-8 -*-
r"""E4.2 / E4.3：候选生成器质量 → 近似比（定理 3 的一致性 + 鲁棒性）。

    # 可控仿真（默认，复现论文现有表格，不联网、不烧 token）
    py -3.14 experiments/run_llm_candidates.py --n-workers 20 --seeds 5

    # 真实 LLM 调用（需要 API key）
    $env:DEEPSEEK_API_KEY = "sk-..."
    py -3.14 experiments/run_llm_candidates.py --mode llm --n-workers 20 --seeds 5

两种模式的口径差别（论文中**必须写明**）：
* ``--mode sim``（默认）——**未调用任何 LLM**。定理 3 的数学对象是「候选池 ``C_u``
  是否包含最优路线」与「污染比例 η」，这与生成器是 LLM、启发式还是随机采样无关。
  本模式实现的是**候选生成器的可控仿真**，验证的是**定理 3 的界结构**
  （一致性 (1−ε)·OPT 与鲁棒性 (1−ε)(1−η′)），**不是某个具体 LLM 的质量**。
* ``--mode llm`` —— 真实调用 LLM 在每位参与者的编号菜单上做选择（见
  ``aprime/llm_gen.py``），命中率与近似比均为实测。此时 E4.3（污染）不跑，
  因为污染实验需要可控基线。

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
import re
import sys
import time
from statistics import mean
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.llm_gen import PRESETS, LLMConfig, LLMConfigError, MenuSelector  # noqa: E402
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


def _gen_llm_pool(selector: MenuSelector, rec: dict, args) -> tuple:
    """让真实 LLM 在每位参与者的全池菜单上选 k 条，返回候选与审计记录。"""
    pool: List[Candidate] = []
    audit: List[dict] = []
    for wid, menu in rec["by_w"].items():
        if len(menu) <= args.k:
            pool.extend(menu)
            continue
        fallback = sorted(menu, key=lambda c: (-c.sat, -len(c.cells)))
        row: dict = {"seed": rec["seed"], "wid": wid, "theta": args.theta}
        picked = selector.select_candidates(
            wid, menu, args.k, theta=args.theta,
            fallback=fallback, record=row, style=args.prompt_style)
        pool.extend(picked)
        audit.append(row)
    return pool, audit


# --------------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--source", default="synthetic",
                    help="参与者来源。synthetic=随机采样（默认，保持既有结果可复现）；"
                         "geolife=GeoLife 1.3 真实轨迹（跨数据集外部效度用）")
    ap.add_argument("--min-cand", type=int, default=0,
                    help="只保留候选池 ≥ 该条数的参与者（默认 0=不过滤）。真实轨迹数据"
                         "（如 GeoLife）出行段短、候选池偏稀，需过滤掉候选不足的参与者；"
                         "被丢弃的人数会写进结果文件，作为有效样本量")
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
    # ---------------- 真实 LLM 模式 ----------------
    ap.add_argument("--mode", choices=("sim", "llm"), default="sim",
                    help="sim=可控仿真（默认，验证界结构）；llm=真实 LLM 调用")
    ap.add_argument("--provider", choices=tuple(PRESETS), default="deepseek")
    ap.add_argument("--model", default=None, help="覆盖预设模型名")
    ap.add_argument("--base-url", default=None,
                    help="覆盖预设端点（自建/本地 mock 用；loopback 会自动绕过代理）")
    ap.add_argument("--cache-dir", default="results/llm_cache")
    ap.add_argument("--no-cache", action="store_true", help="禁用落盘缓存（会重复烧 token）")
    ap.add_argument("--env-file", default="",
                    help="含 API key 的 .env 路径（环境变量缺失时兜底读取）")
    ap.add_argument("--thinking", action="store_true",
                    help="开启推理模式。默认关闭——开启后 completion token 可能涨 100 倍以上")
    ap.add_argument("--prompt-style", choices=("coverage", "diverse"), default="coverage",
                    help="coverage=覆盖最大化优先（与命中率指标一致，默认）；"
                         "diverse=多样性/均衡（早期版本，指标不一致，仅作对照）")
    args = ap.parse_args()

    eps_grid = [float(x) for x in args.eps_grid.split(",") if x]
    eta_grid = [float(x) for x in args.eta_grid.split(",") if x]

    selector = None
    if args.mode == "llm":
        preset = PRESETS[args.provider]
        extra = dict(preset.extra_payload)
        if args.thinking:
            extra["enable_thinking"] = True
        cfg = LLMConfig(
            base_url=args.base_url or preset.base_url,
            model=args.model or preset.model,
            api_key_env=preset.api_key_env,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            trust_env=not (args.base_url or "").startswith(("http://127.", "http://localhost")),
            extra_payload=extra,
            env_file=args.env_file,
        )
        if extra.get("enable_thinking") is True:
            print("[警告] 已开启推理模式：completion token 可能达非推理的 100 倍以上。")
        try:
            cfg.api_key()
        except LLMConfigError as exc:
            raise SystemExit(f"[--mode llm] 凭据缺失：\n{exc}")
        selector = MenuSelector(cfg)
        print(f"[LLM] provider={args.provider} model={cfg.model} "
              f"cache={'off' if args.no_cache else cfg.cache_dir}")

    # 每次 (seed) 一个实例，缓存全池与 OPT
    instances = []
    n_kept = n_dropped = 0
    for seed in range(args.seeds):
        inst = make_dense_instance(args.dataset, args.scale, n_workers=args.n_workers,
                                   seed=seed, index=0, source=args.source)
        cands = build_candidates(inst, k=args.k_full, seed=seed,
                                 lams=(1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0),
                                 n_random=16)
        ppw = _by_worker(cands)
        # 过滤候选不足的参与者：真实轨迹（GeoLife）出行段短，部分参与者可达格太少、
        # 只生成得出 1–3 条不同路线，截断到 k 条没有意义，会让生成器"白送"全池。
        floor = max(args.min_cand, args.k + 1)
        if args.min_cand:
            n_all = len(ppw)
            kept = {wid: v for wid, v in ppw.items() if len(v) >= floor}
            n_kept += len(kept)
            n_dropped += n_all - len(kept)
            cands = [c for v in kept.values() for c in v]
            ppw = kept
            print(f"[seed {seed}] 候选过滤（≥{floor} 条）：保留 {len(kept)}/{n_all} 人，"
                  f"丢弃 {n_all - len(kept)} 人")
        if not ppw:
            print(f"[seed {seed}] 过滤后没有参与者，跳过")
            continue
        sizes = [len(v) for v in ppw.values()]
        if min(sizes) <= args.k:
            raise SystemExit(
                f"候选池太小：每人最少只有 {min(sizes)} 条 ≤ k={args.k}，"
                f"生成器无法真正截断。请增大 --k-full 或 --min-cand。")
        sel_opt, opt_cov, status = _opt(inst, cands, args.theta, args.budget_steps, args.k_full)
        if opt_cov <= 0:
            print(f"[seed {seed}] OPT=0 (status={status})，跳过")
            continue
        opt_routes = {c.wid: c for c in sel_opt}
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
    llm_audit: List[dict] = []
    modes = (("full", "llm") if args.mode == "llm"
             else ("full", "topk_sat", "topk_cov", "random", "oracle_eps"))

    # ---------------- E4.2：命中率 → 近似比 ----------------
    for mode in modes:
        eps_list = eps_grid if mode == "oracle_eps" else [0.0]
        for eps in eps_list:
            for rec in instances:
                rng = random.Random(1000 * rec["seed"] + 17)
                if mode == "llm":
                    pool, audit = _gen_llm_pool(selector, rec, args)
                    llm_audit.extend(audit)
                else:
                    pool = gen_pool(rec["by_w"], args.k, mode, rng,
                                    opt_routes=rec["opt_routes"], eps=eps)
                if not pool:
                    continue
                cov, n_sel = _select(rec["inst"], pool, args.theta,
                                     args.budget_steps, args.k)
                # 命中率：OPT 选中的参与者里，其最优路线是否落进 C_u
                by_w_pool = _by_worker(pool)
                hits, tot = 0, 0
                for wid, tgt in rec["opt_routes"].items():
                    tot += 1
                    if any(abs(c.sat - tgt.sat) < 1e-12 and c.cells == tgt.cells
                           for c in by_w_pool.get(wid, [])):
                        hits += 1
                rows.append({
                    "exp": "E4.2", "mode": mode, "eps": eps, "eta": 0.0,
                    "theta": args.theta, "seed": rec["seed"], "k": args.k,
                    "cov": cov, "opt": rec["opt_cov"],
                    "ratio": round(cov / rec["opt_cov"], 4),
                    "hit_rate": round(hits / tot, 4) if tot else 0.0,
                    "n_selected": n_sel,
                })

    # ---------------- E4.3：污染 η → 近似比 ----------------
    # 基线池统一用 topk_sat（"一个还不错的生成器"），再施加污染
    # 真实 LLM 模式下跳过本节（污染实验需要可控基线，与 LLM 无关）
    for pmode in (() if args.mode == "llm" else ("random", "adversarial")):
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
                    "theta": args.theta, "seed": rec["seed"], "k": args.k,
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
    for mode in modes:
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
    is_llm = args.mode == "llm"
    model_name = args.model or PRESETS[args.provider].model
    # 模型名含 "/"（如 deepseek-ai/DeepSeek-V4-Flash），直接拼进文件名会被当成路径分隔符
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)
    suffix = f"_llm{safe_model}_{args.prompt_style}" if is_llm else ""
    src_tag = "" if args.source == "synthetic" else f"_{args.source}"
    stem = os.path.join(
        args.out_dir,
        f"llm_candidates_{args.dataset}{src_tag}_{args.scale}_n{len(instances)}{suffix}")

    if is_llm:
        note = (f"真实 LLM 调用（provider={args.provider}, model="
                f"{model_name}, prompt={args.prompt_style}）。"
                f"候选池由 LLM 在编号菜单上选择生成，命中率与近似比均为实测。")
    else:
        note = ("候选生成器的可控仿真：未调用任何 LLM，验证的是定理 3 的界结构"
                "（命中率→近似比、污染→退化），不是某个具体 LLM 的质量。")

    payload = {
        "note": note,
        "mode": args.mode,
        "source": args.source, "min_cand": args.min_cand,
        "dataset": args.dataset, "scale": args.scale,
        "n_workers": args.n_workers,
        "n_workers_retained": n_kept or args.n_workers,
        "n_workers_dropped": n_dropped,
        "k_full": args.k_full, "k": args.k,
        "budget_steps": args.budget_steps, "theta": args.theta,
        "seeds": len(instances), "rows": rows,
    }
    if is_llm:
        payload["llm"] = {
            "provider": args.provider,
            "model": model_name,
            "prompt_style": args.prompt_style,
            "stats": selector.stats,
            "n_selections": len(llm_audit),
            "n_padded_total": sum(r.get("n_padded", 0) for r in llm_audit),
            "n_unparsed": sum(1 for r in llm_audit if not r.get("llm_ok")),
            "audit": llm_audit,
        }
        st = selector.stats
        live_tok = st["prompt_tokens"] + st["completion_tokens"]
        cache_tok = st["prompt_tokens_cached"] + st["completion_tokens_cached"]
        payload["llm"]["tokens"] = {
            "live": live_tok, "from_cache": cache_tok, "total": live_tok + cache_tok,
        }
        print(f"\n===== LLM 用量（{payload['llm']['model']}）=====")
        print(f"  实发请求 {st['calls']} 次（{live_tok:,} token），"
              f"缓存命中 {st['cache_hits']} 次（{cache_tok:,} token），失败 {st['failures']} 次")
        print(f"  本次代表总量 {live_tok + cache_tok:,} token；"
              f"实发时延 {st['latency_ms'] / 1000.0:.1f} s")
        print(f"  补足条数合计 {payload['llm']['n_padded_total']}／"
              f"{payload['llm']['n_selections']} 次选择")

    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # markdown
    src_note = ("" if args.source == "synthetic" else
                f"参与者来源 **{args.source}**（真实轨迹）；")
    filt_note = ("" if not args.min_cand else
                 f"仅保留候选池 ≥{max(args.min_cand, args.k + 1)} 条的参与者，"
                 f"累计丢弃 {n_dropped} 人（有效 {n_kept} 人·种子）。")
    lines = [f"# 候选生成器质量 → 近似比（E4.2/E4.3，{args.dataset}/{args.scale}）", "",
             f"{src_note}参与者 {args.n_workers}、全池 k_full={args.k_full}、"
             f"生成器 |C_u|={args.k}、"
             f"步数预算 {args.budget_steps:.0f}、θ={args.theta}、{len(instances)} 个种子。",
             "", f"> {filt_note}" if filt_note else "", "",
             f"> **口径声明**：{note}", ""]
    for mode in modes:
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
    if not is_llm:
        lines += ["", "## E4.3 污染 η → 近似比（鲁棒性，基线 topk_sat）", "",
                  "| 污染模式 | η | A′/OPT |", "|---|---|---|"]
    for pmode in (() if is_llm else ("random", "adversarial")):
        for eta in eta_grid:
            rt = agg("E4.3", pmode, "eta", eta)
            if rt is None:
                continue
            lines.append(f"| {pmode} | {eta:.2f} | **{rt:.3f}** |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    if selector is not None:
        selector.close()

    print(f"\nwrote {stem}.json\nwrote {stem}.md")


if __name__ == "__main__":
    main()
