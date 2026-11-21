# -*- coding: utf-8 -*-
r"""E4.3 污染鲁棒性（真实 LLM 版）：菜单污染下 LLM 与算法基线的对照。

设计（与 ``run_llm_candidates.py --mode sim`` 的 E4.3 同构，但污染对象是**菜单**）：
1. 每位参与者有 ``k_full`` 条菜单候选（small 实际 5–12 条，因去重后不同路线数有限）；
   按比例 η 把其中条目替换成该参与者**覆盖最差**的诱饵（诱饵仍是真实路线、满足同一 θ）。
2. 让真实 LLM 从**污染后的菜单**里选 k 条，实测 A′/OPT、命中率与**污染吸收率**。
3. 在**同一污染菜单**上跑 ``topk_cov`` / ``topk_sat`` / ``random`` 作对照。

为什么必须做「被迫 vs 净吸收」分解：菜单只有 5–12 条，η 高时**好候选可能不足 k 条**，
此时选诱饵是**被迫**而非被骗。故同时报告
``forced = Σ max(0, k − 好候选数)``，净吸收 = 吸收率 − 被迫率。

    py -3.14 experiments/run_llm_pollution.py --provider siliconflow `
      --model deepseek-ai/DeepSeek-V4-Flash --env-file D:\composition\agentsense\.env `
      --eta-grid 0,0.5,0.75 --seeds 10 --n-workers 20 --k-full 12 --k 3 `
      --budget-steps 120 --theta 0.3 --out-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from statistics import mean
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aprime  # noqa: F401,E402
from aprime.candidates import build_candidates  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.llm_gen import PRESETS, LLMConfig, LLMConfigError, MenuSelector  # noqa: E402
from aprime.model import Candidate  # noqa: E402

from run_llm_candidates import _by_worker, _opt, _select, gen_pool  # noqa: E402


def _dedupe(pool: Sequence[Candidate]) -> List[Candidate]:
    """按对象身份去重：污染会在菜单里插入同一诱饵对象的多个副本。"""
    seen = set()
    out: List[Candidate] = []
    for c in pool:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def poison_menu(by_w: Dict[str, List[Candidate]], eta: float, mode: str,
                rng: random.Random) -> Tuple[Dict[str, List[Candidate]], set]:
    """把每位参与者菜单里 η 比例的条目换成"覆盖最差"的诱饵。

    返回 (污染后的菜单, 诱饵对象 id 集合)。至少保留一条真候选，避免菜单全是诱饵。
    """
    poisoned: Dict[str, List[Candidate]] = {}
    decoy_ids: set = set()
    for wid, menu in by_w.items():
        menu = list(menu)
        n = int(round(len(menu) * eta))
        if eta <= 0 or n <= 0 or len(menu) <= 1:
            poisoned[wid] = menu
            continue
        n = min(n, len(menu) - 1)
        worst = min(menu, key=lambda c: (len(c.cells), c.sat))
        if mode == "random":
            idxs = rng.sample(range(len(menu)), n)
        elif mode == "adversarial":
            idxs = sorted(range(len(menu)), key=lambda i: -len(menu[i].cells))[:n]
        else:
            raise ValueError(f"unknown pollution mode {mode}")
        for i in idxs:
            menu[i] = worst
        decoy_ids.add(id(worst))
        poisoned[wid] = menu
    return poisoned, decoy_ids


def _forced_and_slots(poisoned: Dict[str, List[Candidate]], decoy_ids: set,
                      k: int) -> Tuple[int, int]:
    """返回 (被迫选诱饵的最少条数, 总选择槽位数)。

    若某参与者污染后好候选只剩 g 条而 g < k，则至少有 k−g 条诱饵是被迫选的。
    """
    forced = 0
    slots = 0
    for menu in poisoned.values():
        n_decoy_entries = sum(1 for c in menu if id(c) in decoy_ids)
        n_good = len(menu) - n_decoy_entries
        slots += min(k, len(menu))
        forced += max(0, k - n_good)
    return forced, slots


def _hit_rate(pool: Sequence[Candidate],
              opt_routes: Dict[str, Candidate]) -> Tuple[int, int]:
    by_w_pool = _by_worker(pool)
    hits, tot = 0, 0
    for wid, tgt in opt_routes.items():
        tot += 1
        if any(abs(c.sat - tgt.sat) < 1e-12 and c.cells == tgt.cells
               for c in by_w_pool.get(wid, [])):
            hits += 1
    return hits, tot


def llm_select(selector: MenuSelector, rec: dict, args,
               by_w: Dict[str, List[Candidate]],
               decoy_ids: set) -> Tuple[List[Candidate], List[dict], int, int]:
    """返回 (去重后候选池, 审计, 选中诱饵数, 原始选择条数)。"""
    pool: List[Candidate] = []
    audit: List[dict] = []
    n_decoy = 0
    n_picks = 0
    for wid, menu in by_w.items():
        if len(menu) <= args.k:
            pool.extend(menu)
            n_picks += len(menu)
            n_decoy += sum(1 for c in menu if id(c) in decoy_ids)
            continue
        fallback = sorted(menu, key=lambda c: (-c.sat, -len(c.cells)))
        row: dict = {"seed": rec["seed"], "wid": wid, "theta": args.theta,
                     "eta": rec.get("eta"), "pollute": rec.get("pmode")}
        picked = selector.select_candidates(
            wid, menu, args.k, theta=args.theta,
            fallback=fallback, record=row, style=args.prompt_style)
        n_picks += len(picked)
        n_decoy += sum(1 for c in picked if id(c) in decoy_ids)
        pool.extend(picked)
        audit.append(row)
    return _dedupe(pool), audit, n_decoy, n_picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--n-workers", type=int, default=20)
    ap.add_argument("--k-full", type=int, default=12)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget-steps", type=float, default=120.0)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--eta-grid", default="0,0.5,0.75")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--provider", choices=tuple(PRESETS), default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--cache-dir", default="results/llm_cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--env-file", default="")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--prompt-style", choices=("coverage", "diverse"), default="coverage")
    args = ap.parse_args()

    eta_grid = [float(x) for x in args.eta_grid.split(",") if x]

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
        raise SystemExit(f"凭据缺失：\n{exc}")
    selector = MenuSelector(cfg)
    print(f"[LLM] provider={args.provider} model={cfg.model} "
          f"cache={'off' if args.no_cache else cfg.cache_dir}")

    instances = []
    for seed in range(args.seeds):
        inst = make_dense_instance(args.dataset, args.scale, n_workers=args.n_workers,
                                   seed=seed, index=0)
        cands = build_candidates(inst, k=args.k_full, seed=seed,
                                 lams=(1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0),
                                 n_random=16)
        sel_opt, opt_cov, status = _opt(inst, cands, args.theta,
                                        args.budget_steps, args.k_full)
        if opt_cov <= 0:
            print(f"[seed {seed}] OPT=0 (status={status})，跳过")
            continue
        ppw = _by_worker(cands)
        sizes = [len(v) for v in ppw.values()]
        instances.append({
            "seed": seed, "inst": inst, "by_w": ppw, "opt_cov": opt_cov,
            "opt_routes": {c.wid: c for c in sel_opt}, "status": status,
        })
        print(f"[seed {seed}] OPT_full={opt_cov} ({status}) 选中 {len(sel_opt)} 人；"
              f"每人候选 {min(sizes)}–{max(sizes)} 条（k={args.k}）")

    if not instances:
        print("没有可用实例，退出。")
        return 1

    menu_sizes = [len(v) for rec in instances for v in rec["by_w"].values()]
    rows: List[dict] = []
    llm_audit: List[dict] = []
    gens = ("llm", "topk_cov", "topk_sat", "random")

    for pmode in ("random", "adversarial"):
        for eta in eta_grid:
            for rec in instances:
                rng = random.Random(2000 * rec["seed"] + 23)
                poisoned, decoy_ids = poison_menu(rec["by_w"], eta, pmode, rng)
                forced, slots = _forced_and_slots(poisoned, decoy_ids, args.k)

                pool_llm, audit, n_decoy_llm, n_picks_llm = llm_select(
                    selector, {**rec, "eta": eta, "pmode": pmode}, args,
                    poisoned, decoy_ids)
                llm_audit.extend(audit)

                for gen in gens:
                    if gen == "llm":
                        pool, n_decoy, n_picks = pool_llm, n_decoy_llm, n_picks_llm
                    else:
                        raw = gen_pool(poisoned, args.k, gen,
                                       random.Random(3000 * rec["seed"] + 7),
                                       opt_routes=rec["opt_routes"])
                        n_decoy = sum(1 for c in raw if id(c) in decoy_ids)
                        n_picks = len(raw)
                        pool = _dedupe(raw)
                    if not pool:
                        continue
                    cov, n_sel = _select(rec["inst"], pool, args.theta,
                                         args.budget_steps, args.k)
                    hits, tot = _hit_rate(pool, rec["opt_routes"])
                    denom = max(1, n_picks)
                    rows.append({
                        "exp": "E4.3", "mode": pmode, "gen": gen, "eps": 0.0,
                        "eta": eta, "theta": args.theta, "seed": rec["seed"],
                        "k": args.k, "cov": cov, "opt": rec["opt_cov"],
                        "ratio": round(cov / rec["opt_cov"], 4),
                        "hit_rate": round(hits / tot, 4) if tot else 0.0,
                        "uptake": round(n_decoy / denom, 4),
                        "forced_share": round(forced / max(1, slots), 4),
                        "n_picks": n_picks, "n_slots": slots,
                        "n_selected": n_sel,
                    })

    # ---------------- 汇总 ----------------
    def cell(pmode, eta, gen, key):
        v = [r[key] for r in rows
             if r["mode"] == pmode and abs(r["eta"] - eta) < 1e-9
             and r["gen"] == gen and r.get(key) is not None]
        return mean(v) if v else None

    print("\n===== E4.3 菜单污染：LLM vs 算法 =====")
    print(f"{'pollute':12s} {'eta':>5s} {'gen':10s} {'ratio':>7s} {'hit':>6s} "
          f"{'uptake':>7s} {'forced':>7s} {'net':>6s}")
    for pmode in ("random", "adversarial"):
        for eta in eta_grid:
            for gen in gens:
                rt = cell(pmode, eta, gen, "ratio")
                if rt is None:
                    continue
                hr = cell(pmode, eta, gen, "hit_rate")
                up = cell(pmode, eta, gen, "uptake")
                fc = cell(pmode, eta, gen, "forced_share")
                net = None if up is None or fc is None else max(0.0, up - fc)
                print(f"{pmode:12s} {eta:5.2f} {gen:10s} {rt:7.3f} {hr:6.3f} "
                      f"{up:7.3f} {fc:7.3f} {(net if net is not None else float('nan')):6.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    model_name = args.model or preset.model
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)
    stem = os.path.join(
        args.out_dir,
        f"llm_pollution_{args.dataset}_{args.scale}_n{len(instances)}"
        f"_llm{safe_model}_{args.prompt_style}")

    payload = {
        "note": (f"E4.3 菜单污染（真实 LLM：{model_name}，prompt={args.prompt_style}）。"
                 f"按 η 把菜单条目换成覆盖最差的诱饵，LLM 从污染菜单选 k 条；"
                 f"topk_cov/topk_sat/random 在同一污染菜单上作对照。"
                 f"uptake=选中诱饵占比，forced_share=好候选不足 k 时的被迫占比，"
                 f"净吸收=uptake−forced_share。"),
        "mode": "llm", "dataset": args.dataset, "scale": args.scale,
        "n_workers": args.n_workers, "k_full": args.k_full, "k": args.k,
        "budget_steps": args.budget_steps, "theta": args.theta,
        "seeds": len(instances), "eta_grid": eta_grid,
        "menu_sizes": {"min": min(menu_sizes), "max": max(menu_sizes),
                       "mean": round(mean(menu_sizes), 2)},
        "rows": rows,
        "llm": {
            "provider": args.provider, "model": model_name,
            "prompt_style": args.prompt_style, "stats": selector.stats,
            "n_selections": len(llm_audit),
            "n_padded_total": sum(r.get("n_padded", 0) for r in llm_audit),
            "n_unparsed": sum(1 for r in llm_audit if not r.get("llm_ok")),
            "audit": llm_audit,
        },
    }
    st = selector.stats
    live_tok = st["prompt_tokens"] + st["completion_tokens"]
    cache_tok = st["prompt_tokens_cached"] + st["completion_tokens_cached"]
    payload["llm"]["tokens"] = {"live": live_tok, "from_cache": cache_tok,
                                "total": live_tok + cache_tok}

    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = [f"# E4.3 污染鲁棒性（真实 LLM vs 算法，{args.dataset}/{args.scale}）", "",
             f"参与者 {args.n_workers}、菜单 k_full={args.k_full}（实测 "
             f"{payload['menu_sizes']['min']}–{payload['menu_sizes']['max']} 条）、"
             f"选择 |C_u|={args.k}、步数预算 {args.budget_steps:.0f}、θ={args.theta}、"
             f"{len(instances)} 个种子。", "",
             f"> **口径**：{payload['note']}", "",
             "| 污染模式 | η | 生成器 | A′/OPT | 命中率 | 吸收率 | 被迫率 | 净吸收 |",
             "|---|---|---|---|---|---|---|---|"]
    for pmode in ("random", "adversarial"):
        for eta in eta_grid:
            for gen in gens:
                rt = cell(pmode, eta, gen, "ratio")
                if rt is None:
                    continue
                hr = cell(pmode, eta, gen, "hit_rate")
                up = cell(pmode, eta, gen, "uptake")
                fc = cell(pmode, eta, gen, "forced_share")
                net = None if up is None or fc is None else max(0.0, up - fc)
                f = lambda x, w: "—" if x is None else f"{x:.3f}"
                lines.append(f"| {pmode} | {eta:.2f} | `{gen}` | **{rt:.3f}** | "
                             f"{f(hr,3)} | {f(up,3)} | {f(fc,3)} | {f(net,3)} |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n===== LLM 用量（{model_name}）=====")
    print(f"  实发请求 {st['calls']} 次（{live_tok:,} token），"
          f"缓存命中 {st['cache_hits']} 次（{cache_tok:,} token），失败 {st['failures']} 次")
    print(f"  补足条数合计 {payload['llm']['n_padded_total']}／"
          f"{payload['llm']['n_selections']} 次选择")
    selector.close()
    print(f"\nwrote {stem}.json\nwrote {stem}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
