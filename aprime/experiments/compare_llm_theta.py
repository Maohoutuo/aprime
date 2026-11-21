# -*- coding: utf-8 -*-
r"""把多档 θ、多模型的 LLM 候选生成结果聚合成敏感性表（E4.2 的 θ × 模型维度）。

动机：``run_llm_candidates.py`` 一次只跑一个 (模型, θ)，且文件名不含 θ。做敏感性研究时
按 θ 分目录各跑一遍，再用本脚本合并——它**不联网、不重新求解**，只读 JSON。

    py -3.14 experiments/compare_llm_theta.py "results/llm_theta_t*/*.json"

输出 ``results/llm_theta_sensitivity.md`` 与同名 ``.json``。

关键点：
* **必须按模型分组**。若只按 θ 聚合，不同模型的行会被平均掉，跨模型结论就无从谈起。
  模型名取自每个 JSON 顶层的 ``llm.model``；仿真产物没有该字段，记为 ``sim``。
* 若某文件的 ``rows`` 里没有 ``theta`` 字段（早期产物），记为 ``unknown`` 并提示，
  **不会**假装知道它的 θ。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional


def _load(paths: List[str]) -> List[dict]:
    out: List[dict] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[跳过] {p}: {exc}", file=sys.stderr)
            continue
        model = (d.get("llm") or {}).get("model") or "sim"
        for r in d.get("rows", []):
            if r.get("exp") != "E4.2":
                continue
            r = dict(r)
            r["_src"] = p
            r["_model"] = model
            r.setdefault("theta", None)
            out.append(r)
    return out


def _tkey(t) -> str:
    return "unknown" if t is None else ("%.2f" % float(t))


def _verdict(seq: List[tuple]) -> str:
    """seq: [(theta, hit_rate)]，按 θ 升序。给出单调性判定。"""
    if len(seq) < 2:
        return "—（档位不足）"
    hits = [h for _, h in seq]
    non_dec = all(hits[i] <= hits[i + 1] + 1e-9 for i in range(len(hits) - 1))
    non_inc = all(hits[i] >= hits[i + 1] - 1e-9 for i in range(len(hits) - 1))
    if non_dec and non_inc:
        return "基本持平"
    if non_dec:
        return "**单调不减（随 θ 上升）**"
    if non_inc:
        return "单调不增"
    return "**非单调**"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="+", help="JSON 路径或 glob（如 results/*/*.json）")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.pattern:
        hit = sorted(glob.glob(pat))
        paths.extend(hit if hit else ([pat] if os.path.exists(pat) else []))
    paths = list(dict.fromkeys(paths))
    if not paths:
        print("没有匹配到任何 JSON", file=sys.stderr)
        return 1

    rows = _load(paths)
    if not rows:
        print("JSON 里没有 E4.2 行", file=sys.stderr)
        return 1

    # ---------- 聚合：(model, theta, mode, eps) → (hit_rate, ratio)
    hit_acc: Dict[tuple, List[float]] = defaultdict(list)
    ratio_acc: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        key = (r["_model"], _tkey(r["theta"]), r["mode"], float(r.get("eps") or 0.0))
        if r.get("hit_rate") is not None:
            hit_acc[key].append(float(r["hit_rate"]))
        if r.get("ratio") is not None:
            ratio_acc[key].append(float(r["ratio"]))

    models = sorted({r["_model"] for r in rows})
    thetas = sorted({_tkey(r["theta"]) for r in rows},
                    key=lambda s: (s == "unknown", s))
    modes = sorted({r["mode"] for r in rows})

    # ---------- 控制台
    print("%-28s %-6s %-12s %5s %10s %8s %6s" %
          ("model", "theta", "mode", "eps", "hit_rate", "ratio", "n"))
    for m in models:
        for th in thetas:
            for mode in modes:
                eps_list = sorted({float(r.get("eps") or 0.0) for r in rows
                                   if r["_model"] == m and _tkey(r["theta"]) == th
                                   and r["mode"] == mode})
                for eps in eps_list:
                    k = (m, th, mode, eps)
                    n = len(ratio_acc[k])
                    hr = mean(hit_acc[k]) if hit_acc[k] else float("nan")
                    rt = mean(ratio_acc[k]) if ratio_acc[k] else float("nan")
                    print("%-28s %-6s %-12s %5.2f %10.3f %8.3f %6d"
                          % (m, th, mode, eps, hr, rt, n))

    md: List[str] = []
    md.append("# LLM 候选生成：θ 敏感性（E4.2 的 θ × 模型维度）\n")
    md.append("命中率 ＝ OPT 选中的参与者里其最优路线落入 $C_u$ 的比例；"
              "A′/OPT ＝ A′ 在 $C_u$ 上的覆盖 ÷ 全池 ILP 最优。\n")
    if "unknown" in thetas:
        md.append("> **注意**：部分产物的 `rows` 里没有 `theta` 字段（早期版本），"
                  "记为 `unknown`，不能当作已知 θ 解读。\n")

    # ---------- 核心：真实 LLM 跨模型 × θ
    llm_rows = [r for r in rows if r["mode"] == "llm"]
    llm_models = sorted({r["_model"] for r in llm_rows})
    if llm_rows:
        known_thetas = [t for t in thetas if t != "unknown"]
        md.append("## 真实 LLM：命中率随 θ 的变化（跨模型对照）\n")
        md.append("| 模型 | " + " | ".join("θ=%s" % t for t in known_thetas)
                  + " | 单调性（命中率） |")
        md.append("|---" * (len(known_thetas) + 2) + "|")
        for m in llm_models:
            cells = []
            for t in known_thetas:
                k = (m, t, "llm", 0.0)
                cells.append("%.3f" % mean(hit_acc[k]) if hit_acc[k] else "—")
            seq = []
            for t in known_thetas:
                k = (m, t, "llm", 0.0)
                if hit_acc[k]:
                    seq.append((float(t), mean(hit_acc[k])))
            md.append("| `%s` | %s | %s |" % (m, " | ".join(cells), _verdict(seq)))
        md.append("")
        md.append("**读法**：若各模型的「单调性」列一致为「单调不减」，则"
                  "“命中率随 θ 上升”是跨模型成立的性质，而非单个模型的特例。\n")

        # 每个模型的 A′/OPT 明细
        md.append("### 各模型 A′/OPT 随 θ\n")
        md.append("| 模型 | " + " | ".join("θ=%s" % t for t in known_thetas) + " |")
        md.append("|---" * (len(known_thetas) + 1) + "|")
        for m in llm_models:
            cells = []
            for t in known_thetas:
                k = (m, t, "llm", 0.0)
                cells.append("%.3f" % mean(ratio_acc[k]) if ratio_acc[k] else "—")
            md.append("| `%s` | %s |" % (m, " | ".join(cells)))
        md.append("")
    else:
        md.append("## 真实 LLM\n\n本批数据中没有 `mode=llm` 的行，只有可控仿真结果，"
                  "故无法据此讨论真实模型随 θ 的行为。\n")

    # ---------- 全模式明细
    md.append("## 全模式明细\n")
    md.append("| 模型 | θ | 生成器 | ε | 命中率 | A′/OPT | n |")
    md.append("|---|---|---|---|---|---|---|")
    for m in models:
        for th in thetas:
            for mode in modes:
                eps_list = sorted({float(r.get("eps") or 0.0) for r in rows
                                   if r["_model"] == m and _tkey(r["theta"]) == th
                                   and r["mode"] == mode})
                for eps in eps_list:
                    k = (m, th, mode, eps)
                    n = len(ratio_acc[k])
                    hr_s = "—" if not hit_acc[k] else "%.3f" % mean(hit_acc[k])
                    rt_s = "—" if not ratio_acc[k] else "%.3f" % mean(ratio_acc[k])
                    md.append("| `%s` | %s | `%s` | %.2f | %s | %s | %d |"
                              % (m, th, mode, eps, hr_s, rt_s, n))
    md.append("")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "llm_theta_sensitivity")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    payload = {
        "sources": paths,
        "models": models,
        "thetas": thetas,
        "modes": modes,
        "cells": [{"model": k[0], "theta": k[1], "mode": k[2], "eps": k[3],
                   "hit_rate": (round(mean(hit_acc[k]), 4) if hit_acc[k] else None),
                   "ratio": (round(mean(ratio_acc[k]), 4) if ratio_acc[k] else None),
                   "n": len(ratio_acc[k])}
                  for k in sorted(ratio_acc)],
    }
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"\nwrote {stem}.md\nwrote {stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
