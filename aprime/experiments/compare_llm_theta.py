# -*- coding: utf-8 -*-
r"""把多档 θ 的 LLM 候选生成结果聚合成敏感性表（E4.2 的 θ 维度）。

动机：``run_llm_candidates.py`` 一次只跑一个 θ，且文件名不含 θ。做敏感性研究时
按 θ 分目录各跑一遍，再用本脚本合并——它**不联网、不重新求解**，只读 JSON。

    # 收集某个目录（或若干文件）下所有结果
    py -3.14 experiments/compare_llm_theta.py "results/llm_theta/*/*.json"

    # 或显式列出
    py -3.14 experiments/compare_llm_theta.py a.json b.json

输出 ``results/llm_theta_sensitivity.md`` 与同名 ``.json``。

口径提醒：若某文件的 ``rows`` 里没有 ``theta`` 字段（早期产物），本脚本会把它
记为 ``unknown`` 并提示，**不会**假装知道它的 θ。
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
        for r in d.get("rows", []):
            if r.get("exp") != "E4.2":
                continue
            r = dict(r)
            r["_src"] = p
            r.setdefault("theta", None)
            out.append(r)
    return out


def _key(t) -> str:
    return "unknown" if t is None else ("%.2f" % float(t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="+", help="JSON 路径或 glob（如 results/*/*.json）")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.pattern:
        hit = sorted(glob.glob(pat))
        paths.extend(hit if hit else ([pat] if os.path.exists(pat) else []))
    # 去重且只留有内容的
    paths = list(dict.fromkeys(paths))
    if not paths:
        print("没有匹配到任何 JSON", file=sys.stderr)
        return 1

    rows = _load(paths)
    if not rows:
        print("JSON 里没有 E4.2 行", file=sys.stderr)
        return 1

    # ---------- 聚合：(theta, mode, eps) → (hit_rate, ratio)
    hit_acc: Dict[tuple, List[float]] = defaultdict(list)
    ratio_acc: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        key = (_key(r["theta"]), r["mode"], float(r.get("eps") or 0.0))
        if r.get("hit_rate") is not None:
            hit_acc[key].append(float(r["hit_rate"]))
        if r.get("ratio") is not None:
            ratio_acc[key].append(float(r["ratio"]))

    thetas = sorted({_key(r["theta"]) for r in rows})
    modes = sorted({r["mode"] for r in rows})

    # ---------- 控制台
    print("%-8s %-12s %5s %10s %8s %6s" %
          ("theta", "mode", "eps", "hit_rate", "ratio", "n"))
    for th in thetas:
        for mode in modes:
            eps_list = sorted({float(r.get("eps") or 0.0) for r in rows
                               if _key(r["theta"]) == th and r["mode"] == mode})
            for eps in eps_list:
                k = (th, mode, eps)
                n = len(ratio_acc[k])
                hr = mean(hit_acc[k]) if hit_acc[k] else float("nan")
                rt = mean(ratio_acc[k]) if ratio_acc[k] else float("nan")
                print("%-8s %-12s %5.2f %10.3f %8.3f %6d" % (th, mode, eps, hr, rt, n))

    # ---------- llm 模式的 θ 敏感性（核心表）
    llm_rows = [r for r in rows if r["mode"] == "llm"]
    md: List[str] = []
    md.append("# LLM 候选生成：θ 敏感性（E4.2 的 θ 维度）\n")
    md.append("命中率 ＝ OPT 选中的参与者里其最优路线落入 $C_u$ 的比例；"
              "A′/OPT ＝ A′ 在 $C_u$ 上的覆盖 ÷ 全池 ILP 最优。\n")
    if ("unknown" in thetas):
        md.append("> **注意**：部分产物的 `rows` 里没有 `theta` 字段（早期版本），"
                  "记为 `unknown`，不能当作已知 θ 解读。\n")

    if llm_rows:
        md.append("## 真实 LLM（`mode=llm`）随 θ 的变化\n")
        md.append("| θ | 命中率 | A′/OPT | 种子/选择数 |")
        md.append("|---|---|---|---|")
        for th in thetas:
            sub = [r for r in llm_rows if _key(r["theta"]) == th
                   and r.get("hit_rate") is not None]
            if not sub:
                continue
            hr = mean(float(r["hit_rate"]) for r in sub)
            rt = mean(float(r["ratio"]) for r in sub)
            md.append("| %s | %.3f | **%.3f** | %d |" % (th, hr, rt, len(sub)))
        md.append("")
        # 单调性判定（仅当 θ 可解析为数值且有 ≥2 档）
        numeric = sorted({float(r["theta"]) for r in llm_rows
                          if r.get("theta") is not None and r.get("hit_rate") is not None})
        if len(numeric) >= 2:
            seq = []
            for th in numeric:
                sub = [float(r["hit_rate"]) for r in llm_rows
                       if r.get("theta") is not None and abs(float(r["theta"]) - th) < 1e-9
                       and r.get("hit_rate") is not None]
                seq.append((th, mean(sub)))
            mono = all(seq[i][1] >= seq[i + 1][1] - 1e-9 for i in range(len(seq) - 1))
            md.append("**单调性判定**：命中率随 θ "
                      + ("**单调不增**（与“高 θ 可行路线变少”的解释一致）"
                         if mono else "**并非单调不增**——该解释不成立，不能这样写")
                      + "。实测序列："
                      + "、".join("θ=%.2f→%.3f" % (t, h) for t, h in seq) + "。\n")
    else:
        md.append("## 真实 LLM（`mode=llm`）\n\n本批数据中没有 `mode=llm` 的行，"
                  "只有可控仿真结果，故无法据此讨论真实模型随 θ 的行为。\n")

    # ---------- 全模式明细
    md.append("## 全模式明细\n")
    md.append("| θ | 生成器 | ε | 命中率 | A′/OPT | n |")
    md.append("|---|---|---|---|---|---|")
    for th in thetas:
        for mode in modes:
            eps_list = sorted({float(r.get("eps") or 0.0) for r in rows
                               if _key(r["theta"]) == th and r["mode"] == mode})
            for eps in eps_list:
                k = (th, mode, eps)
                n = len(ratio_acc[k])
                hr = mean(hit_acc[k]) if hit_acc[k] else float("nan")
                rt = mean(ratio_acc[k]) if ratio_acc[k] else float("nan")
                hr_s = "—" if hit_acc[k] == [] else "%.3f" % hr
                md.append("| %s | `%s` | %.2f | %s | %.3f | %d |"
                          % (th, mode, eps, hr_s, rt, n))
    md.append("")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "llm_theta_sensitivity")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    payload = {
        "sources": paths,
        "thetas": thetas,
        "modes": modes,
        "cells": [{"theta": k[0], "mode": k[1], "eps": k[2],
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
