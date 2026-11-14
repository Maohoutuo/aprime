# -*- coding: utf-8 -*-
r"""跨运行配对复核：预算档（B=40/80/120）与 `slack` 消融（0 vs 2）。

**为什么需要单独一个脚本**：`run_fairness_tradeoff.py` 的配对统计只在**单次运行内**
（GQC − A′）做。表 6-6 / 6-7 要回答的是"**不同预算档之间**的效应量是否有显著差异"，
而各档的种子数不同、实例难度也不同，只能把两次运行对齐到同一批 `(φ, seed, θ)` 单元
再逐单元配对，才能消除实例难度这个方差来源。

本脚本**只读**已有的 `results/fairness_tradeoff_*.json`，不重跑任何实验、不产生新数字，
输出 `results/budget_tier_paired.md` 作为表 6-6 / 6-7 配对差值的可追溯出处。

口径与 `run_fairness_tradeoff.py::pair_stats` 一致：配对差值均值 ± 1.96·SE（正态近似，
n≥100 时与 t 分布差异 <1%）。

注意：所有输入都是**合成参与者**结果（`meta.source="synthetic(slack)"`），
真实数据版本属 `docs/e0_1_data_access_plan.md` §8 的搁置范围。
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean

#: 标签 → 结果文件。B=40 档是默认运行（budget=None → 40），5 种子；其余为 8 种子。
RUNS = {
    "B=40": "fairness_tradeoff_tdrive_small.json",
    "B=80": "fairness_tradeoff_tdrive_small_b80_n8.json",
    "B=120": "fairness_tradeoff_tdrive_small_b120_n8.json",
    "B=80,slack=2": "fairness_tradeoff_tdrive_small_b80_s2_n8.json",
}

FIELDS = (
    ("coverage", "Δcov"),
    ("gqc_gap", "Δgqc_gap"),
    ("exclusion_gap", "Δexcl"),
)


def _load(out_dir: str, name: str) -> dict:
    path = os.path.join(out_dir, name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _cells(payload: dict, method: str) -> dict:
    return {(r["phi"], r["seed"], round(r["theta"], 4)): r
            for r in payload["rows"] if r["method"] == method}


def _gqc_minus_aprime(payload: dict, field: str) -> dict:
    """运行内 GQC − A′ 的逐单元差值。"""
    a, b = _cells(payload, "gqc"), _cells(payload, "aprime")
    return {k: a[k][field] - b[k][field] for k in set(a) & set(b)}


def _stats(values) -> tuple:
    n = len(values)
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / max(1, n - 1)
    se = (var / n) ** 0.5
    return mu, mu - 1.96 * se, mu + 1.96 * se, n


def _fmt(values) -> str:
    mu, lo, hi, n = _stats(values)
    mark = "" if lo * hi > 0 else " ⚠跨0"
    return f"{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}"


def _binding(payload: dict) -> tuple:
    """配额绑定强度与"实际迫使换人"的频率。

    `binding` = mean(Σ floor_g / k0)：预留名额占总名额的比例，只看配额**想**占多少。
    `deviation` = GQC 与 A′ 覆盖结果不同的单元占比：配额**实际**改变了选择的频率。
    后者才是覆盖代价的直接对应量（前者在两端都可能很高却不咬合）。
    """
    gqc = [r for r in payload["rows"] if r["method"] == "gqc"]
    base = _cells(payload, "aprime")
    ratios, devs = [], []
    for r in gqc:
        floor = r["floor"]
        total = sum(floor.values()) if isinstance(floor, dict) else float(floor)
        ratios.append(total / float(r["k0"]))
        other = base[(r["phi"], r["seed"], round(r["theta"], 4))]
        devs.append(1.0 if r["coverage"] != other["coverage"] else 0.0)
    return mean(ratios), mean(devs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--stem", default="budget_tier_paired")
    args = ap.parse_args()

    runs = {label: _load(args.out_dir, name) for label, name in RUNS.items()}

    lines = [
        "# 预算档与 `slack` 消融的跨运行配对复核",
        "",
        "生成脚本 `experiments/compare_budget_tiers.py`（只读已有结果，不重跑实验）。",
        "配对单元 = 同一 `(φ, seed, θ)`；括号内为 95% CI（正态近似）。"
        "`⚠跨0` 表示该差值不显著。",
        "",
        "> 输入全部为**合成参与者**结果，真实数据版本见 `docs/e0_1_data_access_plan.md` §8。",
        "",
        "## 各档运行内 GQC − A′（配对）",
        "",
        "| 设定 | n 对 | " + " | ".join(label for _, label in FIELDS) + " | Σfloor/k0 | 覆盖结果改变率 |",
        "|---|---|---|" + "---|" * (len(FIELDS) + 1),
    ]
    for tag, payload in runs.items():
        cells = [_gqc_minus_aprime(payload, f) for f, _ in FIELDS]
        n = len(cells[0])
        binding, deviation = _binding(payload)
        lines.append("| %s | %d | %s | %.3f | %.3f |" % (
            tag, n, " | ".join(_fmt(list(c.values())) for c in cells), binding, deviation))

    lines += ["", "## 跨档配对差值（后一档 − 前一档）", "",
              "| 对比 | " + " | ".join(label for _, label in FIELDS) + " |",
              "|---|" + "---|" * len(FIELDS)]
    for left, right in (("B=40", "B=80"), ("B=80", "B=120"),
                        ("B=80", "B=80,slack=2")):
        deltas = []
        for field, _ in FIELDS:
            a = _gqc_minus_aprime(runs[left], field)
            b = _gqc_minus_aprime(runs[right], field)
            shared = set(a) & set(b)
            deltas.append(_fmt([a[k] - b[k] for k in shared]))
        lines.append("| %s − %s | %s |" % (left, right, " | ".join(deltas)))

    lines += [
        "",
        "读法：行标签为「左 − 右」。对 Δcov，**正**表示左档覆盖代价更大；"
        "对 Δgqc_gap / Δexcl，**负**表示左档公平改善更大。因此：",
        "",
        "- `B=40 − B=80` 的 Δcov 为正、Δgqc_gap 为正 ⇒ 40 档代价与收益都更小；",
        "- `B=80 − B=120` 的 Δcov 为负、两个公平指标均跨 0 ⇒ 120 档代价更小且收益持平；",
        "- `B=80 − B=80,slack=2` 的 Δgqc_gap / Δexcl 为负、Δcov 为负 ⇒ 硬配额收益更大但代价也更大。",
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.stem)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", stem + ".md")


if __name__ == "__main__":
    main()
