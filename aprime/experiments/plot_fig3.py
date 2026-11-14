# -*- coding: utf-8 -*-
r"""Fig.3：个性化强度 θ 下的四面板权衡图。

    py -3.14 experiments/plot_fig3.py --json results/fairness_tradeoff_tdrive_small_b140_n30.json

数据来源 `experiments/run_fairness_tradeoff.py` 的长表；本脚本只做聚合与作图，
**不重算任何指标**，保证图与表用的是同一份数据。

四个面板对应论文 §6 的四条主张：

(a) `coverage / budget`  —— θ 越高，可行域碎片化，覆盖如何下降（A′ 与 OPT 的距离）
(b) `feasible_rate`      —— θ 是**硬约束**，这条曲线是"个性化"的代价，与选择规则无关
(c) 能力表示度（默认 `rank_shift`）—— 选中者在能力轴上的百分位偏移，
    0 = 与人数成比例，负 = 偏向低能力者，正 = 偏向高能力者
(d) `exclusion_gap`      —— 结构性挤出（低能力组被排除率 − 高能力组被排除率）

**面板 (c) 为什么不用 `gqc_gap`。** `gqc_gap` 依赖"能力分桶"：当分组塌成一档时
它恒为 0（看着像"完全公平"，实则无信息），最小组过小时份额估计不可信。
实测其跨种子噪声也更大（见 `experiments/compare_fairness_metrics.py` 与
docs §8.7–8.9）。`rank_shift` / `ks_capability` 直接在能力轴上比较分布，
不经过任何桶，因此不会失效、噪声也更低。需要画旧口径时用
``--fair-metric gqc_gap``，脚本会在存在退化格时于面板上打警告。

图中只对 A′ 与 A′+GQC 画 95% 置信带，其余方法画细线，避免 5 条带子糊成一片。
φ = 0 用实线（异质偏好），φ = 0.95 用虚线，且只画在两条主角线上。

**预算口径提醒**：覆盖率是"覆盖单元 / 预算"，预算放大后这个比值会显著变小
（B=140 时约 0.46，B=80 时约 0.74）。跨预算的 `cov/B` **不可直接比较**；
比较覆盖时应看绝对覆盖单元数或 `jain`。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: method -> (图例名, 颜色, 线型, 是否画置信带)
STYLE = {
    "exact":    ("OPT (ILP)",           "#9e9e9e", (0, (1, 1.5)), False),
    "aprime":   ("A′ (ours)",           "#1f77b4", "-",           True),
    "gqc":      ("A′ + GQC (ours)",     "#d62728", "-",           True),
    "capacity": ("CFI-type",            "#2ca02c", "-",           False),
    "mscta":    ("MSCTA-type",          "#9467bd", "-",           False),
    "sa_caftca": ("SA + CFA penalty",   "#ff7f0e", "-",           False),
}
MAIN_METHODS = ["exact", "aprime", "gqc", "capacity", "mscta", "sa_caftca"]
ALT_LINE_METHODS = ["aprime", "gqc"]   # φ = alt 时额外画虚线的两条主角

#: 面板 (c) 可选的公平度量 -> 纵轴标签
FAIR_LABEL = {
    "rank_shift": "capability rank shift\n(0 = proportional, + = favours high-capability)",
    "ks_capability": "KS distance, selected vs pool\n(0 = same capability distribution)",
    "gqc_gap": "GQC gap  (slots vs headcount share)",
}


def _agg(rows, phi, theta, method, key):
    vals = [r[key] for r in rows
            if r["phi"] == phi and r["theta"] == theta and r["method"] == method
            and r.get(key) is not None]
    if not vals:
        return None, None
    mu = mean(vals)
    if len(vals) < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, 1.96 * (var / len(vals)) ** 0.5


def _series(rows, phi, method, key):
    thetas = sorted({r["theta"] for r in rows if r["phi"] == phi})
    xs, mu, half = [], [], []
    for t in thetas:
        m, h = _agg(rows, phi, t, method, key)
        if m is None:
            continue
        xs.append(t)
        mu.append(m)
        half.append(h or 0.0)
    return xs, mu, half


def _panel(ax, rows, key, ylabel, phi_main, phi_alt, ref_line=None, thresholds=None):
    for method in MAIN_METHODS:
        xs, mu, half = _series(rows, phi_main, method, key)
        if not xs:
            continue
        label, color, ls, band = STYLE[method]
        ax.plot(xs, mu, linestyle=ls, color=color, lw=1.9 if method in ("aprime", "gqc") else 1.2,
                label=label, zorder=3 if method in ("aprime", "gqc") else 2)
        if band and any(half):
            ax.fill_between(xs, [m - h for m, h in zip(mu, half)],
                            [m + h for m, h in zip(mu, half)],
                            color=color, alpha=0.16, lw=0, zorder=1)
    if phi_alt is not None:
        for method in ALT_LINE_METHODS:
            xs, mu, _ = _series(rows, phi_alt, method, key)
            if not xs:
                continue
            label, color, _ls, _b = STYLE[method]
            ax.plot(xs, mu, linestyle="--", color=color, lw=1.3, alpha=0.75,
                    label=f"{label}, φ={phi_alt:g}", zorder=2)
    if ref_line is not None:
        ax.axhline(ref_line, color="#555555", lw=0.9, ls=":", zorder=1)
    # 命题 4 的两个闭式阈值：可行域在 θ_all 之后开始碎裂、在 θ_any 之后彻底为空
    if thresholds:
        lo, hi = ax.get_xlim()
        for key_name, txt, ls_ in (("theta_all", r"$\theta_{all}$", (0, (4, 2))),
                                   ("theta_any", r"$\theta_{any}$", (0, (1, 2)))):
            v = thresholds.get(key_name)
            if v is None:
                continue
            if v <= hi:
                ax.axvline(v, color="#7f7f7f", lw=1.0, ls=ls_, zorder=1)
                ax.text(v, ax.get_ylim()[1], f" {txt}", rotation=90, va="top", ha="left",
                        fontsize=8, color="#555555")
            elif key_name == "theta_any":
                # 阈值落在扫描范围之外 ⇒ 整个 θ 网格内可行域都不会空，画在图外会留大片空白
                ax.text(0.985, 0.06, f"{txt} ≈ {v:.2f} > grid max\n(feasible set never empty "
                        f"in scanned range)", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="#666666")
    ax.set_xlabel("personalization threshold θ")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/fairness_tradeoff_tdrive_small_b140_n30.json")
    ap.add_argument("--out", default="figures/fig3_tradeoff_panels.png")
    ap.add_argument("--phi-main", type=float, default=0.0)
    ap.add_argument("--phi-alt", type=float, default=0.95)
    ap.add_argument("--fair-metric", default="rank_shift",
                    choices=["rank_shift", "ks_capability", "gqc_gap"],
                    help="面板 (c) 用的公平度量。默认 rank_shift（不分桶、噪声最低）；"
                         "gqc_gap 为旧口径，存在退化格时脚本会在面板上警告")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    with io.open(args.json, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload["rows"]
    for r in rows:
        if r.get("budget"):
            r["cov_ratio"] = r["coverage"] / r["budget"]
    phis = {r["phi"] for r in rows}
    phi_alt = args.phi_alt if args.phi_alt in phis else None

    # 命题 4 的闭式阈值（跨种子的中位数）：只在主 φ 上标注
    thr_rows = [t for t in (payload.get("theta_thresholds") or []) if t["phi"] == args.phi_main]
    thr = {}
    if thr_rows:
        thr = {"theta_all": mean([t["theta_all"] for t in thr_rows]),
               "theta_any": mean([t["theta_any"] for t in thr_rows])}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True)

    _panel(axes[0][0], rows, "cov_ratio", "coverage / budget steps", args.phi_main, phi_alt,
           thresholds=thr)
    _panel(axes[0][1], rows, "feasible_rate", "share of workers with ≥1 eligible route",
           args.phi_main, phi_alt, thresholds=thr)
    _panel(axes[1][0], rows, args.fair_metric, FAIR_LABEL[args.fair_metric], args.phi_main,
           phi_alt, ref_line=0.0 if args.fair_metric == "rank_shift" else None,
           thresholds=None)
    if any(r.get("grouping_degenerate") for r in rows):
        axes[1][0].text(0.985, 0.92,
                        "⚠ degenerate grouping present\n(n_groups=1 ⇒ gqc_gap ≡ 0)",
                        transform=axes[1][0].transAxes, ha="right", va="top",
                        fontsize=7.5, color="#b00020")
    _panel(axes[1][1], rows, "exclusion_gap", "exclusion gap  (low − high capability)",
           args.phi_main, phi_alt, ref_line=0.0, thresholds=thr)

    for ax, tag in zip(axes.ravel(), "abcd"):
        ax.set_title(f"({tag})", loc="left", fontsize=11, fontweight="bold")

    # 图例合并到整图右侧，避免重复四遍
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes.ravel()[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))

    n = payload.get("n", "?")
    b = payload.get("budget_steps")
    budget_txt = "dataset default budget" if b is None else f"budget = {b:.0f} steps"
    mult = payload.get("budget_mult") or 1.0
    if mult != 1.0:
        budget_txt += f" (×{mult:g})"
    fig.suptitle(f"T-Drive / small, N={payload.get('n_workers')}, k={payload.get('k')}, "
                 f"{budget_txt}, {n} seeds — solid: φ={args.phi_main:g}"
                 + (f", dashed: φ={phi_alt:g}" if phi_alt is not None else ""),
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print("wrote", out)


if __name__ == "__main__":
    main()
