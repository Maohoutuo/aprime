# -*- coding: utf-8 -*-
r"""Fig.3b：GQC 的**尺度健壮性**——T-Drive small vs medium，φ=0。

    py -3.14 experiments/plot_scale_check.py \
        --small results/fairness_tradeoff_tdrive_small_b80_n8.json \
        --medium results/fairness_tradeoff_tdrive_medium_b80_n3.json

三个面板：coverage/budget、gqc_gap、exclusion_gap，横轴 θ。
每个方法画两条：small（粗，n=8）和 medium（细虚线，n=3，仅定性）。
作图只聚合已有长表，不重算任何指标，保证与 .md 表同一份数据。
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

STYLE = {
    "aprime":   ("A′ (ours)",        "#1f77b4"),
    "gqc":      ("A′ + GQC (ours)",  "#d62728"),
}


def _agg(rows, theta, method, key):
    vals = [r[key] for r in rows
            if r["theta"] == theta and r["method"] == method and r.get(key) is not None]
    if not vals:
        return None
    return mean(vals)


def _series(rows, theta_max, method, key):
    xs, ys = [], []
    for t in sorted({r["theta"] for r in rows if r["theta"] <= theta_max}):
        m = _agg(rows, t, method, key)
        if m is None:
            continue
        xs.append(t)
        ys.append(m)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--medium", required=True)
    ap.add_argument("--out", default="figures/fig3b_scale_check.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    with io.open(args.small, encoding="utf-8") as fh:
        small = json.load(fh)
    with io.open(args.medium, encoding="utf-8") as fh:
        medium = json.load(fh)
    small_rows, medium_rows = small["rows"], medium["rows"]

    for r in small_rows:
        if r.get("budget"):
            r["cov_ratio"] = r["coverage"] / r["budget"]
    for r in medium_rows:
        if r.get("budget"):
            r["cov_ratio"] = r["coverage"] / r["budget"]

    phi = 0.0
    panels = [
        ("cov_ratio", "coverage / budget steps"),
        ("gqc_gap", "GQC gap  (slots vs headcount share)"),
        ("exclusion_gap", "exclusion gap  (low − high capability)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=False)
    for ax, (key, ylabel) in zip(axes, panels):
        for scale_name, rows, lw, ls in (("small", small_rows, 2.0, "-"),
                                         ("medium", medium_rows, 1.1, ":")):
            xs_max = max(r["theta"] for r in rows)
            for method, (label, color) in STYLE.items():
                xs, ys = _series(rows, xs_max, method, key)
                if not xs:
                    continue
                meta = small if scale_name == "small" else medium
                ax.plot(xs, ys, linestyle=ls, color=color, lw=lw,
                        label=f"{label} ({meta.get('scale')} grid, N={meta.get('n_workers')}, "
                              f"{meta.get('n')} seeds)")
        ax.set_xlabel("personalization threshold θ")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    axes[0].set_title("(a) coverage", loc="left", fontsize=10)
    axes[1].set_title("(b) GQC gap", loc="left", fontsize=10)
    axes[2].set_title("(c) exclusion gap", loc="left", fontsize=10)

    handles, labels = {}, {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles[ll] = hh
                labels[ll] = True
    fig.legend(list(handles.values()), list(labels.keys()), loc="lower center", ncol=4,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.14))

    b = small.get("budget_steps")
    fig.suptitle(f"T-Drive, φ={phi:g}, budget = {b:.0f} steps — thick: small grid "
                 f"(statistical), thin dotted: medium grid (indicative only)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print("wrote", out)


if __name__ == "__main__":
    main()
