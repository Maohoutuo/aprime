# -*- coding: utf-8 -*-
r"""把 results/*.json 画成论文里的图（图 3/4/9/11 + 协商前后对比）。

    py -3.14 experiments\make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join("results", "figures")
DPI = 160
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _load(name: str):
    path = os.path.join("results", name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _pick(prefix: str, prefer: str = "tdrive") -> dict | None:
    """同名前缀的结果文件里优先用真实 LLM 的那份（其次偏好指定数据集）。"""
    cands = [c for c in sorted(os.listdir("results"))
             if c.startswith(prefix) and c.endswith(".json")]
    if not cands:
        return None
    real = [c for c in cands if "surrogate" not in c.lower()]
    pool = real or cands
    pool.sort(key=lambda c: (prefer not in c, "qwen" not in c))
    return _load(pool[0])


def fig_ablation() -> str | None:
    """图 3：消融（Coverage / PSS × 三个规模）。"""
    data = _pick("ablation_")
    if not data:
        return None
    scales = list(data["results"].keys())
    variants = ["w/o PRG", "w/o FPS", "w/o NRR", "Full"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(scales))
    w = 0.2
    for ax, key, std, title in ((axes[0], "coverage", "coverage_std", "Coverage Utility"),
                                (axes[1], "pss", "pss_std", "Path Satisfaction Score")):
        for i, v in enumerate(variants):
            vals = [data["results"][s].get(v, {}).get(key, np.nan) for s in scales]
            errs = [data["results"][s].get(v, {}).get(std, 0) for s in scales]
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=2, label=v)
        ax.set_xticks(x, scales)
        ax.set_title(title)
        if key == "coverage":
            ax.set_ylim(min(4.3, np.nanmin([data["results"][s][v]["coverage"] for s in scales for v in variants]) - 0.1),
                        None)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Ablation on {data['dataset']} (n={data['n']}, backend={data['backend']})", y=0.99)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_ablation.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_fairness() -> str | None:
    """图 4：四种选择策略的公平性分析。"""
    data = _pick("fairness_")
    if not data:
        return None
    strategies = ["UR", "LS", "CG", "FPS"]
    colors = {"UR": "#9aa5b1", "LS": "#5b8def", "CG": "#e8833a", "FPS": "#2f9e6f"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for name in strategies:
        counts = np.array(data["strategies"][name]["counts"])
        axes[0].bar(np.arange(len(counts)), counts, label=name, color=colors[name], alpha=0.75)
    axes[0].set_xlabel("Worker index")
    axes[0].set_ylabel("# selections over 60 tasks")
    axes[0].set_title("(a) per-worker selection count")
    axes[0].legend(frameon=False, fontsize=7)

    for name in strategies:
        counts = np.sort(np.array(data["strategies"][name]["counts"]))
        cdf = np.arange(1, len(counts) + 1) / len(counts)
        axes[1].plot(counts, cdf, "-o", ms=2.5, label=name, color=colors[name])
    axes[1].set_xlabel("# selections over 60 tasks")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("(b) CDF of selection counts")

    cov = [data["strategies"][s]["coverage"] for s in strategies]
    pss = [data["strategies"][s]["pss"] for s in strategies]
    axes[2].bar(strategies, cov, color=[colors[s] for s in strategies], alpha=0.85)
    for i, v in enumerate(cov):
        axes[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    axes[2].set_title("(c) coverage utility")
    axes[2].set_ylim(min(cov) - 0.4, max(cov) + 0.15)
    axes[3].bar(strategies, pss, color=[colors[s] for s in strategies], alpha=0.85)
    for i, v in enumerate(pss):
        axes[3].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    axes[3].set_title("(d) path satisfaction score")
    axes[3].set_ylim(min(pss) - 0.1, max(pss) + 0.05)
    fig.suptitle(f"Fairness-aware participant selection (T-Drive Medium, {data['tasks']} tasks, "
                 f"pool {data['pool']}, backend={data['backend']})", y=1.0)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_fairness.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_land_use() -> str | None:
    """图 9：土地利用与犯罪分布。"""
    from mapus.data.instances import make_mapus_instance
    inst = make_mapus_instance("tdrive_medium", 0)
    grab = make_mapus_instance("grab_medium", 0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    cmap = matplotlib.colors.ListedColormap(
        ["#4a8f4a", "#8c6d4f", "#4f6d9e", "#b5484d", "#d9b26a", "#c56fa4"])
    axes[0].imshow(inst.urban.category.T, cmap=cmap, origin="lower", vmin=-0.5, vmax=5.5)
    axes[0].set_title("(a) Land use, T-Drive")
    im = axes[1].imshow(inst.urban.crime_norm.T, cmap="Reds", origin="lower")
    axes[1].set_title("(b) Crime risk, T-Drive")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(grab.urban.category.T, cmap=cmap, origin="lower", vmin=-0.5, vmax=5.5)
    axes[2].set_title("(c) Land use, Grab-Posisi")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(6)]
    axes[0].legend(handles, ["vegetation", "industrial", "institutional", "medical",
                             "residential", "commercial"], fontsize=6, loc="upper center",
                   bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_land_use.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_base_llm() -> str | None:
    """图 11：性能—时延权衡（气泡大小 = 运行时间）。"""
    data = _load("base_llm_n2.json") or _load("base_llm.json")
    if not data:
        return None
    datasets = sorted({d for m in data["models"].values() for d in m})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for model, entry in data["models"].items():
            if dataset not in entry:
                continue
            v = entry[dataset]
            ax.scatter(v["pss"], v["coverage"], s=max(20, v["runtime"] * 1.6),
                       alpha=0.55, edgecolor="k", linewidth=0.4)
            ax.annotate(model.split("/")[-1], (v["pss"], v["coverage"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=6.5)
        ax.set_xlabel("Path satisfaction score")
        ax.set_ylabel("Coverage utility")
        ax.set_title(dataset)
    fig.suptitle(f"Trade-off of base LLMs ({data['scale']}, n={data['n']}; bubble = runtime)", y=1.0)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_base_llm.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_negotiation() -> str | None:
    """表 4 的柱状对照：协商前后的重叠与覆盖。"""
    data = _pick("negotiation_")
    if not data:
        return None
    ds = list(data["datasets"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    x = np.arange(len(ds))
    w = 0.35
    specs = [("overlap", "pre_neg_overlap", "avg_overlap", "Average route overlap"),
             ("coverage", "pre_neg_coverage", "coverage", "Coverage utility"),
             ("pss", "pre_neg_pss", "pss", "Path satisfaction score")]
    for ax, (name, before_key, after_key, title) in zip(axes, specs):
        before = [data["datasets"][d][before_key] for d in ds]
        after = [data["datasets"][d][after_key] for d in ds]
        ax.bar(x - w / 2, before, w, label="before NRR")
        ax.bar(x + w / 2, after, w, label="after NRR")
        for i, (b, a) in enumerate(zip(before, after)):
            ax.text(i - w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ds)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Negotiation-based route refinement (Medium, n={data['n']}, "
                 f"backend={data['backend']})", y=1.0)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_negotiation.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    made = []
    for fn in (fig_ablation, fig_fairness, fig_land_use, fig_base_llm, fig_negotiation):
        try:
            path = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{fn.__name__}: skipped ({exc})")
            continue
        if path:
            made.append(path)
            print("wrote", path)
    print(f"{len(made)} figures")


if __name__ == "__main__":
    main()
