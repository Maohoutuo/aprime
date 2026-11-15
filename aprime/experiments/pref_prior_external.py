# -*- coding: utf-8 -*-
r"""用外部真实数据检验 ``mapus/profiles.py`` 里的偏好先验。

背景
----
``PREF_MIXTURES`` 的 5 个 archetype 与其 6 维土地利用偏好向量均为手写设计值，
``source`` 一律标 ``unvalidated``。本脚本用公开的 **Foursquare NYC 签到数据**
做一次外部校验：把每个人真实的签到类别份额当作「显示偏好向量」，看

1. 手写的 5 类人是否在真实数据里有对应物；
2. 等概率的人群构成是否站得住；
3. 数据支持「恰好 5 类」这个聚类数吗。

数据来源
--------
``huggingface.co/datasets/yyh11/FourSquare_checkin``（Apache-2.0，
``FS_NYC.csv``，227,428 条签到 / 1,083 名用户 / 251 个类别，纽约）。
下载：``https://huggingface.co/datasets/yyh11/FourSquare_checkin/resolve/main/FS_NYC.csv``

口径限制（结论必须带这些前提读）
--------------------------------
* **显示偏好 ≠ 陈述偏好**：签到只记录"人在哪打卡"，不记录"人喜欢什么"。
* **签到有系统性偏差**：人们常在餐饮/商店打卡，很少在公园、医院打卡，
  故 ``vegetation``/``medical`` 会被系统性低估。这一点对结论方向有实质影响。
* **纽约 ≠ 北京**：只能用于判断"这类人是否存在"，不能作点估计，也不能替代
  目标城市的调查数据。

用法
----
    py experiments/pref_prior_external.py --csv %TEMP%\\fs_nyc.csv
"""

from __future__ import annotations

import argparse
import os
import unicodedata

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

DIMS = ("vegetation", "industrial", "institutional", "medical",
        "residential", "commercial")

#: 有序映射：先命中先归类，顺序即优先级。
#: 这套映射是本文的分析选择，不是数据自带口径——换一套映射结论会变。
RULES = [
    ("medical", ["hospital", "medical", "clinic", "pharmacy", "drugstore",
                 "dentist", "doctor", "physician", "urgent care", "wellness center",
                 "veterinar", "optician", "physical therapy"]),
    ("residential", ["home (private)", "residential", "apartment", "condo",
                     "housing", "assisted living"]),
    ("industrial", ["office", "factory", "industrial", "warehouse",
                    "manufactur", "business service", "corporate", "printing",
                    "shipping center", "distribution"]),
    ("institutional", ["college", "university", "school", "academic building",
                       "library", "government", "courthouse", "police",
                       "fire station", "embassy", "church", "mosque", "temple",
                       "synagogue", "religious", "community center", "museum",
                       "dormitor", "fraternit", "sororit", "military",
                       "post office", "convention center"]),
    ("vegetation", ["park", "garden", "botanical", "beach", "trail", "scenic",
                    "nature preserve", "forest", "mountain", "lake",
                    "plaza", "cemetery", "monument", "outdoors", "farm",
                    "harbor", "river", "island"]),
    ("commercial", ["bar", "pub", "restaurant", "cafe", "coffee", "shop", "store",
                    "deli", "bodega", "bakery", "bank", "hotel", "motel", "hostel",
                    "gym", "fitness", "salon", "barbershop", "spa", "mall",
                    "market", "cinema", "movie theater", "theater", "nightclub",
                    "lounge", "food", "drink", "brewery", "wine", "liquor",
                    "music venue", "stadium", "arena", "casino", "bowling",
                    "arcade", "bookstore", "gas station", "garage", "electronics",
                    "department", "clothing", "jewelry", "shoe", "toy",
                    "pet store", "florist", "supermarket", "butcher", "food truck",
                    "pizza", "sandwich", "burger", "taco", "sushi", "ramen",
                    "noodle", "cafeter", "breakfast", "brunch", "dessert",
                    "ice cream", "donut", "juice", "tea room", "steakhouse",
                    "seafood", "barbecue", "bbq", "diner", "food court"]),
]

#: 手写 archetype，与 ``mapus/profiles.py::ARCHETYPES`` 保持一致。
ARCHETYPES = {
    "eco_enthusiast": (0.46, 0.02, 0.07, 0.03, 0.27, 0.15),
    "city_dweller": (0.07, 0.05, 0.05, 0.04, 0.39, 0.40),
    "industrial_worker": (0.06, 0.48, 0.10, 0.04, 0.24, 0.08),
    "community_helper": (0.12, 0.04, 0.24, 0.22, 0.28, 0.10),
    "explorer": (0.17, 0.17, 0.16, 0.17, 0.17, 0.16),
}


def classify(name: str) -> str | None:
    s = unicodedata.normalize("NFKD", name.lower())
    for dim, keys in RULES:
        if any(k in s for k in keys):
            return dim
    return None


def build_vectors(df: pd.DataFrame, min_visits: int = 30,
                  min_cov: float = 0.70):
    """每人 6 维签到份额向量；剔除映射覆盖率过低或样本过少的用户。"""
    df = df.copy()
    df["dim"] = df.venueCategory.map(classify)
    m = df[df["dim"].notna()]
    cnt = m.groupby(["userId", "dim"]).size().unstack(fill_value=0)
    for d in DIMS:
        if d not in cnt.columns:
            cnt[d] = 0
    cnt = cnt[list(DIMS)]
    tot = cnt.sum(axis=1)
    cov = df.groupby("userId")["dim"].apply(lambda s: s.notna().mean())
    keep = (tot >= min_visits) & (cov >= min_cov)
    vec = cnt[keep].div(tot[keep], axis=0).to_numpy()
    return vec, df


def silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    """平均轮廓系数（无 sklearn 依赖）。"""
    ks = np.unique(lab)
    if len(ks) < 2:
        return -1.0
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    scores = []
    for i in range(len(x)):
        same = lab == lab[i]
        same[i] = False
        if same.sum() == 0:
            scores.append(0.0)
            continue
        a = d[i, same].mean()
        b = min(d[i, lab == k].mean() for k in ks if k != lab[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def best_kmeans(x: np.ndarray, k: int, seeds: int = 8):
    best = None
    for seed in range(seeds):
        cent, lab = kmeans2(x, k, minit="++", seed=seed)
        inertia = float(((x - cent[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, cent, lab)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(os.environ.get("TEMP", "."),
                                                  "fs_nyc.csv"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "pref_prior_external.md"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    vec, dfm = build_vectors(df)
    L = []
    L.append("# 偏好先验的外部校验（Foursquare NYC 签到）\n")
    L.append(f"数据：227,428 条签到 / {df.userId.nunique()} 名用户 / "
             f"{df.venueCategory.nunique()} 个类别（Apache-2.0）。")
    L.append(f"类别→6 维映射覆盖率：{dfm['dim'].notna().mean()*100:.1f}%；"
             f"保留用户 {len(vec)} 名（≥30 条已映射签到、覆盖率≥70%）。\n")

    mean_vec = vec.mean(axis=0)
    L.append("## 全样本平均显示偏好\n")
    L.append("| 维度 | 均值 | 跨用户 sd | 中位 | 90 分位 |")
    L.append("|---|---|---|---|---|")
    for j, d in enumerate(DIMS):
        L.append(f"| {d} | {mean_vec[j]:.3f} | {vec[:, j].std(axis=0):.3f} | "
                 f"{np.median(vec[:, j]):.3f} | {np.quantile(vec[:, j], 0.9):.3f} |")

    # k 选择：轮廓系数，避免"因为先验有 5 类所以取 k=5"的循环论证
    L.append("\n## 聚类数是否支持 5？\n")
    L.append("| k | 轮廓系数 |")
    L.append("|---|---|")
    sil = {}
    for k in range(2, 8):
        _, _, lab = best_kmeans(vec, k)
        sil[k] = silhouette(vec, lab)
        L.append(f"| {k} | {sil[k]:.3f} |")
    kbest = max(sil, key=sil.get)
    L.append(f"\n轮廓系数最优 k={kbest}。")

    # k=5 详细结果
    _, cent, lab = best_kmeans(vec, 5)
    sizes = np.bincount(lab, minlength=5) / len(lab)
    order = np.argsort(-sizes)
    L.append("\n## k=5 的簇中心与构成（按占比降序）\n")
    L.append("| 组 | 占比 | " + " | ".join(DIMS) + " |")
    L.append("|---|---|" + "---|" * len(DIMS))
    for rank, k in enumerate(order, 1):
        row = " | ".join(f"{cent[k][j]:.3f}" for j in range(len(DIMS)))
        L.append(f"| {rank} | {sizes[k]*100:.1f}% | {row} |")

    L.append("\n## 手写 archetype 与真实簇的最近匹配（L1 距离）\n")
    L.append("| archetype | 最近簇 | 簇占比 | L1 | 最远簇 L1 |")
    L.append("|---|---|---|---|---|")
    for name, v in ARCHETYPES.items():
        d = [float(np.abs(np.array(v) - cent[k]).sum()) for k in range(5)]
        k = int(np.argmin(d))
        L.append(f"| {name} | {k+1} | {sizes[k]*100:.1f}% | {d[k]:.3f} | {max(d):.3f} |")

    L.append("\n## 各专向类型的真实尾部占比\n")
    L.append("| 类型 | 判据 | 达到的用户占比 |")
    L.append("|---|---|---|")
    for j, name, thr in [(0, "vegetation", 0.30), (1, "industrial", 0.30),
                         (3, "medical", 0.15), (4, "residential", 0.30)]:
        f = float((vec[:, j] >= thr).mean())
        L.append(f"| {name} 主导 | ≥{thr:.2f} | {f*100:.1f}% |")
    f = float((vec[:, 5] >= 0.80).mean())
    L.append(f"| commercial 主导 | ≥0.80 | {f*100:.1f}% |")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
