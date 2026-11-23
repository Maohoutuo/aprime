# -*- coding: utf-8 -*-
"""用 NHTS 2022 的出行目的份额，对 ``PREF_MIXTURES`` 的 3 个 archetype 做边缘校准。

背景
----
``mapus/docs/calibration.md`` §4 指出偏好先验有四项缺口，其中三项需要
"居民出行或活动调查的出行目的份额"，映射到 ``city_dweller`` /
``community_helper`` / ``explorer``。本脚本用美国 NHTS 2022 的公开微观数据
产出这三项的实测份额。

**口径限制（必须随结果一起引用）**：
  * NHTS 是美国全国样本，论文目标城市是北京/新加坡——这是**代理边缘校准**，
    用于替换"5 个 archetype 各 0.20"的等概率先验，不等于"偏好模型已按目标城市标定"；
  * 份额按**出行次数**加权（``WTTRDFIN``），不是按人加权；
  * 出行目的 → archetype 的映射是**判断**，见 ``PURPOSE_MAP``，可改可议；
  * ``Home``（返程）与 ``Something else`` 不作去处，归一前已排除。

编码来源
--------
``WHYTRP1S`` 的码值含义取自站点自带的 ``assets/2022/doc/codebook.xlsx``
（Trip 表），**不是凭记忆**。初版曾误以为 70=家庭事务、80=接送，实际是
``70=Transport someone``、``80=Meals``——已按字典更正。

用法
----
    py -3.14 experiments/calibrate_archetype_weights_nhts.py
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import httpx
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(DATA_DIR, "nhts", "raw")
OUT = os.path.join(DATA_DIR, "context")
ZIP = os.path.join(RAW, "NHTS_2022_CSV.zip")
URL = "https://nhts.ornl.gov/assets/2022/download/CSV.zip"

#: WHYTRP1S 码值 → 语义（来自 codebook.xlsx / Trip 表）
LABELS = {
    1: "Home",
    10: "Work",
    20: "School/Daycare/Religious",
    30: "Medical/Dental services",
    40: "Shopping/Errands",
    50: "Social/Recreational",
    70: "Transport someone",
    80: "Meals",
    97: "Something else",
}

#: 出行目的 → archetype（判断，非字典给定）
PURPOSE_MAP = {
    "city_dweller": ["Work", "Shopping/Errands"],
    "community_helper": ["School/Daycare/Religious", "Medical/Dental services",
                         "Transport someone"],
    "explorer": ["Social/Recreational", "Meals"],
}

#: 不作去处、归一前排除
EXCLUDE = {"Home", "Something else"}


def ensure_zip() -> str:
    if os.path.exists(ZIP) and os.path.getsize(ZIP) > 1_000_000:
        return ZIP
    os.makedirs(RAW, exist_ok=True)
    print(f"下载 {URL} …")
    with httpx.Client(timeout=180.0, trust_env=True, follow_redirects=True) as c:
        with c.stream("GET", URL) as r:
            r.raise_for_status()
            with open(ZIP, "wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
    print(f"已保存 {ZIP} ({os.path.getsize(ZIP):,} bytes)")
    return ZIP


def load_trips() -> pd.DataFrame:
    with zipfile.ZipFile(ensure_zip()) as z:
        with z.open("tripv2pub.csv") as fh:
            df = pd.read_csv(fh, low_memory=False)
    df = df[df["WHYTRP1S"].isin(LABELS)].copy()
    df["purpose"] = df["WHYTRP1S"].map(LABELS)
    return df


def main() -> int:
    df = load_trips()
    if df.empty:
        print("没有可用记录", file=sys.stderr)
        return 1

    n = len(df)
    share = df["purpose"].value_counts(normalize=True)
    wsum = df.groupby("purpose")["WTTRDFIN"].sum()
    wshare = wsum / wsum.sum()

    print(f"NHTS 2022 出行记录: {n:,}（WTTRDFIN 加权）\n")
    print(f"{'出行目的':<28}{'未加权':>9}{'加权':>9}")
    rows = []
    for k in LABELS.values():
        if k in wshare.index:
            print(f"{k:<28}{share[k] * 100:8.2f}%{wshare[k] * 100:8.2f}%")
            rows.append({"purpose": k, "share_unweighted_pct": round(share[k] * 100, 4),
                         "share_weighted_pct": round(wshare[k] * 100, 4)})

    dest = {k: v for k, v in wshare.items() if k not in EXCLUDE}
    denom = sum(dest.values())
    if denom <= 0:
        print("非回家去处为空", file=sys.stderr)
        return 1

    print("\n映射到 archetype（仅非回家去处，归一）:")
    weights = {}
    for arch, purposes in PURPOSE_MAP.items():
        v = sum(wshare.get(p, 0.0) for p in purposes)
        weights[arch] = v / denom
        print(f"  {arch:<20}{v / denom * 100:6.2f}%   (占全部出行 {v * 100:5.2f}%)")
    print(f"  非回家去处合计占全部出行 {denom * 100:.2f}%")

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "nhts2022_trip_purpose.csv"),
                              index=False, encoding="utf-8")
    json.dump({
        "source": "NHTS 2022 (nhts.ornl.gov/assets/2022/download/CSV.zip)",
        "variable": "WHYTRP1S (Trip purpose summary), weight=WTTRDFIN",
        "codebook": "assets/2022/doc/codebook.xlsx, sheet=Trip",
        "n_trips": n,
        "shares_weighted": {k: round(v, 4) for k, v in wshare.items()},
        "archetype_weights_normalized": {k: round(v, 4) for k, v in weights.items()},
        "purpose_map": PURPOSE_MAP,
        "excluded": sorted(EXCLUDE),
        "caveat": ("美国全国样本，非北京/新加坡；按出行次数而非人加权；"
                   "目的→archetype 映射为判断。属代理边缘校准。"),
    }, open(os.path.join(OUT, "nhts2022_archetype_weights.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
