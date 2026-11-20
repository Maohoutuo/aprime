# -*- coding: utf-8 -*-
r"""跨尺度复核：把 small / medium / large 两次试跑的**结论**逐条对照。

## 为什么需要这个脚本

小网格上得到的效应，可能只是网格太粗导致的伪影。人眼比对两张表时，
很容易只看见"方向差不多"就下结论，而漏掉"方向反了"或"量级差三倍"。
这里把每条结论写成一个**带符号的效应量**，再用统一规则判定：

* **复现**：两次同号，且相对差 < 50%；
* **仅方向复现**：两次同号，但相对差 ≥ 50%（量级不稳，论文只能讲方向）；
* **反转**：两次异号 —— 该结论**至少有一次是伪影**，不能写进论文。

所有效应量都是**同一文件内**相对「全合成」基线的差，所以能跨尺度比较
（不同尺度的 PSS 基线本身不同，直接比绝对值没有意义）。

用法::

    py -3.14 experiments/compare_cross_scales.py \
        --files results/cross_trial_small60.json,results/cross_trial_medium60.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from statistics import mean, stdev
from typing import Dict, List, Tuple

#: (轨迹来源, 用地来源, 犯罪来源) → 展示名。与 run_cross_trial.COMBOS 保持一致。
LABELS = {
    ("synthetic", "synthetic", "synthetic"): "全合成",
    ("tdrive", "synthetic", "synthetic"): "仅T-Drive轨迹",
    ("geolife", "synthetic", "synthetic"): "仅GeoLife轨迹",
    ("synthetic", "osm", "synthetic"): "仅OSM用地",
    ("synthetic", "synthetic", "zhang2025"): "仅犯罪实测",
    ("geolife", "osm", "synthetic"): "轨迹+用地均实测",
    ("geolife", "osm", "zhang2025"): "轨迹+用地+犯罪均实测",
    ("tdrive", "osm", "zhang2025"): "T-Drive+OSM+犯罪实测+显示偏好",
}
BASE = ("synthetic", "synthetic", "synthetic")


def _agg(rows: List[dict], wsrc: str, usrc: str, csrc: str, key: str,
         pref_source: str = "archetype") -> float:
    vals = [r[key] for r in rows
            if r["worker_source"] == wsrc and r["urban_source"] == usrc
            and r["crime_source"] == csrc
            and r.get("pref_source", "archetype") == pref_source]
    return float(mean(vals)) if vals else float("nan")


def _mean_se(rows: List[dict], wsrc: str, usrc: str, csrc: str,
             key: str, pref_source: str = "archetype") -> Tuple[float, float]:
    """均值与标准误（跨种子 × θ 的格子）。"""
    vals = [r[key] for r in rows
            if r["worker_source"] == wsrc and r["urban_source"] == usrc
            and r["crime_source"] == csrc
            and r.get("pref_source", "archetype") == pref_source]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) < 2:
        return float(mean(vals)), float("nan")
    return float(mean(vals)), float(stdev(vals) / math.sqrt(len(vals)))


def effects(rows: List[dict], thetas: List[float]) -> Dict[str, Tuple[float, float]]:
    """把一份结果压成一组**带符号的效应量及其标准误**。

    返回 ``{名称: (效应值, 标准误)}``。标准误用于回答"这个效应是不是伪影"——
    只比两个尺度上的点估计大小是不够的，必须知道它是否能与 0 区分开。
    """
    def _p(wsrc: str, usrc: str, csrc: str, key: str) -> Tuple[float, float]:
        return _mean_se(rows, wsrc, usrc, csrc, key)

    def d(wsrc: str, usrc: str, csrc: str, key: str) -> Tuple[float, float]:
        """相对基线的差值及其标准误（两独立均值之差的合并 SE）。"""
        m1, s1 = _p(wsrc, usrc, csrc, key)
        m0, s0 = _p(*BASE, key)
        if s1 != s1 or s0 != s0:
            return m1 - m0, float("nan")
        return m1 - m0, math.sqrt(s1 * s1 + s0 * s0)

    def d_rel(wsrc: str, usrc: str, csrc: str, key: str) -> Tuple[float, float]:
        """相对基线的**比例**变化。覆盖数会随网格变大，绝对值不能跨尺度比。"""
        m0, _ = _p(*BASE, key)
        if not m0:
            return float("nan"), float("nan")
        v, se = d(wsrc, usrc, csrc, key)
        return v / m0, se / abs(m0)

    e = {
        # T-Drive 与 GeoLife 分开列：两者都是真实轨迹但结论可能不同——
        # small 上 T-Drive 抬高 PSS 而 GeoLife 压低，到 medium 两者都压低。
        # 只报其中一份会漏掉这个反转。
        "PSS·仅换T-Drive": d("tdrive", "synthetic", "synthetic", "pss_mean_a"),
        "PSS·仅换GeoLife": d("geolife", "synthetic", "synthetic", "pss_mean_a"),
        "PSS·仅换用地": d("synthetic", "osm", "synthetic", "pss_mean_a"),
        "PSS·仅换犯罪": d("synthetic", "synthetic", "zhang2025", "pss_mean_a"),
        "PSS·两者同换": d("geolife", "osm", "synthetic", "pss_mean_a"),
        "PSS·三者同换": d("geolife", "osm", "zhang2025", "pss_mean_a"),
        "覆盖·仅换T-Drive(相对)": d_rel("tdrive", "synthetic", "synthetic", "cov_a"),
        "覆盖·仅换GeoLife(相对)": d_rel("geolife", "synthetic", "synthetic", "cov_a"),
        "覆盖·仅换用地(相对)": d_rel("synthetic", "osm", "synthetic", "cov_a"),
        "覆盖·仅换犯罪(相对)": d_rel("synthetic", "synthetic", "zhang2025", "cov_a"),
        "覆盖·两者同换(相对)": d_rel("geolife", "osm", "synthetic", "cov_a"),
        "覆盖·三者同换(相对)": d_rel("geolife", "osm", "zhang2025", "cov_a"),
        "公平差距·仅换轨迹": d("geolife", "synthetic", "synthetic", "gap_a"),
        "公平差距·仅换用地": d("synthetic", "osm", "synthetic", "gap_a"),
        "公平差距·仅换犯罪": d("synthetic", "synthetic", "zhang2025", "gap_a"),
        "公平差距·两者同换": d("geolife", "osm", "synthetic", "gap_a"),
        "公平差距·三者同换": d("geolife", "osm", "zhang2025", "gap_a"),
        # 满意度尺度：水平下移与离散变宽（这是 θ 不可比的机制）
        "满意度水平下移": (
            _p(*BASE, "sat_best_med")[0] - _p("synthetic", "osm", "synthetic", "sat_best_med")[0],
            math.sqrt(_p(*BASE, "sat_best_med")[1] ** 2
                      + _p("synthetic", "osm", "synthetic", "sat_best_med")[1] ** 2)),
    }

    def pref_delta(key: str) -> Tuple[float, float]:
        """同一真实输入下，显示性偏好相对手写偏好的变化。"""
        m_r, s_r = _mean_se(rows, "tdrive", "osm", "zhang2025", key, "revealed")
        m_a, s_a = _mean_se(rows, "tdrive", "osm", "zhang2025", key, "archetype")
        if s_r != s_r or s_a != s_a:
            return m_r - m_a, float("nan")
        return m_r - m_a, math.sqrt(s_r * s_r + s_a * s_a)

    e["PSS·显示性偏好−手写偏好"] = pref_delta("pss_mean_a")
    e["覆盖·显示性偏好−手写偏好"] = pref_delta("cov_a")
    e["公平差距·显示性偏好−手写偏好"] = pref_delta("gap_a")

    w_osm = (_p("synthetic", "osm", "synthetic", "sat_p95")[0]
             - _p("synthetic", "osm", "synthetic", "sat_p05")[0])
    w_base = (_p(*BASE, "sat_p95")[0] - _p(*BASE, "sat_p05")[0])
    se_w = math.sqrt((_p("synthetic", "osm", "synthetic", "sat_p95")[1] ** 2
                      + _p("synthetic", "osm", "synthetic", "sat_p05")[1] ** 2)
                     + (_p(*BASE, "sat_p95")[1] ** 2
                        + _p(*BASE, "sat_p05")[1] ** 2))
    e["满意度离散变宽"] = (w_osm - w_base, se_w)

    # 交互项：两者同换 − (仅轨迹 + 仅用地)
    it, se_it = e["PSS·两者同换"]
    a1, se_a1 = e["PSS·仅换GeoLife"]
    a2, se_a2 = e["PSS·仅换用地"]
    e["PSS·交互项"] = (it - a1 - a2, math.sqrt(se_it ** 2 + se_a1 ** 2 + se_a2 ** 2))
    it, se_it = e["公平差距·两者同换"]
    a1, se_a1 = e["公平差距·仅换轨迹"]
    a2, se_a2 = e["公平差距·仅换用地"]
    e["公平差距·交互项"] = (it - a1 - a2, math.sqrt(se_it ** 2 + se_a1 ** 2 + se_a2 ** 2))

    # 三因子交互项：三者同换 − (仅轨迹 + 仅用地 + 仅犯罪)
    it, se_it = e["PSS·三者同换"]
    a1, se_a1 = e["PSS·仅换GeoLife"]
    a2, se_a2 = e["PSS·仅换用地"]
    a3, se_a3 = e["PSS·仅换犯罪"]
    e["PSS·三因子交互"] = (it - a1 - a2 - a3,
                          math.sqrt(se_it ** 2 + se_a1 ** 2 + se_a2 ** 2 + se_a3 ** 2))

    # 合法池在最大 θ 下的绝对人数（这是最直观的"约束强度"证据）
    t_hi = max(thetas)
    hi_rows = [r for r in rows if abs(r["theta"] - t_hi) < 1e-9]
    e["合法池·全合成@θmax"] = _mean_se(hi_rows, *BASE, "eligible")
    e["合法池·两者均实测@θmax"] = _mean_se(hi_rows, "geolife", "osm", "synthetic", "eligible")
    e["合法池·三者均实测@θmax"] = _mean_se(hi_rows, "geolife", "osm", "zhang2025", "eligible")
    return e


#: 显著性门槛：效应必须超过 2 倍标准误才算"与 0 可区分"。
_SIG_Z = 2.0


def verdict(a: Tuple[float, float], b: Tuple[float, float], tol: float = 0.5) -> str:
    """判定一条结论是否跨尺度成立。``a``/``b`` 是 ``(效应值, 标准误)``。

    先做**显著性**过滤，再看方向：

    * 任一尺度上效应与 0 不可区分 → 「不显著」（该尺度没有可比信号）；
    * 两个尺度都显著且异号 → 「反转」（至少一次是伪影）；
    * 同号 → 按相对差是否 ≤ tol 判「复现」或「仅方向复现」。

    先做显著性再比方向，是为了避免把"两边都在噪声里抖"误判成反转。
    门槛用的是数据自身的离散度（跨种子/θ 的标准误），不是人为常数。
    """
    va, sa = a
    vb, sb = b
    if va != va or vb != vb:
        return "缺数据"
    sig_a = sa == sa and abs(va) > _SIG_Z * sa
    sig_b = sb == sb and abs(vb) > _SIG_Z * sb
    if not (sig_a and sig_b):
        return "不显著（噪声级）"
    if va * vb < 0:
        return "**反转**"
    scale = max(abs(va), abs(vb), 1e-9)
    if abs(va - vb) / scale <= tol:
        return "复现"
    return "仅方向复现"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True,
                    help="逗号分隔的 cross_trial_*.json，按尺度从小到大")
    ap.add_argument("--out", default="results/cross_scale_check.md")
    args = ap.parse_args()

    blobs = []
    for p in args.files.split(","):
        with open(p, encoding="utf-8") as fh:
            blobs.append((p, json.load(fh)))
    small_name = "%s/%s" % (blobs[0][1]["scale"], blobs[0][1]["n_workers"])
    large_name = "%s/%s" % (blobs[-1][1]["scale"], blobs[-1][1]["n_workers"])

    es = [(b["scale"], effects(b["rows"], b["thetas"])) for _, b in blobs]

    lines = ["# 跨尺度复核：哪些结论是伪影", "",
             f"*`experiments/compare_cross_scales.py` 生成；对比 "
             f"**{small_name}** 与 **{large_name}**。*", "",
             "所有数值都是相对「全合成」基线的差（覆盖类为相对比例），故可跨尺度比较。",
             "括号内是 **z 值**（效应 / 标准误）；|z| ≤ 2 表示该效应与 0 无法区分。", "",
             "判定顺序：先看显著性 → 再看方向 → 再看量级。",
             "* 任一尺度不显著 → **不显著（噪声级）**；",
             "* 两尺度都显著且异号 → **反转**（至少一次是伪影）；",
             "* 同号且相对差 ≤ 50% → **复现**；同号但差更多 → **仅方向复现**。", "",
             "| 效应量 | %s | %s | 判定 |" % (small_name, large_name),
             "|---|---|---|---|"]

    def cell(vs: Tuple[float, float]) -> str:
        v, se = vs
        if v != v:
            return "—"
        z = v / se if se and se == se else float("nan")
        zs = "—" if z != z else "%+.1f" % z
        f = "%.4f" if abs(v) < 10 else "%.1f"
        return "%s (z=%s)" % (f % v, zs)

    counts: Dict[str, int] = {"复现": 0, "仅方向复现": 0, "**反转**": 0,
                              "不显著（噪声级）": 0, "缺数据": 0}
    for key in es[0][1]:
        v = verdict(es[0][1][key], es[-1][1][key])
        counts[v] = counts.get(v, 0) + 1
        lines.append("| %s | %s | %s | %s |"
                     % (key, cell(es[0][1][key]), cell(es[-1][1][key]), v))

    lines += ["", "## 汇总", ""]
    for k in ("复现", "仅方向复现", "**反转**", "不显著（噪声级）", "缺数据"):
        if counts.get(k):
            lines.append(f"* {k.replace('**', '')}：**{counts[k]}** 项")
    lines.append("")

    # 反转项单独点出来——这些是绝对不能写进论文的
    bad = [k for k in es[0][1] if verdict(es[0][1][k], es[-1][1][k]) == "**反转**"]
    if bad:
        lines += ["## 反转项（必须处理，不能写进论文）", ""]
        for k in bad:
            lines.append(f"* **{k}**：{es[0][0]} 为 {es[0][1][k][0]:+.4f}"
                         f"（z={es[0][1][k][0] / es[0][1][k][1]:+.1f}），"
                         f"{es[-1][0]} 为 {es[-1][1][k][0]:+.4f}"
                         f"（z={es[-1][1][k][0] / es[-1][1][k][1]:+.1f}）—— "
                         f"方向相反且两边都显著，至少有一次是伪影。")
        lines.append("")
    else:
        lines += ["## 反转项", "", "无。所有效应方向在两个尺度上一致。", ""]

    # 未显著项：这些结论在两个尺度上都拿不到信号，写进论文会被审稿人打
    weak = [k for k in es[0][1]
            if verdict(es[0][1][k], es[-1][1][k]) == "不显著（噪声级）"]
    if weak:
        lines += ["## 未显著项（两个尺度上都与 0 无法区分）", "",
                  "下列结论**不要写进论文**，或必须先扩到更大尺度/更多种子：", ""]
        for k in weak:
            a, b = es[0][1][k], es[-1][1][k]
            za = a[0] / a[1] if a[1] else float("nan")
            zb = b[0] / b[1] if b[1] else float("nan")
            sa = "%+.1f" % za if za == za else "—（SE=0）"
            sb = "%+.1f" % zb if zb == zb else "—（SE=0）"
            lines.append(f"* **{k}**：z = {sa} / {sb}")
        lines.append("")

    # 稳健项：可以直接进论文的
    solid = [k for k in es[0][1] if verdict(es[0][1][k], es[-1][1][k]) == "复现"]
    if solid:
        lines += ["## 可复现项（两个尺度上方向与量级都一致，可进论文）", ""]
        for k in solid:
            a, b = es[0][1][k], es[-1][1][k]
            lines.append(f"* **{k}**：{a[0]:+.4f} → {b[0]:+.4f}"
                         f"（相对变化 {abs(a[0] - b[0]) / max(abs(a[0]), abs(b[0])):.0%}）")
        lines.append("")

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("\n写出 %s" % out)


if __name__ == "__main__":
    main()
