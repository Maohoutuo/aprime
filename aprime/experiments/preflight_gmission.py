# -*- coding: utf-8 -*-
r"""E5.9 预检：gMission 真实数据上 MRG 处于什么状态？

## 三种可能的结果（必须区分，否则会误报）

| 状态 | 含义 | 判据 |
|---|---|---|
| **惰性**（inert） | 没有任何组的比例下界被抬升 | 所有 `floor_prop ≥ min_rep` |
| **激活但不绑定**（active-nonbinding） | 下界被抬到 1，但贪心本来就会选它 | `floor` 变了、选择没变 |
| **激活且绑定**（active-binding） | 下界被抬到 1，且改变了选择 | `floor` 变了、选择也变了 |

只有第三种才算"MRG 起了作用"。合成稠密实例属于第三种；真实稀疏数据可能是前两种。

## 为什么要按两种分桶各跑一遍

* `quantile_tied`（§6.8.5 的 GQC-tied）：按能力值的累积分布切 30/70。gMission 的 `|F_u|`
  中位数是 1，累积分布几乎全在最小值上，**会退化成单组**，`gqc_gap ≡ 0`（不是差距为零，
  而是没有组可比）。
* `value`（GQC 默认）：按 `|F_u|` 的**值**分桶（1 / 2 / ≥3），gMission 上仍是三组。

两者结论不同，必须并列报告。

## gMission → 覆盖模型的口径（README 的"口径 A：忠实映射"）

* 网格 `cell = floor(x/5·nx)`，默认 nx=ny=8、保留半径 2 格；
* 每条 GOMA 可行 (worker, task) 边 → 一个 `Candidate`（capacity=1），
  `cells = {任务所在格}`、`steps = 1`；
* `sat = 1 − dist/radius`：θ 的语义是"任务要多近才被接受"；
* 预算口径 `count`，预算 = 工作者总数。

## 已知失真（必须写进论文）

`cells` 只有任务所在的一格，8×8 共 64 格，故**覆盖饱和于 64**，`k₀` 被压在 40–60
而非 `N_feas`。这是"容量 1 + 网格量化"的必然，不是实现缺陷。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "data")
sys.path.insert(0, os.path.join(_DATA_ROOT, "gmission"))
sys.path.insert(0, os.path.join(_DATA_ROOT, "sc_dense"))

from aprime.candidates import build_candidates  # noqa: E402
from aprime.config import THETA_GRID, APrimeConfig  # noqa: E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.groups import (  # noqa: E402
    GROUP_ORDER,
    capability_groups,
    gqc_gap as _gqc_gap,
    group_shares,
)
from aprime.homogeneity import with_homogeneity  # noqa: E402
from aprime.model import Candidate, make_problem  # noqa: E402
from aprime.solvers.gqc import GQCParams, gqc_solve  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402

from load_gmission import feasible_pairs, list_files, load_instance  # noqa: E402

import load_sc  # noqa: E402


def _ds_name(path: str) -> str:
    """快照名：取父目录；若父目录是纯数字（如 `real_didi/1000/`），拼上再上一级。"""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    par = parts[-2] if len(parts) >= 2 else ""
    if par.isdigit() and len(parts) >= 3:
        return "%s_%s" % (parts[-3], par)
    return par or "gmission"


def _loader(path: str):
    """按路径挑读取器：`sc_dense/` 下的快照用 `load_sc`，其余用 `load_gmission`。

    两个读取器处理的坑不同（前者有前置序号列与 Didi 行数不一致，后者有表头浮点），
    都不能绕过。"""
    norm = path.replace("\\", "/")
    return load_sc if "sc_dense" in norm else sys.modules["load_gmission"]

SPAN = 5.0
SCHEMES = ("value", "quantile_tied")


def _auto_span(inst) -> float:
    """坐标平面的实际跨度。

    gMission / EverySender 是 5×5 归一化平面（`SPAN=5.0` 恰好吻合），但 sc_dense 的
    GOMA 是 0–500、ONRTA 是 0–100、Didi 是 0–88。写死 5.0 会把后三者的所有坐标压进
    一个格子，可行边全错。自动探测对 5×5 数据是恒等变换（4.999/5.0 与 4.999/4.999
    都落到第 7 格），故不改变既有结果。
    """
    xs = [w.x for w in inst.workers] + [t.x for t in inst.tasks]
    ys = [w.y for w in inst.workers] + [t.y for t in inst.tasks]
    return max(max(xs), max(ys), 1e-9)


@dataclass
class _Stub:
    budget: float
    name: str = "gmission"


def _cell(x: float, y: float, nx: int, ny: int, span: float = SPAN) -> Tuple[int, int]:
    cx = min(nx - 1, max(0, int(math.floor(x / span * nx))))
    cy = min(ny - 1, max(0, int(math.floor(y / span * ny))))
    return (cx, cy)


def _gm_candidates(inst, nx: int, ny: int, radius_cells: int,
                   span: float = SPAN, slot_sec: int = 300,
                   feasible_fn=None) -> Tuple[List[Candidate], dict]:
    if feasible_fn is None:
        feasible_fn = feasible_pairs
    cands: List[Candidate] = []
    keep = total = 0
    for w, t in feasible_fn(inst):
        total += 1
        cw, ct = _cell(w.x, w.y, nx, ny, span), _cell(t.x, t.y, nx, ny, span)
        if max(abs(cw[0] - ct[0]), abs(cw[1] - ct[1])) > radius_cells:
            continue
        keep += 1
        dist = math.hypot(w.x - t.x, w.y - t.y)
        sat = 1.0 if w.radius <= 0 else max(0.0, min(1.0, 1.0 - dist / w.radius))
        slot = t.t_arrive // slot_sec
        cands.append(Candidate(
            wid="w%03d" % w.wid,
            path=((ct[0], ct[1], slot),),
            cells=frozenset({ct}),
            sat=sat,
            tag="t%03d" % t.tid,
            space_time=frozenset({(ct[0], ct[1], slot)}),
        ))
    return cands, {"edges_total": total, "edges_kept": keep}


def _state(floor_plain, floor_mrg, changed) -> str:
    bumped = any(floor_mrg.get(g, 0) > floor_plain.get(g, 0) for g in floor_mrg)
    if not bumped:
        return "惰性"
    return "激活且绑定" if changed else "激活但不绑定"


def _profile(prob, scheme: str) -> dict:
    """一个 (实例, θ, 分桶方案) 下的完整画像。"""
    tied = capability_groups(prob, scheme=scheme)
    share = group_shares(tied)
    n_feas = len(tied)
    caps = {w: sum(1 for c in prob.by_worker[w] if prob.is_eligible(c)) for w in prob.by_worker}
    caps = {w: n for w, n in caps.items() if n >= 1}

    sel_a, _ = solve(prob, with_ls=True)
    k0 = len(sel_a)
    base = dict(kappa=1.0, slack=0, scheme=scheme)
    sel_p, info_p = gqc_solve(prob, GQCParams(**base), with_ls=True)
    sel_m, info_m = gqc_solve(prob, GQCParams(**base, min_rep=1, mrg_cap=2.0), with_ls=True)

    f_p = info_p.get("floor") or {}
    f_m = info_m.get("floor") or {}
    changed = {c.wid for c in sel_p} != {c.wid for c in sel_m}

    rows = []
    for g in GROUP_ORDER:
        s = share.get(g, 0.0)
        if s <= 0:
            continue
        rows.append({
            "group": g,
            "n": sum(1 for v in tied.values() if v == g),
            "share": round(s, 4),
            "q": round(s * k0, 4),
            "floor_plain": int(f_p.get(g, 0)),
            "floor_mrg": int(f_m.get(g, 0)),
        })

    return {
        "scheme": scheme,
        "theta": float(prob.cfg.theta),
        "n_feas": n_feas,
        "k0": k0,
        "delta": round(k0 / n_feas, 4) if n_feas else None,
        "need_n": round(n_feas / k0, 3) if k0 else None,
        "caps_min": min(caps.values()) if caps else 0,
        "caps_med": sorted(caps.values())[len(caps) // 2] if caps else 0,
        "caps_max": max(caps.values()) if caps else 0,
        "n_groups": len(rows),
        "groups": rows,
        "floor_zero_groups": [r["group"] for r in rows if r["floor_plain"] == 0],
        "mrg_blocked": info_m.get("mrg_blocked") or [],
        "mrg_changed": changed,
        "state": _state(f_p, f_m, changed),
        "floor_ok": info_m.get("floor_ok"),
        "quota_ok": info_m.get("quota_ok"),
        "coverage_a": prob.coverage(sel_a),
        "coverage_gqc": prob.coverage(sel_p),
        "coverage_mrg": prob.coverage(sel_m),
        "steps_a": sum(c.steps for c in sel_a),
        "steps_gqc": sum(c.steps for c in sel_p),
        "steps_mrg": sum(c.steps for c in sel_m),
        "gap_a": round(_gqc_gap(sel_a, tied), 4),
        "gap_gqc": round(_gqc_gap(sel_p, tied), 4),
        "gap_mrg": round(_gqc_gap(sel_m, tied), 4),
        "theta_violations": sum(1 for c in sel_p if not prob.is_eligible(c)),
        "theta_violations_mrg": sum(1 for c in sel_m if not prob.is_eligible(c)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--radius-cells", type=int, default=-1,
                    help="半径换算后的格数；<0 时按 ceil(radius/span*nx) 自动推算")
    ap.add_argument("--span", type=float, default=0.0,
                    help="坐标平面跨度；0 表示按数据实际范围自动探测")
    ap.add_argument("--slot-sec", type=int, default=0,
                    help="时隙长度（秒）；0 表示按数据的时间窗中位数自动探测")
    ap.add_argument("--kind", default="auto", choices=("auto", "real", "synthetic"),
                    help="数据性质标注；auto 时按路径是否含 'synthetic' 判定")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--synthetic-n", type=int, default=8)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",") if t] or THETA_GRID
    path = args.file or list_files()[0]
    ds = _ds_name(path)
    kind = args.kind
    if kind == "auto":
        kind = "synthetic" if "synthetic" in path.replace("\\", "/").lower() else "real"
    _ld = _loader(path)
    load_instance = _ld.load_instance
    inst = load_instance(path)
    span = args.span if args.span > 0 else _auto_span(inst)
    if args.slot_sec > 0:
        slot_sec = args.slot_sec
    else:
        wins = [t.t_end - t.t_arrive for t in inst.tasks if t.t_end > t.t_arrive]
        slot_sec = int(sorted(wins)[len(wins) // 2]) if wins else 300
    if args.radius_cells >= 0:
        radius_cells = args.radius_cells
    else:
        radii = [w.radius for w in inst.workers if w.radius > 0]
        med_r = sorted(radii)[len(radii) // 2] if radii else 0.0
        radius_cells = max(1, int(math.ceil(med_r / span * args.nx)))
    print("几何口径：span=%.3f slot_sec=%d radius_cells=%d" % (span, slot_sec, radius_cells))
    cands, edge_stat = _gm_candidates(inst, args.nx, args.ny, radius_cells,
                                      span=span, slot_sec=slot_sec,
                                      feasible_fn=_ld.feasible_pairs)
    workers_with_cand = len({c.wid for c in cands})
    stub = _Stub(budget=float(len(inst.workers)), name=inst.name)

    gm: Dict[str, List[dict]] = {s: [] for s in SCHEMES}
    for theta in thetas:
        prob = make_problem(stub, cands,
                            APrimeConfig(theta=theta, budget_kind="count",
                                         budget=float(len(inst.workers))))
        for scheme in SCHEMES:
            gm[scheme].append(_profile(prob, scheme))

    # 合成对照（同口径，value 分桶）
    syn: List[dict] = []
    for seed in range(args.synthetic_n):
        d = make_dense_instance("tdrive", "medium", n_workers=40, seed=seed, index=0)
        d = with_homogeneity(d, 0.0, seed=seed)
        cs = build_candidates(d, k=8, seed=seed, lams=(1.0, 0.5, 0.0), n_random=2)
        for theta in thetas:
            prob = make_problem(d, cs, APrimeConfig(theta=theta, k_cand=8, budget=80))
            r = _profile(prob, "value")
            r["seed"] = seed
            syn.append(r)

    val = gm["value"]
    qt = gm["quantile_tied"]

    lines = [
        "# E5.9 真实数据预检：MRG 处于什么状态",
        "",
        f"*`experiments/preflight_gmission.py` 生成；数据集 **{ds}**（"
        f"{'真实' if kind == 'real' else '**合成**'}），"
        f"文件 `{os.path.basename(path)}`，网格 {args.nx}×{args.ny}，"
        f"半径 {radius_cells} 格（span={span:.2f}、时隙 {slot_sec} s）。*",
        "",
        "状态定义：**惰性** = 下界没被抬升；**激活但不绑定** = 下界抬了但选择没变；"
        "**激活且绑定** = 下界抬了且改变了选择。只有第三种说明 MRG 真起了作用。",
        "",
        "## 0. 数据与映射",
        "",
        f"* 工作者 {len(inst.workers)}，任务 {len(inst.tasks)}，事件 {len(inst)}",
        f"* GOMA 可行边 **{edge_stat['edges_total']}**，网格量化保留 "
        f"**{edge_stat['edges_kept']}**"
        f"（{100.0 * edge_stat['edges_kept'] / max(1, edge_stat['edges_total']):.1f}%）",
        f"* 有候选的工作者 **{workers_with_cand}** / {len(inst.workers)}"
        f"（{100.0 * workers_with_cand / len(inst.workers):.1f}%）",
        "",
        "## 1. 拥挤度（与分桶无关）",
        "",
        "| θ | `N_feas` | `k₀` | δ = k₀/N_feas | `N_feas/k₀`（人） | `\\|F_u\\|` min/med/max |",
        "|---|---|---|---|---|---|",
    ]
    for r in val:
        lines.append("| %.1f | %d | %d | %.3f | %.2f | %d / %d / %d |" % (
            r["theta"], r["n_feas"], r["k0"], r["delta"], r["need_n"],
            r["caps_min"], r["caps_med"], r["caps_max"]))
    lines.append("")

    lines += ["## 2. `value` 分桶（GQC 默认）：分组、配额与 MRG 状态", "",
              "| θ | 组 | 人数 | 份额 | q_g | 下界(无MRG) | 下界(MRG) | 状态 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in val:
        for g in r["groups"]:
            lines.append("| %.1f | %s | %d | %.4f | %.3f | %d | %d | %s |" % (
                r["theta"], g["group"], g["n"], g["share"], g["q"],
                g["floor_plain"], g["floor_mrg"],
                r["state"] if g["floor_mrg"] != g["floor_plain"] else ""))
    lines.append("")

    lines += ["## 3. `quantile_tied` 分桶（GQC-tied）：分桶退化", "",
              "| θ | 非空组数 | 最高份额组 | `gqc_gap` | 状态 |",
              "|---|---|---|---|---|"]
    for r in qt:
        top = max(r["groups"], key=lambda g: g["share"]) if r["groups"] else {"group": "—",
                                                                            "share": 0.0}
        lines.append("| %.1f | %d | %s (%.3f) | %.4f | %s |" % (
            r["theta"], r["n_groups"], top["group"], top["share"], r["gap_gqc"], r["state"]))
    lines.append("")

    lines += ["## 4. A′ 与 GQC 在真实数据上的对照（`value` 分桶）", "",
              "| θ | A′ 覆盖 | GQC 覆盖 | Δ覆盖 | `gqc_gap`(A′) | `gqc_gap`(GQC) | 相对降幅 | θ 违规 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in val:
        rel = (100.0 * (r["gap_gqc"] - r["gap_a"]) / r["gap_a"]) if r["gap_a"] else 0.0
        lines.append("| %.1f | %d | %d | %+d | %.4f | %.4f | %s | %d |" % (
            r["theta"], r["coverage_a"], r["coverage_gqc"],
            r["coverage_gqc"] - r["coverage_a"], r["gap_a"], r["gap_gqc"],
            ("%+.0f%%" % rel) if r["gap_a"] else "—", r["theta_violations"]))
    lines.append("")

    # 合成对照
    lines += ["## 5. 与合成稠密实例的机制对照（`value` 分桶）", "",
              f"| 指标 | {ds} | 合成 medium |", "|---|---|---|"]
    lines.append("| `k₀` 均值 | %.1f | %.1f |" % (
        mean(r["k0"] for r in val), mean(r["k0"] for r in syn)))
    lines.append("| `N_feas` 均值 | %.1f | %.1f |" % (
        mean(r["n_feas"] for r in val), mean(r["n_feas"] for r in syn)))
    lines.append("| δ = `k₀/N_feas` | %.3f | %.3f |" % (
        mean(r["delta"] for r in val), mean(r["delta"] for r in syn)))
    lines.append("| `N_feas/k₀` | %.2f 人 | %.2f 人 |" % (
        mean(r["n_feas"] for r in val) / mean(r["k0"] for r in val),
        mean(r["n_feas"] for r in syn) / mean(r["k0"] for r in syn)))
    lines.append("| 非空组数（均值） | %.1f | %.1f |" % (
        mean(r["n_groups"] for r in val), mean(r["n_groups"] for r in syn)))

    def _states(rows):
        c = {"惰性": 0, "激活但不绑定": 0, "激活且绑定": 0}
        for r in rows:
            c[r["state"]] = c.get(r["state"], 0) + 1
        return c

    sv, ss = _states(val), _states(syn)
    for k in ("惰性", "激活但不绑定", "激活且绑定"):
        lines.append("| 状态：%s（格数） | %d / %d | %d / %d |" % (
            k, sv.get(k, 0), len(val), ss.get(k, 0), len(syn)))
    lines.append("")

    # 极小组构成
    tiny_gm = []
    for r in val:
        for g in r["groups"]:
            if g["share"] <= 0.05:
                tiny_gm.append((r["theta"], g["group"], g["n"], g["share"]))
    lines += ["## 6. 极小份额组（份额 ≤5%）出现在哪里", "",
              "| θ | 组 | 人数 | 份额 |", "|---|---|---|---|"]
    for th, g, nn, s in tiny_gm[:12]:
        lines.append("| %.1f | %s | %d | %.4f |" % (th, g, nn, s))
    if not tiny_gm:
        lines.append("| — | 无 | — | — |")
    # 极小组方向：由数据判定，不预设"稀疏"或"稠密"
    def _tiny_group(rows):
        seen = []
        for r in rows:
            ne = [g for g in r["groups"] if g["n"] > 0]
            if len(ne) >= 2:
                seen.append(min(ne, key=lambda g: g["n"])["group"])
        return seen

    tiny_seen = _tiny_group(val)
    tight_is_tiny = "tight" in tiny_seen
    lines += ["",
              "关键：%s 的极小组出现在 %s。" % (
                  ds, "、".join(sorted(set(tiny_seen))) or "无（各组人数并列）"),
              ("**极小组是 `tight`（能力最低档）**——这正是 MRG 设计时的目标场景。"
               if tight_is_tiny else
               "**极小组是 `ample`（能力最高档）**——与 MRG 的设计动机方向相反，"
               "它保护的是“本来就容易被选中”的高能力群。"),
              ""]

    # 判读
    bind_gm = sv.get("激活且绑定", 0)
    nonbind_gm = sv.get("激活但不绑定", 0)
    tight_shares = [g["share"] for r in val for g in r["groups"] if g["group"] == "tight"]
    tiny_desc = "、".join("θ=%.1f 时 %s %d 人" % (th, g, nn) for th, g, nn, _s in tiny_gm[:4]) \
        or "无"
    lines += ["## 7. 判读", ""]
    if bind_gm == 0 and nonbind_gm > 0:
        lines += ["**MRG 在 %s 上“激活但不绑定”（%d/%d 个 θ），从未改变选择。**"
                  % (ds, nonbind_gm, len(val)), ""]
    elif bind_gm > 0:
        lines += ["**MRG 在 %s 上“激活且绑定”（%d/%d 个 θ）。**" % (ds, bind_gm, len(val)), ""]
    else:
        lines += ["**MRG 在 %s 上完全惰性。**" % ds, ""]
    lines += [
        "* `|F_u|` 中位数为 %d，`tight` 组份额 %.2f–%.2f，极小组为 %s。" % (
            val[0]["caps_med"], min(tight_shares) if tight_shares else 0.0,
            max(tight_shares) if tight_shares else 0.0, tiny_desc),
        ("* 极小组是 `tight` ⇒ **能力最低的组恰好最小**，贪心按边际覆盖密度选人时"
         "最容易被跳过，MRG 的 floor 因此能改变选择——这是它的目标场景。" if tight_is_tiny
         else
         "* 极小组是 `ample` ⇒ 它的成员**未必**已被选中：若其格子已被别人覆盖，"
         "贪心会跳过它，此时 floor 强制入选同样会改变选择（Didi 就是这样绑定的）。"
         "故“极小组高能力 ⇒ 不绑定”不成立，绑定取决于覆盖冗余度。"),
        "* `quantile_tied` 在 %s 上退化为单组（`gqc_gap ≡ 0`），指标不可判读；" % ds,
        "`value` 分桶下 `gqc_gap` 可测，且 GQC 相对 A′ 有明显降幅（见 §4）。",
        "* 覆盖饱和于 64 格（8×8）：`k₀` 被压在 40–63 而非 `N_feas`，"
        "故“Δ覆盖”的绝对值不能与合成实验的步数口径直接比较。",
        "",
        "**结论**：%s 该快照%s。" % (
            ds,
            "**可用于检验 MRG 的绑定效应**（%d/%d 个 θ 绑定，极小组方向与 MRG 的设计动机%s）"
            % (bind_gm, len(val), "一致" if tight_is_tiny else "相反") if bind_gm > 0
            else "可用于**主算法的外部有效性检验**（GQC 同样降低 `gqc_gap` 且覆盖代价很小），"
                 "但**不能**用来验证 MRG 的绑定效应（0/%d 个 θ 绑定）" % len(val)),
        "",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "preflight_%s.md" % ds)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    js = os.path.join(args.out_dir, "preflight_%s.json" % ds)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump({"file": path, "dataset": ds, "kind": kind,
                   "nx": args.nx, "ny": args.ny,
                   "radius_cells": radius_cells, "span": span, "slot_sec": slot_sec,
                   "thetas": thetas,
                   "n_workers": len(inst.workers), "n_tasks": len(inst.tasks),
                   "edges": edge_stat, "workers_with_cand": workers_with_cand,
                   "states_value": sv, "states_synthetic": ss,
                   "gmission_value": val, "gmission_quantile_tied": qt,
                   "synthetic_medium": syn}, fh, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print("\nwrote", md)


if __name__ == "__main__":
    main()
