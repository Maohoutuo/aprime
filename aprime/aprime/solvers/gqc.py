# -*- coding: utf-8 -*-
"""分组配额一致（GQC）选择规则——创新点 3 的算法部分。

目标：**名额 ∝ 合法池内人数份额**（一人一票），替代东华 CFI 型的
"名额 ∝ 能力份额"，从而停止对"刚好达标"能力组的结构性挤出。

核心机制（带可行性保证的配额贪心）：

1. 先按合法路线条数把合法工人分为三组：`tight`(=1), `two`(=2), `ample`(≥3)。
2. 目标名额 `floor_g = ⌊κ·share_g·k0⌋`，`share_g` = 组 g 在合法池内的人数份额，
   `k0` = 无约束贪心的自然规模（已验证可行），κ ∈ (0,1] 控制配额松紧。
3. 在线贪心时，每条待选路线属于一个组；选择它需满足：
   * 该组名额不超 `ceil_g = k0 − Σ_{g'≠g} floor_{g'}`（即"其余各组下界仍可满足"推出的上界）；
   * 其余各组的缺口**求和**后不超过剩余名额（Hall 型 catch-up 检验）。
   满足两者者中，选边际覆盖（或密度）最大者。
4. （可选）局部搜索：只接受覆盖严格提升、且新旧元素属于同一组（保持名额上下界）的移动，因此
   不会破坏配额一致性。

两个公平观点不可兼容的实证体现：GQC 把 `gqc_gap` 压到最小（≈0–1/6 粒度），
代价是 `cfi_gap` 变大——论文正文要写明白，否则会被误读成"GQC 公平性更差"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .. import groups as G
from ..model import Candidate, SelectionProblem
from .greedy import greedy


@dataclass
class GQCParams:
    kappa: float = 1.0          # 配额紧度：1.0 = 严格按人数份额分配
    slack: int = 0              # 下界松弛量（>0 时允许配额贪心让位于覆盖，换取更高覆盖）
    target_k: Optional[int] = None  # 目标名额数；None 时由纯贪心规模估计
    scheme: str = "value"       # 能力分桶方案，见 `groups.GROUP_SCHEMES`；
    #   "value"（默认）保持历史行为，"quantile_tied" 为并列一致变体（GQC-tied）
    min_rep: int = 0            # 最小代表兜底（MRG）：>0 时**每个非空组**至少 `min_rep` 个名额。
    #   只对"比例下界取整为 0"的极小组成效（`max(⌊κ·s·k₀⌋, min_rep)`），
    #   因此对大组不改变比例下界。默认 0 = 关闭，保持历史行为。
    mrg_cap: float = 2.0        # MRG 的过度代表上限：倍数 `min_rep/(s_g·k₀)` 超过它就不兜底。
    #   **默认 2.0 恰是 Pareto 分界**：`min_rep/(s·k₀) ≤ 2 ⟺ s·k₀ ≥ min_rep/2`，此区间内
    #   "把 0 抬到 min_rep" 使名额份额偏差**变小**（`|1/k₀ − s| < s`），故兜底不会恶化 `gqc_gap`。
    #   超限的组保持比例结果（0 名额）并记入 `mrg_blocked`，即"显式声明不可表示"而非静默排除。
    #   设 0 可关闭上限（保护所有非空组，按 §6.8.6 至多过度代表 `1/(s·k₀)` 倍）。


def _over_rep(params: GQCParams, s: float, k: int) -> float:
    """兜底把组的名额份额抬到 `min_rep/k` 时的过度代表倍数 `min_rep/(s·k)`。"""
    sk = s * k
    return float("inf") if sk <= 0 else params.min_rep / sk


def _mrg_blocked(params: GQCParams, share: Dict[str, float], k: int) -> List[str]:
    """返回因超过过度代表上限而**未被兜底**的组名列表（与 `q_g` 分界对应）。"""
    if params.min_rep <= 0 or params.mrg_cap <= 0 or k <= 0:
        return []
    out: List[str] = []
    for g, s in share.items():
        if s <= 0:
            continue
        prop = max(0, int(math.floor(params.kappa * s * k)) - params.slack)
        if prop >= params.min_rep:
            continue
        if _over_rep(params, s, k) > params.mrg_cap + 1e-12:
            out.append(g)
    return out


def _floors_for(params: GQCParams, share: Dict[str, float], k: int,
                kappa: float) -> Dict[str, int]:
    """比例下界，叠加最小代表兜底（MRG）。

    MRG 的必要性：`⌊κ·s·k₀⌋` 在 `s·k₀ < 1` 时取整为 0，极小低组（1–3 人）在任何
    可行 `k₀` 下都拿不到名额。兜底把下界抬到 `min_rep`，代价是**该组被过度代表**
    （名额份额 > 人数份额），故 MRG 与 `gqc_gap` 在极小组上直接冲突——这是规范性取舍，
    不是实现缺陷。

    上限 `mrg_cap`（默认 2.0）：`min_rep/(s·k₀)` 超过它的组不被兜底，保持比例结果
    并记入 `mrg_blocked`。2.0 恰是 Pareto 分界，故默认配置下兜底不会抬高 `gqc_gap`。

    可行性：若抬升后 `Σ floor > k`（组数多于名额数），**放弃兜底**退回纯比例，
    因为此时"每组至少 1 个"本身不可行。
    """
    floors = {g: max(0, int(math.floor(kappa * s * k)) - params.slack)
              for g, s in share.items()}
    if params.min_rep <= 0 or not floors:
        return floors
    bumped = {}
    for g, f in floors.items():
        if f >= params.min_rep:
            bumped[g] = f
            continue
        if params.mrg_cap > 0 and _over_rep(params, share[g], k) > params.mrg_cap + 1e-12:
            bumped[g] = f                  # 超限：不兜底，保持比例下界
        else:
            bumped[g] = params.min_rep
    if sum(bumped.values()) <= k:
        return bumped
    return floors


def _estimate_k(problem: SelectionProblem, params: GQCParams) -> int:
    if params.target_k is not None:
        return max(1, params.target_k)
    sel, _ = greedy(problem)
    return max(1, len(sel))


def _floor_ceil(params: GQCParams, share: Dict[str, float], k: int):
    """下界 `floor_g = max(0, ⌊κ·share_g·k⌋ − slack)`，上界由"其余各组下界仍可满足"推出。

    **上界不能简单取 `floor+bandwidth`**：Day-6 测试抓到过反例（3 组各 floor=2、k=6 时
    `floor+1=3` 会允许某组占 3 个，剩下 3 个名额塞不下另两组各 2 个 floor，算法被迫违约）。
    正确写法是把上界定义为 `k − Σ_{g'≠g} floor_{g'}`，这样上下界**联合可行**：
    只要 `Σ floor ≤ k`（κ ≤ 1 时恒成立），就总存在满足全部 floor 与 ceil 的分配。
    """
    floors = _floors_for(params, share, k, params.kappa)
    total = sum(floors.values())
    ceil_: Dict[str, int] = {}
    for g in floors:
        ceil_[g] = max(floors[g], k - (total - floors[g]))
    return floors, ceil_


def _quota_ok(cnt: Dict[str, int], floor_: Dict[str, int],
              ceil_: Dict[str, int]) -> bool:
    """下界**是保证**，上界只是防集中；两者分开报（`floor_ok` / `ceil_ok`）。"""
    return _floor_ok(cnt, floor_) and _ceil_ok(cnt, ceil_)


def _floor_ok(cnt: Dict[str, int], floor_: Dict[str, int]) -> bool:
    return all(cnt.get(g, 0) >= f for g, f in floor_.items())


def _ceil_ok(cnt: Dict[str, int], ceil_: Dict[str, int]) -> bool:
    return all(cnt.get(g, 0) <= c for g, c in ceil_.items())


def _can_catch_up(cnt: Dict[str, int], floor_: Dict[str, int],
                  chosen_g: str, remaining_steps: int) -> bool:
    """选完 `chosen_g` 后，其余各组是否还有足够名额把 floor 补齐。

    必须用**求和**（Hall 型）条件：每组的名额都从同一池子里出，
    逐组比较 `need ≤ remaining` 会漏掉"两组各需 2 个而只剩 2 个"的情形（Day-6 踩过）。
    """
    need = sum(max(0, f - cnt.get(g, 0)) for g, f in floor_.items() if g != chosen_g)
    return need <= remaining_steps


def _reserve_cost(unused_by_group: Dict[str, List[Tuple[float, str, int]]],
                  floor_: Dict[str, int], cnt_after: Dict[str, int],
                  skip_key: Optional[Tuple[str, int]] = None) -> Optional[float]:
    """按"选完某个候选之后"的组内名额（`cnt_after`），算补齐**所有**组的 floor 的最小总代价。

    关键点：**不能跳过候选自己那一组**的余下欠账。
    Day-7 的第二个坑：只预留"其他组"的欠账时，同一组可以被反复超支，
    直到最后买不起自己的 floor（2/160 的失败正是这样来的）。
    """
    total = 0.0
    for g, f in floor_.items():
        need = f - cnt_after.get(g, 0)
        if need <= 0:
            continue
        pool = unused_by_group.get(g, [])
        picked_w, got, sub = set(), 0, 0.0
        for cst, wid, idx in pool:
            if skip_key is not None and (wid, idx) == skip_key:
                continue
            if wid in picked_w:
                continue
            picked_w.add(wid)
            sub += cst
            got += 1
            if got == need:
                break
        if got < need:
            return None
        total += sub
    return total


#: 自动放宽配额时尝试的 κ 序列（从严格比例往下退）
KAPPA_LADDER = tuple(round(1.0 - 0.05 * i, 2) for i in range(20))


def _effective_floors(problem: SelectionProblem, elems: Sequence[Candidate],
                      group_of: Dict[str, str], share: Dict[str, float],
                      k0: int, params: GQCParams
                      ) -> Tuple[Dict[str, int], Dict[str, int], float]:
    """求出**预算可承受**的最大配额：退回 κ 直到 floor 的总最小代价 ≤ 预算。

    为什么需要：`k0` 是"名额数"（来自贪心），而步数预算是"代价"预算。θ 高、路线变长时，
    12 个名额的最小总代价可能超过 80 步，此时**严格比例 quotas 本身不可行**——
    这不是算法输了，是定义没把预算装进去。Day-7 的 2/160 失败正是这一类。

    返回 `(floor_, ceil_, kappa_eff)`；`kappa_eff` 是最终采用的 κ（≤ params.kappa）。
    """
    kind = problem.cfg.budget_kind
    unused: Dict[str, List[Tuple[float, str, int]]] = {}
    size: Dict[str, int] = {}
    for idx, e in enumerate(elems):
        g = group_of.get(e.wid)
        if g is None:
            continue
        unit = 1.0 if kind == "count" else float(e.steps)
        unused.setdefault(g, []).append((unit, e.wid, idx))
    for g in unused:
        unused[g].sort()
        size[g] = len({wid for _c, wid, _i in unused[g]})

    for step in KAPPA_LADDER:
        kappa = min(params.kappa, step)
        if kappa <= 0:
            break
        floors = _floors_for(params, share, k0, kappa)
        # 名额不可能超过该组合法人数
        floors = {g: min(f, size.get(g, 0)) for g, f in floors.items()}
        need_cost = _reserve_cost(unused, floors, {}, skip_key=None)
        if need_cost is not None and need_cost <= problem.budget + 1e-9:
            total = sum(floors.values())
            ceil_ = {g: max(floors[g], k0 - (total - floors[g])) for g in floors}
            return floors, ceil_, kappa
    # 连最小的 κ 都不行：只保留"防集中"的上界，不承诺下界
    floors = {g: 0 for g in share}
    ceil_ = {g: k0 for g in share}
    return floors, ceil_, 0.0


def gqc_greedy(problem: SelectionProblem, params: Optional[GQCParams] = None,
               rng=None) -> Tuple[List[Candidate], dict]:
    """配额驱动的贪心（配额不可满足时退回到无配额纯增益填充，但**绝不破坏 floor**）。

    `kappa == 0` 退化为纯贪心，不做任何配额约束（用于与 A′ 做“同入口”的
    对照：此时 gqc 与 greedy 应作出完全相同的选择）。
    """
    params = params or GQCParams()
    kind = problem.cfg.budget_kind
    ratio = kind != "count"
    elems = problem.eligible
    if not elems:
        return [], {"solver": "gqc_greedy", "reason": "no eligible"}

    group_of = G.capability_groups(problem, scheme=params.scheme)
    share = G.group_shares(group_of)
    if not share or params.kappa <= 0:
        sel, info = greedy(problem)
        info["solver"] = "gqc_greedy(kappa=0)" if params.kappa <= 0 else "gqc_greedy"
        info["reason"] = "kappa<=0 -> plain greedy" if params.kappa <= 0 else "no groups"
        info["quota_ok"] = True
        info["k0"] = len(sel)
        info["final_counts"] = G.group_counts(sel, group_of) if group_of else {}
        info["gqc_gap"] = round(G.gqc_gap(sel, group_of), 4) if group_of else 0.0
        info["coverage"] = problem.coverage(sel)
        return sel, info

    k0 = _estimate_k(problem, params)
    floor_, ceil_, kappa_eff = _effective_floors(problem, elems, group_of, share, k0, params)
    quota_gap_target = {g: share[g] for g in share}

    used: set = set()
    sel: List[Candidate] = []
    covered: set = set()
    cost = 0.0
    cnt: Dict[str, int] = {g: 0 for g in share}

    def unit_of(c: Candidate) -> float:
        return 1.0 if kind == "count" else float(c.steps)

    while len(sel) < k0:
        remaining_after = k0 - len(sel) - 1
        # 每步重建一次"各组未用候选按代价升序"，供预算 catch-up 检验使用
        unused_by_group: Dict[str, List[Tuple[float, str, int]]] = {}
        for idx, e in enumerate(elems):
            if e.wid in used:
                continue
            g = group_of.get(e.wid)
            if g is None:
                continue
            unused_by_group.setdefault(g, []).append((unit_of(e), e.wid, idx))
        for pool in unused_by_group.values():
            pool.sort()

        def pick(require_gain: bool, enforce_quota: bool):
            """返回候选；`enforce_quota=False` 时只查可行性（兜底用）。"""
            cur_best = None
            cur_key = None
            for idx, e in enumerate(elems):
                if e.wid in used:
                    continue
                g = group_of.get(e.wid)
                if g is None:
                    continue
                if enforce_quota:
                    if cnt[g] + 1 > ceil_[g]:
                        continue
                    if not _can_catch_up(cnt, floor_, g, remaining_after):
                        continue
                unit = unit_of(e)
                if cost + unit > problem.budget + 1e-9:
                    continue
                if enforce_quota:
                    # 预算 catch-up：选完这一条之后，**所有组**（包括 g 的余下欠账）的
                    # floor 都还要买得起（Hall 型预留）。
                    cnt_after = dict(cnt)
                    cnt_after[g] = cnt_after.get(g, 0) + 1
                    need_cost = _reserve_cost(unused_by_group, floor_, cnt_after,
                                              (e.wid, idx))
                    if need_cost is None or cost + unit + need_cost > problem.budget + 1e-9:
                        continue
                gain = len(e.cells - covered)
                if require_gain and gain <= 0:
                    continue
                key = (gain / unit) if (ratio and unit > 0) else float(gain)
                if cur_best is None or key > cur_key:
                    cur_best, cur_key = e, key
            return cur_best

        # ① 优先"有覆盖增益 + 满足配额"；② 配额欠账必须还，允许零增益元素；
        # ③ 都没有才退回无配额贪心（会记进 floor_ok / ceil_ok，供实验如实报告）
        best = pick(require_gain=True, enforce_quota=True)
        if best is None:
            best = pick(require_gain=False, enforce_quota=True)
        if best is None:
            best = pick(require_gain=True, enforce_quota=False)
        if best is None:
            best = pick(require_gain=False, enforce_quota=False)
        if best is None:
            break
        sel.append(best)
        used.add(best.wid)
        covered |= best.cells
        cost += unit_of(best)
        g = group_of.get(best.wid)
        if g is not None:
            cnt[g] = cnt.get(g, 0) + 1

    info = {
        "solver": "gqc_greedy",
        "kappa": params.kappa,
        "slack": params.slack,
        "kappa_eff": round(kappa_eff, 3),
        "k0": k0,
        "quota_share": {g: round(s, 3) for g, s in share.items()},
        "floor": floor_,
        "floor_raw": {g: max(0, int(math.floor(params.kappa * s * k0)) - params.slack)
                      for g, s in share.items()},
        "min_rep": params.min_rep,
        "mrg_cap": params.mrg_cap,
        "mrg_active": any(floor_.get(g, 0) > max(0, int(math.floor(params.kappa * s * k0))
                                               - params.slack)
                          for g, s in share.items()),
        "mrg_blocked": _mrg_blocked(params, share, k0),
        "scheme": params.scheme,
        "final_counts": cnt,
        "gqc_gap": round(G.gqc_gap(sel, group_of), 4),
        "floor_ok": _floor_ok(cnt, floor_),
        "ceil_ok": _ceil_ok(cnt, ceil_),
        "quota_ok": _quota_ok(cnt, floor_, ceil_),
        "coverage": problem.coverage(sel),
    }
    return sel, info


def gqc_solve(problem: SelectionProblem, params: Optional[GQCParams] = None,
              with_ls: bool = True,
              rng=None) -> Tuple[List[Candidate], dict]:
    """GQC 贪心 + 受限局部搜索（不接受破坏名额带宽的移动）。"""
    params = params or GQCParams()
    sel, info = gqc_greedy(problem, params)
    if not sel:
        return sel, info

    from .greedy import local_search as _ls
    group_of = G.capability_groups(problem, scheme=params.scheme)
    share = G.group_shares(group_of)
    k0 = info["k0"]
    # 局部搜索必须用**同一套有效配额**（可能已按预算放宽过 κ），否则会把 floor 判成失守
    floor_ = info.get("floor") or _floor_ceil(params, share, k0)[0]
    ceil_ = _floor_ceil(params, share, k0)[1] if not info.get("floor") else None
    if ceil_ is None:
        total = sum(floor_.values())
        ceil_ = {g: max(floor_[g], k0 - (total - floor_[g])) for g in floor_}

    cur = list(sel)
    cur_cov = problem.coverage(cur)
    cnt = G.group_counts(cur, group_of)
    moves = 0
    kind = problem.cfg.budget_kind
    elems = problem.eligible
    used_w = {c.wid for c in cur}

    for _ in range(200):
        improved = False
        # 1-swap：只接受覆盖严格提升，且新旧均属同一组，配额带宽不变
        for i, old in enumerate(cur):
            rest = [c for j, c in enumerate(cur) if j != i]
            g_old = group_of.get(old.wid)
            for new in elems:
                if new in cur or new.wid in {c.wid for c in rest}:
                    continue
                if group_of.get(new.wid) != g_old:
                    continue        # 换组会改变名额分布，拒绝
                trial = rest + [new]
                if not problem.feasible(trial):
                    continue
                cov = problem.coverage(trial)
                if cov > cur_cov:
                    cur, cur_cov, improved, moves = trial, cov, True, moves + 1
                    used_w = {c.wid for c in cur}
                    break
            if improved:
                break
        if improved:
            continue
        # drop-refill：refill 只能补回被移走元素所属的同一组
        for i in range(len(cur)):
            rest = [c for j, c in enumerate(cur) if j != i]
            g_old = group_of.get(cur[i].wid)
            covered_set = set()
            for c in rest:
                covered_set |= c.cells
            cost_rest = problem.cost(rest)
            cnt_rest = G.group_counts(rest, group_of)
            trial = list(rest)
            while True:
                best_e, best_key = None, None
                for e in elems:
                    if e.wid in used_w - {cur[i].wid} or e.wid in {c.wid for c in trial}:
                        continue
                    if group_of.get(e.wid) != g_old:
                        continue
                    unit = 1.0 if kind == "count" else float(e.steps)
                    if cost_rest + unit > problem.budget + 1e-9:
                        continue
                    gain = len(e.cells - covered_set)
                    if gain <= 0:
                        continue
                    if best_e is None or gain > best_key:
                        best_e, best_key = e, gain
                if best_e is None:
                    break
                trial.append(best_e)
                covered_set |= best_e.cells
                cost_rest += 1.0 if kind == "count" else float(best_e.steps)
            cnt_trial = G.group_counts(trial, group_of)
            cov = problem.coverage(trial)
            if cov > cur_cov and _quota_ok(cnt_trial, floor_, ceil_) and problem.feasible(trial):
                cur, cur_cov, improved, moves = trial, cov, True, moves + 1
                break
        if not improved:
            break

    info.update({
        "solver": "gqc_solve" if with_ls else "gqc_greedy",
        "moves": moves,
        "gqc_gap": round(G.gqc_gap(cur, group_of), 4),
        "floor_ok": _floor_ok(G.group_counts(cur, group_of), floor_),
        "ceil_ok": _ceil_ok(G.group_counts(cur, group_of), ceil_),
        "quota_ok": _quota_ok(G.group_counts(cur, group_of), floor_, ceil_),
        "coverage": problem.coverage(cur),
        "n_selected": len(cur),
        "theta_violations": sum(1 for c in cur if not problem.is_eligible(c)),
    })
    return cur, info



