# -*- coding: utf-8 -*-
"""A′ Day-1 回归测试：模型/精确/贪心/α 估计的最小自洽性。

这些测试不依赖真实轨迹（用合成候选集），因此 1 秒内跑完；真实数据管线单独用
``test_candidates_from_tdrive`` 覆盖（缺数据时自动跳过）。
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.analysis.submodularity import apx_bound, submodularity_ratio  # noqa: E402
from aprime.config import APrimeConfig  # noqa: E402
from aprime.model import Candidate, SelectionProblem  # noqa: E402
from aprime.solvers.baselines import capacity_greedy  # noqa: E402
from aprime.solvers.exact import brute_force, exact  # noqa: E402
from aprime.solvers.greedy import solve  # noqa: E402


def _c(wid: str, cells, sat: float) -> Candidate:
    """合成候选：``cells`` 传 (x, y) 空间格子；时间戳按顺序补，仅供审计。"""
    xy = [(int(c[0]), int(c[1])) for c in cells]
    path = tuple((x, y, i) for i, (x, y) in enumerate(xy))
    return Candidate(wid=wid, path=path, cells=frozenset(xy), sat=sat,
                     space_time=frozenset((x, y, i) for i, (x, y) in enumerate(xy)))


def _fake_instance(budget: float = 100.0):
    return types.SimpleNamespace(budget=budget)


def _problem(cands, theta=0.0, kind="steps", budget=100.0, **kw):
    cfg = APrimeConfig(theta=theta, budget_kind=kind, budget=budget, **kw)
    return SelectionProblem(instance=_fake_instance(budget), candidates=list(cands), cfg=cfg)


# --------------------------------------------------------------------- 基本可行性
def test_one_route_per_worker_and_budget():
    a = _c("u1", [(0, 0, 0), (0, 0, 1)], 0.9)
    b = _c("u1", [(5, 5, 0)], 0.9)
    p = _problem([a, b], budget=10.0)
    assert not p.feasible([a, b])          # 同一人两条 → 不可行
    assert p.feasible([a])
    p2 = _problem([a], budget=1.0)
    assert not p2.feasible([a])            # 步数 2 > 预算 1


def test_theta_is_a_hard_constraint():
    good = _c("u1", [(0, 0, 0)], 0.9)
    bad = _c("u2", [(1, 1, 0)], 0.2)
    p = _problem([good, bad], theta=0.5)
    assert [c.wid for c in p.eligible] == ["u1"]
    assert not p.feasible([bad])           # 低于 θ 的路线直接非法
    sel, _ = exact(p)
    assert [c.wid for c in sel] == ["u1"]


def test_per_worker_theta():
    a = _c("u1", [(0, 0, 0)], 0.4)
    b = _c("u2", [(9, 9, 0)], 0.4)
    p = _problem([a, b], theta=0.0, per_worker_theta={"u1": 0.5})
    assert [c.wid for c in p.eligible] == ["u2"]


# --------------------------------------------------------------------- 求解器
def test_greedy_matches_bruteforce_on_tiny():
    cands = [
        _c("u1", [(0, 0), (0, 0)], 0.8),   # 空间上只覆盖 {(0,0)}（停留不增加覆盖）
        _c("u2", [(0, 0), (0, 1)], 0.7),
        _c("u3", [(7, 7)], 0.6),
    ]
    p = _problem(cands, kind="count", budget=2.0)
    sel_bf, _ = brute_force(p)
    sel_gr, _ = solve(p, with_ls=True)
    assert p.coverage(sel_gr) == p.coverage(sel_bf) == 3
    assert p.feasible(sel_gr)


def test_exact_beats_or_ties_greedy():
    A = _c("u1", [(i, 0, i % 4) for i in range(6)], 0.9)
    B = _c("u2", [(0, 1, 0), (0, 1, 1), (0, 1, 2)], 0.9)
    C = _c("u3", [(1, 1, 0), (1, 1, 1), (1, 1, 2)], 0.9)
    p = _problem([A, B, C], kind="count", budget=2.0)
    sel_ex, _ = exact(p)
    sel_gr, _ = solve(p, with_ls=False)
    sel_bf, _ = brute_force(p)
    assert p.coverage(sel_ex) >= p.coverage(sel_gr)
    assert p.coverage(sel_ex) == p.coverage(sel_bf)


def test_exact_matches_bruteforce_random_instances():
    import random
    rng = random.Random(7)
    for _ in range(5):
        cands = []
        for u in range(4):
            for r in range(2):
                cells = [(rng.randrange(4), rng.randrange(4), rng.randrange(3)) for _ in range(3)]
                cands.append(_c(f"u{u}", cells, round(rng.uniform(0.3, 1.0), 2)))
        p = _problem(cands, kind="count", budget=2.0, theta=0.0)
        sel_ex, _ = exact(p)
        sel_bf, _ = brute_force(p)
        assert p.coverage(sel_ex) == p.coverage(sel_bf)


def test_capacity_baseline_may_violate_theta():
    """CFI 型基线故意忽略 θ：存在低满意度者被选中即算"命中靶子"。"""
    weak = _c("weak", [(0, 0, 0), (0, 0, 1)], 0.1)
    strong = _c("strong", [(5, 5, 0)], 0.95)
    p = _problem([weak, strong], theta=0.5, budget=3.0)
    sel, info = capacity_greedy(p)
    assert info["violates_theta"] >= 1 or any(c.wid == "weak" for c in sel)


# --------------------------------------------------------------------- 定理 2 的结构性测试
def test_theta_thresholds_closed_form():
    """命题 4：两个闭式阈值夹住"个性化的可行区间"。

    * `θ_all = min_{u∈U⁺} max_{r∈R_u} S_u(r)`：θ ≤ θ_all ⇔ **每个**参与者都至少有一条合法路线；
    * `θ_any = max_{u∈U⁺} max_{r∈R_u} S_u(r)`：θ > θ_any ⇔ 可行域为空（问题不可行）。

    早先版本把这两个搞混了（写成 `min_u` 决定空集），Day-5 被这个测试抓到。
    """
    cands = [_c("u1", [(0, 0), (0, 1)], 0.8), _c("u2", [(1, 1), (1, 2)], 0.6),
             _c("u2", [(2, 2), (2, 3)], 0.4)]
    p = _problem(cands, kind="count", budget=2.0)
    th = p.theta_thresholds()
    assert th["theta_all"] == pytest.approx(0.6) and th["theta_any"] == pytest.approx(0.8)

    # θ ≤ θ_all：所有人都保留至少一条
    p = _problem(cands, theta=th["theta_all"] - 1e-9, kind="count", budget=2.0)
    assert len(p.eligible_workers()) == 2

    # θ_all < θ ≤ θ_any：只剩高能力者，但问题仍然可行
    p = _problem(cands, theta=th["theta_all"] + 1e-6, kind="count", budget=2.0)
    assert [w for w in p.by_worker if any(p.is_eligible(c) for c in p.by_worker[w])] == ["u1"]

    # θ > θ_any：彻底不可行
    p = _problem(cands, theta=th["theta_any"] + 1e-6, kind="count", budget=2.0)
    assert not p.eligible


def test_theta_only_shrinks_feasible_set():
    """定理 2 证明第 1 步：`I(θ) ⊆ I(0)`，且 θ 不进目标函数（覆盖与 θ 无关）。"""
    cands = [_c("u1", [(0, 0), (0, 1)], 0.9), _c("u2", [(1, 1), (1, 2)], 0.3)]
    base = _problem(cands, kind="count", budget=2.0)
    strict = _problem(cands, theta=0.5, kind="count", budget=2.0)
    assert {c.wid for c in strict.eligible} < {c.wid for c in base.eligible}
    sel = list(strict.eligible)
    assert base.coverage(sel) == strict.coverage(sel), "覆盖函数不得随 θ 变化"


def test_lemma2_gamma_version_holds_on_stress_case():
    """带 γ 的引理版本在"嵌套覆盖"压力例上成立；同时记录 ρ_oc 被嵌套结构顶到 1。"""
    from aprime.analysis.alpha_exact import exact_submodularity_ratio
    from aprime.analysis.submodularity import alpha_min_bound, overlap_stats

    b, m_routes = 3, 3
    blocks = [[1 + i * b + j for j in range(b)] for i in range(m_routes)]
    z = [1 + m_routes * b + i for i in range(m_routes)]
    cands = []
    for i in range(m_routes):
        cands.append(_c(f"r{i}", [(0, 0)] + [(c, 0) for c in blocks[i]] + [(z[i], 0)], 0.9))
        cands.append(_c(f"h{i}", [(c, 0) for c in blocks[i]] + [(z[i], 0)], 0.9))
    p = _problem(cands, kind="count", budget=2.0)
    ex = exact_submodularity_ratio(p, max_elements=11)
    st = overlap_stats(p, samples=400)
    am = alpha_min_bound(p, rho=st["max_oc"], rho_stat="max_oc", m_mode="eligible")
    assert st["max_oc"] == pytest.approx(1.0), "嵌套（h_i ⊂ r_i）会把重叠系数顶到 1"
    assert am["alpha_min"] <= ex["alpha"] + 1e-9, "引理 2 在此例上仍成立（更保守）"


# ----------------------------------------------------------------- GQC（创新点 3 算法）
def _gqc_fixture():
    """3 组能力（1/2/3 条合法路线）各 4 人，格子互不重叠，count 预算 6。"""
    cands = []
    for u in range(4):                       # tight：每人 1 条
        cands.append(_c(f"t{u}", [(u, 0), (u, 1)], 0.9))
    for u in range(4):                       # two：每人 2 条
        for r in range(2):
            cands.append(_c(f"w{u}", [(10 + u, 2 * r), (10 + u, 2 * r + 1)], 0.9))
    for u in range(4):                       # ample：每人 3 条
        for r in range(3):
            cands.append(_c(f"a{u}", [(20 + u, 2 * r), (20 + u, 2 * r + 1)], 0.9))
    return _problem(cands, kind="count", budget=6.0)


def test_groups_bucket_by_eligible_route_count():
    from aprime.groups import GROUP_AMPLE, GROUP_TIGHT, GROUP_TWO, capability_groups

    p = _gqc_fixture()
    g = capability_groups(p)
    assert g["t0"] == GROUP_TIGHT and g["w0"] == GROUP_TWO and g["a0"] == GROUP_AMPLE
    assert len(g) == 12, "合法池内应覆盖全部 12 人"


def test_gqc_satisfies_quota_floors_and_theta():
    """GQC 的输出必须：①满足各组 floor；②不违反 θ；③可行。"""
    import math

    from aprime.groups import capability_groups, group_counts, group_shares
    from aprime.solvers.gqc import GQCParams, gqc_solve

    p = _gqc_fixture()
    sel, info = gqc_solve(p, GQCParams(kappa=1.0, slack=0))
    assert info["quota_ok"] is True, f"配额未被满足：{info}"
    assert info["theta_violations"] == 0
    assert p.feasible(sel)

    g = capability_groups(p)
    cnt = group_counts(sel, g)
    share = group_shares(g)
    for grp, s in share.items():
        assert cnt.get(grp, 0) >= math.floor(s * info["k0"]) - 1e-9, f"{grp} 未达 floor"


def test_gqc_gap_beats_plain_greedy_on_skewed_instance():
    """偏斜实例上（能力强的人格子更大），纯贪心会超配 ample 组，GQC 不会。"""
    from aprime.groups import capability_groups
    from aprime.groups import gqc_gap as gap_fn
    from aprime.solvers.gqc import GQCParams, gqc_solve
    from aprime.solvers.greedy import solve

    cands = []
    for u in range(6):                       # tight：6 人，每条 1 格
        cands.append(_c(f"t{u}", [(u, 0)], 0.9))
    for u in range(2):                       # ample：2 人，每人 3 条且每条 4 格（贪心偏爱）
        for r in range(3):
            cands.append(_c(f"a{u}", [(50 + 4 * u + r, c) for c in range(4)], 0.9))
    p = _problem(cands, kind="count", budget=4.0)
    g = capability_groups(p)

    sel_a, _ = solve(p, with_ls=True)
    sel_q, _ = gqc_solve(p, GQCParams(kappa=1.0, slack=0))
    assert gap_fn(sel_q, g) <= gap_fn(sel_a, g) + 1e-9, (
        f"GQC 的 gqc_gap({gap_fn(sel_q, g):.3f}) 应不大于纯贪心({gap_fn(sel_a, g):.3f})")


def test_gqc_kappa_zero_degenerates_to_plain_greedy():
    """κ=0 时必须完全退化为纯贪心（同入口对照，便于说明"只加了配额"）。"""
    from aprime.solvers.gqc import GQCParams, gqc_greedy
    from aprime.solvers.greedy import greedy

    p = _gqc_fixture()
    sel_q, info = gqc_greedy(p, GQCParams(kappa=0.0))
    sel_g, _ = greedy(p)
    assert [c.wid + str(sorted(c.cells)) for c in sel_q] == \
           [c.wid + str(sorted(c.cells)) for c in sel_g]
    assert "kappa<=0" in info["reason"]


def test_greedy_meets_theorem2_guarantee():
    """定理 2 的回归测试：随机小实例上 A′ 必须达到 (a) `½`、(b) `½(1−1/e)`。

    这是把定理 2 从"文档里的措辞"变成"代码里的不变式"：任何改动一旦破坏
    "贪心 + 只改进的局部搜索"结构，这里会立刻红。
    """
    import math
    import random

    from aprime.solvers.exact import exact
    from aprime.solvers.greedy import solve

    rng = random.Random(20261110)
    const_steps = 0.5 * (1.0 - 1.0 / math.e)
    checked = 0
    for _ in range(25):
        cands = []
        for u in range(rng.randint(2, 4)):
            for _r in range(rng.randint(1, 3)):
                cells = [(rng.randint(0, 7), rng.randint(0, 7))
                         for _ in range(rng.randint(1, 4))]
                cands.append(_c(f"u{u}", cells, 0.9))
        for kind, budget, const in (("count", 2.0, 0.5), ("steps", 6.0, const_steps)):
            p = _problem(cands, kind=kind, budget=budget)
            if not p.eligible:
                continue
            opt, _ = exact(p)
            sel, _ = solve(p, with_ls=True)
            assert p.feasible(sel)
            got, best = p.coverage(sel), p.coverage(opt)
            assert got >= const * best - 1e-9, (
                f"违反定理 2（{kind}）：A′={got} < {const:.4f}·OPT={const * best:.3f}")
            checked += 1
    assert checked >= 30, f"有效实例太少（{checked}），测试没有覆盖力"


def test_gqc_pays_zero_gain_route_to_meet_floor():
    """配额欠账时必须敢选"零覆盖增益"的路线——否则覆盖饱和后配额会悄悄失守。

    Day-7 在 `budget_steps=80`、θ=0.9 上真实踩到过：覆盖已饱和，`tight` 组欠 1 个名额，
    而该组唯一可选的路线边际增益为 0，贪心直接跳过 → `floor_ok` 掉到 0.933。
    """
    from aprime.groups import capability_groups, group_counts
    from aprime.solvers.gqc import GQCParams, gqc_solve
    from aprime.solvers.greedy import solve

    cands = [
        _c("t0", [(0, 0), (0, 1), (1, 0), (1, 1)], 0.9),   # tight：一次盖住 4 格
        _c("t1", [(2, 0), (2, 1)], 0.9),                    # tight：另外 2 格
    ]
    for u in range(2):                                       # ample：每人 3 条，每条只 1 格
        for cell in ([(0, 0)], [(0, 1)], [(1, 1)]):         # 全部落在 t0 的覆盖里
            cands.append(_c(f"a{u}", cell, 0.9))
    p = _problem(cands, kind="count", budget=2.0)
    g = capability_groups(p)

    # 纯贪心：t0 + t1 已拿满 2 个名额与 6 格覆盖，ample 的路线增益全为 0 → 配额失守
    sel_a, _ = solve(p, with_ls=False)
    assert p.coverage(sel_a) == 6
    assert group_counts(sel_a, g).get("ample", 0) == 0, "前置条件：纯贪心确实不给 ample 名额"

    # GQC：必须在"增益为 0"的前提下仍然给 ample 一个名额
    sel_q, info = gqc_solve(p, GQCParams(kappa=1.0, slack=0))
    cnt = group_counts(sel_q, g)
    assert info["floor_ok"] is True, f"配额失守：{info}"
    assert cnt.get("ample", 0) >= 1
    # 公平的代价要如实暴露：本例中 GQC 用 2 格覆盖换 1 个 ample 名额
    assert p.coverage(sel_q) == 4


# ----------------------------------------------------------------- 公平性指标
def test_fairness_metrics_endpoints():
    """公平指标的端点必须对：完全均等 → gini 0 / jain 1；极度不均 → gini 大。"""
    from aprime.analysis.fairness import _gini, _jain

    assert _gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)
    assert _jain([1, 1, 1, 1]) == pytest.approx(1.0, abs=1e-9)
    assert _gini([0, 0, 0, 10]) > 0.6
    assert _jain([0, 0, 0, 10]) == pytest.approx(0.25, abs=1e-9)


def test_fairness_report_separates_feasibility_from_selection():
    """θ 让一半人不可行时：feasible_rate 反映代价，exclusion_gap 只在合法池内算。

    这是 Day-4 抓到的口径问题：把不可行的人算进"被排除"，会让高 θ 端出现虚假的公平。
    """
    from aprime.analysis.fairness import fairness_report

    cands = []
    # 4 个"高能力"者（各 3 条 0.9 的路线）+ 4 个"低能力"者（各 1 条 0.9 的路线）
    for u in range(4):
        for r in range(3):
            cands.append(_c(f"h{u}", [(u, r), (u, r + 1)], 0.9))
        cands.append(_c(f"l{u}", [(u + 5, 0), (u + 5, 1)], 0.9))
    p = _problem(cands, kind="count", budget=2.0)
    sel = [p.by_worker["h0"][0], p.by_worker["h1"][0]]
    rep = fairness_report(p, sel)
    assert rep["feasible_rate"] == pytest.approx(1.0), "θ=0 时所有人都应有合法路线"
    assert rep["theta_violations"] == 0
    # 低能力组（1 条路线）全被排除、高能力组只被排除一半 → gap 应为正
    assert rep["exclusion_gap"] > 0.3

    # 把 θ 提到 0.95：所有人都不合法 → feasible_rate 掉到 0，但不该伪造"更公平"
    p2 = _problem(cands, theta=0.95, kind="count", budget=2.0)
    rep2 = fairness_report(p2, [])
    assert rep2["feasible_rate"] == pytest.approx(0.0)
    assert rep2["exclusion_gap"] == pytest.approx(0.0), "合法池为空时不该给出排除率结论"


def test_mscta_greedy_ignores_theta_and_uses_multiple_slots():
    """东华式基线故意不看 θ、允许每人多名额——这两个性质必须被测试钉住。"""
    from aprime.solvers.baselines import mscta_greedy

    cands = [_c("u1", [(0, 0), (0, 1)], 0.2), _c("u1", [(1, 1), (1, 2)], 0.2),
             _c("u2", [(2, 2), (2, 3)], 0.2)]
    p = _problem(cands, theta=0.9, kind="count", budget=2.0)
    sel, info = mscta_greedy(p, cap_per_worker=2)
    assert info["ignores_theta"] is True
    assert any(not p.is_eligible(c) for c in sel), "它应当选中违反 θ 的路线"
    assert info["max_slots_per_worker"] <= 2
def test_gqc_gap_is_zero_and_uninformative_when_grouping_collapses():
    """单档分组下 gqc_gap 恒为 0 —— 是**指标失效**，不是"完全公平"。

    实测踩过的坑：绝对口径下基线所有 60 人都在 ample，
    于是任何让分组分化的改动都显得在"抬高公平差距"。必须能识别出来。
    """
    from aprime.groups import (GROUP_AMPLE, GROUP_TIGHT, gqc_gap,
                               grouping_is_degenerate, min_gap_floor)

    one_group = {f"u{i}": GROUP_AMPLE for i in range(60)}
    assert grouping_is_degenerate(one_group)
    sel = [_c(f"u{i}", [(i, 0)], 0.9) for i in range(5)]
    assert gqc_gap(sel, one_group) == 0.0

    two_groups = {f"u{i}": (GROUP_AMPLE if i < 30 else GROUP_TIGHT) for i in range(60)}
    assert not grouping_is_degenerate(two_groups)
    sel2 = [_c(f"u{i}", [(i, 1)], 0.9) for i in range(5)]  # 全从 ample 取
    assert gqc_gap(sel2, two_groups) > 0.0


def test_min_gap_floor_is_pure_integer_artifact():
    """整数下限只由"组规模 × 选中数"决定，与算法、与数据源都无关。"""
    from aprime.groups import min_gap_floor

    # 1 人小组 + 选 5 条：份额 1/60=1.67%，最小偏差就是 1.67%（给它 0 条）
    assert min_gap_floor({"tight": 1, "ample": 59}, 5) == pytest.approx(1 / 60, abs=1e-6)
    # 组规模恰好能整除时下限应为 0：30/30 且选 6 条（3+3）
    assert min_gap_floor({"tight": 30, "ample": 30}, 6) == pytest.approx(0.0, abs=1e-9)
    # 组越偏斜，下限越高——这正是"组规模伪影"的量化形式
    small = min_gap_floor({"tight": 2, "ample": 58}, 5)
    large = min_gap_floor({"tight": 12, "ample": 48}, 5)
    assert small > large


def test_alpha_is_one_for_disjoint_sets():
    cands = [_c(f"u{i}", [(i, 0, 0), (i, 0, 1)], 0.9) for i in range(4)]
    p = _problem(cands)
    assert submodularity_ratio(p, samples=80)["alpha"] == pytest.approx(1.0)


def test_alpha_drops_with_redundancy():
    dup = [_c(f"u{i}", [(0, 0, t) for t in range(4)], 0.9) for i in range(4)]
    p = _problem(dup)
    assert submodularity_ratio(p, samples=120)["alpha"] < 0.6


def test_apx_bound_monotone_and_bounded():
    vals = [apx_bound(a) for a in (0.1, 0.3, 0.5, 0.8, 1.0)]
    assert vals == sorted(vals)
    assert apx_bound(1.0) == pytest.approx(1 - 2.718281828 ** -1, abs=1e-6)


def test_alpha_min_uses_worst_case_rho():
    """Lemma 2 的 ρ 是最坏成对重叠，用均值会把界说强（这是 Day-2 抓到的一个真 bug）。"""
    from aprime.analysis.submodularity import alpha_min_bound, overlap_stats

    cands = [
        _c("u1", [(0, 0), (0, 1), (0, 2)], 0.9),
        _c("u2", [(0, 0), (0, 1), (0, 2)], 0.9),   # 与 u1 完全重叠
        _c("u3", [(5, 5), (5, 6)], 0.9),           # 与别人不重叠
        _c("u4", [(5, 5), (7, 7)], 0.9),
    ]
    p = _problem(cands, budget=6.0)
    st = overlap_stats(p, samples=300)
    assert st["p95_oc"] >= st["mean_oc"], "重叠系数的 p95 必须不低于均值"
    assert st["max_oc"] >= st["p95_oc"]
    a_max = alpha_min_bound(p, samples=300, rho_stat="max_oc")["alpha_min"]
    a_mean = alpha_min_bound(p, samples=300, rho_stat="mean_oc")["alpha_min"]
    assert a_max <= a_mean + 1e-12, "用最坏重叠算出的界必须更保守"


def test_lemma2_is_a_genuine_lower_bound_on_alpha():
    """**关键回归测试**：引理 2 的下界必须 ≤ 真实 α，否则整个定理 2 无效。

    Day-3 抓到过两种写法都不成立：
    * ρ 用 Jaccard（应为重叠系数）；
    * m 用预算 ⌊B/w⌋（应为 |E(θ)|，因为 α 定义里 T 不受预算限制）。
    """
    import itertools
    import random

    from aprime.analysis.alpha_exact import exact_submodularity_ratio
    from aprime.analysis.submodularity import alpha_min_bound, overlap_stats

    rng = random.Random(11)
    checked = 0
    for trial in range(12):
        n_workers = rng.choice([3, 3, 4])
        routes = rng.choice([1, 2])
        universe = rng.choice([5, 6, 7])
        cands = []
        for u in range(n_workers):
            for _ in range(routes):
                w = rng.choice([2, 3])
                cells = rng.sample(range(universe), min(w, universe))
                cands.append(_c(f"u{u}", [(c, 0) for c in cells], 0.9))
        p = _problem(cands, kind="count", budget=float(rng.choice([2, 3])))
        ex = exact_submodularity_ratio(p, max_elements=11)
        if ex is None:
            continue
        st = overlap_stats(p, samples=400)
        am = alpha_min_bound(p, rho=st["max_oc"], rho_stat="max_oc", m_mode="eligible")
        checked += 1
        assert am["alpha_min"] <= ex["alpha"] + 1e-9, (
            f"引理 2 失效：α_min={am['alpha_min']} > α_true={ex['alpha']}；"
            f"ρ_oc={st['max_oc']} m={am['m']}")
        # 顺带确认"按预算截断 m"确实会破坏下界（保留反例意识，不当作正确用法）
        am_budget = alpha_min_bound(p, rho=st["max_oc"], rho_stat="max_oc", m_mode="budget")
        assert am_budget["alpha_min"] >= am["alpha_min"] - 1e-12
    assert checked >= 8, f"有效样本太少（{checked}），测试没有实际约束力"


# --------------------------------------------------------------------- 真实数据
def test_candidates_from_tdrive():
    """真实轨迹上的候选池：每人至少 1 条可行路线，且路线满足时间窗/速度约束。"""
    from aprime.candidates import build_candidates
    from mapus.data.instances import make_mapus_instance
    from mapus.routing import validate_route

    inst = make_mapus_instance("tdrive_small", 0)
    cands = build_candidates(inst, k=6, seed=0)
    assert cands, "候选池为空"
    by_w = {}
    for c in cands:
        by_w.setdefault(c.wid, []).append(c)
    assert len(by_w) == len(inst.workers), "有参与者没有候选路线"
    assert all(len(v) <= 6 for v in by_w.values())
    wmap = {w.wid: w for w in inst.workers}
    for c in cands[:40]:
        ok, why = validate_route(wmap[c.wid], [list(s) for s in c.path], inst.nx, inst.ny)
        assert ok, f"{c.wid} 路线非法: {why}"


# --------------------------------------------------------------------- 密度 / 偏好
def test_dense_instance_worker_count():
    from aprime.dense import make_dense_instance

    inst = make_dense_instance("tdrive", "small", n_workers=12, budget=25.0, seed=0)
    assert len(inst.workers) == 12
    assert len({w.wid for w in inst.workers}) == 12, "参与者 id 必须唯一"
    assert inst.budget == 25.0
    assert "dense" in inst.meta["source"]


def test_homogeneity_blend_endpoints():
    import numpy as np

    from aprime.dense import make_dense_instance
    from aprime.homogeneity import anchor_pref, blend_profiles

    inst = make_dense_instance("tdrive", "small", n_workers=4, seed=0)
    same = blend_profiles(inst.profiles, 0.0)
    for wid, p in inst.profiles.items():
        assert np.allclose(same[wid].pref, p.pref), "φ=0 不应改变任何偏好"

    anchor = anchor_pref("eco_enthusiast", seed=0)
    homog = blend_profiles(inst.profiles, 1.0, archetype="eco_enthusiast", seed=0)
    for p in homog.values():
        assert np.allclose(p.pref, anchor, atol=1e-9), "φ=1 后所有偏好应等于锚点"
        assert p.pref.sum() == pytest.approx(1.0)


def test_phi_proxy_monotone_in_knob():
    """φ 的可测代理必须随旋钮单调上升，否则定理里的 φ 没法从数据估。"""
    from aprime.analysis.preference import pref_concentration
    from aprime.dense import make_dense_instance
    from aprime.homogeneity import blend_profiles

    inst = make_dense_instance("tdrive", "small", n_workers=20, seed=0)
    vals = [pref_concentration(blend_profiles(inst.profiles, phi, seed=0))["phi_cos"]
            for phi in (0.0, 0.5, 1.0)]
    assert vals == sorted(vals), f"phi_cos 应随 φ 单调上升，实际 {vals}"
    assert vals[-1] > vals[0] + 0.1, f"端点差异太小，代理无判别力: {vals}"
    assert vals[-1] == pytest.approx(1.0, abs=1e-6), "φ=1 时平均余弦应为 1"

