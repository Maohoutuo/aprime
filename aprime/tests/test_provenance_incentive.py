# -*- coding: utf-8 -*-
"""来源闸门 + 激励层的回归测试。

这两件事的共同点：**一旦悄悄失效，论文里就会出现“合成数字冒充真实数据”**，
而那种错误在结果文件里看不出来。所以这里全部用"必须抛错"的形式钉住。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.dense import make_dense_instance  # noqa: E402
from aprime.provenance import (KIND_REAL, KIND_SYNTHETIC, Provenance,  # noqa: E402
                               SyntheticDataError, provenance_of, require_real,
                               require_stamped, stamp)
from aprime.synth.incentive import (IncentiveParams, count_budget,  # noqa: E402
                                   make_incentive_layer, match_money_to_quota,
                                   quota_equivalent)


# ------------------------------------------------------------------ 来源解析
def test_default_mapus_instance_is_synthetic():
    """默认构造必须是合成的——这是 README §0 警告的那个经典事故源。"""
    from mapus.data.instances import make_mapus_instance

    prov = provenance_of(make_mapus_instance("tdrive_small", 0))
    assert prov.workers_kind == KIND_SYNTHETIC
    assert prov.synthetic
    assert "synthetic" in prov.workers_label


def test_dense_label_does_not_confuse_n_workers_with_worker_source():
    """`dense(n_workers=160, workers=synthetic(...))` 里有两个 `workers=`。

    用不带前置断言的 `re.search("workers=...")` 会先命中 `n_workers=160`，
    把参与者数当成来源标签——那时任何 dense 实例都不会被判成合成。
    """
    inst = make_dense_instance("tdrive", "small", n_workers=12, seed=0)
    prov = provenance_of(inst)
    assert prov.workers_kind == KIND_SYNTHETIC
    assert prov.workers_label.startswith("synthetic"), prov.workers_label


def test_unknown_label_fails_closed():
    """未知标签不许猜成 real。"""
    class _Fake:
        meta = {"source": "cryptic_thing"}

    assert provenance_of(_Fake()).workers_kind == "unknown"
    with pytest.raises(SyntheticDataError):
        require_real(_Fake(), "E1.1")


def test_require_real_rejects_synthetic_and_mentions_the_fix():
    from mapus.data.instances import make_mapus_instance

    inst = make_mapus_instance("tdrive_small", 0)
    with pytest.raises(SyntheticDataError) as ei:
        require_real(inst, "E1.1")
    msg = str(ei.value)
    assert "E1.1" in msg and "source='tdrive'" in msg


def test_require_real_accepts_real_tdrive_instance():
    from mapus.data.instances import make_mapus_instance

    try:
        inst = make_mapus_instance("tdrive_small", 0, source="tdrive")
    except Exception:  # pragma: no cover - 数据缺失环境
        pytest.skip("T-Drive 清洗文件不可用")
    prov = require_real(inst, "E1.1")
    assert prov.workers_kind == KIND_REAL


def test_allow_env_marks_forced_but_still_warns(monkeypatch, capsys):
    """放行开关只能“不拦”，不能把 provenance 洗白：必须留 forced=True。"""
    from mapus.data.instances import make_mapus_instance

    monkeypatch.setenv("APRIME_ALLOW_SYNTHETIC_IN_REAL", "1")
    inst = make_mapus_instance("tdrive_small", 0)
    prov = require_real(inst, "E1.1")
    assert prov.forced is True
    assert prov.workers_kind == KIND_SYNTHETIC
    assert "不得" in capsys.readouterr().err


def test_shelved_switch_does_not_bypass_provenance(monkeypatch):
    """**关键**：`APRIME_ALLOW_SHELVED` 是“允许跑”，不是“允许拿合成数据冒充真实数据”。

    两道闸门必须独立，否则调试用的放行会顺手关掉诚实性检查。
    """
    from aprime.availability import require_available
    from mapus.data.instances import make_mapus_instance

    monkeypatch.setenv("APRIME_ALLOW_SHELVED", "1")
    monkeypatch.delenv("APRIME_ALLOW_SYNTHETIC_IN_REAL", raising=False)
    inst = make_mapus_instance("tdrive_small", 0)
    with pytest.raises(SyntheticDataError):
        require_available("E1.1", instance=inst)


def test_require_available_ignores_provenance_for_ready_experiments():
    """不依赖真实数据的实验不该被来源闸门管（E2.1 是抽象实例）。"""
    from aprime.availability import require_available
    from mapus.data.instances import make_mapus_instance

    require_available("E2.1", instance=make_mapus_instance("tdrive_small", 0))


# ------------------------------------------------------------------ 结果戳
def test_stamp_and_require_stamped_roundtrip():
    from mapus.data.instances import make_mapus_instance

    inst = make_mapus_instance("tdrive_small", 0)
    payload = stamp({"rows": []}, instance=inst, models={"incentive": "flat"})
    block = require_stamped(payload)
    assert block["workers"]["kind"] == KIND_SYNTHETIC
    assert block["models"] == {"incentive": "flat"}
    assert block["profiles"]["kind"] == KIND_SYNTHETIC, "偏好层无真实来源，必须如实标合成"


def test_require_stamped_rejects_bare_payload():
    with pytest.raises(SyntheticDataError):
        require_stamped({"rows": [1, 2, 3]})
    with pytest.raises(SyntheticDataError):
        require_stamped({"provenance": {"label": "x"}})  # 缺字段同样要拦


def test_stamp_without_instance_is_marked_payload_level():
    block = require_stamped(stamp({"x": 1}, models={"incentive": True}))
    assert block["label"].startswith("<payload-level")


# ------------------------------------------------------------------ 激励层
@pytest.fixture(scope="module")
def _inst():
    try:
        return make_dense_instance("tdrive", "small", n_workers=20, seed=0,
                                   source="tdrive", urban_source="osm")
    except Exception:  # pragma: no cover
        pytest.skip("T-Drive / OSM 属性层不可用")


def test_three_models_share_total_purchasing_power(_inst):
    """三模型必须同购买力，否则“结构性差异”会被尺度差异污染。"""
    means = [sum(make_incentive_layer(_inst, m, seed=0).pay.values()) / 20
             for m in ("flat", "dist_cost", "supply_demand")]
    assert means[1] == pytest.approx(means[0], rel=1e-9)
    assert means[2] == pytest.approx(means[0], rel=1e-9)


def test_flat_is_uniform_and_gives_pure_cardinality(_inst):
    L = make_incentive_layer(_inst, "flat", seed=0)
    assert len(set(round(v, 9) for v in L.pay.values())) == 1
    pp = count_budget(L)
    assert pp.k_min == pp.k_max, "同价时背包退化，k_min 必须等于 k_max"
    assert pp.backpack_factor == pytest.approx(1.0)


def test_dist_cost_is_monotone_in_distance(_inst):
    """per_cell 系数为正 ⇒ 同等时间窗下，距离越长的要价不应更低。"""
    L = make_incentive_layer(_inst, "dist_cost", seed=0)
    by_win: Dict[int, List[Tuple[int, float]]] = {}
    for w in _inst.workers:
        d = abs(w.origin[0] - w.dest[0]) + abs(w.origin[1] - w.dest[1])
        by_win.setdefault(w.window, []).append((d, L.pay[w.wid]))
    for dps in by_win.values():
        dps.sort()
        pays = [p for _d, p in dps]
        assert all(b >= a - 1e-9 for a, b in zip(pays, pays[1:])), dps


def test_supply_demand_moves_pay_away_from_dist_cost(_inst):
    a = make_incentive_layer(_inst, "dist_cost", seed=0)
    b = make_incentive_layer(_inst, "supply_demand", seed=0)
    diff = sum(abs(a.pay[w] - b.pay[w]) for w in a.pay)
    assert diff > 1e-6, "地段溢价没起作用（urban 属性缺失？）"
    assert count_budget(b).backpack_factor >= 1.0


def test_backpack_factor_grows_with_unequal_pay(_inst):
    flat = count_budget(make_incentive_layer(_inst, "flat", seed=0))
    dist = count_budget(make_incentive_layer(_inst, "dist_cost", seed=0))
    assert dist.backpack_factor > flat.backpack_factor


def test_match_money_to_quota_lands_exactly_on_k(_inst):
    """`worst` 锚定后，k_min 必须恰好等于想要的名额数——否则换算不可用。"""
    for model in ("flat", "dist_cost", "supply_demand"):
        base = make_incentive_layer(_inst, model, seed=0)
        for k in (3, 6, 9):
            M = match_money_to_quota(base, k, mode="worst")
            L = make_incentive_layer(_inst, model, seed=0, money_budget=M)
            assert count_budget(L).k_min == k, (model, k, count_budget(L).k_min)


def test_quota_equivalent_anchors_steps_to_slots():
    assert quota_equivalent(40.0, 5.0) == 8
    assert quota_equivalent(60.0, 5.0) == 12
    assert quota_equivalent(0.5, 5.0) == 1, "预算极小时至少留 1 个名额"
    with pytest.raises(ValueError):
        quota_equivalent(40.0, 0.0)


def test_incentive_layer_is_deterministic_and_jitter_aware(_inst):
    a = make_incentive_layer(_inst, "dist_cost", seed=7)
    b = make_incentive_layer(_inst, "dist_cost", seed=7)
    assert a.pay == b.pay
    p1 = IncentiveParams(jitter_sigma=0.2)
    c = make_incentive_layer(_inst, "dist_cost", seed=1, params=p1)
    d = make_incentive_layer(_inst, "dist_cost", seed=2, params=p1)
    assert c.pay != d.pay


def test_incentive_layer_never_needs_real_payments(_inst):
    """激励层的出身必须可被机读为“假设”，不能被当成观测到的支付数据。"""
    L = make_incentive_layer(_inst, "supply_demand", seed=0)
    prov = provenance_of(_inst)
    assert L.capacity and all(v == 1 for v in L.capacity.values()), "每人至多一条"
    assert set(L.pay) == {w.wid for w in _inst.workers}
    assert prov.urban_kind == KIND_REAL, "osm 用地应被识别为真实层（犯罪层仍合成）"
    assert "crime=synthetic" in prov.urban_label
