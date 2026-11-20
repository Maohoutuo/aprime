# -*- coding: utf-8 -*-
"""搁置机制的自洽性测试。

E0.1 阻塞期间，"哪些实验能跑"必须由 ``aprime/availability.py`` 单点判定；
这里守住三件事：
  1. 登记表自身不漂移（引用了未声明的数据集、blocked 却没写解冻条件）；
  2. 搁置的实验确实拦得住（``require_available`` 抛错并带解冻条件）；
  3. 不依赖真实数据的实验不被误伤（E2.1/E2.2 仍可跑）。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime import availability as avail  # noqa: E402


def test_registry_is_self_consistent():
    for eid, exp in avail.EXPERIMENTS.items():
        assert exp.id == eid, "键与 id 不一致: %r" % eid
        for name in exp.datasets:
            assert name in avail.DATASETS, "%s 引用了未声明的数据集 %r" % (eid, name)
        # blocked 的实验必须写清楚怎么解冻，否则"搁置"会退化成"遗忘"
        if avail.status_of(eid) == "blocked":
            assert exp.unblock or exp.blockers, "%s 是 blocked 但没有解冻条件" % eid


def test_status_matches_blockers():
    for eid in avail.EXPERIMENTS:
        status = avail.status_of(eid)
        assert status in {"ready", "deferred", "blocked"}
        if status == "blocked":
            assert avail.blockers_of(eid), "%s 标 blocked 但列不出阻塞项" % eid
        else:
            assert not avail.blockers_of(eid), "%s 有阻塞项却不是 blocked" % eid


def test_abstract_experiments_stay_runnable():
    # E2.1/E2.2 用抽象实例（run_bound_tightness.py），不该被真实数据搁置波及
    for eid in ("E2.1", "E2.2"):
        assert avail.status_of(eid) == "ready"
        avail.require_available(eid)  # 不抛错即通过


def test_e0_1_readiness_flows_from_signed_verdict(monkeypatch, tmp_path):
    """E0.1 就绪由签收凭据决定：没有 pass 就是 blocked，有 pass 才能 require_available。"""
    fake = tmp_path / "manifest.json"
    monkeypatch.setattr(avail, "_TDRIVE_MANIFEST", str(fake))

    fake.write_text(json.dumps({"time_field_verdict": "partial: 等 D5"}), encoding="utf-8")
    assert avail.status_of("E0.1") == "blocked"
    with pytest.raises(avail.DataUnavailable):
        avail.require_available("E0.1")

    fake.write_text(json.dumps({"time_field_verdict": "pass: A1–A8 全过"}), encoding="utf-8")
    assert avail.status_of("E0.1") == "ready"
    avail.require_available("E0.1")  # 不抛错即通过

    # 注：本机 data/tdrive/manifest.json 已实测 pass（audit_time_field.py 跑过），
    # 所以本仓库工作副本上的 E0.1 是 ready；新检出需先重跑审计再产出真实数据结果。


def test_e0_1b_is_blocked_and_says_why():
    """gMission 不能承载轨迹级 E0.1，降级为匹配级且持续 blocked。"""
    assert avail.status_of("E0.1b") == "blocked"
    with pytest.raises(avail.DataUnavailable) as excinfo:
        avail.require_available("E0.1b")
    msg = str(excinfo.value)
    assert "blocked" in msg
    assert "gmission" in msg
    assert "数据集 gmission 不可用" in msg


def test_shelved_lists_real_data_experiments_only():
    shelved = dict(avail.shelved())
    # 剩下 blocked 的：E0.1b（数据结构不可满足）、E1.2（管线未实现）；
    # E0.1/E1.1/E2.3/E3.1/E5.1/SEC6/CROSS 已解冻为 ready
    assert shelved == {"E0.1b": "blocked", "E1.2": "blocked"}, shelved
    assert "E2.1" not in shelved
    ids = [eid for eid, _ in avail.shelved()]
    assert ids == sorted(ids)


def test_shelve_switch_lifts_only_deferred(monkeypatch):
    """关掉全局闸门：deferred 变 ready，blocked 仍然拦住。"""
    monkeypatch.setattr(avail, "SHELVE_REAL_DATA", True)
    assert avail.status_of("E1.1") == "deferred"
    assert avail.status_of("CROSS") == "deferred"
    assert avail.status_of("E0.1b") == "blocked"
    assert avail.status_of("E1.2") == "blocked"


def test_unknown_ids_are_rejected():
    with pytest.raises(KeyError):
        avail.status_of("E9.9")
    with pytest.raises(KeyError):
        avail.dataset_available("no_such_dataset")


def test_tdrive_verdict_probe_needs_pass_manifest(tmp_path, monkeypatch):
    """只有 time_field_verdict 为 pass 的 manifest 才算 E0.1 签收。"""
    monkeypatch.setattr(avail, "_TDRIVE_MANIFEST", str(tmp_path / "manifest.json"))
    assert avail._probe_tdrive_verdict() is False  # 文件不存在

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert avail._probe_tdrive_verdict() is False  # 有 manifest 但没填 verdict

    (tmp_path / "manifest.json").write_text(
        json.dumps({"time_field_verdict": "partial: 车次不足"}), encoding="utf-8")
    assert avail._probe_tdrive_verdict() is False  # 未验收不得解冻

    (tmp_path / "manifest.json").write_text(
        json.dumps({"time_field_verdict": "pass: A1–A8 全过"}), encoding="utf-8")
    assert avail._probe_tdrive_verdict() is True


def test_gmission_probe_needs_manifest_and_nonempty_raw(tmp_path, monkeypatch):
    """只建目录不算接入；**manifest 存在但 verdict 不是 pass 也不算**。

    后半条是 2026-11-14 收紧的：``docs/e0_1_data_access_plan.md`` §5 写着
    "``time_field_verdict`` 是 E0.1 的签收字段，没有它 E0.1 视为未完成"，
    而当时 ``data/gmission/manifest.json`` 的实测值是 ``partial``（D1 已做、
    D2–D5 未做）。若只校验"文件存在"，写个骨架就会误触发 E0.1 解冻。
    """
    monkeypatch.setattr(avail, "_GMISSION_DIR", str(tmp_path))
    assert avail._probe_gmission() is False  # 目录都不存在

    (tmp_path / "raw").mkdir()
    assert avail._probe_gmission() is False  # raw 为空

    (tmp_path / "raw" / "task.csv").write_text("task_id\n", encoding="utf-8")
    assert avail._probe_gmission() is False  # 有数据但没有 manifest

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert avail._probe_gmission() is False  # 有 manifest 但没填 verdict

    (tmp_path / "manifest.json").write_text(
        json.dumps({"time_field_verdict": "partial: 无时区/纪元"}), encoding="utf-8")
    assert avail._probe_gmission() is False  # 骨架/未验收不得解冻

    (tmp_path / "manifest.json").write_text(
        json.dumps({"time_field_verdict": "pass: 99.97% parseable"}), encoding="utf-8")
    assert avail._probe_gmission() is True  # 签收字段为 pass 才算接入


def test_geolife_probe_rejects_missing_file(tmp_path, monkeypatch):
    """GeoLife 的判据是**归一化产物**，不是原始 zip。

    未归一化的 GeoLife 是 1–5 s 采样，重复 slot 会把 ``_contiguous_runs`` 全断
    （实测 363 万点只切出 15 条出行段），等于不可用，故不能只看 zip 在不在。
    """
    monkeypatch.setenv("GEOLIFE_PATH", str(tmp_path / "nope.txt.gz"))
    assert avail._probe_geolife() is False

    empty = tmp_path / "empty.txt.gz"
    empty.write_bytes(b"")
    monkeypatch.setenv("GEOLIFE_PATH", str(empty))
    assert avail._probe_geolife() is False  # 0 字节不算

    real = tmp_path / "geolife_norm.txt.gz"
    real.write_bytes(b"x")
    monkeypatch.setenv("GEOLIFE_PATH", str(real))
    assert avail._probe_geolife() is True


def test_summary_is_printable():
    text = avail.summary()
    assert "E0.1" in text and "blocked" in text
    assert "gmission=" in text
