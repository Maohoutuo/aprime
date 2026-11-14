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


def test_e0_1_is_blocked_and_says_how_to_unblock():
    assert avail.status_of("E0.1") == "blocked"
    with pytest.raises(avail.DataUnavailable) as excinfo:
        avail.require_available("E0.1")
    msg = str(excinfo.value)
    assert "blocked" in msg
    assert "解冻条件" in msg
    assert "gmission" in msg
    assert "e0_1_data_access_plan.md" in msg


def test_shelved_lists_real_data_experiments_only():
    shelved = dict(avail.shelved())
    assert "E0.1" in shelved and shelved["E0.1"] == "blocked"
    assert "E1.1" in shelved
    assert "E2.1" not in shelved
    # 列表按 id 排序，便于直接贴进汇报
    ids = [eid for eid, _ in avail.shelved()]
    assert ids == sorted(ids)


def test_shelve_switch_only_lifts_deferred(monkeypatch):
    """关掉全局闸门：deferred 变 ready，但 blocked 仍然拦住。"""
    monkeypatch.setattr(avail, "SHELVE_REAL_DATA", False)
    assert avail.status_of("E1.1") == "ready"
    assert avail.status_of("E0.1") == "blocked"  # 缺数据不是开关能解决的


def test_unknown_ids_are_rejected():
    with pytest.raises(KeyError):
        avail.status_of("E9.9")
    with pytest.raises(KeyError):
        avail.dataset_available("no_such_dataset")


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
