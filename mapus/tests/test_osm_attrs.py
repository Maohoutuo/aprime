# -*- coding: utf-8 -*-
"""OSM 属性层读取器的测试。

重点守三件事：
  1. 逐格和为 1（``route_land_use`` 依赖它，不归一化会让满意度算错）；
  2. **缺格填均匀**而不是填 0（0 会让加权平均除以 0，且"无数据"≠"六类都没有"）；
  3. 未标注的车道数/限速是 ``nan`` 而不是 0（0 会被读成"零车道"）。

用临时构造的小 CSV 测，不依赖真实构建产物——否则测试会在别的机器上失败。
"""

from __future__ import annotations

import gzip
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapus.config import LAND_USE  # noqa: E402
from mapus.data import osm_attrs  # noqa: E402

_HEADER = (["n", "x", "y", "n_road_seg", "road_len_m", "n_intersection",
            "n_building", "n_poi", "lanes_mean", "maxspeed_mean"]
           + ["road_share_" + g for g in ("motorway", "primary", "secondary",
                                          "residential", "service", "foot")]
           + ["lu_area_m2"]
           + ["lu_area_share_" + c for c in LAND_USE]
           + ["lu_poi_share_" + c for c in LAND_USE]
           + ["lu_share_" + c for c in LAND_USE])


def _row(n, x, y, shares, **over):
    r = {k: "0" for k in _HEADER}
    r.update({"n": str(n), "x": str(x), "y": str(y),
              "n_road_seg": "10", "road_len_m": "1234.5", "n_intersection": "3",
              "n_building": "5", "n_poi": "7",
              "lanes_mean": "", "maxspeed_mean": "",      # 未标注
              "road_share_primary": "0.5", "road_share_residential": "0.5",
              "lu_area_m2": "1000.0"})
    for c, v in zip(LAND_USE, shares):
        r["lu_share_" + c] = str(v)
    r.update(over)
    return r


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(",".join(_HEADER) + "\n")
        for r in rows:
            fh.write(",".join(r[k] for k in _HEADER) + "\n")


@pytest.fixture()
def built(tmp_path, monkeypatch):
    path = tmp_path / "attrs.csv.gz"
    # 2×2 网格，只填 (0,0) 与 (1,1) 两格，另两格故意缺席
    _write(path, [
        _row(2, 0, 0, [0.6, 0.0, 0.0, 0.0, 0.4, 0.0]),
        _row(2, 1, 1, [0.0, 0.0, 0.0, 0.0, 0.2, 0.8]),
    ])
    monkeypatch.setenv("OSM_ATTRS_PATH", str(path))
    return path


def test_available_reflects_file(built, tmp_path, monkeypatch):
    assert osm_attrs.available() is True

    monkeypatch.setenv("OSM_ATTRS_PATH", str(tmp_path / "nope.csv.gz"))
    assert osm_attrs.available() is False

    empty = tmp_path / "empty.csv.gz"
    empty.write_bytes(b"")
    monkeypatch.setenv("OSM_ATTRS_PATH", str(empty))
    assert osm_attrs.available() is False  # 0 字节不算


def test_land_use_grid_normalizes_and_fills_missing_uniform(built):
    lu = osm_attrs.land_use_grid(2)
    assert lu.shape == (2, 2, len(LAND_USE))
    # 每一格（含缺席格）都必须和为 1
    np.testing.assert_allclose(lu.sum(axis=2), np.ones((2, 2)), rtol=1e-9)
    # 缺席格 = 均匀分布
    np.testing.assert_allclose(lu[0, 1], np.full(len(LAND_USE), 1.0 / len(LAND_USE)))
    # 有数据的格保持原样
    assert lu[0, 0, LAND_USE.index("vegetation")] == pytest.approx(0.6)
    assert lu[1, 1, LAND_USE.index("commercial")] == pytest.approx(0.8)


def test_land_use_grid_renormalizes_dirty_shares(built):
    """原始份额之和不为 1 时要重新归一，不能原样透传。"""
    path = built
    _write(path, [_row(2, 0, 0, [0.5, 0.0, 0.0, 0.0, 0.5, 0.0], lu_area_m2="1")])
    lu = osm_attrs.land_use_grid(2)
    np.testing.assert_allclose(lu.sum(axis=2), np.ones((2, 2)), rtol=1e-9)


def test_coverage_reports_missing_cells(built):
    assert osm_attrs.coverage(2) == pytest.approx(0.5)  # 4 格里 2 格有数据


def test_road_features_keeps_nan_for_unlabelled(built):
    rf = osm_attrs.road_features(2)
    assert rf["road_len_m"].shape == (2, 2)
    assert rf["road_len_m"][0, 0] == pytest.approx(1234.5)
    # 未标注 → nan，不是 0
    assert np.isnan(rf["lanes_mean"][0, 0])
    assert np.isnan(rf["maxspeed_mean"][0, 0])
    # 缺席格是 0（计数型字段）
    assert rf["n_road_seg"][0, 1] == 0.0
    assert rf["road_share_primary"][0, 0] == pytest.approx(0.5)


def test_load_attrs_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OSM_ATTRS_PATH", str(tmp_path / "nope.csv.gz"))
    with pytest.raises(FileNotFoundError):
        osm_attrs.load_attrs()


def test_make_urban_grid_osm_uses_argmax_and_reports_coverage(built):
    """``source="osm"`` 时类别 = 实测占比的 argmax，并带上真实覆盖率。

    不能套 ``_assign_categories`` 的 TARGET_SHARE 配额——那会把实测占比
    改写成设计占比。
    """
    from mapus.env.urban import _URBAN_CACHE, make_urban_grid

    _URBAN_CACHE.clear()
    g = make_urban_grid("tdrive", 2, 2, source="osm")
    assert g.coverage == pytest.approx(0.5)
    np.testing.assert_array_equal(g.category, g.land_use.argmax(axis=2))
    assert g.category[1, 1] == LAND_USE.index("commercial")
    _URBAN_CACHE.clear()


def test_make_urban_grid_osm_rejects_non_square(built):
    """属性层按 n×n 构建，非方布局必须显式报错而不是悄悄切错。"""
    from mapus.env.urban import _URBAN_CACHE, make_urban_grid

    _URBAN_CACHE.clear()
    with pytest.raises(ValueError, match="方形网格"):
        make_urban_grid("grab", 8, 4, source="osm")
    _URBAN_CACHE.clear()


def test_make_urban_grid_synthetic_unchanged(built):
    """``source="synthetic"`` 必须与历史行为一致（覆盖率恒为 1）。"""
    from mapus.env.urban import _URBAN_CACHE, make_urban_grid

    _URBAN_CACHE.clear()
    g = make_urban_grid("tdrive", 4, 4, source="synthetic")
    assert g.coverage == 1.0
    np.testing.assert_allclose(g.land_use.sum(axis=2), np.ones((4, 4)), rtol=1e-9)
    _URBAN_CACHE.clear()
