# -*- coding: utf-8 -*-
"""数据与实验的可用性登记表：把"拿不到数据"变成可执行的状态。

背景见 ``docs/e0_1_data_access_plan.md``。E0.1（真实时间戳接入）停在 D0，依赖真实数据的
实验全部搁置；本模块是**唯一的状态来源**，目的是防止"某个脚本忘了改默认值就悄悄跑出
合成数字"，以及防止把合成结果当成真实数据复现写进论文。

三种状态：

* ``ready``    —— 不依赖缺失数据、且未被搁置，可以跑；
* ``deferred`` —— 数据在磁盘上，但按本次决定暂不产出真实数据结果（可通过
  ``SHELVE_REAL_DATA = False`` 解除）；
* ``blocked``  —— 缺数据或缺管线，跑不了，必须满足解冻条件。

用法::

    from aprime.availability import require_available, status_of, shelved
    require_available("E2.1")   # ready，正常通过
    require_available("E0.1")   # 抛 DataUnavailable，消息里带解冻条件
    for eid, st in shelved():
        print(eid, st)

放行开关：``APRIME_ALLOW_SHELVED=1`` 只跳过拦阻，不改变状态判定本身
（解冻前试跑用；解冻应改 ``SHELVE_REAL_DATA`` / 补齐数据）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

#: 全局闸门：真实数据实验是否暂停产出结果。
#:
#: 2026-11-20 由 True 改为 False：E0.1 改由 T-Drive 承载后，A1–A8 硬门全部通过，
#: 签收凭据落在 ``data/tdrive/manifest.json``（``time_field_verdict`` 以 pass 开头，
#: 由 ``data/tdrive/audit_time_field.py`` 可重跑）。闸门的作用是"防止在验收未通过时
#: 产出真实数据结果"，验收通过后继续关着就变成无意义的自我阻塞。
#:
#: 注意：置 False 只解除 ``deferred``，``blocked``（数据/管线缺失）仍然拦住。
SHELVE_REAL_DATA = False

#: 显式放行开关（仅用于解冻前的试跑）。例如：
#:   $env:APRIME_ALLOW_SHELVED=1; py -3.14 experiments/run_theta_curve.py
_ALLOW_ENV = "APRIME_ALLOW_SHELVED"

_TDRIVE_DEFAULT = r"D:\composition\data\tdrive\tdrive_clean.txt.gz"
_GMISSION_DIR = r"D:\composition\data\gmission"
_DIDI_PATH = r"D:\composition\data\sc_dense\raw\real_didi\1000\data_1000.txt"
_ONRTA_SYNTH_PATH = (r"D:\composition\data\sc_dense\raw\synthetic_onrta"
                     r"\3000_15000_1_10_0.5_6_10\data_00.txt")
_GEOLIFE_DEFAULT = r"D:\composition\data\geolife\geolife_norm.txt.gz"
#: 道路/POI 属性重建管线的产出物（B1 OSM+OSMnx / B2 Overture Places）。
_OSM_ATTRS_PATH = r"D:\composition\data\osm_attrs\attrs.csv.gz"
#: E0.1 的签收凭据（T-Drive 承载）。由 ``data/tdrive/audit_time_field.py`` 实测生成。
_TDRIVE_MANIFEST = r"D:\composition\data\tdrive\manifest.json"


class DataUnavailable(RuntimeError):
    """尝试运行一个被搁置（blocked / deferred）的实验。"""


@dataclass(frozen=True)
class Dataset:
    name: str
    #: 是否为真实世界数据（True）还是合成/同分布替身（False）。
    real: bool
    #: 就位判据。用函数而非常量，避免"目录存在但数据是空的"被当成可用。
    probe: Callable[[], bool]
    reason: str
    plan: str = ""


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    #: 依赖的数据集名（``DATASETS`` 的键）。
    datasets: Tuple[str, ...] = ()
    #: 是否必须真实数据才能成立。合成实例上跑出来的东西不算它的结果。
    requires_real_data: bool = False
    #: 非数据类阻塞项（未实现的管线、未获得授权的原文等）。
    blockers: Tuple[str, ...] = ()
    note: str = ""
    #: 解冻条件，写进异常消息，避免"搁置"变成"遗忘"。
    unblock: str = ""


def _tdrive_path() -> str:
    try:  # 与 mapus 保持同一来源（含 TDRIVE_PATH 环境变量覆盖）
        from mapus.data.instances import TDRIVE_PATH

        return TDRIVE_PATH
    except Exception:
        return os.environ.get("TDRIVE_PATH", _TDRIVE_DEFAULT)


def _probe_tdrive() -> bool:
    path = _tdrive_path()
    return bool(path) and os.path.exists(path)


def _probe_tdrive_verdict() -> bool:
    """E0.1 的签收凭据：T-Drive 的 ``time_field_verdict`` 必须以 ``pass`` 开头。

    只看 ``tdrive_clean.txt.gz`` 在不在是不够的——那只证明"文件存在"，
    不证明"时间字段验收通过"。凭据由 ``data/tdrive/audit_time_field.py``
    对 A1–A9 硬门实测生成，可重跑。
    """
    path = os.environ.get("TDRIVE_MANIFEST", _TDRIVE_MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            verdict = str(json.load(fh).get("time_field_verdict", "")).strip().lower()
    except Exception:
        return False
    return verdict.startswith("pass")


def _probe_didi() -> bool:
    return os.path.exists(_DIDI_PATH)


def _probe_onrta_synthetic() -> bool:
    return os.path.exists(_ONRTA_SYNTH_PATH)


def _probe_osm_attrs() -> bool:
    """道路/POI 属性重建层是否已产出。

    判据是**产出物**而非"装了 osmnx"：管线只要跑过就该留下这张表。
    """
    path = os.environ.get("OSM_ATTRS_PATH", _OSM_ATTRS_PATH)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _probe_geolife() -> bool:
    """GeoLife 是否可用：归一化文件在位且非空。

    只看原始 zip 不算——未归一化的 GeoLife 是 1–5 s 采样，重复 slot 会让下游
    ``_contiguous_runs`` 全断（实测 363 万点只切出 15 条出行段），等于不可用。
    """
    try:
        from mapus.data.instances import _geolife_source

        path = _geolife_source()
    except Exception:
        path = os.environ.get("GEOLIFE_PATH", _GEOLIFE_DEFAULT)
        if not os.path.exists(path):
            path = None
    return bool(path) and os.path.getsize(path) > 0


def _probe_gmission() -> bool:
    """gMission 到位的唯一判据：``manifest.json`` 存在、``raw/`` 非空、
    **且 ``time_field_verdict`` 为 ``pass``**。

    只看目录是否存在会把"建了个空目录"误判为可用；只看 manifest 存在则会
    把"写了骨架但没验收"误判为解冻——而 ``docs/e0_1_data_access_plan.md`` §5
    明确写着"``time_field_verdict`` 是 E0.1 的签收字段，没有它 E0.1 视为未完成"。
    故这里按契约校验取值，而不是只校验文件在不在。

    **注意**：`raw/` 下的原始快照其实已经在位（见 ``data/README.md``），
    本判据管的是"E0.1 时间戳接入是否签收"，不是"原始文件是否存在"。
    当前 manifest 的 verdict 是 ``partial``（D1 实测已做、D2–D5 未做），
    故仍返回 False——这是**如实的未接入**，不是假阻塞。
    """
    manifest = os.path.join(_GMISSION_DIR, "manifest.json")
    raw = os.path.join(_GMISSION_DIR, "raw")
    if not (os.path.exists(manifest) and os.path.isdir(raw)):
        return False
    if not any(True for _ in os.scandir(raw)):
        return False
    try:
        with open(manifest, "r", encoding="utf-8") as fh:
            verdict = str(json.load(fh).get("time_field_verdict", "")).strip().lower()
    except Exception:
        return False
    return verdict.startswith("pass")


DATASETS: Dict[str, Dataset] = {
    "tdrive": Dataset(
        name="tdrive",
        real=True,
        probe=_probe_tdrive,
        reason="T-Drive 清洗文件在位，真实轨迹通路已实测可用",
        plan="docs/e0_1_data_access_plan.md §2.1",
    ),
    "tdrive_verdict": Dataset(
        name="tdrive_verdict",
        real=True,
        probe=_probe_tdrive_verdict,
        reason="E0.1 的签收凭据：data/tdrive/manifest.json 的 time_field_verdict 必须以 "
               "pass 开头。该凭据由 data/tdrive/audit_time_field.py 对 A1–A8 硬门实测生成，"
               "可重跑（由 audit 脚本按运行时实测值填写）",
        plan="docs/e0_1_data_access_plan.md §5/§7",
    ),
    "gmission": Dataset(
        name="gmission",
        real=True,
        probe=_probe_gmission,
        reason="原始快照已在位（gMission_cap1 / EverySender_cap1），但缺 E0.1 签收用的 "
               "manifest.json（time_field_verdict），故仍判为未接入",
        plan="docs/e0_1_data_access_plan.md §4",
    ),
    "onrta_didi": Dataset(
        name="onrta_didi",
        real=True,
        probe=_probe_didi,
        reason="ONRTA 的 Didi 快照在位（874 工作者 / 5000 任务，覆盖率 11.5%），"
               "E5.9b 用作第三份真实对照",
        plan="data/README.md §2",
    ),
    "grab": Dataset(
        name="grab",
        real=True,
        probe=lambda: False,
        reason="Grab-Posisi 需**邮件**授权（grab.posisi@grabtaxi.com，非 Kaggle），"
               "且论文里的 13 项道路/POI 属性层**从未随数据发布**——即使拿到轨迹，"
               "S(·) 的语义属性仍然缺位。现有 grab_* 实例为合成",
        plan="docs/data_alternatives_grab_crowdsensim.md §2",
    ),
    "geolife": Dataset(
        name="geolife",
        real=True,
        probe=_probe_geolife,
        reason="GeoLife 1.3（MSRA，182 用户 / 北京 / 2007-04…2012-08）已归一化为 "
               "T-Drive 四列格式并落盘：10,055 条（用户×日）轨迹、98,709 个 15 分钟槽点，"
               "加载 0.8 s、可切出 11k–15k 条出行段。作为 Grab-Posisi 的轨迹替代",
        plan="data/README.md §8；docs/data_alternatives_grab_crowdsensim.md §3.1",
    ),
    "onrta_synthetic": Dataset(
        name="onrta_synthetic",
        real=False,
        probe=_probe_onrta_synthetic,
        reason="ONRTA 合成匹配快照（3000 工作者 / 15000 任务 / 28218 可行边，"
               "|F_u| 中位 9）——README 标为'主实验'，是唯一满足 MRG 设计动机的稠密形状；"
               "但它是合成数据，只支持机制",
        plan="data/README.md §3",
    ),
    "osm_attrs": Dataset(        name="osm_attrs",
        real=True,
        probe=_probe_osm_attrs,
        reason="道路/POI 属性层已用 OpenStreetMap 重建：研究范围内 69,247 条道路、"
               "15,561 个用地多边形、85,823 条兴趣点/建筑，产出 8/16/32 三种分辨率的"
               "1344 行属性表。土地利用按**面积**（128×128 细网格切分）与兴趣点个数"
               "各半合成；道路等级构成、交叉口密度、车道/限速均值齐备。"
               "**注意：犯罪层仍是合成的**——北京无公开细粒度犯罪数据",
        plan="data/README.md §9；docs/data_alternatives_grab_crowdsensim.md §3.2",
    ),
}

EXPERIMENTS: Dict[str, Experiment] = {
    "E0.1": Experiment(
        id="E0.1",
        title="真实时间戳数据接入与验收（T-Drive 承载）",
        datasets=("tdrive", "tdrive_verdict"),
        requires_real_data=True,
        note="gMission 与 EverySender 快照实测为 worker–task 匹配型（归一化 0–5 坐标、"
             "整数秒相对偏移、无时区/纪元、无日期字段、每个 worker 只出现一次），"
             "A2/A5/A6 结构性不可满足、A7 不适用，故**不能**承载轨迹级 E0.1，已降级为 "
             "E0.1b（仅匹配级，由 E5.9b 消费）。E0.1 改由 T-Drive 承载：轨迹带经纬度与"
             "明确时区、跨 7 个自然日、通行路线管线已实测可用"
             "（见 docs/e0_1_data_access_plan.md §10 的退路）",
        unblock="data/tdrive/manifest.json 的 time_field_verdict 以 pass 开头"
                "（重跑：py -3.14 data/tdrive/audit_time_field.py）",
    ),
    "E0.1b": Experiment(
        id="E0.1b",
        title="gMission 匹配级接入（轨迹级 E0.1 的降级替代）",
        datasets=("gmission",),
        requires_real_data=True,
        note="这份快照只支持 worker–task 匹配型实验（E5.9b）。它的 time_field_verdict 是 "
             "partial：100% 的 begTime 可解析为整数秒，但无时区/纪元、无日期字段、"
             "坐标为归一化 0–5 平面而非经纬度。**这是如实的未接入，不是待办漏项**——"
             "要签 pass 必须换数据源，不能在 gMission 上补手续",
        unblock="换一份带经纬度、已发布时区/纪元与日期字段的匹配快照；"
                "这三项在 gMission/EverySender 上都结构性缺失（见 data/gmission/manifest.json）",
    ),
    "E1.1": Experiment(
        id="E1.1",
        title="θ 权衡曲线（真实轨迹版）",
        datasets=("tdrive",),
        requires_real_data=True,
        note="现版 run_theta_curve.py 走 make_mapus_instance() 默认合成参与者；"
             "gMission 版本属于 E0.1 解冻后的追加重跑",
        unblock="真实实例冒烟通过 + 同 seed/同候选池重跑",
    ),
    "E1.2": Experiment(
        id="E1.2",
        title="大规模密度（500→5000 workers 的真实批量管线）",
        datasets=("tdrive",),
        requires_real_data=True,
        blockers=("500→5000 workers 的批量管线未实现",),
        note="aprime/dense.py 明确写着不要当作规模复现",
        unblock="批量管线就绪 + 真实参与者来源",
    ),
    "E2.1": Experiment(
        id="E2.1",
        title="界的紧性（精确 α 枚举）",
        note="run_bound_tightness.py 用抽象实例，不涉及任何数据",
    ),
    "E2.2": Experiment(
        id="E2.2",
        title="界紧性的补充配置",
        note="同 E2.1，抽象实例",
    ),
    "E2.3": Experiment(
        id="E2.3",
        title="α(θ) 的真实实例分析",
        datasets=("tdrive",),
        requires_real_data=True,
        note="与 E1.1 同批；α 已降级为实例诊断量",
        unblock="同 E1.1",
    ),
    "E3.1": Experiment(
        id="E3.1",
        title="低能力子群（θ 更严）",
        datasets=("tdrive",),
        requires_real_data=True,
        note="config.py 已留 per_worker_theta 接口，缺真实能力信号",
        unblock="真实能力分层可用",
    ),
    "E5.1": Experiment(
        id="E5.1",
        title="exclusion_gap 的统计功效（指标定义缺陷）",
        datasets=("tdrive",),
        requires_real_data=True,
        note="合成机制实验：能力高度并列时该指标不可测（medium 并列率 0.63），"
             "见 results/excl_gap_power.md；回答 §6.11 开放问题 1",
        unblock="同 SEC6：G1/G2/G3 三条解冻 gate 同时满足",
    ),
    "E5.9b": Experiment(
        id="E5.9b",
        title="三份真实快照预检（gMission / EverySender / Didi）与跨快照对照",
        note="直接读 data/gmission/raw/*/data_00.txt 与 data/sc_dense/raw/real_didi/1000/，"
             "不经过 E0.1 的 manifest 签收；产物 results/preflight_<快照>.md 与 "
             "results/preflight_compare.md",
    ),
    "E5.10": Experiment(
        id="E5.10",
        title="命题 4 两个闭式阈值的三段验证",
        note="合成实例，θ 网格扩到 1.3；结论：theta_any 恒 >1、在 [0,1] 内不可达",
    ),
    "E5.11": Experiment(
        id="E5.11",
        title="偏好异质性 φ 与 θ 的联合定标",
        note="合成实例，7 档 φ × 10 档 θ；结论：可行性可定标、覆盖不可定标",
    ),
    "E5.12": Experiment(
        id="E5.12",
        title="路线级建模收益随 θ 的衰减律",
        note="合成实例，19 档 θ × 2 尺度；结论：方向由预算松弛度决定",
    ),
    "E5.13": Experiment(
        id="E5.13",
        title="真实参与者复核：E5.10–E5.12 换 source='tdrive' 重跑",
        datasets=("tdrive",),
        note="参与者改为 T-Drive 真实出行段（tdrive_workers），栅格/时槽/速度口径不变；"
             "偏好与城市属性仍是合成的。产物 results/*_tdrive.md",
    ),
    "E5.14": Experiment(
        id="E5.14",
        title="稠密匹配快照上的 MRG 主实验（ONRTA，10 θ × 8 种子）",
        datasets=("onrta_synthetic",),
        note="用 README 标为'主实验'但此前无人引用的 synthetic_onrta；"
             "种子由工作者子采样 50% 制造。产物 results/gqc_mrg_onrta.md",
    ),
    "E5.15": Experiment(
        id="E5.15",
        title="偏好先验的敏感性界（人群构成 × jitter × mix_uniform）",
        datasets=("tdrive",),
        note="偏好向量与人群构成权重均无调查数据支撑，故把未标定的部分当自变量扫，"
             "界定哪些结论先验无关。含 --space calibrated：第二产业权重按北京年鉴 2023 "
             "边缘校准（16.43%/15.13%），其余 4 项仍自由。"
             "产物 results/pref_mixture_sensitivity_*.md",
    ),
    "SEC6": Experiment(
        id="SEC6",
        title="§6 表 6-1…6-9 的真实数据复现批",
        datasets=("tdrive",),
        requires_real_data=True,
        note="现有数字均为合成参与者，真实版待重跑",
        unblock="G1/G2/G3 三条解冻 gate 同时满足",
    ),
    "CROSS": Experiment(
        id="CROSS",
        title="跨数据集验证（真实稠密轨迹 + 重建属性层）",
        datasets=("geolife", "osm_attrs"),
        requires_real_data=True,
        note="Grab-Posisi 官方轨迹需邮件授权、其 13 项道路/POI 属性层从未公开发布，故改设计："
             "轨迹用 GeoLife 1.3（已归一化落盘）；属性层用 OpenStreetMap 重建"
             "（道路等级/交叉口/车道/限速 + 用地面积占比 + 兴趣点类别）。"
             "两侧数据均已就位，只剩全局闸门。Porto（UCI #339）本机实测仅 19 KB/s 且"
             "不支持 Range 续传，本轮未下完，叫车语义侧仍缺。"
             "详见 docs/data_alternatives_grab_crowdsensim.md §4",
        unblock="数据侧已满足（geolife=ok、osm_attrs=ok）；剩余条件只有 E0.1 解冻后"
                "解除 SHELVE_REAL_DATA。**不再以'拿到 Grab 官方轨迹 + POI'为条件——"
                "该条件实际不可满足**",
    ),
}


def dataset_available(name: str) -> bool:
    ds = DATASETS.get(name)
    if ds is None:
        raise KeyError("未登记的数据集: %r（请先在 DATASETS 里声明）" % name)
    return bool(ds.probe())


def available_datasets() -> Tuple[str, ...]:
    return tuple(n for n, ds in DATASETS.items() if ds.probe())


def blockers_of(eid: str) -> Tuple[str, ...]:
    exp = _experiment(eid)
    items: List[str] = list(exp.blockers)
    for name in exp.datasets:
        if not dataset_available(name):
            items.append("数据集 %s 不可用：%s" % (name, DATASETS[name].reason))
    return tuple(items)


def status_of(eid: str) -> str:
    """返回 ``ready`` / ``deferred`` / ``blocked``。"""
    exp = _experiment(eid)
    if blockers_of(eid):
        return "blocked"
    if exp.requires_real_data and SHELVE_REAL_DATA:
        return "deferred"
    return "ready"


def is_runnable(eid: str) -> bool:
    return status_of(eid) == "ready"


def require_available(eid: str, instance=None) -> None:
    """搁置中的实验直接抛错，并带上解冻条件。

    ``APRIME_ALLOW_SHELVED=1`` 可显式放行（仅用于解冻前的试跑/调试），
    放行不会改变状态判定本身，只是不拦。

    ``instance`` 给定时追加**来源闸门**（``aprime.provenance.require_real``）：
    `requires_real_data=True` 的实验拿到合成参与者的实例会直接抛错。
    这道闸门**故意放在 ``_ALLOW_ENV`` 早退之前**——搁置开关是"允许跑"，
    不是"允许拿合成数据冒充真实数据"，两者必须独立（见 provenance 模块文档）。
    """
    exp = _experiment(eid)
    if instance is not None and exp.requires_real_data:
        from .provenance import require_real

        require_real(instance, eid)
    if os.environ.get(_ALLOW_ENV):
        return
    status = status_of(eid)
    if status == "ready":
        return
    exp = _experiment(eid)
    lines = ["实验 %s（%s）当前状态：%s，不能运行。" % (exp.id, exp.title, status)]
    if exp.note:
        lines.append("说明：%s" % exp.note)
    for item in blockers_of(eid):
        lines.append("阻塞：%s" % item)
    if status == "deferred":
        lines.append("搁置原因：E0.1 阻塞期间暂停产出真实数据结果"
                     "（解除开关：aprime.availability.SHELVE_REAL_DATA = False）")
    if exp.unblock:
        lines.append("解冻条件：%s" % exp.unblock)
    lines.append("详见 docs/e0_1_data_access_plan.md §8")
    raise DataUnavailable("\n".join(lines))


def shelved() -> Tuple[Tuple[str, str], ...]:
    """当前不可运行的实验清单 ``(id, status)``，按 id 排序。"""
    return tuple((eid, status_of(eid)) for eid in sorted(EXPERIMENTS)
                 if status_of(eid) != "ready")


def summary() -> str:
    """给内部汇报用的一段纯文本状态表。"""
    width = max(len(e.id) for e in EXPERIMENTS.values())
    rows = ["%-*s  %-9s  %s" % (width, "ID", "STATUS", "TITLE")]
    for eid in sorted(EXPERIMENTS):
        rows.append("%-*s  %-9s  %s" % (width, eid, status_of(eid),
                                        EXPERIMENTS[eid].title))
    rows.append("")
    rows.append("数据集：%s" % ", ".join(
        "%s=%s" % (n, "ok" if ds.probe() else "missing")
        for n, ds in DATASETS.items()))
    return "\n".join(rows)


def _experiment(eid: str) -> Experiment:
    try:
        return EXPERIMENTS[eid]
    except KeyError:
        raise KeyError("未登记的实验: %r（请先在 EXPERIMENTS 里声明）" % eid) from None
