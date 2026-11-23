# -*- coding: utf-8 -*-
"""数据来源的机读证明：**合成数据不能流进"需要真实数据"的实验**。

## 为什么需要这个模块

`availability.py` 管的是"数据集在不在盘上"，它回答不了"**手上这个实例**是什么来源"。
典型事故是：数据全都在，脚本跑得通，于是有人用

    make_mapus_instance("tdrive_small", 0)      # 默认 source="synthetic"

跑出 E1.1 的数字，结果文件里 `dataset=tdrive`、文件名也带 `tdrive` 前缀，
但参与者是**随机抽的**。`data/README.md` 和 `docs/data_alternatives_grab_crowdsensim.md`
都专门警告过这件事，可它一直只靠"人记得"来防。

本模块把这件事变成代码里的不变式：

* `provenance_of(instance)` —— 从 `instance.meta` 里解析出各层的真实/合成身份；
* `require_real(instance, eid)` —— `requires_real_data=True` 的实验拿到合成实例直接抛错；
* `stamp(payload, ...)` / `require_stamped(payload)` —— 结果文件必须带 `provenance` 块。

## 两道闸门必须独立（关键设计）

`availability.require_available` 有一个开发用放行开关 `APRIME_ALLOW_SHELVED=1`。
**来源闸门绝不能复用它**——否则"为了调试临时放行"会顺手把"合成数据可以冒充真实数据"也放开。
故本模块用**另一个**环境变量 `APRIME_ALLOW_SYNTHETIC_IN_REAL=1`，且使用时会在 stderr
打横幅、并在 provenance 里记 `forced=True`，让事后审计看得见。

## 一个必须说清的口径

A′ 的实例由**三层**拼成，它们的真实程度不一样，不能一句话概括：

| 层 | 来源 | 现在是什么 |
|---|---|---|
| 参与者（`workers`） | T-Drive / GeoLife 真实出行段 | 可真实（`source="tdrive"`），也可合成（默认） |
| 城市属性（`urban`） | OSM 实测用地 + **合成犯罪层** | `urban_source="osm"` 时用地真实、犯罪仍合成 |
| 偏好向量（`profiles`） | **无任何真实来源**（AgentSense / MAPUS 都是合成的） | 恒为合成 |

`require_real` 只对**参与者**这一层较真——那是 E1.1/E1.2/E2.3/E3.1/SEC6 真正要的东西；
另外两层如实记录在 provenance 里，供论文的"数据可得性声明"引用，但不假装它们是真的。
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

#: 独立于 `APRIME_ALLOW_SHELVED` 的显式放行开关（见模块文档「两道闸门必须独立」）。
_ALLOW_SYNTHETIC_ENV = "APRIME_ALLOW_SYNTHETIC_IN_REAL"

#: 可作为"真实参与者来源"的标签前缀。与 `availability.DATASETS` 里 `real=True`
#: 的条目保持同序；这里写死一份是有意的——解析标签不该依赖 import 顺序，
#: 也不该因为某个数据集探测失败就把"曾经的真实来源"降级成合成。
REAL_WORKER_SOURCES = frozenset({"tdrive", "geolife", "gmission", "onrta_didi", "didi"})

#: 只提供**属性**、不可能充当参与者来源的数据集（防止被误当 worker 标签）。
ATTRIBUTE_ONLY_SOURCES = frozenset({"osm_attrs"})

#: 城市属性层的真实来源。`osm` 只代表**土地利用**实测；犯罪层仍是合成的，
#: 故归类为 real 但标签里必须保留 `crime=synthetic` 的信息（见 `urban_note`）。
REAL_URBAN_SOURCES = frozenset({"osm"})

#: `dense(...)` 标签里抓 `workers=` 的那段。必须带前置断言：`n_workers=160` 也含
#: 子串 `workers=`，若用 `re.search("workers=...")` 会先命中它，把人数当成来源标签。
_WORKERS_RE = re.compile(r"(?<![A-Za-z0-9_])workers=([^,)]+(?:\([^)]*\))?)")

KIND_REAL = "real"
KIND_SYNTHETIC = "synthetic"
KIND_UNKNOWN = "unknown"


class SyntheticDataError(RuntimeError):
    """把合成实例喂给了要求真实数据的实验。"""


@dataclass(frozen=True)
class Provenance:
    """一个实例的三层来源身份 + 由谁标定。"""

    #: `meta["source"]` 原文（不加工，便于与结果文件对账）
    label: str
    workers_kind: str
    workers_label: str
    urban_kind: str
    urban_label: str
    #: 偏好层恒为合成（无真实来源），如实记录而不是省略
    profiles_kind: str = KIND_SYNTHETIC
    profiles_label: str = "synthetic(no public survey source exists)"
    #: 被显式放行开关强制放行过（事后审计用）
    forced: bool = False
    #: 其他需要留痕的口径（slot_minutes / cells_per_step / urban_note …）
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def workers_real(self) -> bool:
        return self.workers_kind == KIND_REAL

    @property
    def synthetic(self) -> bool:
        """参与者是否为合成。**这是 `require_real` 唯一较真的判据**。"""
        return not self.workers_real

    @property
    def fully_real(self) -> bool:
        """三层全真。当前**不可能**为 True（偏好层没有真实来源），保留给未来。"""
        return (self.workers_real
                and self.urban_kind == KIND_REAL
                and self.profiles_kind == KIND_REAL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "workers": {"kind": self.workers_kind, "label": self.workers_label},
            "urban": {"kind": self.urban_kind, "label": self.urban_label},
            "profiles": {"kind": self.profiles_kind, "label": self.profiles_label},
            "forced": self.forced,
            "notes": dict(self.notes),
        }

    def describe(self) -> str:
        """一行摘要，可以直接贴进汇报/表注。"""
        return ("参与者=%s | 城市属性=%s | 偏好=%s"
                % (self.workers_label, self.urban_label, self.profiles_kind))


# --------------------------------------------------------------------- 解析
def _split_dense(label: str) -> Optional[str]:
    """从 ``dense(n_workers=..., workers=X, ...)`` 里取出 ``X``；不是 dense 则 None。"""
    m = _WORKERS_RE.search(label or "")
    return m.group(1).strip() if m else None


def _classify(token: str) -> str:
    """把来源标签归类成 real / synthetic / unknown。

    宁可判 unknown 也不猜 real：unknown 会让 `require_real` 拦下来，
    而误判成 real 会直接污染论文。
    """
    t = (token or "").strip()
    if not t:
        return KIND_UNKNOWN
    low = t.lower()
    if "synthetic" in low:
        return KIND_SYNTHETIC
    base = low.split("(")[0].strip()
    if base in ATTRIBUTE_ONLY_SOURCES:
        return KIND_SYNTHETIC
    if base in REAL_WORKER_SOURCES:
        return KIND_REAL
    return KIND_UNKNOWN


def _classify_urban(token: str) -> str:
    """城市属性层用另一套词汇：`osm` = 用地实测（犯罪层仍合成）。"""
    t = (token or "").strip().lower()
    if not t:
        return KIND_UNKNOWN
    if "synthetic" in t or "默认" in t:
        return KIND_SYNTHETIC
    return KIND_REAL if t.split("(")[0].strip() in REAL_URBAN_SOURCES else KIND_UNKNOWN


def _meta_of(instance) -> Mapping[str, Any]:
    """实例对象可能是 `Instance`、`MapusInstance` 或带 `.base` 的包装。"""
    base = getattr(instance, "base", instance)
    meta = getattr(base, "meta", None)
    return meta if isinstance(meta, Mapping) else {}


def provenance_of(instance) -> Provenance:
    """解析一个实例的层来源。对未知标签 fail-closed（判 synthetic/unknown，不判 real）。"""
    meta = _meta_of(instance)
    label = str(meta.get("source", "") or "")
    dense_workers = _split_dense(label)
    workers_label = dense_workers if dense_workers is not None else label

    # 专门覆盖：`make_dense_instance` 已经把 worker 来源单独写进 `worker_source`
    ws = meta.get("worker_source")
    if isinstance(ws, str) and ws.strip():
        workers_label = ws.strip()

    urban_source = meta.get("urban_source")
    if isinstance(urban_source, str) and urban_source.strip():
        urban_label = urban_source.strip()
    else:
        # 未显式标注的时刻意默认 `make_urban_grid(source="synthetic")`
        urban_label = "synthetic(默认)"
    urban_kind = _classify_urban(urban_label)

    # 解析城市属性层的两个子层（土地利用 / 犯罪），把混合状态写进标签。
    urban_note = str(meta.get("urban_note", ""))
    lu_real = urban_kind == KIND_REAL  # "osm" 等
    crime_real = ("crime=real" in urban_note) or ("张等2025" in urban_note)
    lu_tag = "landuse=实测" if lu_real else "landuse=synthetic"
    crime_tag = "crime=实测(张等2025)" if crime_real else "crime=synthetic"
    if lu_real or crime_real:
        # 只要有一个子层是实测，就把两子层状态都写进标签，不让它藏在 notes 里
        urban_label = "%s(%s, %s)" % (urban_label, lu_tag, crime_tag)

    notes: Dict[str, Any] = {}
    for key in ("slot_minutes", "horizon_minutes", "cells_per_step",
                "cells_per_step_source", "urban_note", "dataset", "scale"):
        if key in meta:
            notes[key] = meta[key]

    return Provenance(
        label=label or "<未标注>",
        workers_kind=_classify(workers_label),
        workers_label=workers_label or "<未标注>",
        urban_kind=urban_kind,
        urban_label=urban_label,
        notes=notes,
    )


def is_synthetic(instance) -> bool:
    return provenance_of(instance).synthetic


# --------------------------------------------------------------------- 闸门
def require_real(instance, eid: str = "") -> Provenance:
    """要求**参与者**来自真实轨迹，否则抛 `SyntheticDataError`。

    与 `availability.require_available` 的分工：那个管"数据集在不在"，
    这个管"手上这个实例是不是真的"。两者都不满足才叫真的跑不了。

    放行：``APRIME_ALLOW_SYNTHETIC_IN_REAL=1``（仅调试；会在 stderr 打横幅、
    并在返回的 provenance 里记 ``forced=True``，事后可查）。
    """
    prov = provenance_of(instance)
    if prov.workers_real:
        return prov

    if os.environ.get(_ALLOW_SYNTHETIC_ENV):
        print("[provenance] ⚠ 已用 %s 强制放行：实验 %s 拿到的是**合成参与者**"
              "（source=%r）。此结果**不得**作为真实数据结果写入论文。"
              % (_ALLOW_SYNTHETIC_ENV, eid or "?", prov.workers_label),
              file=sys.stderr)
        return Provenance(
            label=prov.label,
            workers_kind=prov.workers_kind,
            workers_label=prov.workers_label,
            urban_kind=prov.urban_kind,
            urban_label=prov.urban_label,
            profiles_kind=prov.profiles_kind,
            profiles_label=prov.profiles_label,
            forced=True,
            notes=dict(prov.notes),
        )

    where = ("make_mapus_instance(..., source='tdrive') / "
             "make_dense_instance(..., source='tdrive'|'geolife')")
    raise SyntheticDataError(
        "实验 %s 要求**真实参与者**，但实例的参与者来源是 %s。\n"
        "  实例标签：%s\n"
        "  修正：显式传 %s\n"
        "  说明：仅换成 name/prefix 不算——文件名叫 `tdrive_*` 而 `meta['source']` 是 "
        "`synthetic(...)` 正是本模块要拦的经典事故（见 data/README.md §0）。\n"
        "  若确实要试跑：设环境变量 %s=1（结果会被标记 forced=True）。\n"
        "  背景：docs/data_alternatives_grab_crowdsensim.md §8、"
        "docs/e0_1_data_access_plan.md §8"
        % (eid or "?", prov.workers_label, prov.label, where, _ALLOW_SYNTHETIC_ENV)
    )


# --------------------------------------------------------------------- 结果戳
#: 时间/随机性无法证伪，但至少要能追到"哪个脚本、什么口径"。
_REQUIRED_KEYS = ("label", "workers", "urban", "profiles")


def stamp(payload: Dict[str, Any], instance=None, prov: Optional[Provenance] = None,
          models: Optional[Mapping[str, Any]] = None,
          assumed: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """给结果 payload 盖上 `provenance` 块，**原地写回并返回同一个 dict**。

    参数
    ----
    instance : 产生该结果的实例（取三层的真实/合成身份）
    models   : 本次用到但**不可观测**的模型假设，例如 ``{"incentive": "dist_cost"}``
    assumed  : 参数取值与其依据，例如 ``{"per_cell": 0.45}``

    如果既不传 ``instance`` 也不传 ``prov``，会用一个“payload 级占位”
    兜底——只用于没有具体实例的元结果文件，正文不得引用它作为数据来源。
    """
    if prov is None:
        if instance is not None:
            prov = provenance_of(instance)
        else:
            prov = Provenance(
                label="<payload-level annotation>",
                workers_kind=KIND_UNKNOWN,
                workers_label="payload-level(no instance provenance)",
                urban_kind=KIND_UNKNOWN,
                urban_label="—",
                notes={"note": "provenance 由元结果脚本补齐，非具体实例的数据身份"},
            )
    block = prov.to_dict()
    block["models"] = dict(models or {})
    block["assumed"] = dict(assumed or {})
    payload["provenance"] = block
    return payload


def require_stamped(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """校验结果文件带齐了 provenance。缺了就不许落盘。"""
    block = payload.get("provenance") if isinstance(payload, Mapping) else None
    if not isinstance(block, Mapping):
        raise SyntheticDataError(
            "结果 payload 缺 `provenance` 块；请用 aprime.provenance.stamp(...) 包一层。"
            "（这是为了防止合成数字在没有来源标注的情况下流进论文）")
    missing = [k for k in _REQUIRED_KEYS if k not in block]
    if missing:
        raise SyntheticDataError("provenance 块缺字段：%s" % ", ".join(missing))
    return block


def summary(instance) -> str:
    return provenance_of(instance).describe()
