# -*- coding: utf-8 -*-
"""E4.3 污染菜单的可复现性回归测试。

历史 bug：``poison_menu`` 用 ``id(worst)`` 判断「这条是不是诱饵」。``id()`` 是
内存地址，跨进程不可比、对象被回收后还会被复用，导致同一 ``(seed, eta, pmode)``
在不同批次给出不同的 ``uptake``，而 ``ratio`` / ``hit`` 不变——这种"只有部分指标
漂移"的故障极难察觉（表现为净吸收在不同批次间不可比）。

修复方式：改用由候选内容派生的稳定键 ``_ckey``，并给 ``min()`` / ``sorted()``
的并列情形加确定性兜底。本测试锁住修复后的行为。
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from aprime.model import Candidate  # noqa: E402
from run_llm_pollution import (  # noqa: E402
    _ckey,
    _forced_and_slots,
    poison_menu,
)


def _cand(wid: str, cells: set, sat: float) -> Candidate:
    return Candidate(wid=wid, path=((0, 0, 0),), cells=frozenset(cells), sat=sat)


def _fixture() -> dict:
    """6 位参与者、每人 6 条候选；覆盖数与满意度都可控，便于制造并列。"""
    by_w = {}
    for u in range(6):
        wid = f"w{u}"
        by_w[wid] = [
            _cand(wid, {(u, j), (u, j + 1)}, 0.9 - 0.1 * j) for j in range(6)
        ]
    return by_w


def test_ckey_ignores_object_identity():
    """数据相同但对象不同 ⇒ 同一身份。这是替换 id() 的核心保证。"""
    a = _cand("w0", {(0, 0), (0, 1)}, 0.5)
    b = _cand("w0", {(0, 0), (0, 1)}, 0.5)
    assert a is not b
    assert _ckey(a) == _ckey(b)


def test_ckey_distinguishes_worker_and_satisfaction():
    base = _cand("w0", {(0, 0)}, 0.5)
    assert _ckey(base) != _ckey(_cand("w1", {(0, 0)}, 0.5))
    assert _ckey(base) != _ckey(_cand("w0", {(0, 0)}, 0.6))
    assert _ckey(base) != _ckey(_cand("w0", {(0, 1)}, 0.5))


def test_poison_menu_random_is_deterministic():
    by_w = _fixture()
    p1, d1 = poison_menu(by_w, 0.5, "random", random.Random(23))
    p2, d2 = poison_menu(by_w, 0.5, "random", random.Random(23))
    assert d1 == d2
    for wid in by_w:
        assert [_ckey(c) for c in p1[wid]] == [_ckey(c) for c in p2[wid]]


def test_poison_menu_adversarial_is_deterministic():
    by_w = _fixture()
    p1, d1 = poison_menu(by_w, 0.5, "adversarial", random.Random(0))
    p2, d2 = poison_menu(by_w, 0.5, "adversarial", random.Random(0))
    assert d1 == d2
    for wid in by_w:
        assert [_ckey(c) for c in p1[wid]] == [_ckey(c) for c in p2[wid]]


def test_eta_zero_leaves_menu_untouched():
    by_w = _fixture()
    poisoned, keys = poison_menu(by_w, 0.0, "random", random.Random(23))
    assert keys == set()
    for wid in by_w:
        assert [_ckey(c) for c in poisoned[wid]] == [_ckey(c) for c in by_w[wid]]


def test_forced_independent_of_object_identity():
    """把诱饵换成"数据相同的新对象"（模拟跨进程重建），forced 必须不变。"""
    by_w = _fixture()
    poisoned, keys = poison_menu(by_w, 0.5, "random", random.Random(23))
    f1, s1 = _forced_and_slots(poisoned, keys, 3)

    rebuilt = {
        wid: [_cand(c.wid, set(c.cells), c.sat) for c in menu]
        for wid, menu in poisoned.items()
    }
    f2, s2 = _forced_and_slots(rebuilt, keys, 3)
    assert (f1, s1) == (f2, s2)


def test_forced_bounds():
    by_w = _fixture()
    for eta in (0.0, 0.5, 0.75):
        for mode in ("random", "adversarial"):
            poisoned, keys = poison_menu(by_w, eta, mode, random.Random(7))
            forced, slots = _forced_and_slots(poisoned, keys, 3)
            assert 0 <= forced <= slots


def test_poison_keeps_at_least_one_good_candidate():
    """η 再高也至少留一条真候选，否则菜单全是诱饵、实验失去意义。"""
    by_w = _fixture()
    poisoned, keys = poison_menu(by_w, 1.0, "random", random.Random(5))
    for menu in poisoned.values():
        n_decoy = sum(1 for c in menu if _ckey(c) in keys)
        assert n_decoy < len(menu)


_SNIPPET = '''
import random, sys, zlib
sys.path.insert(0, __ROOT__)
sys.path.insert(0, __EXP__)
from aprime.model import Candidate
from run_llm_pollution import _ckey, _forced_and_slots, poison_menu

def cand(wid, cells, sat):
    return Candidate(wid=wid, path=((0, 0, 0),), cells=frozenset(cells), sat=sat)

by_w = {}
for u in range(6):
    wid = "w%d" % u
    by_w[wid] = [cand(wid, {(u, j), (u, j + 1)}, 0.9 - 0.1 * j) for j in range(6)]

out = []
for mode in ("random", "adversarial"):
    p, d = poison_menu(by_w, 0.5, mode, random.Random(23))
    f, s = _forced_and_slots(p, d, 3)
    sig = "|".join(
        wid + ":" + ",".join(str(_ckey(c)) for c in menu)
        for wid, menu in sorted(p.items())
    )
    out.append("%s forced=%d slots=%d decoys=%d crc=%d"
               % (mode, f, s, len(d), zlib.crc32(sig.encode())))
print("\\n".join(out))
'''


def test_cross_process_stable_under_hash_seed():
    """换 PYTHONHASHSEED 重跑，输出必须逐字一致（跨进程确定性）。

    Python 默认随机化 str 的哈希，这会改变 set/dict 的迭代顺序。旧实现依赖
    ``id()`` 与输入顺序，因此在该测试下会漂移。
    """
    code = _SNIPPET.replace("__ROOT__", repr(ROOT)).replace(
        "__EXP__", repr(os.path.join(ROOT, "experiments")))
    outs = []
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=ROOT)
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout)
    assert outs[0] == outs[1] == outs[2], outs
