# -*- coding: utf-8 -*-
"""真实 LLM 候选生成器的离线测试：用本地 mock 服务器验证整条链路。

不消耗任何 token，也不访问外网；验证的是提示词确定性、响应解析、缓存命中、
补足披露与凭据缺失时的显式失败。
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aprime  # noqa: F401,E402
from aprime.llm_gen import (
    LLMConfig,
    LLMConfigError,
    MenuSelector,
    build_prompt,
    parse_picks,
)
from aprime.model import Candidate  # noqa: E402


def _c(wid: str, cells, sat: float) -> Candidate:
    xy = [(int(c[0]), int(c[1])) for c in cells]
    path = tuple((x, y, i) for i, (x, y) in enumerate(xy))
    return Candidate(wid=wid, path=path, cells=frozenset(xy), sat=sat,
                     space_time=frozenset((x, y, i) for i, (x, y) in enumerate(xy)))


def _menu():
    return [
        _c("w0", [(0, 0), (0, 1), (0, 2)], 0.90),
        _c("w0", [(1, 0), (1, 1)], 0.70),
        _c("w0", [(2, 0), (2, 1), (2, 2), (2, 3)], 0.55),
        _c("w0", [(3, 0)], 0.40),
    ]


# ------------------------------------------------------------------ 纯函数
def test_prompt_is_deterministic():
    m = _menu()
    a = build_prompt("w0", m, 2, theta=0.3, phi=0.95)
    b = build_prompt("w0", m, 2, theta=0.3, phi=0.95)
    assert a == b
    assert "至多 2 条" in a
    assert "[3] 覆盖 1 格" in a


def test_prompt_changes_with_k():
    m = _menu()
    assert build_prompt("w0", m, 2) != build_prompt("w0", m, 3)


@pytest.mark.parametrize("text,expect", [
    ('{"pick": [1, 3]}', [1, 3]),
    ('```json\n{"pick": [0, 2]}\n```', [0, 2]),
    ("好的，我选 [1,2]", [1, 2]),
    ('{"pick": [1, 1, 9, -1, 3]}', [1, 3]),      # 去重 + 越界丢弃
    ('{"pick": [0,1,2,3]}', [0, 1]),             # 超过 k 截断
    ("完全无法解析", []),
    ("", []),
])
def test_parse_picks(text, expect):
    assert parse_picks(text, n_menu=4, k=2) == expect


# ------------------------------------------------------------------ mock 服务
class _Handler(BaseHTTPRequestHandler):
    calls = 0
    reply = '{"pick": [1, 3]}'
    last_payload = None

    def do_POST(self):  # noqa: N802
        _Handler.calls += 1
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            _Handler.last_payload = json.loads(raw.decode())
        except Exception:  # noqa: BLE001
            _Handler.last_payload = None
        body = json.dumps({
            "choices": [{"message": {"content": _Handler.reply}}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 9},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 静音
        pass


@pytest.fixture
def mock_server():
    _Handler.calls = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _cfg(base_url: str, cache_dir: str) -> LLMConfig:
    return LLMConfig(base_url=base_url, model="mock-1", api_key_env="MOCK_KEY",
                     cache_dir=cache_dir, max_retries=2, timeout=5.0,
                     trust_env=False)  # loopback 必须绕过系统代理


# ------------------------------------------------------------------ 端到端
def test_end_to_end_and_cache(mock_server, tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_KEY", "sk-test")
    sel = MenuSelector(_cfg(mock_server, str(tmp_path / "cache")))
    m = _menu()

    r1 = sel.select("w0", m, 3, theta=0.0)
    assert r1.ok and r1.picks == [1, 3]
    assert r1.prompt_tokens == 128 and r1.completion_tokens == 9
    assert _Handler.calls == 1

    # 第二次必须命中缓存，不再发请求
    r2 = sel.select("w0", m, 3, theta=0.0)
    assert r2.picks == [1, 3] and r2.cached is True
    assert _Handler.calls == 1, "缓存未生效，重复烧 token"

    # 换 k 应产生新缓存键
    sel.select("w0", m, 2, theta=0.0)
    assert _Handler.calls == 2

    sel.close()


def test_padding_is_disclosed(mock_server, tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_KEY", "sk-test")
    sel = MenuSelector(_cfg(mock_server, str(tmp_path / "c")))
    rec = {}
    got = sel.select_candidates("w0", _menu(), 4, record=rec)
    sel.close()
    assert len(got) == 4
    # LLM 只给了 2 条，补 2 条；必须如实记录
    assert rec["n_padded"] == 2
    assert rec["llm_picks"] == [1, 3]


def test_prompt_styles_differ_and_are_stable():
    m = _menu()
    cov = build_prompt("w0", m, 2, style="coverage")
    div = build_prompt("w0", m, 2, style="diverse")
    assert cov != div
    assert build_prompt("w0", m, 2, style="coverage") == cov   # 仍需确定性
    assert "并集" in cov
    with pytest.raises(ValueError):
        build_prompt("w0", m, 2, style="no-such-style")


def test_cache_is_style_isolated(mock_server, tmp_path, monkeypatch):
    """不同 prompt 风格必须各自缓存，否则会串味。"""
    monkeypatch.setenv("MOCK_KEY", "sk-test")
    sel = MenuSelector(_cfg(mock_server, str(tmp_path / "c")))
    sel.select("w0", _menu(), 2, style="coverage")
    sel.select("w0", _menu(), 2, style="diverse")
    sel.close()
    assert _Handler.calls == 2


def test_missing_api_key_raises(mock_server, tmp_path, monkeypatch):
    monkeypatch.delenv("MOCK_KEY", raising=False)
    sel = MenuSelector(_cfg(mock_server, str(tmp_path / "c")))
    with pytest.raises(LLMConfigError):
        sel.select("w0", _menu(), 2)
    sel.close()


def test_unparseable_reply_falls_back(mock_server, tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_KEY", "sk-test")
    _Handler.reply = "抱歉，我无法完成。"
    try:
        sel = MenuSelector(_cfg(mock_server, str(tmp_path / "c")))
        rec = {}
        got = sel.select_candidates("w0", _menu(), 2, record=rec)
        sel.close()
        assert rec["llm_ok"] is False
        assert rec["n_padded"] == 2          # 全部靠补足
        assert len(got) == 2
    finally:
        _Handler.reply = '{"pick": [1, 3]}'


def test_extra_payload_is_forwarded(mock_server, tmp_path, monkeypatch):
    """enable_thinking 之类的 provider 参数必须真的进到请求体。

    这一项若失效，推理模型会在 ``max_tokens=32`` 下仍产出上千 reasoning token，
    单次成本高出两个量级，而表面上完全看不出异常。
    """
    monkeypatch.setenv("MOCK_KEY", "sk-test")
    cfg = _cfg(mock_server, str(tmp_path / "c"))
    cfg.extra_payload = {"enable_thinking": False}
    sel = MenuSelector(cfg)
    sel.select("w0", _menu(), 2)
    sel.close()
    assert _Handler.last_payload is not None
    assert _Handler.last_payload.get("enable_thinking") is False


def test_env_file_fallback(tmp_path, monkeypatch):
    """环境变量缺失时应能从 .env 兜底读取。"""
    monkeypatch.delenv("MOCK_KEY", raising=False)
    ef = tmp_path / ".env"
    ef.write_text("# comment\nMOCK_KEY=sk-from-file\nOTHER=x\n", encoding="utf-8")
    cfg = LLMConfig(api_key_env="MOCK_KEY", env_file=str(ef), trust_env=False)
    assert cfg.api_key() == "sk-from-file"


def test_env_file_missing_still_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("MOCK_KEY", raising=False)
    cfg = LLMConfig(api_key_env="MOCK_KEY", env_file=str(tmp_path / "nope.env"),
                    trust_env=False)
    with pytest.raises(LLMConfigError):
        cfg.api_key()
