# -*- coding: utf-8 -*-
r"""真实 LLM 候选生成器（DeepSeek / 任意 OpenAI 兼容端点）。

设计要点
--------
1. **LLM 只做选择，不做生成。** ``Candidate`` 需要 ``path`` / ``cells`` / ``sat`` /
   ``space_time`` 四个字段，LLM 无法凭空给出带真实栅格与步数的路线。因此把每位
   参与者已有的候选池编成**编号菜单**，让 LLM 返回编号子集，再映射回真实
   ``Candidate``。这正是定理 3 的口径「候选生成器输出 :math:`C_u \subseteq R_u`」，
   并且命中率（最优路线是否被选中）天然可测。

2. **提示词确定性。** 同一 ``(model, prompt_version, wid, 菜单签名)`` 生成的提示词
   逐字节一致，故可安全落盘缓存，重复运行不再消耗 token。

3. **失败显式。** 缺 API key 时抛 ``LLMConfigError``，绝不静默退回仿真——
   仿真结果与真实 LLM 结果混在同一张表里属于学术不端。

用法
----
    export DEEPSEEK_API_KEY=sk-...
    py -3.14 experiments/run_llm_candidates.py --mode llm --live --n-workers 20

依赖 ``httpx``（仓库已装，无需新增）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from .model import Candidate

#: 提示词版本号。**改动模板时必须递增**，否则旧缓存会被错误复用。
PROMPT_VERSION = "menu-select-v2"

#: 默认 prompt 风格：与度量指标（OPT 路线命中）保持一致的覆盖最大化
DEFAULT_STYLE = "coverage"


class LLMConfigError(RuntimeError):
    """凭据缺失或配置非法。"""


class LLMCallError(RuntimeError):
    """HTTP 层失败且重试耗尽。"""


@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 60.0
    max_retries: int = 4
    cache_dir: str = "results/llm_cache"
    use_cache: bool = True
    #: 是否读取系统代理设置。真实调用通常需要 True（本机就是经系统代理出网的）；
    #: 指向本地 mock/自建端点时应设 False，否则 loopback 请求会被代理拦成 502。
    trust_env: bool = True
    #: provider 特有参数，原样合并进请求体。注意 ``enable_thinking``：推理模型
    #: 即使把 ``max_tokens`` 设成 32，仍会产出上千个 reasoning token，成本差 100 倍以上。
    extra_payload: Dict[str, Any] = field(default_factory=dict)
    #: 可选 .env 路径；环境变量缺失时从这里兜底读取，免得每次都 export
    env_file: str = ""

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key and self.env_file:
            key = _read_env_file(self.env_file).get(self.api_key_env, "").strip()
        if not key:
            hint = f"（也尝试过读取 {self.env_file}）" if self.env_file else ""
            raise LLMConfigError(
                f"未找到环境变量 {self.api_key_env}{hint}。\n"
                f"  请先设置（PowerShell）：$env:{self.api_key_env}=\"sk-...\"\n"
                f"  或加 --env-file 指向含该键的 .env 文件\n"
                f"  若不想联网，请改用可控仿真：--mode sim（默认）"
            )
        return key


def _read_env_file(path: str) -> Dict[str, str]:
    """极简 .env 解析：``KEY=VALUE`` 逐行，忽略注释与空行。"""
    out: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


#: 常用 provider 预设
PRESETS: Dict[str, LLMConfig] = {
    "deepseek": LLMConfig(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "zhipu": LLMConfig(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
    ),
    "qwen": LLMConfig(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "moonshot": LLMConfig(
        base_url="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
    ),
    # 硅基流动：聚合多家模型，OpenAI 兼容。注意其推理模型默认开 thinking，
    # 成本会高出两个量级，故这里显式关闭。
    "siliconflow": LLMConfig(
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V4-Flash",
        api_key_env="SILICONFLOW_API_KEY",
        extra_payload={"enable_thinking": False},
    ),
}


@dataclass
class Selection:
    """一次菜单选择的结果，含审计所需的用量信息。"""

    picks: List[int]
    raw: str = ""
    ok: bool = True
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ------------------------------------------------------------------ 提示词
def _menu_line(i: int, c: Candidate) -> str:
    return f"[{i}] 覆盖 {len(c.cells)} 格 | 满意度 {c.sat:.3f} | 步数 {c.steps}"


def build_prompt(wid: str, menu: Sequence[Candidate], k: int,
                 theta: float = 0.0, phi: float = 0.0,
                 style: str = "coverage") -> str:
    """构造确定性提示词。同样的入参必须产出逐字节相同的字符串。

    ``style`` 决定给 LLM 的任务指令：

    * ``coverage``（默认）—— 以**最大化总覆盖（并集）**为目标。度量指标
      （OPT 路线是否被选中）是覆盖驱动的，指令必须与之一致，否则测出来的是
      指令与指标的错配，而不是模型能力。
    * ``diverse`` —— 强调覆盖不重复、满意度与步数均衡。保留用于对照。
    """
    if style == "coverage":
        task = [
            f"任务：从清单中选出**至多 {k} 条**作为该参与者的候选池 C_u，目标是",
            "  (a) 让这批路线覆盖的**栅格并集尽可能大**；",
            "  (b) 因此优先选覆盖格数多的路线，并避免多条路线重复覆盖同一批栅格；",
            "  (c) 覆盖率接近时，取满意度更高者。",
        ]
    elif style == "diverse":
        task = [
            f"任务：从清单中选出**至多 {k} 条**作为该参与者的候选池 C_u，使",
            "  (a) 入选路线彼此覆盖的栅格单元尽量不重复；",
            "  (b) 满意度尽量高；",
            "  (c) 路线步数不要过度集中，保留长短搭配。",
        ]
    else:
        raise ValueError(f"未知 prompt style: {style!r}")

    lines = [
        "你是移动众感平台的路线筛选器。下面是一位参与者的候选路线清单，",
        "每条路线已标注「覆盖栅格数 / 个性化满意度 / 步数」。",
        "",
        f"参与者编号：{wid}",
        f"个性化阈值 θ：{theta:.2f}（满意度低于 θ 的路线应尽量排除）",
        f"偏好同质性 φ：{phi:.2f}",
        "",
        *task,
        "",
        "候选路线：",
    ]
    lines.extend(_menu_line(i, c) for i, c in enumerate(menu))
    lines += [
        "",
        '只输出一个 JSON 对象，不要任何解释或 markdown 代码块：{"pick": [0, 2, 5]}',
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ 解析
_JSON_RE = re.compile(r"\{[^{}]*\"pick\"[^{}]*\}", re.S)
_ARR_RE = re.compile(r"\[[\s\d,]*\]")


def parse_picks(text: str, n_menu: int, k: int) -> List[int]:
    """从模型输出里抽出编号列表。返回去重、合法的前 k 个编号。

    容忍三种常见形态：纯 JSON、```json 包裹、只给了数组。
    """
    if not text:
        return []
    picks: Optional[List[Any]] = None
    m = _JSON_RE.search(text)
    if m:
        try:
            picks = json.loads(m.group(0)).get("pick")
        except json.JSONDecodeError:
            picks = None
    if picks is None:
        m2 = _ARR_RE.search(text)
        if m2:
            try:
                picks = json.loads(m2.group(0))
            except json.JSONDecodeError:
                picks = None
    if not isinstance(picks, list):
        return []

    out: List[int] = []
    for v in picks:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= iv < n_menu and iv not in out:
            out.append(iv)
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------------ 主体
@dataclass
class MenuSelector:
    """把一批候选编成菜单，问 LLM 选哪些，返回真实 Candidate 子集。"""

    cfg: LLMConfig = field(default_factory=LLMConfig)
    _client: Any = field(default=None, init=False, repr=False)
    stats: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.stats = {
            "calls": 0, "cache_hits": 0, "failures": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0,
            # 缓存命中省掉的用量单列：否则全命中的重跑会把真实成本记成 0
            "prompt_tokens_cached": 0, "completion_tokens_cached": 0,
        }
        if self.cfg.use_cache:
            os.makedirs(self.cfg.cache_dir, exist_ok=True)

    # -------------------------------------------------- 缓存
    def _cache_path(self, wid: str, menu: Sequence[Candidate], k: int, theta: float,
                    style: str) -> str:
        sig = hashlib.sha256(
            ("\n".join(_menu_line(i, c) for i, c in enumerate(menu))
             + f"|k={k}|theta={theta:.4f}|style={style}").encode()
        ).hexdigest()[:32]
        key = hashlib.sha256(
            f"{self.cfg.model}|{PROMPT_VERSION}|{wid}|{sig}".encode()
        ).hexdigest()
        return os.path.join(self.cfg.cache_dir, key + ".json")

    def _cache_read(self, path: str) -> Optional[Selection]:
        if not (self.cfg.use_cache and os.path.exists(path)):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            return Selection(
                picks=list(rec["picks"]), raw=rec.get("raw", ""), cached=True,
                prompt_tokens=rec.get("prompt_tokens", 0),
                completion_tokens=rec.get("completion_tokens", 0),
            )
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _cache_write(self, path: str, sel: Selection) -> None:
        if not self.cfg.use_cache:
            return
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "picks": sel.picks, "raw": sel.raw,
                    "prompt_tokens": sel.prompt_tokens,
                    "completion_tokens": sel.completion_tokens,
                    "model": self.cfg.model, "prompt_version": PROMPT_VERSION,
                }, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError:
            pass  # 缓存失败不影响主流程

    # -------------------------------------------------- HTTP
    def _post(self, prompt: str) -> Dict[str, Any]:
        if self._client is None:
            self._client = httpx.Client(timeout=self.cfg.timeout,
                                        trust_env=self.cfg.trust_env)
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        payload.update(self.cfg.extra_payload)
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key()}",
            "Content-Type": "application/json",
        }
        last: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:  # 网络层：可重试
                last = exc
                time.sleep(min(2.0 ** attempt * 0.8, 8.0))
                continue
            # 4xx 中除 429 外都是永久错误，重试只会浪费时间
            if r.status_code in (401, 403):
                raise LLMConfigError(
                    f"凭据被拒绝（HTTP {r.status_code}）：{r.text[:200]}")
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise LLMCallError(
                    f"请求被拒（HTTP {r.status_code}）：{r.text[:200]}")
            if r.status_code == 429 or 500 <= r.status_code < 600:
                last = LLMCallError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(min(2.0 ** attempt * 0.8, 8.0))
                continue
            r.raise_for_status()
            return r.json()
        raise LLMCallError(f"{url} 重试 {self.cfg.max_retries} 次后仍失败：{last}")

    # -------------------------------------------------- 对外
    def select(self, wid: str, menu: Sequence[Candidate], k: int,
               theta: float = 0.0, phi: float = 0.0,
               style: str = DEFAULT_STYLE) -> Selection:
        """让 LLM 从 ``menu`` 里选至多 k 条。返回编号（已映射为 menu 的下标）。"""
        if not menu:
            return Selection(picks=[], ok=False, error="空菜单")
        k = min(k, len(menu))

        cpath = self._cache_path(wid, menu, k, theta, style)
        hit = self._cache_read(cpath)
        if hit is not None:
            self.stats["cache_hits"] += 1
            self.stats["prompt_tokens_cached"] += hit.prompt_tokens
            self.stats["completion_tokens_cached"] += hit.completion_tokens
            return hit

        prompt = build_prompt(wid, menu, k, theta=theta, phi=phi, style=style)
        self.stats["calls"] += 1
        t0 = time.perf_counter()
        try:
            body = self._post(prompt)
        except LLMConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.stats["failures"] += 1
            return Selection(picks=[], ok=False, error=str(exc)[:300])

        latency = (time.perf_counter() - t0) * 1000.0
        choices = body.get("choices") or []
        content = ""
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        sel = Selection(
            picks=parse_picks(content, len(menu), k),
            raw=content,
            ok=bool(parse_picks(content, len(menu), k)),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=round(latency, 1),
        )
        if not sel.ok:
            self.stats["failures"] += 1
            sel.error = "无法从响应中解析 pick 列表"

        self.stats["prompt_tokens"] += sel.prompt_tokens
        self.stats["completion_tokens"] += sel.completion_tokens
        self.stats["latency_ms"] += sel.latency_ms
        # 只缓存成功结果：一次偶发失败若被固化成空选择，后续所有运行都会继承这个错误
        if sel.ok:
            self._cache_write(cpath, sel)
        return sel

    def select_candidates(self, wid: str, menu: Sequence[Candidate], k: int,
                          theta: float = 0.0, phi: float = 0.0,
                          fallback: Optional[Sequence[Candidate]] = None,
                          record: Optional[Dict[str, Any]] = None,
                          style: str = DEFAULT_STYLE) -> List[Candidate]:
        """返回真实 ``Candidate`` 子集。

        若 LLM 选出的条数不足 ``k``，按 ``fallback`` 的顺序**补齐**（默认按满意度降序），
        并把补了几条记进 ``record["n_padded"]``——补足是为了让 ``|C_u|`` 与其它
        生成器可比，而补了几条必须在报告里披露。
        """
        sel = self.select(wid, menu, k, theta=theta, phi=phi, style=style)
        picked = [menu[i] for i in sel.picks]
        n_padded = 0
        if len(picked) < k:
            pool = list(fallback) if fallback is not None else sorted(
                menu, key=lambda c: (-c.sat, -len(c.cells)))
            for c in pool:
                if len(picked) >= k:
                    break
                if c not in picked:
                    picked.append(c)
                    n_padded += 1
        if record is not None:
            record.update({
                "llm_ok": sel.ok, "llm_cached": sel.cached,
                "llm_picks": sel.picks, "n_padded": n_padded,
                "prompt_tokens": sel.prompt_tokens,
                "completion_tokens": sel.completion_tokens,
                "latency_ms": sel.latency_ms,
                "llm_error": sel.error,
            })
        return picked[:k]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
