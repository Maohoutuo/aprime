# -*- coding: utf-8 -*-
"""LLM 客户端：mock / OpenAI 兼容（含硅基流动）/ Anthropic，全部走标准库 urllib。"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


def load_dotenv(path: Optional[str] = None) -> None:#加载dotenv文件
    """极简 .env 读取（KEY=VALUE），不覆盖已存在的环境变量。"""
    candidates = [Path(path)] if path else [Path.cwd() / ".env",
                                            Path(__file__).resolve().parent.parent / ".env"]
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
        break


load_dotenv()

PRESETS = {
    # 硅基流动：OpenAI 兼容
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}


class LLMClient:#LLMClient类
    def __init__(self, provider: str = "mock", model: str = "mock", api_key: Optional[str] = None, base_url: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 2048, mock_policy: Optional[Callable] = None, retries: int = 3, timeout: float = 120.0, verbose: bool = False):
        self.provider = provider#生成一个提供者
        self.model = model#生成一个模型
        self.temperature = temperature#生成一个温度
        self.max_tokens = max_tokens#生成一个最大令牌数
        self.mock_policy = mock_policy#生成一个模拟策略
        self.retries = retries#生成一个重试次数
        self.timeout = timeout#生成一个超时时间
        self.verbose = verbose#生成一个是否verbose
        self.prompt_tokens = 0#生成一个提示令牌数
        self.completion_tokens = 0#生成一个完成令牌数
        self.calls = 0#生成一个调用次数

        if provider in PRESETS:#如果提供者是预设的
            default_base, key_env = PRESETS[provider]#生成一个默认基础URL和环境变量
            self.api_key = api_key or os.environ.get(key_env)#生成一个API密钥
            self.base_url = (base_url or os.environ.get(
                "SILICONFLOW_BASE_URL" if provider == "siliconflow" else f"{provider.upper()}_BASE_URL")
                or default_base).rstrip("/")#生成一个基础URL
            self.kind = "openai"#生成一个类型
        elif provider == "anthropic":#如果提供者是anthropic
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")#生成一个API密钥
            self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")#生成一个基础URL
            self.kind = "anthropic"#生成一个类型
        elif provider == "mock":#如果提供者是mock
            self.api_key = None#生成一个API密钥
            self.base_url = None#生成一个基础URL
            self.kind = "mock"#生成一个类型
        else:
            raise ValueError(f"unknown provider: {provider}")#生成一个错误

    # ------------------------------------------------------------------ public
    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        if self.kind == "mock":#如果类型是mock
            return self._mock(system, user)
        last_err = None#生成一个错误
        use_json = json_mode#生成一个是否使用JSON
        for attempt in range(self.retries):#生成一个重试次数
            try:
                if self.verbose:#如果verbose为真
                    print(f"[llm] {self.model}: request ({len(system)+len(user)} chars), "
                          f"JSON={use_json}, attempt {attempt + 1}/{self.retries}", flush=True)
                start = time.time()#生成一个开始时间
                if self.kind == "openai":#如果类型是openai
                    result = self._openai(system, user, use_json)#生成一个结果
                else:
                    result = self._anthropic(system, user)#生成一个结果
                if self.verbose:#如果verbose为真
                    print(f"[llm] {self.model}: response {len(result)} chars in "
                          f"{time.time()-start:.1f}s", flush=True)
                return result#返回结果
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:300]#生成一个body
                last_err = f"HTTP {e.code}: {body}"#生成一个错误
                # json_object 不被支持时降级重试
                if e.code in (400, 422) and use_json and "json" in body.lower():#如果代码是400或422且使用JSON且body包含json
                    use_json = False#生成一个是否使用JSON
                    continue#继续
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)#生成一个睡眠时间
                    continue#继续
                raise RuntimeError(last_err) from e#生成一个错误
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = str(e)#生成一个错误
                time.sleep(2 ** attempt)#生成一个睡眠时间
        raise RuntimeError(f"LLM call failed after {self.retries} retries: {last_err}")#生成一个错误

    @property
    def usage(self) -> dict:#usage属性
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}#返回使用情况

    # ------------------------------------------------------------------ backends
    def _mock(self, system: str, user: str) -> str:
        if self.mock_policy is not None:#如果模拟策略不为空
            return self.mock_policy(system, user)#生成一个结果
        return json.dumps({"think_process": "mock: no change", "refined_solution": None})#返回一个结果

    def _openai(self, system: str, user: str, json_mode: bool) -> str:#openai函数
        payload = {
            "model": self.model,#生成一个模型
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,#生成一个温度
            "max_tokens": self.max_tokens,#生成一个最大令牌数
        }
        if json_mode:#如果使用JSON模式
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),#生成一个数据
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},#生成一个头
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())#生成一个输出
        self.calls += 1#更新调用次数
        usage = out.get("usage") or {}#生成一个使用情况
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))#更新提示令牌数
        self.completion_tokens += int(usage.get("completion_tokens", 0))#更新完成令牌数
        return out["choices"][0]["message"]["content"]#返回内容

    def _anthropic(self, system: str, user: str) -> str:#anthropic函数
        payload = {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": user}],#生成一个消息
        }
        req = urllib.request.Request(
            self.base_url + "/v1/messages",#生成一个URL
            data=json.dumps(payload).encode(),#生成一个数据
            headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},#生成一个头
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())#生成一个输出
        self.calls += 1
        usage = out.get("usage") or {}#生成一个使用情况
        self.prompt_tokens += int(usage.get("input_tokens", 0))#更新提示令牌数
        self.completion_tokens += int(usage.get("output_tokens", 0))#更新完成令牌数
        return out["content"][0]["text"]#返回内容
