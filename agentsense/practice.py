# -*- coding: utf-8 -*-
"""一个最小可用的 LLM agent —— 用 LongCat 回答提问，需要时调用工具。

它把 agentsense 包里四个文件的机制压缩进一个脚本，方便对照阅读：

    prompts.py  →  SYSTEM_PROMPT / NO_TOOLS_PROMPT / 工具的说明书
    llm.py      →  LLMClient（urllib + 指数退避重试 + usage 统计 + 截断检查）
    tools.py    →  TOOLS 注册表 + call_tool()
    agents.py   →  MiniAgent.run()（结构化输出 + parse_json_block 容错 + 循环）

核心思想与 AgentSense 一致：模型只负责"决定做什么"和"用自然语言说"；
算术、时间这类精确事实交给工具；格式错误由代码兜底。

用法：
    py practice.py "帮我算一下 sqrt(2)*100 是多少"
    py practice.py                     # 进入交互模式
    py practice.py "..." --verbose     # 打印每轮的思考与工具调用
    py practice.py "..." --no-tools    # 关掉工具，退化成单轮问答（对照用）

API key 从同目录 .env 读取：LONGCAT_API_KEY / LONGCAT_BASE_URL / LONGCAT_MODEL
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_BASE_URL = "https://api.longcat.chat/openai"
DEFAULT_MODEL = "LongCat-2.5"
MAX_ROUNDS = 6


# --------------------------------------------------------------------------- 环境变量
def load_env(path: Path = ENV_FILE) -> None:
    """把 .env 中的 KEY=VALUE 读进 os.environ（不覆盖已存在的变量）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# --------------------------------------------------------------------------- 工具实现
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST) -> float:
    """只允许数字、四则运算和白名单函数——不用 eval()，避免执行任意代码。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字常量")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS):
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"不支持的表达式片段：{type(node).__name__}")


def tool_calc(expression: str) -> str:
    try:
        value = _eval_node(ast.parse(expression, mode="eval"))
    except Exception as exc:
        return f"错误：{exc}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


def tool_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S（本地时区）")


def tool_count_text(text: str) -> str:
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return f"总字符数={len(text)}，空白分隔词数={len(text.split())}，中文字数={han}"


TOOLS: Dict[str, Dict[str, Any]] = {
    "calc": {
        "desc": '计算数学表达式，例如 "(3+4)*2"、"sqrt(2)"、"2**10"',
        "args": {"expression": "str"},
        "fn": tool_calc,
    },
    "now": {
        "desc": "获取当前日期时间，不需要参数",
        "args": {},
        "fn": tool_now,
    },
    "count_text": {
        "desc": "统计一段文本的字符数与字数",
        "args": {"text": "str"},
        "fn": tool_count_text,
    },
}


def tools_manual() -> str:
    lines = []
    for name, spec in TOOLS.items():
        args = ", ".join(f'"{k}": {v}' for k, v in spec["args"].items()) or "无参数"
        lines.append(f"- {name}: {spec['desc']}；参数 {{{args}}}")
    return "\n".join(lines)


def call_tool(name: str, args: Dict[str, Any]) -> str:
    """执行工具。任何异常都变成字符串返回给模型，绝不让循环崩掉。"""
    spec = TOOLS.get(name)
    if spec is None:
        return f"错误：没有名为 {name} 的工具。可用工具：{', '.join(TOOLS)}"
    try:
        return str(spec["fn"](**args))
    except TypeError as exc:
        return f"错误：参数与工具签名不匹配（{exc}）"
    except Exception as exc:
        return f"错误：{exc}"


# --------------------------------------------------------------------------- 提示词
SYSTEM_PROMPT = """你是一个会使用工具的助手。

可用工具：
{tools}

工作方式：
- 需要精确计算、当前时间或字数统计时，先调用工具，不要自己心算。
- 每轮只输出一个 JSON 对象，JSON 之外不要写任何文字。

要调用工具时，输出：
{{"thought": "一句话说明为什么要调用它", "tool": "工具名", "args": {{"参数名": "参数值"}}}}

已经可以回答用户时，输出：
{{"thought": "一句话说明结论是怎么得出的", "final": "给用户的最终回答，用中文，简洁但完整"}}
"""

NO_TOOLS_PROMPT = "你是一个乐于助人的助手。请用中文简洁、完整地回答用户的问题。"


# --------------------------------------------------------------------------- 输出解析
def parse_json_block(text: str) -> Optional[dict]:
    """从模型输出里抠出第一个 JSON 对象，容忍 ```json 包裹和前后废话。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- LLM 客户端
class LLMClient:
    """LongCat 的 OpenAI 兼容端点。与 agentsense/llm.py 同构，但支持多轮 messages。"""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, temperature: float = 0.5,
                 max_tokens: int = 4096, timeout: float = 120.0,
                 retries: int = 3, verbose: bool = False):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def complete(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error = ""
        for attempt in range(self.retries):
            try:
                if self.verbose:
                    print(f"  [llm] {self.model} 第 {attempt + 1}/{self.retries} 次请求，"
                          f"{len(body)} 字节", flush=True)
                started = time.time()
                request = urllib.request.Request(self.url, data=body,
                                                 headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if self.verbose:
                    print(f"  [llm] 返回于 {time.time() - started:.1f}s", flush=True)
                return self._consume(result)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code == 401:
                    raise RuntimeError(
                        "API key 无效或已过期（401）。请检查 .env 里的 LONGCAT_API_KEY。"
                    ) from exc
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(last_error) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"调用失败（已重试 {self.retries} 次）：{last_error}")

    def _consume(self, payload: dict) -> str:
        self.calls += 1
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

        choice = (payload.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if choice.get("finish_reason") == "length":
            print("  [!] 输出被 max_tokens 截断，回答可能不完整", file=sys.stderr, flush=True)
        return content

    @property
    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


# --------------------------------------------------------------------------- Agent 本体
class MiniAgent:
    """Role + Tools + 结构化输出 + 循环，四件套的最小实现。"""

    def __init__(self, client: LLMClient, max_rounds: int = MAX_ROUNDS,
                 use_tools: bool = True, verbose: bool = False):
        self.client = client
        self.max_rounds = max_rounds
        self.use_tools = use_tools
        self.verbose = verbose

    def run(self, question: str) -> str:
        if not self.use_tools:
            answer = self.client.complete([
                {"role": "system", "content": NO_TOOLS_PROMPT},
                {"role": "user", "content": question},
            ])
            return answer.strip() or "（模型没有返回内容）"

        messages: List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_manual())},
            {"role": "user", "content": question},
        ]

        for round_no in range(1, self.max_rounds + 1):
            raw = self.client.complete(messages)
            action = parse_json_block(raw)

            if not isinstance(action, dict):
                # 模型没按协议输出：直接把它的话当答案，避免死循环
                return raw.strip() or "（模型没有返回内容）"

            if action.get("final"):
                return str(action["final"]).strip()

            tool_name = str(action.get("tool") or "").strip()
            if not tool_name:
                return str(action.get("thought") or raw).strip()

            raw_args = action.get("args")
            tool_args = raw_args if isinstance(raw_args, dict) else {}
            observation = call_tool(tool_name, tool_args)

            if self.verbose:
                print(f"  [第 {round_no} 轮] {action.get('thought', '')}", flush=True)
                print(f"           {tool_name}({json.dumps(tool_args, ensure_ascii=False)})"
                      f" -> {observation}", flush=True)

            messages.append({"role": "assistant", "content": raw})
            # 用 user 角色回传观察结果：兼容性最好，不依赖 tool_call_id
            messages.append({
                "role": "user",
                "content": f"[工具 {tool_name} 的执行结果]\n{observation}",
            })

        return f"（达到最大轮数 {self.max_rounds} 仍未得出最终回答，可用 --max-rounds 调大上限）"


# --------------------------------------------------------------------------- 入口
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    load_env()

    parser = argparse.ArgumentParser(description="用 LongCat 大模型回答问题的最小 agent")
    parser.add_argument("question", nargs="*", help="要问的问题；留空则进入交互模式")
    parser.add_argument("--model", default=os.environ.get("LONGCAT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("LONGCAT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--no-tools", action="store_true",
                        help="关闭工具，退化成单轮问答（对照实验）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每轮的思考、工具调用与请求耗时")
    args = parser.parse_args()

    api_key = os.environ.get("LONGCAT_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_NEW_LONGCAT_API_KEY_HERE":
        raise SystemExit(
            "还没有配置可用的 key。\n"
            "1) 打开 https://longcat.chat/platform/api_keys 创建 API Key\n"
            f"2) 把 key 写进 {ENV_FILE}：LONGCAT_API_KEY=你的key\n"
            "3) 重新运行本脚本"
        )

    client = LLMClient(
        api_key=api_key, base_url=args.base_url, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens, verbose=args.verbose,
    )
    agent = MiniAgent(client, max_rounds=args.max_rounds,
                      use_tools=not args.no_tools, verbose=args.verbose)

    try:
        if args.question:
            print(agent.run(" ".join(args.question)))
            return

        print("交互模式：输入问题后回车，输入 exit 退出。")
        while True:
            try:
                question = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in ("exit", "quit", "q", ""):
                break
            print("\n" + agent.run(question))
    finally:
        used = client.usage
        print(f"\n[usage] calls={used['calls']} prompt={used['prompt_tokens']} "
              f"completion={used['completion_tokens']} total={used['total_tokens']}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
