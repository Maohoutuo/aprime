# -*- coding: utf-8 -*-
"""practice2.py —— 最短的大模型 Agent 示例（新手向）。

一个 Agent 其实只要四样东西：
    1. 角色提示词：告诉模型它是谁、手上有什么工具
    2. 工具：模型算不准的事（算术、时间）交给 Python 做
    3. 固定输出格式：模型必须回一个 JSON，代码才读得懂
    4. 循环：模型要工具 → 代码执行 → 结果回给模型 → 直到给出答案

和 practice.py 是同一个思路，但去掉了所有抽象、类型注解和重试细节，
只留最短能跑的骨架。

用法：
    py practice2.py "帮我算 (17*23+45)/4"
    py practice2.py            # 交互模式，输入 exit 退出
"""

import json
import math
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

URL = "https://api.longcat.chat/openai/v1/chat/completions"
MODEL = "LongCat-2.5"

# ---------- 1. 读 key（写在同目录 .env 里：LONGCAT_API_KEY=你的key） ----------
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("LONGCAT_API_KEY="):
            os.environ.setdefault("LONGCAT_API_KEY", line.split("=", 1)[1].strip())

API_KEY = os.environ.get("LONGCAT_API_KEY", "")


# ---------- 2. 工具：模型不会算数，让它调 Python ----------
# __builtins__ 置空，模型就算写出 __import__('os') 也执行不了
SAFE = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}#生成一个安全字典


def calc(expression):#calc函数
    """算数学表达式。"""
    return str(eval(expression, SAFE, {}))#返回一个字符串


def now():#now函数
    """当前日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")#返回一个字符串


TOOLS = {"calc": calc, "now": now}#生成一个工具字典


# ---------- 3. 角色提示词：定义身份、工具、输出格式 ----------
SYSTEM = """你是一个会使用工具的助手。

可用工具：
- calc(expression)：计算数学表达式，例如 "sqrt(2)*100"
- now()：获取当前日期时间，无参数

规则：
- 需要计算或需要当前时间时，必须先调用工具，不要自己心算。
- 每次只输出一个 JSON 对象，不要输出 JSON 以外的任何文字。

需要调用工具时，输出：
{"tool": "calc", "args": {"expression": "1+1"}}

已经可以回答用户时，输出：
{"final": "给用户的最终回答，用中文，简洁完整"}
"""


# ---------- 4. 发请求 ----------
def ask(messages):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.5,
    }).encode("utf-8")
    request = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY,
    })
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------- 5. Agent 主循环 ----------
def run(question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, 5):                 # 最多 4 轮，防止死循环
        reply = ask(messages)

        # 模型可能用 ```json 包起来，也可能前后带废话，所以只抠出 {...}
        found = re.search(r"\{.*\}", reply, re.S)
        if found is None:
            return reply                     # 没按格式回，就把它的话当答案

        action = json.loads(found.group())
        if "final" in action:                # 模型认为可以回答了
            return action["final"]

        name = action.get("tool")
        args = action.get("args") or {}
        if name not in TOOLS:
            return reply                     # 用了不存在的工具，别硬撑

        result = TOOLS[name](**args)         # 关键：由 Python 算，不是模型算
        print("  [第 %d 轮] %s(%s) -> %s" % (step, name, args, result))

        # 把模型这轮的输出和工具结果都塞回对话，让它基于事实继续
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "工具结果：" + result})

    return "（轮数用完了也没得出结论）"


# ---------- 6. 入口 ----------
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not API_KEY or API_KEY == "PASTE_NEW_LONGCAT_API_KEY_HERE":
        sys.exit("请先在 .env 里填 LONGCAT_API_KEY=你的key（https://longcat.chat/platform/api_keys）")

    if len(sys.argv) > 1:
        print(run(" ".join(sys.argv[1:])))
    else:
        while True:
            question = input("\n你> ").strip()
            if question in ("", "exit", "quit"):
                break
            print(run(question))
