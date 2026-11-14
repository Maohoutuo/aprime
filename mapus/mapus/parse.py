# -*- coding: utf-8 -*-
"""LLM 输出解析：容错 JSON 抽取 + 各类响应解析。

LLM 经常给出 ```json fenced 块、前后夹带解释、或把坐标写成元组，因此这里先做
宽松抽取，再由调用方做严格的可行性校验（算法 1 第 6 行）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    # 直接把 (x, y, t) 元组写法改成 JSON 数组，兼容 Fig.5 的示意格式
    cleaned = re.sub(r"\((\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*)\)", r"[\1]", cleaned)
    decoder = json.JSONDecoder()
    for candidate in (cleaned,):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {"value": obj}
        except json.JSONDecodeError:
            pass
    # 找第一个 JSON 对象再做 raw_decode
    for m in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _coerce_path(value: Any) -> List[List[int]]:
    out: List[List[int]] = []
    if not isinstance(value, (list, tuple)):
        return out
    for step in value:
        if isinstance(step, dict):
            step = [step.get("x"), step.get("y"), step.get("t")]
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            continue
        try:
            x, y = int(float(step[0])), int(float(step[1]))
            t = int(float(step[2])) if len(step) > 2 else len(out)
        except (TypeError, ValueError):
            continue
        out.append([x, y, t])
    return out


def parse_refine(text: str) -> Tuple[List[List[int]], str]:
    obj = extract_json(text) or {}
    path = _coerce_path(obj.get("final_path") or obj.get("path") or obj.get("route") or [])
    return path, str(obj.get("explanation", ""))[:400]


def parse_accept(text: str) -> Tuple[bool, str]:
    """解析同意/拒绝。

    解析失败时**默认接受**：论文的参与者只在行程不可行时才退出，把"模型输出格式
    错误"当成拒绝会让接受率被解析噪声主导（实测 Qwen 的回答常带解释性前缀）。
    """
    obj = extract_json(text) or {}
    if "accept" in obj:
        val = obj["accept"]
    elif "agreement" in obj:
        val = obj["agreement"]
    elif "accept_task" in obj:
        val = obj["accept_task"]
    else:
        low = (text or "").strip().lower()
        if any(k in low for k in ("false", "reject", "decline", "cannot accept")):
            return False, (text or "")[:200]
        return True, (text or "accepted (unstructured response)")[:200]
    if isinstance(val, str):
        val = val.strip().lower() in ("true", "yes", "accept", "1")
    return bool(val), str(obj.get("reason", obj.get("feedback", "")))[:300]


def parse_tiebreak(text: str, allowed: Sequence[str]) -> Optional[str]:
    obj = extract_json(text) or {}
    for key in ("selected", "worker_id", "id", "participant"):
        val = obj.get(key)
        if isinstance(val, str) and val in allowed:
            return val
        if isinstance(val, list) and val and str(val[0]) in allowed:
            return str(val[0])
    # 兜底：在文本里找第一个出现的 id
    for wid in allowed:
        if wid in text:
            return wid
    return None


def parse_offer(text: str, u: str, v: str) -> Tuple[Dict[str, List[List[int]]], Dict[str, str]]:
    obj = extract_json(text) or {}
    routes = obj.get("refined_routes") or obj.get("routes") or {}
    routes = routes if isinstance(routes, dict) else {}
    out: Dict[str, List[List[int]]] = {}
    for wid in (u, v):
        path = _coerce_path(routes.get(wid, []))
        if path:
            out[wid] = path
    incentives = obj.get("incentives") or {}
    incentives = {k: str(v)[:300] for k, v in incentives.items()} if isinstance(incentives, dict) else {}
    return out, incentives


def parse_single(text: str) -> Dict[str, List[List[int]]]:
    obj = extract_json(text) or {}
    selected = obj.get("selected") or obj.get("plan") or []
    out: Dict[str, List[List[int]]] = {}
    if isinstance(selected, dict):
        for wid, path in selected.items():
            parsed = _coerce_path(path)
            if parsed:
                out[str(wid)] = parsed
        return out
    for item in selected if isinstance(selected, list) else []:
        if not isinstance(item, dict):
            continue
        wid = str(item.get("worker_id") or item.get("id") or "")
        path = _coerce_path(item.get("path") or item.get("final_path") or [])
        if wid and path:
            out[wid] = path
    return out


def parse_tool_calls(text: str) -> List[dict]:
    obj = extract_json(text) or {}
    calls = obj.get("tool_calls")
    return calls if isinstance(calls, list) else []
