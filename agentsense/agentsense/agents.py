# -*- coding: utf-8 -*-
"""三个 Agent：Disturbance Parser / Solver Agent / Eval Agent / Memory Agent。"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import prompts, tools
from .advice import suggest_actions
from .baselines.planners import _candidate_cells, _dp_path, _weights_from, build_path, pad_to_window, path_to_solution
from .config import DEFAULT_ENV, EnvConfig
from .env.model import Instance, Solution, validate_solution
from .env.objective import Coverage, coverage_of, evaluate
from .llm import LLMClient
from .repair import repair_solution


# --------------------------------------------------------------------------- 工具函数
def solution_to_json(solution: Solution, max_chars: int = 6000) -> str:
    """紧凑序列化，超长时截断（大模型的 context 有限）。"""
    out = {}
    used = 0
    for wid, path in solution.items():
        item = [[int(p[0]), int(p[1]), int(p[2])] for p in path]
        s = json.dumps({wid: item})
        if used + len(s) > max_chars:
            out["..."] = "truncated"
            break
        out[wid] = item
        used += len(s)
    return json.dumps(out)


def instance_summary(inst: Instance) -> str:
    return (f"setting={inst.setting_name} index={inst.index} grid={inst.nx}x{inst.ny} "
            f"slots={inst.n_slots} budget={inst.budget:.1f} workers={len(inst.workers)} "
            f"tasks={inst.n_tasks}")


def workers_spec(inst: Instance, solution: Optional[Solution] = None, max_workers: int = 80) -> str:
    """把每个工作者的硬约束交给模型——缺了这段，模型必然造出非法路径。"""
    valid = {w.wid for w in inst.workers}
    ids = [wid for wid in (solution or {}) if wid in valid]
    ids += [w.wid for w in inst.workers if w.wid not in set(ids)]
    lines = []
    for wid in ids[:max_workers]:
        w = inst.worker(wid)
        lines.append(f"{wid}: {w.origin}->{w.dest}, t={w.t_start}..{w.t_end} "
                     f"(len={w.window}, grid 0..{inst.nx-1}x0..{inst.ny-1})")
    return "\n".join(lines)


def parse_json_block(text: str) -> Optional[dict]:
    """从模型输出中抽取第一个 JSON 对象（容忍 ```json 包裹）。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- 1. 扰动解析器
class DisturbanceParser:
    KEYWORDS = {
        "bad_weather": ["weather", "rain", "storm", "snow", "雾", "雨", "天气"],
        "area_blocked": ["blocked", "closed", "restricted", "封", "禁行", "封锁"],
        "worker_unavailable": ["unavailable", "offline", "sick", "quit", "不可用", "离岗"],
        "traffic_congestion": ["congestion", "traffic", "jam", "拥堵", "交通"],
        "time_window_shift": ["shift", "delay", "window", "延后", "时间窗"],
        "budget_cut": ["budget", "cost cut", "预算"],
        "task_priority_change": ["priority", "weight", "优先级"],
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def parse(self, text: str) -> Dict[str, Any]:#扰动解析函数
        if self.client.provider != "mock":#如果客户端提供者不是mock
            raw = self.client.complete(prompts.PARSER_SYSTEM, text, json_mode=True)#生成一个原始文本
            parsed = parse_json_block(raw)#解析一个JSON对象
            if parsed and parsed.get("type"):#如果解析成功
                return parsed#返回解析结果
        return self._rule_based(text)#返回规则基于解析结果

    def _rule_based(self, text: str) -> Dict[str, Any]:#规则基于解析函数
        low = text.lower()#生成一个低级文本
        for dtype, kws in self.KEYWORDS.items():
            if any(k in low for k in kws):#如果低级文本包含关键字
                cells = re.findall(r"\((\d+),\s*(\d+)\)", text)#生成一个细胞列表
                return {"type": dtype, "description": text[:200],#生成一个扰动描述
                        "value": [list(map(int, c)) for c in cells]}#生成一个扰动值
        return {"type": "task_priority_change", "description": text[:200], "value": None}#生成一个扰动描述


# --------------------------------------------------------------------------- 2. Solver Agent
class SolverAgent:#SolverAgent类
    def __init__(self, client: LLMClient, cfg: EnvConfig = DEFAULT_ENV):#SolverAgent初始化函数
        self.client = client
        self.cfg = cfg
        self.tokens_used = 0#生成一个令牌使用量

    def propose(self, inst: Instance, solution: Solution, disturbance: Dict[str, Any],
                advice: List[str], memory: List[str], rng: random.Random,
                use_tools: bool = True) -> Tuple[Solution, str]:#SolverAgent提议函数
        if self.client.provider == "mock":#如果客户端提供者是mock
            return self._mock_step(inst, solution, rng, disturbance)

        # 扰动（如 worker_unavailable）可能已移除某些工作者，先清洗引用
        valid = {w.wid for w in inst.workers}
        solution = {k: v for k, v in solution.items() if k in valid}
        metrics = evaluate(inst, solution, self.cfg)#生成一个指标
        if use_tools:
            vr = tools.validate_worker_paths(inst, solution)#生成一个验证工作者路径
            vis = tools.visual_analysis(inst, solution, solution)#生成一个可视化分析
            tool_results = json.dumps({
                "validate": {k: vr[k] for k in ("overall_feasible", "cost")},
                "check_disturbance": tools.check_disturbance_handling(inst, solution, disturbance),
                "visual": {k: vis[k] for k in ("improvement", "gain_cells", "loss_cells")},
            }, ensure_ascii=False)
        else:
            tool_results = "(tools disabled: single-agent baseline)"#生成一个工具结果
        user = prompts.solver_user(instance_summary(inst), solution_to_json(solution), metrics,
                                   disturbance.get("description", str(disturbance)),
                                   tool_results, advice, memory,
                                   workers_spec(inst, solution))#生成一个用户文本
        raw = self.client.complete(prompts.SOLVER_SYSTEM, user, json_mode=True)#生成一个原始文本
        self.tokens_used += len(raw) // 4#更新令牌使用量
        parsed = parse_json_block(raw)#解析一个JSON对象
        if not parsed or not isinstance(parsed.get("refined_solution"), dict):#如果解析失败或不是字典
            return solution, "solver: unparsable output, keep current solution"#返回解决方案
        cand = {wid: [[int(v[0]), int(v[1]), int(v[2])] for v in path]
                for wid, path in parsed["refined_solution"].items() if isinstance(path, list) and path}#生成一个候选解决方案路径

        # 自校正：确定性修复 + 一次带违规信息的重试（对应论文的 validate 循环）
        cand, violations = repair_solution(inst, cand)#修复解决方案
        if violations:#如果违规信息不为空
            retry_user = user + (
                f"\n\n## Your previous answer violated these constraints\n"
                + "\n".join(f"- {v}" for v in violations[:8])
                + "\nFix them and return the FULL solution again.")#生成一个重试用户
            raw2 = self.client.complete(prompts.SOLVER_SYSTEM, retry_user, json_mode=True)#生成一个原始文本
            self.tokens_used += len(raw2) // 4#更新令牌使用量
            parsed2 = parse_json_block(raw2)#解析一个JSON对象
            if parsed2 and isinstance(parsed2.get("refined_solution"), dict):#如果解析成功
                cand2 = {wid: [[int(v[0]), int(v[1]), int(v[2])] for v in path]
                         for wid, path in parsed2["refined_solution"].items()
                         if isinstance(path, list) and path}#生成一个候选解决方案
                cand2, viol2 = repair_solution(inst, cand2)#修复解决方案
                if len(viol2) <= len(violations):#如果违规信息长度小于等于原始违规信息长度
                    cand, violations = cand2, viol2
        return cand, str(parsed.get("think_process", ""))[:400]#返回候选解决方案和思考过程 400字符长度

    # ---- mock：先修复可行性，再做局部改进（模拟 LLM 的 edit-and-verify）
    def _mock_step(self, inst: Instance, solution: Solution, rng: random.Random,
                   disturbance: Dict[str, Any]) -> Tuple[Solution, str]:#mock步骤函数
        blocked = inst.meta.get("blocked_cells", set())#生成一个blocked集合
        # 1) 修复：移除不可用工作者、重规划失效路径、裁剪预算
        repaired = self._repair_solution(inst, solution, blocked)#修复解决方案
        cur_obj = evaluate(inst, repaired, self.cfg)["objective"] if repaired else 0.0
        best_sol, best_obj = (dict(repaired), cur_obj) if repaired else ({}, 0.0)#生成一个最佳解决方案

        # 2) 局部改进：GraphDP 重规划某个工作者
        if repaired:
            for wid in rng.sample(list(repaired), min(3, len(repaired))):
                w = inst.worker(wid)
                cov = coverage_of(inst, repaired)#生成一个覆盖率
                cells = _dp_path(inst, w, _weights_from(inst, cov))#生成一个细胞列表
                if cells is None:
                    continue
                trial = dict(repaired)#生成一个试验解决方案
                trial[wid] = path_to_solution(w, cells)
                vr = validate_solution(inst, trial)#验证试验解决方案可行性
                if not vr.feasible:
                    continue
                obj = evaluate(inst, trial, self.cfg)["objective"]#生成一个目标值
                if obj > best_obj:
                    best_obj, best_sol = obj, trial#更新最佳解决方案

        # 3) 替换贡献最低的工作者
        if repaired:
            free = [w for w in inst.workers if w.wid not in repaired]#生成一个自由工作者列表
            freed_budget = sum(len(repaired.get(wid, [])) for wid in [])  # placeholder#生成一个自由预算
            for w in free[:6]:
                if not repaired:
                    break
                worst = min(repaired, key=lambda k: len(repaired[k]))#生成一个最差工作者
                if inst.worker(worst).window + w.window > inst.budget:#如果最差工作者窗口加上自由工作者窗口大于预算
                    pass
                trial = {k: v for k, v in repaired.items() if k != worst}#生成一个试验解决方案
                cov_t = coverage_of(inst, trial)#生成一个覆盖率
                cells = _dp_path(inst, w, _weights_from(inst, cov_t))#生成一个细胞列表
                if cells is None:
                    continue
                trial[w.wid] = path_to_solution(w, cells)#生成一个试验解决方案
                vr = validate_solution(inst, trial)#验证试验解决方案可行性
                if not vr.feasible:
                    continue
                obj = evaluate(inst, trial, self.cfg)["objective"]#生成一个目标值
                if obj > best_obj:
                    best_obj, best_sol = obj, trial#更新最佳解决方案

        # 4) 在某条路径上插入一个更值钱的路点
        if repaired:
            cov = coverage_of(inst, repaired)#生成一个覆盖率
            for wid in rng.sample(list(repaired), min(2, len(repaired))):
                w = inst.worker(wid)
                for cand in _candidate_cells(cov, w, rng, 6):#生成一个候选细胞列表
                    built = build_path(w, [cand])
                    if built is None:
                        continue
                    trial = dict(repaired if best_sol is repaired else best_sol)
                    trial[wid] = path_to_solution(w, pad_to_window(built, w, prefer=cand))#生成一个试验解决方案
                    vr = validate_solution(inst, trial)#验证试验解决方案可行性
                    if not vr.feasible:
                        continue
                    obj = evaluate(inst, trial, self.cfg)["objective"]#生成一个目标值
                    if obj > best_obj:
                        best_obj, best_sol = obj, trial#更新最佳解决方案
                        break

        delta = best_obj - (evaluate(inst, solution, self.cfg)["objective"] if solution else 0.0)
        repaired_base = evaluate(inst, solution, self.cfg)["objective"] if solution else 0.0
        delta = best_obj - repaired_base
        return best_sol if best_sol else repaired, f"mock: applied local edit, delta_J={delta:+.4f}"

    @staticmethod
    def _repair_solution(inst: Instance, solution: Solution, blocked: set) -> Solution:
        out: Solution = {}
        for wid, path in (solution or {}).items():
            try:
                w = inst.worker(wid)
            except KeyError:
                continue
            ok = (len(path) == w.window and w.min_steps <= w.window - 1
                  and (int(path[0][0]), int(path[0][1])) == w.origin
                  and (int(path[-1][0]), int(path[-1][1])) == w.dest
                  and all(int(p[2]) == w.t_start + i for i, p in enumerate(path))
                  and not any((int(p[0]), int(p[1])) in blocked for p in path))
            if ok:
                out[wid] = path
        # 重新规划被移除的工作者
        removed = [wid for wid, path in (solution or {}).items()
                   if wid not in out and wid in {w.wid for w in inst.workers}]
        for wid in removed:
            w = inst.worker(wid)
            if w.min_steps > w.window - 1:
                continue
            cov_t = coverage_of(inst, out)
            cells = _dp_path(inst, w, _weights_from(inst, cov_t))
            if cells is not None and len(cells) == w.window:
                out[wid] = path_to_solution(w, cells)
        # 预算裁剪：移除窗口最小的工作者直到不超预算
        while sum(len(p) for p in out.values()) > inst.budget and out:
            worst = min(out, key=lambda k: inst.worker(k).window)
            del out[worst]
        return out


# --------------------------------------------------------------------------- 3. Eval Agent
class EvalAgent:#EvalAgent类
    def __init__(self, client: LLMClient, cfg: EnvConfig = DEFAULT_ENV):
        self.client = client
        self.cfg = cfg
        self.tokens_used = 0#生成一个令牌使用量

    def evaluate(self, inst: Instance, baseline: Solution, candidate: Solution,
                 disturbance: Dict[str, Any], use_vision: bool = True) -> Dict[str, Any]:
        base_m = evaluate(inst, baseline, self.cfg)#生成一个基线指标
        cand_m = evaluate(inst, candidate, self.cfg)#生成一个候选指标
        vr = tools.validate_worker_paths(inst, candidate)#生成一个验证工作者路径
        dist_check = tools.check_disturbance_handling(inst, candidate, disturbance)#生成一个扰动处理
        vis_text = "disabled"
        if use_vision:
            vis = tools.visual_analysis(inst, baseline, candidate)#生成一个可视化分析
            vis_text = (f"delta_J={vis['improvement']:+.4f}, gain_cells={vis['gain_cells']}, "
                        f"loss_cells={vis['loss_cells']}, png={vis['png']}")#生成一个可视化文本
        valid = bool(vr["overall_feasible"] and dist_check["handled"]
                     and cand_m["objective"] > base_m["objective"] + 1e-12)#生成一个有效性
        result = {
            "baseline": base_m,
            "candidate": cand_m,
            "feasible": vr["overall_feasible"],#生成一个可行性
            "complies": bool(vr["overall_feasible"] and dist_check["handled"]),#生成一个合规性
            "violations": vr["violations"][:5],#生成一个违规信息
            "disturbance_handled": dist_check["handled"],#生成一个扰动处理
            "valid_improvement": valid,#生成一个有效性
            "advice": self._heuristic_advice(inst, candidate, cand_m, vr, dist_check),#生成一个建议
        }
        if self.client.provider != "mock":#如果客户端提供者不是mock
            raw = self.client.complete(
                prompts.EVAL_SYSTEM,
                prompts.eval_user(instance_summary(inst), base_m, cand_m,
                                  json.dumps({"feasible": vr["overall_feasible"], "violations": vr["violations"][:5]}),
                                  vis_text, disturbance.get("description", str(disturbance))),
                json_mode=True)
            self.tokens_used += len(raw) // 4
            parsed = parse_json_block(raw)#解析一个JSON对象
            if parsed:#如果解析成功
                result["eval_summary"] = str(parsed.get("eval_summary", ""))[:300]
                if isinstance(parsed.get("advice"), list) and parsed["advice"]:#如果建议是列表且不为空
                    concrete = result["advice"]
                    llm_advice = [str(a)[:200] for a in parsed["advice"]][:3]#生成一个LLM建议
                    merged, seen = [], set()
                    for a in concrete[:2] + llm_advice:#生成一个合并建议
                        if a not in seen:
                            merged.append(a)
                            seen.add(a)#生成一个合并建议集合
                    result["advice"] = merged[:3]#生成一个合并建议
        return result

    @staticmethod
    def _heuristic_advice(inst: Instance, candidate: Solution, cand: dict, vr, dist_check) -> List[str]:#启发式建议函数
        advice = []
        if not vr["overall_feasible"]:#如果可行性不为真
            advice.append("Fix feasibility first: " + "; ".join(vr["violations"][:2]))
        if not dist_check["handled"]:#如果扰动处理不为真
            advice.append("The solution does not handle the disturbance: "
                          + "; ".join(dist_check["violations"][:2]))
        if cand["cost"] < inst.budget:
            advice.extend(suggest_actions(inst, candidate, k=3))#生成一个建议
        return advice[:3] or ["Fine-tune paths of the least-contributing workers."]#生成一个建议


# --------------------------------------------------------------------------- 4. Memory Agent
@dataclass
class MemoryRecord:
    ops: List[dict]
    lesson: str
    context: str
    phi: float
    delta_j: float


class MemoryAgent:#MemoryAgent类
    """差分 → 元操作 → φ 打分 → 持久化 → top-k 检索（与论文 3.5 节对应）。"""

    def __init__(self, client: Optional[LLMClient] = None, k: int = 3):
        self.client = client#生成一个客户端
        self.k = k
        self.records: List[MemoryRecord] = []#生成一个记忆记录列表

    @staticmethod
    def diff(inst: Instance, before: Solution, after: Solution) -> List[dict]:
        ops = []#生成一个操作列表
        for wid in after.keys() - before.keys():
            ops.append({"op": "add_worker", "worker": wid})#生成一个操作
        for wid in before.keys() - after.keys():
            ops.append({"op": "remove_worker", "worker": wid})#生成一个操作
        for wid in before.keys() & after.keys():
            a = [(p[0], p[1], p[2]) for p in before[wid]]#生成一个a列表
            b = [(p[0], p[1], p[2]) for p in after[wid]]#生成一个b列表
            if a != b:
                changed = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))#生成一个改变步骤
                ops.append({"op": "reroute_path", "worker": wid, "changed_steps": changed})
        return ops#返回操作列表

    def record(self, inst: Instance, before: Solution, after: Solution, delta_j: float,
               disturbance: Dict[str, Any]) -> MemoryRecord:
        ops = self.diff(inst, before, after)#生成一个操作列表
        phi = 1.0 / (1.0 + math.exp(-20.0 * delta_j)) if delta_j > 0 else 0.0
        ctx = f"{disturbance.get('type','none')}|{inst.setting_name}"#生成一个上下文
        lesson = self._lesson(ops, delta_j, disturbance)#生成一个课程
        rec = MemoryRecord(ops=ops, lesson=lesson, context=ctx, phi=phi, delta_j=delta_j)
        self.records.append(rec)#添加到记忆记录列表
        return rec

    def _lesson(self, ops: List[dict], delta_j: float, disturbance: Dict[str, Any]) -> str:#课程函数
        if self.client is not None and self.client.provider != "mock":
            raw = self.client.complete(prompts.MEMORY_SYSTEM,#生成一个原始文本
                                       json.dumps({"ops": ops, "delta_j": delta_j,
                                                   "disturbance": disturbance}, ensure_ascii=False),
                                       json_mode=True)
            parsed = parse_json_block(raw)#解析一个JSON对象
            if parsed and parsed.get("lesson"):
                return str(parsed["lesson"])[:120]#返回课程 120字符长度
        kinds = {o["op"] for o in ops}
        if "reroute_path" in kinds and delta_j > 0:#如果改变路径
            return "re-routing the least-contributing worker improves balance with no extra budget"
        if "add_worker" in kinds and delta_j > 0:#如果添加工作者
            return "recruiting a worker in uncovered regions pays off when budget allows"
        if "remove_worker" in kinds and delta_j > 0:#如果删除工作者
            return "dropping a redundant worker frees budget for better covered regions"
        return f"no reliable lesson (delta_J={delta_j:+.4f})"#返回课程

    def _tokens(self, text: str) -> set:#tokens函数
        return set(re.findall(r"[a-z_]+|[0-9]+", text.lower()))

    def retrieve(self, inst: Instance, disturbance: Dict[str, Any]) -> List[str]:#检索函数
        if not self.records:
            return []
        query = self._tokens(f"{disturbance.get('type','none')}|{inst.setting_name}")#生成一个查询
        scored = []
        for r in self.records:#生成一个得分列表
            toks = self._tokens(r.context)
            sim = len(query & toks) / max(1, len(query | toks))#生成一个相似度
            scored.append((sim * 0.5 + r.phi * 0.5, r))
        scored.sort(key=lambda x: -x[0])#生成一个得分列表
        out = []
        for s, r in scored[: self.k]:#生成一个输出列表
            if s <= 0:
                continue
            ops = ", ".join(o["op"] for o in r.ops[:4])#生成一个操作列表
            out.append(f"[{r.context}] {ops} -> {r.lesson} (phi={r.phi:.2f})")
        return out#返回输出列表
