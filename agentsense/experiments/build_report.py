# -*- coding: utf-8 -*-
"""把 results/ 下的实验输出汇总回 report.html（单文件网页，双击即可打开）。

原理：
    report.html 里三个数据块用注释包围，例如

        /* T4:START */
        const T4 = { ... };
        /* T4:END */

    本脚本读取 results/*.json，生成新的 const 声明，替换两个标记之间的内容。
    只要标记还在，重复执行就是幂等的。

用法：
    py -3.14 experiments/build_report.py            # 汇总并写回 report.html
    py -3.14 experiments/build_report.py --dry-run  # 只看会写什么，不改文件
    py -3.14 experiments/build_report.py --report docs/report.html

run_table4.py / run_llm.py 跑完会自动调用 rebuild()，一般不需要手动执行。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report.html"

# 与 agentsense/disturbances.py 保持一致的机制说明
MECH = {
    "bad_weather": "30% 的格子价值降到 0.3",
    "area_blocked": "10% 的格子被封锁，不可进入",
    "worker_unavailable": "随机移除 N/8 名工作者（≤3）",
    "traffic_congestion": "时间窗缩短 30%",
    "time_window_shift": "所有时间窗后移 2 个时段",
    "budget_cut": "预算削减 25%",
    "task_priority_change": "α 由 0.5 变为 0.3",
}


# ----------------------------------------------------------------- 读文件
def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


# ----------------------------------------------------------------- 表 4
def collect_table4(results: Path = RESULTS) -> dict:
    """{"n=20": {"small/RN": {ours, paper, err, cost, sec}, ...}, ...}"""
    found: Dict[str, dict] = {}
    for path in sorted(results.glob("table4_*.json")):
        m = re.search(r"table4_n(\d+)", path.name)
        if not m:
            continue
        found[f"n={m.group(1)}"] = {
            key: {
                "ours": v["ours"],
                "paper": v["paper"],
                "err": v["err_pct"],
                "cost": v["cost"],
                "sec": v["seconds"],
            }
            for key, v in _load(path).items()
        }
    # n 大的排前面
    return dict(sorted(found.items(), key=lambda kv: -int(kv[0].split("=")[1])))


# ----------------------------------------------------------------- 表 3
def _row_out(row: dict) -> dict:
    return {k: row.get(k) for k in ("SR", "AIR", "ANI", "ACS", "tokens")}


def collect_table3(results: Path = RESULTS) -> dict:
    """每次都独立成 tab，按文件修改时间排序；内容完全相同的只保留一份。"""
    paths = sorted(results.glob("real_table3_*.json"), key=_mtime)
    out: Dict[str, dict] = {}
    seen: set[str] = set()

    for path in paths:
        data = _load(path)
        rows = data.get("rows") or {}
        if not rows:
            continue
        fingerprint = json.dumps(rows, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        stamp = datetime.fromtimestamp(_mtime(path)).strftime("%H:%M")
        label = f"运行 {chr(ord('A') + len(out))} · {stamp}"
        out[label] = {
            "src": (f"{path.name} · n={data.get('n')} · {data.get('scale')} · "
                    f"T={data.get('temperature')}"),
            "rows": {
                "单 Agent": _row_out(rows.get("single-agent") or {}),
                "AgentSense": _row_out(rows.get("AgentSense") or {}),
            },
        }
    return out


# ----------------------------------------------------------------- 表 5
def _t5_rows(rows: dict) -> dict:
    return {
        name: {
            "mech": MECH.get(name, "—"),
            "SR": v.get("SR", 0),
            "AIR": v.get("AIR", 0),
            "ANI": v.get("ANI", 0),
            "ACS": v.get("ACS", 0),
            "tokens": v.get("tokens") or None,
        }
        for name, v in rows.items()
        if name in MECH
    }


def collect_table5(results: Path = RESULTS) -> dict:
    out: Dict[str, dict] = {}

    reals = sorted(results.glob("real_table5_*.json"), key=_mtime)
    if reals:
        path = reals[-1]                      # 取最新一次
        data = _load(path)
        rows = data.get("rows") or {}
        if rows:
            out[f"真实 LLM · n={data.get('n')}"] = {
                "src": (f"{path.name} · {data.get('model', '')} · "
                        f"{data.get('scale', '')} · T={data.get('temperature')}"),
                "rows": _t5_rows(rows),
            }

    mock = results / "mock_table5.json"
    if mock.exists():
        data = _load(mock)
        rows = data.get("rows") or data       # mock 文件可能是扁平结构
        rows = {k: v for k, v in rows.items() if isinstance(v, dict)}
        if rows:
            n = next((v.get("n") for v in rows.values() if v.get("n")), None)
            out[f"Mock 替身 · n={n}"] = {
                "src": f"{mock.name} · 确定性算法假扮 LLM，不代表方法有效性",
                "rows": _t5_rows(rows),
            }
    return out


# ----------------------------------------------------------------- 写回
def _js(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _patch(html: str, tag: str, block: str) -> str:
    start, end = f"/* {tag}:START */", f"/* {tag}:END */"
    i = html.index(start)
    j = html.index(end, i)
    return f"{html[:i]}{start}\n{block}\n{end}{html[j + len(end):]}"


def rebuild(report: Path = REPORT, results: Path = RESULTS, dry_run: bool = False) -> dict:
    """重新生成三个数据块并写回 report.html，返回 {块名: 条目数}。"""
    html = report.read_text(encoding="utf-8")
    stats: Dict[str, int] = {}

    for tag, collect in (("T4", collect_table4), ("T3", collect_table3), ("T5", collect_table5)):
        data = collect(results)
        stats[tag] = len(data)
        if not data:
            print(f"[skip] {tag}：results/ 下没找到对应数据，保持原样")
            continue
        html = _patch(html, tag, f"const {tag} = {_js(data)};")

    if dry_run:
        print("[dry-run] 未写入文件")
    else:
        report.write_text(html, encoding="utf-8")
        print(f"wrote {report}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 results/*.json 到 report.html")
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats = rebuild(Path(args.report), Path(args.results), args.dry_run)
    print("数据块：" + "  ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
