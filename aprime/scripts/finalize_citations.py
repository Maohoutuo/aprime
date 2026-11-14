# -*- coding: utf-8 -*-
r"""把正文里的 `[CITE-n]` 占位符替换成完整引用。

用法：
    1. 由 A 填好 `docs/citation_numbers_alignment.md` 的总表最后两列；
    2. 把结果填进本文件的 `CITATIONS`（只改字符串，不动别的）；
    3. 运行 `py -3.14 scripts/finalize_citations.py`；
    4. 脚本报告替换了几处、是否还有残留占位符。

设计约束：
    * **只替换占位符，不改任何正文措辞**——定理 2 的措辞与代码回归测试绑定，改动会连带测试；
    * 未填条的条目保持占位符原样，并明确报出"待 A 填"，绝不静默补一个号（错号比没号更糟）；
    * 幂等：重复运行不会重复插入。
"""

from __future__ import annotations

import io
import os
import re
import sys

# ---------------------------------------------------------------------------
# 本机无外网，**定理号/页码无法核实，故一律不编**：文献元数据（作者–年份–标题–
# 期刊卷页）已按可确定程度填全，具体定理号标注为「〔定理号待核〕」。
# 若后续拿到确切定理号，只需改下面字符串再重跑本脚本（幂等）。
# ---------------------------------------------------------------------------
CITATIONS = {
    # 划分拟阵的 B-截断仍是拟阵（H2 + 证明骨架步骤 2）
    "CITE-1": ("Oxley, Matroid Theory, 2nd ed., Oxford University Press, 2011, "
               "§1.2（截断/truncation）〔定理号待核〕"),
    # KMN 1999 纯密度贪心的 ½(1−1/e)（H3 + 证明骨架步骤 3）
    "CITE-2": ("Khuller, Moss & Naor (1999), The budgeted maximum coverage problem, "
               "Information Processing Letters 70(1):39–45〔定理号待核〕"),
    # Fisher–Nemhauser–Wolsey 1978 拟阵上边际贪心 ≥ ½（证明骨架步骤 2）
    "CITE-3": ("Fisher, Nemhauser & Wolsey (1978), An analysis of approximations for "
               "maximizing submodular set functions — I, Mathematical Programming "
               "14(1):265–294〔定理号待核〕"),
    # 可选：引理 1 的外部经典出处；留空即保持"引理 1（本文证明）"
    "CITE-4": "",
}

DOCS = [
    "docs/theory_alpha_min.md",
    "docs/paper_sec6_experiments.md",
]

PATTERN = re.compile(r"\*\*\[CITE-(\d+)\]\*\*")


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved = {k: v.strip() for k, v in CITATIONS.items() if v.strip()}
    pending = [k for k, v in CITATIONS.items() if not v.strip()]

    if not resolved:
        print("没有任何条目被填写。请先在 scripts/finalize_citations.py 的 CITATIONS 里填入 A 给的编号。")
        print("待填条目：" + ", ".join(sorted(pending)))
        return 1

    total = 0
    touched: set = set()
    for rel in DOCS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"[skip] 不存在：{rel}")
            continue
        src = io.open(path, encoding="utf-8").read()

        def repl(m: re.Match) -> str:
            key = f"CITE-{m.group(1)}"
            if key not in resolved:
                return m.group(0)
            touched.add(key)
            return resolved[key]

        new, _n_matched = PATTERN.subn(repl, src)
        # subn 的计数含"保持原样"的匹配，故用 touched 判断真实改动，避免虚报
        real = sum(1 for m in PATTERN.finditer(src) if f"CITE-{m.group(1)}" in resolved)
        if real:
            io.open(path, "w", encoding="utf-8", newline="\n").write(new)
        total += real
        print(f"[ok] {rel}: 实际替换 {real} 处")

    for key in sorted(set(resolved) - touched):
        print(f"[注意] {key} 已填写，但正文里没有对应占位符（CITE-4 属可选条目，可忽略）")

    left = []
    for rel in DOCS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        new = io.open(path, encoding="utf-8").read()
        left += [f"{rel}:{m.group(0)}" for m in PATTERN.finditer(new)]

    print(f"\n合计替换 {total} 处。")
    if pending:
        print("仍未填写（A 待回）：" + ", ".join(sorted(pending)))
    if left:
        print("文档内仍有占位符：")
        for item in left:
            print("  " + item)
    else:
        print("文档内已无残留占位符。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
