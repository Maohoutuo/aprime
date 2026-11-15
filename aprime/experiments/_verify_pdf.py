# -*- coding: utf-8 -*-
"""检查导出的 PDF：LaTeX 残留、公式符号、结构。"""
import re

import pymupdf as fitz

doc = fitz.open(r"d:/composition/aprime/paper_draft.pdf")
txt = "".join(p.get_text() for p in doc)
print("页数:", doc.page_count, "字符数:", len(txt))

print("\n--- LaTeX 源码残留（应为 0）---")
checks = ["\\theta", "\\frac", "\\Delta", "\\text{", "\\ge ", "\\le ",
          "\\(", "\\[", "$"]
for c in checks:
    print(f"  {c!r:<12} {txt.count(c)}")

print("\n--- 数学符号是否出现在 PDF 文本中 ---")
for sym, name in [("θ", "theta"), ("φ", "phi"), ("Δ", "Delta"),
                  ("≤", "le"), ("≥", "ge"), ("Ω", "Omega"), ("∈", "in")]:
    print(f"  {name:<8} {sym!r} count={txt.count(sym)}")

print("\n--- 结构抽查 ---")
for key in ["摘要", "6.5 GQC 的效果", "6.11", "参考文献"]:
    print(f"  {key!r:<20} {'FOUND' if key in txt else 'MISSING'}")

print("\n--- 第 1 页前 300 字 ---")
print(doc[0].get_text()[:300])
