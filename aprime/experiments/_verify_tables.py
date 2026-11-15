# -*- coding: utf-8 -*-
"""核对 docx 表格列数与 markdown 是否一致，并检查数学码位。"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx import Document                      # noqa: E402
import pymupdf as fitz                         # noqa: E402

from md_to_docx import split_row, is_sep_row   # noqa: E402

MD = r"d:/composition/aprime/paper_draft.md"

# markdown 侧：收集所有表格及其列数
lines = open(MD, encoding="utf-8").read().split("\n")
md_tables = []
i = 0
while i < len(lines):
    if lines[i].strip().startswith("|"):
        buf = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            buf.append(lines[i])
            i += 1
        rows = [split_row(b) for b in buf]
        if len(rows) >= 2 and is_sep_row(rows[1]):
            md_tables.append(len(rows[0]))
        else:
            md_tables.append(max(len(r) for r in rows))
    else:
        i += 1

doc = Document(r"d:/composition/aprime/paper_draft.docx")
docx_tables = [len(t.columns) for t in doc.tables]

print("markdown 表格数:", len(md_tables), " docx 表格数:", len(docx_tables))
bad = 0
for idx, (a, b) in enumerate(zip(md_tables, docx_tables), 1):
    flag = "OK " if a == b else "MISMATCH"
    if a != b:
        bad += 1
        print(f"  表{idx:<3} md={a} docx={b}  <-- {flag}")
print("列数一致:", "全部一致" if bad == 0 and len(md_tables) == len(docx_tables)
      else f"{bad} 处不一致")

# 数学码位检查
pdf = fitz.open(r"d:/composition/aprime/paper_draft.pdf")
txt = "".join(p.get_text() for p in pdf)
print("\n数学斜体码位：")
for name, ch in [("theta", "\U0001d703"), ("phi", "\U0001d719"),
                 ("Omega", "\U0001d6fa"), ("Delta", "\U0001d6ab")]:
    print(f"  {name:<7} U+{ord(ch):05X} count={txt.count(ch)}")
print("\n希腊直体（应少见于公式）: φ=%d Ω=%d" % (txt.count("φ"), txt.count("Ω")))
