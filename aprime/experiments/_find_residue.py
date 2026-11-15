# -*- coding: utf-8 -*-
"""定位 PDF 里残留的 LaTeX 片段上下文。"""
import pymupdf as fitz

doc = fitz.open(r"d:/composition/aprime/paper_draft.pdf")
txt = "".join(p.get_text() for p in doc)

for needle in ["\\le ", "\\(", "\\[", "\\frac", "\\theta"]:
    start = 0
    found = False
    while True:
        idx = txt.find(needle, start)
        if idx < 0:
            break
        found = True
        print(f"=== {needle!r} @ {idx} ===")
        print(repr(txt[max(0, idx - 110): idx + 110]))
        print()
        start = idx + 1
    if not found:
        print(f"=== {needle!r}: 无 ===\n")
