# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "课程论文_待补个人信息.docx"
PDF = ROOT / "课程论文_待补个人信息.pdf"
MD = ROOT / "course_paper.md"


def main():
    assert DOCX.exists() and DOCX.stat().st_size > 100_000
    assert PDF.exists() and PDF.stat().st_size > 100_000

    doc = Document(str(DOCX))
    paragraphs = [p.text for p in doc.paragraphs]
    cell_text = [c.text for t in doc.tables for row in t.rows for c in row.cells]
    text = "\n".join(paragraphs + cell_text)

    forbidden = [
        "说 明",
        "内容要求与数字语音信号相关",
        "正式上交课程论文时",
        "×××（五号字）",
        "[FIG ",
        "[EQ ",
        "[TABLECAP]",
    ]
    found_forbidden = [item for item in forbidden if item in text]
    assert not found_forbidden, found_forbidden

    refs = re.findall(r"^\[(\d+)\]\s", text, flags=re.M)
    assert len(refs) == 36 and refs[0] == "1" and refs[-1] == "36", refs
    assert len(doc.inline_shapes) == 2, len(doc.inline_shapes)
    # 6 equation tables + 4 result tables + 1 author bio table + 1 assessment table.
    assert len(doc.tables) == 12, len(doc.tables)
    assert text.count("研究生课程论文评定") == 1
    assert "作者照片与简介" in text
    assert "T-Drive参数标定的合成参与者" in text

    source = MD.read_text(encoding="utf-8")
    zh = re.search(r"^\[ABSTRACT_ZH\]\s*(.+)$", source, re.M).group(1)
    assert 300 <= len(zh) <= 500, len(zh)
    assert len(re.findall(r"^\[\d+\]\s", source, flags=re.M)) == 36

    pdf = pymupdf.open(str(PDF))
    assert len(pdf) == 18, len(pdf)
    page_texts = [page.get_text() for page in pdf]
    assert all(t.strip() for t in page_texts)
    assert "研究生课程论文" in page_texts[0]
    assert "个性化满意度硬约束" in page_texts[1]
    assert "Fair Route Selection" in page_texts[2]
    assert "作者照片与简介" in page_texts[-2]
    assert "研究生课程论文评定" in page_texts[-1]
    assert (ROOT / "figures" / "fig3_tradeoff_panels_course.png").exists()
    assert (ROOT / "figures" / "fig3b_scale_check_course.png").exists()

    out_of_bounds = []
    for page_no, page in enumerate(pdf, 1):
        rect = page.rect
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < -1 or y0 < -1 or x1 > rect.width + 1 or y1 > rect.height + 1:
                out_of_bounds.append((page_no, [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]))
    assert not out_of_bounds, out_of_bounds

    placeholders = sorted(set(re.findall(r"【[^】]+】", text)))
    report = {
        "docx_bytes": DOCX.stat().st_size,
        "pdf_bytes": PDF.stat().st_size,
        "pages": len(pdf),
        "references": len(refs),
        "inline_figures": len(doc.inline_shapes),
        "tables_total": len(doc.tables),
        "equations_expected": 6,
        "chinese_abstract_characters": len(zh),
        "placeholders_to_fill": placeholders,
        "forbidden_template_text": found_forbidden,
        "out_of_bounds_blocks": out_of_bounds,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
