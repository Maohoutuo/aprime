# -*- coding: utf-8 -*-
r"""把 paper_draft.md 转成 Word .docx。

公式走 LaTeX -> MathML(latex2mathml) -> OMML(Word 自带 MML2OMML.XSL)，
产出的是 **Word 原生公式**，在 Word 里可以继续双击编辑，而不是截图。

用法：
    py experiments/md_to_docx.py
    py experiments/md_to_docx.py --src paper_draft.md --out paper_draft.docx
"""

from __future__ import annotations

import argparse
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt
from latex2mathml.converter import convert
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

XSL_CANDIDATES = [
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL",
]

_TRANSFORM = None


def _transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        for path in XSL_CANDIDATES:
            if os.path.exists(path):
                _TRANSFORM = etree.XSLT(etree.parse(path))
                break
        else:
            raise RuntimeError("找不到 MML2OMML.XSL（需要安装 Word）")
    return _TRANSFORM


STATS = {"math_omml": 0, "math_fallback": 0, "tables": 0, "images": 0,
         "headings": 0, "paras": 0}


def latex_to_omml(latex: str):
    """LaTeX -> OMML 元素；失败返回 None。"""
    try:
        mathml = convert(latex)
        node = etree.fromstring(mathml.encode("utf-8"))
        out = _transform()(node)
        xml = etree.tostring(out, encoding="unicode")
        # XSLT 有时包一层 oMathPara，取里面的 oMath 便于内联插入
        return parse_xml(xml)
    except Exception:
        return None


# ---- 富文本切分：代码 / 粗体 / 斜体 -------------------------------------
RICH = re.compile(r"(`[^`]+`)|(\*\*.+?\*\*)|(\*[^*\n]+?\*)")


def rich_segments(text: str):
    out, pos = [], 0
    for m in RICH.finditer(text):
        if m.start() > pos:
            out.append(("plain", text[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append(("code", tok[1:-1]))
        elif tok.startswith("**"):
            out.append(("bold", tok[2:-2]))
        else:
            out.append(("italic", tok[1:-1]))
        pos = m.end()
    if pos < len(text):
        out.append(("plain", text[pos:]))
    return out


MATH_SPLIT = re.compile(r"(\\\(.*?\\\)|\\\[.*?\\\])", re.S)
MATH_SPAN = MATH_SPLIT
PROTECT = "\x01"


def add_rich(paragraph, text: str, base_size: Pt | None = None, italic_all=False):
    """把一段 markdown 文本写入 paragraph，公式转 OMML。"""
    for part in MATH_SPLIT.split(text):
        if not part:
            continue
        stripped = part.strip()
        if stripped.startswith("\\(") and stripped.endswith("\\)"):
            latex = stripped[2:-2]
            omml = latex_to_omml(latex)
            if omml is not None:
                paragraph.add_run()._r.append(omml)
                STATS["math_omml"] += 1
            else:
                r = paragraph.add_run(latex)
                r.italic = True
                STATS["math_fallback"] += 1
            continue
        if stripped.startswith("\\[") and stripped.endswith("\\]"):
            latex = stripped[2:-2]
            omml = latex_to_omml(latex)
            if omml is not None:
                paragraph.add_run()._r.append(omml)
                STATS["math_omml"] += 1
            else:
                paragraph.add_run(latex)
                STATS["math_fallback"] += 1
            continue
        for kind, seg in rich_segments(part):
            run = paragraph.add_run(seg)
            if base_size is not None:
                run.font.size = base_size
            if kind == "bold":
                run.bold = True
            elif kind == "italic":
                run.italic = True
            elif kind == "code":
                run.font.name = "Consolas"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            if italic_all and kind == "plain":
                run.italic = True


# ---- 表格 ---------------------------------------------------------------
def split_row(line: str):
    """切表格行。注意：不能切在公式内的 `|` 上（如 \\(|\\Omega|\\le B\\) 的基数竖线）。"""
    line = MATH_SPAN.sub(lambda m: m.group(0).replace("|", PROTECT), line.strip())
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip().replace(PROTECT, "|") for c in line.split("|")]


def is_sep_row(cells):
    return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c != "")


def add_table(doc, lines):
    rows = [split_row(ln) for ln in lines]
    if len(rows) >= 2 and is_sep_row(rows[1]):
        header, body = rows[0], rows[2:]
    else:
        header, body = None, rows
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=(1 if header else 0) + len(body), cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    def fill(cells, values, bold=False):
        for i in range(ncols):
            cell = cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            txt = values[i] if i < len(values) else ""
            add_rich(p, txt, base_size=Pt(8.5))
            if bold:
                for r in p.runs:
                    r.bold = True

    start = 0
    if header:
        fill(table.rows[0].cells, header, bold=True)
        start = 1
    for j, r in enumerate(body):
        fill(table.rows[start + j].cells, r)
    STATS["tables"] += 1
    doc.add_paragraph()


IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def add_image(doc, target: str):
    path = target.strip()
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        p = doc.add_paragraph()
        p.add_run(f"[缺图: {target}]").italic = True
        return
    doc.add_picture(path, width=Inches(5.9))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    STATS["images"] += 1


# ---- 主流程 -------------------------------------------------------------
def convert_file(src: str, out: str):
    text = open(src, encoding="utf-8").read()
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(20)
    sec.top_margin = sec.bottom_margin = Mm(20)

    lines = text.split("\n")
    n = len(lines)
    i = 0

    def is_blank(x):
        return x.strip() == ""

    def special(line):
        t = line.strip()
        if not t:
            return True
        if re.match(r"^#{1,4}\s+", t):
            return True
        if t.startswith("|") or t.startswith(">"):
            return True
        if re.match(r"^[-*]\s+", t) or re.match(r"^\d+\.\s+", t):
            return True
        if IMG.fullmatch(t):
            return True
        if re.fullmatch(r"-{3,}|\*{3,}", t):
            return True
        return t.startswith("```")

    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        # 围栏代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(buf))
            r.font.name = "Consolas"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            r.font.size = Pt(9)
            continue

        # 标题（可连续出现，不要求空行分隔）
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            p = doc.add_heading("", level=min(len(m.group(1)), 4))
            add_rich(p, m.group(2))
            STATS["headings"] += 1
            i += 1
            continue

        if re.fullmatch(r"-{3,}|\*{3,}", s):
            doc.add_paragraph()
            i += 1
            continue

        mi = IMG.fullmatch(s)
        if mi:
            add_image(doc, mi.group(1))
            i += 1
            continue

        if s.startswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            add_table(doc, buf)
            continue

        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i].strip()))
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Mm(6)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            add_rich(p, " ".join(buf), base_size=Pt(9.5), italic_all=True)
            STATS["paras"] += 1
            continue

        if re.match(r"^[-*]\s+", s):
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                p = doc.add_paragraph(style="List Bullet")
                add_rich(p, re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            STATS["paras"] += 1
            continue

        if re.match(r"^\d+\.\s+", s):
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                p = doc.add_paragraph(style="List Number")
                add_rich(p, re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            STATS["paras"] += 1
            continue

        # 普通段落：连续非特殊行；行尾两个空格视为硬换行
        buf = []
        while i < n and not is_blank(lines[i]) and not special(lines[i]):
            cur = lines[i]
            hard = cur.endswith("  ")
            buf.append(cur.strip())
            i += 1
            if hard:
                break
        if buf:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            add_rich(p, " ".join(buf))
            STATS["paras"] += 1

    doc.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "paper_draft.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper_draft.docx"))
    args = ap.parse_args()
    convert_file(args.src, args.out)
    print(f"wrote {args.out}")
    print("统计:", STATS)


if __name__ == "__main__":
    main()
