# -*- coding: utf-8 -*-
"""Build the course paper inside the university Word template.

The script keeps the original cover and final assessment form, removes the
instruction/example pages, and inserts the research paper in between.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt
from latex2mathml.converter import convert
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "course_paper.md"
TEMPLATE = ROOT / "course_paper_template.docx"
OUT = ROOT / "课程论文_待补个人信息.docx"

XSL_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL"),
]

TRANSFORM = None
STATS = {"equations": 0, "tables": 0, "figures": 0, "references": 0, "paragraphs": 0}


def get_transform():
    global TRANSFORM
    if TRANSFORM is None:
        for path in XSL_CANDIDATES:
            if path.exists():
                TRANSFORM = etree.XSLT(etree.parse(str(path)))
                break
        if TRANSFORM is None:
            raise RuntimeError("MML2OMML.XSL not found; Microsoft Word is required")
    return TRANSFORM


def latex_to_omml(latex: str):
    mathml = convert(latex)
    node = etree.fromstring(mathml.encode("utf-8"))
    out = get_transform()(node)
    return parse_xml(etree.tostring(out, encoding="unicode"))


def set_run_font(run, size=12, bold=None, italic=None, east="宋体", west="Times New Roman"):
    run.font.name = west
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), east)
    rfonts.set(qn("w:ascii"), west)
    rfonts.set(qn("w:hAnsi"), west)
    return run


def clear_paragraph(paragraph):
    p = paragraph._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)


def set_keep_with_next(paragraph, value=True):
    paragraph.paragraph_format.keep_with_next = value


def set_body_format(paragraph, first_line=True, line_spacing=2.0, size=12):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = line_spacing
    fmt.first_line_indent = Pt(size * 2) if first_line else Pt(0)
    fmt.widow_control = True


def add_plain(paragraph, text, size=12, bold=False, italic=False, east="宋体", west="Times New Roman"):
    run = paragraph.add_run(text)
    return set_run_font(run, size=size, bold=bold, italic=italic, east=east, west=west)


def add_labeled_paragraph(marker, label, text, english=False):
    p = marker.insert_paragraph_before()
    set_body_format(p, first_line=False, line_spacing=2.0, size=12)
    add_plain(p, label, size=12, bold=True, east="宋体" if not english else "Times New Roman")
    add_plain(p, text, size=12, east="宋体" if not english else "Times New Roman")
    STATS["paragraphs"] += 1
    return p


def add_body_paragraph(marker, text):
    p = marker.insert_paragraph_before()
    set_body_format(p, first_line=True, line_spacing=2.0, size=12)
    add_plain(p, text, size=12)
    STATS["paragraphs"] += 1
    return p


def add_heading(marker, text, level=1):
    p = marker.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = p.paragraph_format
    fmt.first_line_indent = Pt(0)
    fmt.line_spacing = 2.0
    fmt.space_before = Pt(10 if level == 1 else 6)
    fmt.space_after = Pt(0)
    set_keep_with_next(p, True)
    add_plain(p, text, size=12, bold=True, east="黑体")
    STATS["paragraphs"] += 1
    return p


def add_centered(marker, text, size, bold=False, english=False, page_break_before=False):
    p = marker.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = p.paragraph_format
    fmt.first_line_indent = Pt(0)
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(3)
    fmt.line_spacing = 1.5
    fmt.page_break_before = page_break_before
    set_keep_with_next(p, True)
    add_plain(
        p,
        text,
        size=size,
        bold=bold,
        east="Times New Roman" if english else ("黑体" if bold else "宋体"),
        west="Times New Roman",
    )
    STATS["paragraphs"] += 1
    return p


def add_top_border(paragraph):
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is None:
        pbdr = OxmlElement("w:pBdr")
        ppr.append(pbdr)
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), "8")
    top.set(qn("w:space"), "4")
    top.set(qn("w:color"), "000000")
    pbdr.append(top)


def set_table_borders(table, value="nil"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), value)
        node.set(qn("w:sz"), "0" if value == "nil" else "4")
        node.set(qn("w:color"), "auto")


def move_table_before(table, marker):
    marker._p.addprevious(table._tbl)


def add_equation(doc, marker, number, latex):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Mm(145)
    table.columns[1].width = Mm(15)
    set_table_borders(table, "nil")
    move_table_before(table, marker)

    left = table.cell(0, 0)
    right = table.cell(0, 1)
    left.width = Mm(145)
    right.width = Mm(15)
    p = left.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    try:
        node = latex_to_omml(latex)
        if node.tag == qn("m:oMathPara"):
            p._p.append(node)
        else:
            p.add_run()._r.append(node)
    except Exception:
        add_plain(p, latex, size=11, italic=True)
    p2 = right.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p2.paragraph_format.space_before = Pt(3)
    p2.paragraph_format.space_after = Pt(3)
    add_plain(p2, f"({number})", size=11)
    STATS["equations"] += 1


def split_md_row(line):
    line = line.strip().strip("|")
    return [cell.strip() for cell in line.split("|")]


def is_separator_row(cells):
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells)


def set_repeat_table_header(row):
    trpr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    trpr.append(tbl_header)


def add_markdown_table(doc, marker, lines):
    rows = [split_md_row(line) for line in lines]
    if len(rows) > 1 and is_separator_row(rows[1]):
        header, body = rows[0], rows[2:]
    else:
        header, body = None, rows
    ncols = max(len(row) for row in rows)
    table = doc.add_table(rows=(1 if header else 0) + len(body), cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table, "single")
    move_table_before(table, marker)

    def fill(row, values, bold=False):
        for idx, cell in enumerate(row.cells):
            cell.text = ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            text = values[idx] if idx < len(values) else ""
            add_plain(p, text, size=8.5, bold=bold)

    offset = 0
    if header:
        fill(table.rows[0], header, bold=True)
        set_repeat_table_header(table.rows[0])
        offset = 1
    for idx, row in enumerate(body):
        fill(table.rows[offset + idx], row)
    STATS["tables"] += 1


def add_table_caption(marker, text):
    p = marker.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    set_keep_with_next(p, True)
    add_plain(p, text, size=10.5)
    STATS["paragraphs"] += 1


def add_figure(marker, path, caption):
    p = marker.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Mm(154))
    cap = marker.insert_paragraph_before()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Pt(0)
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.line_spacing = 1.0
    add_plain(cap, caption, size=10.5)
    STATS["figures"] += 1


def add_reference(marker, text):
    p = marker.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = p.paragraph_format
    fmt.first_line_indent = Pt(-18)
    fmt.left_indent = Pt(18)
    fmt.line_spacing = 1.0
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(2)
    add_plain(p, text, size=9, east="宋体")
    STATS["references"] += 1


def add_page_break(marker):
    p = marker.insert_paragraph_before()
    p.add_run().add_break(WD_BREAK.PAGE)


def add_bio_table(doc, marker, photo_text, bio_text):
    title = marker.insert_paragraph_before()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(6)
    add_plain(title, "作者照片与简介", size=12, bold=True, east="黑体")

    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, "single")
    table.columns[0].width = Mm(38)
    table.columns[1].width = Mm(118)
    move_table_before(table, marker)
    row = table.rows[0]
    row.height = Mm(48)
    row.height_rule = WD_ROW_HEIGHT_RULE.EXACTLY
    left, right = row.cells
    left.width = Mm(38)
    right.width = Mm(118)
    left.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    right.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    p1 = left.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = add_plain(p1, photo_text, size=10.5, bold=True)
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW

    p2 = right.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p2.paragraph_format.first_line_indent = Pt(24)
    p2.paragraph_format.line_spacing = 1.5
    add_plain(p2, bio_text, size=12)


def fill_cover(doc):
    paragraphs = doc.paragraphs
    fields = {
        7: ("课  程：", "【请填写课程名称】"),
        9: ("题  目：", "个性化满意度硬约束下的移动众感公平路线选择方法"),
        11: ("学  院：", "【请填写学院】"),
        13: ("专  业：", "【请填写专业】"),
        15: ("姓  名：", "【请填写姓名】"),
        17: ("学  号：", "【请填写学号】"),
        19: ("任课教师：", "【请填写任课教师】"),
    }
    for idx, (label, value) in fields.items():
        p = paragraphs[idx]
        clear_paragraph(p)
        p.paragraph_format.line_spacing = 1.5
        label_run = add_plain(p, label + " ", size=12, bold=True, east="黑体")
        value_run = add_plain(p, value, size=12, bold=True, east="宋体")
        value_run.underline = True
        if "【" in value:
            value_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    p = paragraphs[21]
    clear_paragraph(p)
    p.paragraph_format.line_spacing = 1.5
    add_plain(p, "提交日期： ", size=12, bold=True, east="黑体")
    value_run = add_plain(p, "2026 年 11 月 26 日", size=12, bold=True, east="宋体")
    value_run.underline = True


def remove_template_instructions(doc):
    paragraphs = list(doc.paragraphs)
    marker = paragraphs[64]
    for paragraph in paragraphs[26:64]:
        element = paragraph._element
        element.getparent().remove(element)
    return marker


def read_source():
    text = SRC.read_text(encoding="utf-8")
    meta_names = {
        "TITLE_ZH", "AUTHOR_ZH", "AFFIL_ZH", "EMAIL_ZH", "ABSTRACT_ZH", "KEYWORDS_ZH",
        "TITLE_EN", "AUTHOR_EN", "AFFIL_EN", "EMAIL_EN", "ABSTRACT_EN", "KEYWORDS_EN",
        "AUTHOR_NOTE",
    }
    meta = {}
    body = []
    for line in text.splitlines():
        match = re.match(r"^\[([A-Z_]+)\]\s*(.*)$", line)
        if match and match.group(1) in meta_names:
            meta[match.group(1)] = match.group(2)
        else:
            body.append(line)
    return meta, body


def add_front_matter(marker, meta):
    add_centered(marker, meta["TITLE_ZH"], size=18, bold=True, page_break_before=True)
    add_centered(marker, meta["AUTHOR_ZH"], size=14, bold=True)
    add_centered(marker, meta["AFFIL_ZH"], size=10.5)
    add_centered(marker, meta["EMAIL_ZH"], size=10.5)
    add_labeled_paragraph(marker, "摘  要：", meta["ABSTRACT_ZH"])
    add_labeled_paragraph(marker, "关键词：", meta["KEYWORDS_ZH"])

    add_centered(marker, meta["TITLE_EN"], size=16, bold=True, english=True)
    add_centered(marker, meta["AUTHOR_EN"], size=12, bold=True, english=True)
    add_centered(marker, meta["AFFIL_EN"], size=10.5, english=True)
    add_centered(marker, meta["EMAIL_EN"], size=10.5, english=True)
    add_labeled_paragraph(marker, "Abstract: ", meta["ABSTRACT_EN"], english=True)
    add_labeled_paragraph(marker, "Key words: ", meta["KEYWORDS_EN"], english=True)

    note = marker.insert_paragraph_before()
    add_top_border(note)
    note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    note.paragraph_format.first_line_indent = Pt(0)
    note.paragraph_format.line_spacing = 1.0
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(4)
    r = add_plain(note, meta["AUTHOR_NOTE"], size=9)
    if "【" in meta["AUTHOR_NOTE"]:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def parse_body(doc, marker, lines):
    index = 0
    in_references = False
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue

        heading = re.match(r"^(#{1,2})\s+(.+)$", line)
        if heading:
            text = heading.group(2)
            if text == "参考文献":
                in_references = True
            add_heading(marker, text, level=len(heading.group(1)))
            index += 1
            continue

        eq = re.match(r"^\[EQ\s+(\d+)\](.+)$", line)
        if eq:
            add_equation(doc, marker, eq.group(1), eq.group(2).strip())
            index += 1
            continue

        fig = re.match(r"^\[FIG\s+([^|]+)\|(.+)\]$", line)
        if fig:
            path = Path(fig.group(1).strip())
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                raise FileNotFoundError(path)
            add_figure(marker, path, fig.group(2).strip())
            index += 1
            continue

        cap = re.match(r"^\[TABLECAP\]\s*(.+)$", line)
        if cap:
            add_table_caption(marker, cap.group(1))
            index += 1
            continue

        if line == "[PAGEBREAK]":
            add_page_break(marker)
            index += 1
            continue

        bio = re.match(r"^\[BIO\]\s*([^|]+)\|(.+)$", line)
        if bio:
            add_bio_table(doc, marker, bio.group(1).strip(), bio.group(2).strip())
            index += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            add_markdown_table(doc, marker, table_lines)
            continue

        if in_references and re.match(r"^\[\d+\]\s+", line):
            add_reference(marker, line)
            index += 1
            continue

        add_body_paragraph(marker, line)
        index += 1


def normalize_section(doc):
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Mm(25)
        section.bottom_margin = Mm(25)
        section.left_margin = Mm(25)
        section.right_margin = Mm(25)
        section.gutter = Mm(0)


def style_assessment_form(doc):
    # Preserve the form structure but make sure its text remains readable after margin normalization.
    if not doc.tables:
        return
    table = doc.tables[-1]
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    set_run_font(run, size=9, east="宋体")


def validate_source(meta, lines):
    zh_len = len(meta["ABSTRACT_ZH"])
    refs = [line for line in lines if re.match(r"^\[\d+\]\s+", line.strip())]
    if not 300 <= zh_len <= 500:
        raise ValueError(f"Chinese abstract length out of range: {zh_len}")
    if len(refs) < 30:
        raise ValueError(f"Too few references: {len(refs)}")


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    meta, lines = read_source()
    validate_source(meta, lines)

    doc = Document(str(TEMPLATE))
    fill_cover(doc)
    marker = remove_template_instructions(doc)
    normalize_section(doc)
    add_front_matter(marker, meta)
    parse_body(doc, marker, lines)
    style_assessment_form(doc)

    doc.core_properties.title = meta["TITLE_ZH"]
    doc.core_properties.subject = "研究生课程论文"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.keywords = meta["KEYWORDS_ZH"]
    doc.core_properties.comments = "个人信息及照片尚待作者补充；实验参与者为合成数据。"
    doc.save(str(OUT))
    print(f"wrote: {OUT}")
    print(f"stats: {STATS}")


if __name__ == "__main__":
    main()
