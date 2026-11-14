# -*- coding: utf-8 -*-
r"""Build paper_draft.pdf from paper_draft.md using PyMuPDF Story.

Pipeline:
    markdown -> HTML (with LaTeX->Unicode math, tables, lists, inline formatting)
             -> PyMuPDF Story (flow layout, A4, page numbers)

LaTeX math is rendered as Unicode text (PyMuPDF Story has no TeX engine).
This is acceptable for a reviewable draft; final typesetting can re-insert
proper math in the journal template.
"""

from __future__ import annotations

import io
import re
import sys
from html import escape
from pathlib import Path

import pymupdf


# ---------------------------------------------------------------------------
# LaTeX -> Unicode text
# ---------------------------------------------------------------------------

SYMBOLS = {
    # relations / operators
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\in": "∈", r"\notin": "∉", r"\ni": "∋",
    r"\subseteq": "⊆", r"\subset": "⊂", r"\supseteq": "⊇", r"\supset": "⊃",
    r"\cup": "∪", r"\bigcup": "∪", r"\cap": "∩", r"\bigcap": "∩",
    r"\sum": "Σ", r"\prod": "Π", r"\int": "∫",
    r"\cdot": "·", r"\times": "×", r"\pm": "±", r"\mp": "∓", r"\div": "÷",
    r"\infty": "∞", r"\emptyset": "∅", r"\varnothing": "∅",
    r"\forall": "∀", r"\exists": "∃",
    r"\to": "→", r"\rightarrow": "→", r"\Rightarrow": "⇒", r"\implies": "⇒",
    r"\leftarrow": "←", r"\Leftarrow": "⇐", r"\leftrightarrow": "↔",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉",
    r"\mid": "|", r"\ldots": "…", r"\cdots": "…", r"\dots": "…",
    r"\quad": " ", r"\qquad": "  ", r"\,": " ",
    # function names
    r"\min": "min", r"\max": "max", r"\log": "log", r"\ln": "ln",
    r"\exp": "exp", r"\arg": "arg", r"\argmin": "argmin", r"\argmax": "argmax",
    r"\OPT": "OPT", r"\st": " s.t. ",
    # greek
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

# longest keys first so \leq is consumed before \le
_SYMBOL_KEYS = sorted(SYMBOLS, key=len, reverse=True)


def latex_to_text(raw: str) -> str:
    """Best-effort LaTeX -> readable Unicode (draft quality, not TeX)."""
    s = raw.strip()

    # \text{...} / \mathrm{...} / \operatorname{...} -> bare content
    for _ in range(4):
        prev = s
        s = re.sub(r"\\(?:text|mathrm|operatorname|mathbf|mathit)\{([^{}]*)\}", r"\1", s)
        if s == prev:
            break

    # \frac{A}{B} -> A/B when both are simple tokens, else (A)/(B)
    def _frac(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        simple = re.compile(r"^[\w\.\+\-\u03b1-\u03c9]+$")
        if simple.match(a) and simple.match(b):
            return f"{a}/{b}"
        return f"({a})/({b})"

    for _ in range(6):
        prev = s
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", _frac, s)
        if s == prev:
            break

    # \sqrt{A} -> √(A)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)

    # sizing / spacing primitives that carry no meaning in plain text
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\;", " ")

    # symbols (longest first)
    for key in _SYMBOL_KEYS:
        if key in s:
            s = s.replace(key, SYMBOLS[key])

    # sub/superscripts: keep readable as _( ... ) and ^( ... )
    s = re.sub(r"_\{([^{}]*)\}", r"_(\1)", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)

    # leftover braces / backslashes
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
    s = s.replace("\\\\", " ")
    s = s.replace("\\", "")

    return re.sub(r"[ \t]+", " ", s).strip()


# ---------------------------------------------------------------------------
# Inline markdown -> HTML
# ---------------------------------------------------------------------------

_MATH_INLINE = re.compile(r"\\\((.*?)\\\)", re.S)
_MATH_DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.S)


def inline(text: str) -> str:
    """Escape, then apply inline math / bold / italic / code."""
    t = escape(text)

    t = _MATH_INLINE.sub(lambda m: latex_to_text(m.group(1)), t)
    t = _MATH_DISPLAY.sub(
        lambda m: '<span class="math">' + latex_to_text(m.group(1)) + "</span>", t
    )

    # bold before italic; use lookarounds so ** is not eaten by *
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t, flags=re.S)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


# ---------------------------------------------------------------------------
# Block-level markdown -> HTML
# ---------------------------------------------------------------------------

_MATH_INLINE = re.compile(r"\\\((.*?)\\\)", re.S)
_MATH_DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.S)
_SEP_CELL = re.compile(r"^\s*:?-{2,}:?\s*$")


def _protect_math(t: str):
    """Replace math spans with placeholders so pipe-splitting won't break them."""
    store: list[str] = []

    def rep(m: re.Match) -> str:
        store.append(latex_to_text(m.group(1)))
        return f"\x00{len(store)-1}\x00"

    out = _MATH_INLINE.sub(rep, t)
    out = _MATH_DISPLAY.sub(rep, out)
    return out, store


def _restore(t: str, store: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], t)


def _split_row(line: str, store: list[str] | None = None) -> list[str]:
    if store is not None:
        return [_restore(c, store) for c in _split_row(line.strip())]
    return [c for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    if not line.strip().startswith("|"):
        return False
    cells = [c for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(_SEP_CELL.match(c) for c in cells)


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # horizontal rule
        if stripped in ("---", "***", "___"):
            out.append("<hr>")
            i += 1
            continue

        # table: header row followed by separator row
        if stripped.startswith("|") and i + 1 < n and _is_separator(lines[i + 1]):
            prot_hdr, hdr_store = _protect_math(line)
            header = [c for c in prot_hdr.strip().strip("|").split("|")]
            header = [_restore(c, hdr_store) for c in header]
            i += 2
            body: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                prot_row, row_store = _protect_math(lines[i])
                cells = [c for c in prot_row.strip().strip("|").split("|")]
                cells = [_restore(c, row_store) for c in cells]
                body.append(cells)
                i += 1
            out.append("<table>")
            out.append("<thead><tr>" +
                       "".join(f"<th>{inline(c)}</th>" for c in header) +
                       "</tr></thead><tbody>")
            for row in body:
                row = (row + [""] * len(header))[: len(header)]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue

        # figure: ![caption](path)
        m = re.match(r"^!\[(.*?)\]\((.+?)\)$", stripped)
        if m:
            cap, src = m.group(1), m.group(2)
            out.append('<div class="figure">'
                       f'<img src="{src}">'
                       f'<div class="caption">{inline(cap)}</div>'
                       "</div>")
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        # blockquote (consecutive > lines merge)
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue

        # unordered list
        if re.match(r"^[\*\-]\s+", stripped):
            items = []
            while i < n and re.match(r"^[\*\-]\s+", lines[i].strip()):
                items.append(re.sub(r"^[\*\-]\s+", "", lines[i].strip()))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue

        # ordered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
            continue

        # paragraph: accumulate until a block boundary
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt:
                break
            if (nxt.startswith("|") or nxt.startswith(">") or nxt.startswith("#")
                    or re.match(r"^[\*\-]\s+", nxt) or re.match(r"^\d+\.\s+", nxt)
                    or nxt in ("---", "***", "___")):
                break
            buf.append(nxt)
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
body { font-family: "Microsoft YaHei"; font-size: 10.5pt; line-height: 1.5; color: #111; }
h1 { font-size: 18pt; font-weight: bold; margin: 1.2em 0 0.5em 0; }
h2 { font-size: 14pt; font-weight: bold; margin: 1.1em 0 0.45em 0; }
h3 { font-size: 12pt; font-weight: bold; margin: 0.95em 0 0.4em 0; }
h4 { font-size: 11pt; font-weight: bold; margin: 0.85em 0 0.35em 0; }
p  { margin: 0 0 0.7em 0; text-align: justify; }
li { margin-bottom: 0.3em; text-align: justify; }
ul, ol { margin: 0.4em 0 0.7em 0; padding-left: 1.4em; }
blockquote { margin: 0.55em 0.6em; padding: 0.3em 0.7em; border-left: 3px solid #bbb;
             color: #333; font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0 0.9em 0; font-size: 8.6pt; }
th { background: #ececec; border: 1px solid #888; padding: 3px 5px; text-align: left; }
td { border: 1px solid #888; padding: 3px 5px; vertical-align: top; }
hr { border: none; border-top: 1px solid #ccc; margin: 0.9em 0; }
code { font-family: "Consolas", monospace; font-size: 9pt; }
.math { font-style: italic; }
.figure { text-align: center; margin: 0.9em 0; page-break-inside: avoid; }
.figure img { width: 100%; border: 1px solid #ddd; }
.caption { font-size: 8.5pt; color: #555; text-align: center; margin-top: 3px; }
"""


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def build_pdf(md_path: str, out_path: str) -> int:
    with io.open(md_path, encoding="utf-8") as fh:
        md = fh.read()

    html = ("<html><head><meta charset='utf-8'></head><body>"
            + md_to_html(md) + "</body></html>")

    archive = pymupdf.Archive(str(Path(md_path).resolve().parent))
    story = pymupdf.Story(html=html, user_css=CSS, archive=archive)
    writer = pymupdf.DocumentWriter(out_path)

    def rectfn(rect_num: int, filled: bool):
        mediabox = pymupdf.paper_rect("a4")
        rect = mediabox + (56, 60, -56, -62)   # margins: L, T, R, B
        ctm = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        return mediabox, rect, ctm

    story.write(writer, rectfn)
    writer.close()

    # stamp page numbers
    doc = pymupdf.open(out_path)
    total = doc.page_count
    for idx, page in enumerate(doc, start=1):
        r = page.rect
        page.insert_text((r.width / 2 - 12, r.height - 34), f"{idx} / {total}",
                         fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
    doc.saveIncr()
    doc.close()
    return total


def main() -> int:
    md_path = r"D:\composition\aprime\paper_draft.md"
    out_path = r"D:\composition\aprime\paper_draft.pdf"
    pages = build_pdf(md_path, out_path)
    print(f"Wrote {out_path} ({pages} pages).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
