# -*- coding: utf-8 -*-
"""统计 paper_draft.md 里的公式定界符与宏使用，用于评估 mathtext 覆盖度。"""
import re
import collections

SRC = r"d:/composition/aprime/paper_draft.md"
text = open(SRC, encoding="utf-8").read()

inline = re.findall(r"\\\((.*?)\\\)", text, re.S)
display = re.findall(r"\\\[(.*?)\\\]", text, re.S)
dollar = re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", text, re.S)

print("内联 \\(...\\) :", len(inline))
print("块级 \\[...\\]  :", len(display))
print("美元 $...$     :", len(dollar))

macros = collections.Counter()
for blob in inline + display + dollar:
    for m in re.finditer(r"\\([a-zA-Z]+)", blob):
        macros[m.group(1)] += 1

print("\n宏使用（降序）:")
for name, cnt in macros.most_common():
    print(f"  {name:<16} {cnt}")

# mathtext 已知不支持/易出错的构造
risky = ["xrightarrow", "xleftarrow", "boldsymbol", "operatorname", "text{",
         "begin{", "substack", "stackrel", "underset", "overset", "mbox",
         "mathrm", "mathbb", "mathcal", "forall", "exists", "lfloor", "rfloor",
         "lceil", "rceil", "sim", "approx", "le", "ge", "ne", "cdot", "qquad",
         "quad", "Delta", "Omega", "kappa", "phi", "theta", "lambda", "alpha"]
print("\n风险构造:")
for r in risky:
    n = sum(blob.count("\\" + r) if not r.endswith("{") else blob.count(r)
            for blob in inline + display + dollar)
    if n:
        print(f"  {r:<16} {n}")
