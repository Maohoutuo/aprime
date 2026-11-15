# -*- coding: utf-8 -*-
"""快速验证 LaTeX -> MathML -> OMML 管线。"""
from latex2mathml.converter import convert
from lxml import etree
import os

XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"

latex_samples = [
    r"\theta_{\text{all}}",
    r"\frac{1}{2}(1-\frac{1}{e})",
    r"\Delta\text{cov}",
    r"\theta \ge 0.6",
    r"\lfloor \kappa \cdot share_g \cdot k_0 \rfloor",
    r"\phi=0.95",
]

xslt = etree.parse(XSL_PATH)
transform = etree.XSLT(xslt)

for latex in latex_samples:
    try:
        mathml = convert(latex)
        mm = etree.fromstring(mathml.encode("utf-8"))
        omml = transform(mm)
        omml_str = str(omml)
        ok = "m:oMath" in omml_str or "<m:oMath" in omml_str
        print(f"OK   {latex!r:<45} -> omml len={len(omml_str)} is_omml={ok}")
    except Exception as e:
        print(f"FAIL {latex!r:<45} -> {type(e).__name__}: {e}")
