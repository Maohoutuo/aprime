# -*- coding: utf-8 -*-
"""用本机 Word（COM）把 docx 导出为 PDF。需要 pywin32。"""
import os
import sys

import win32com.client

DOCX = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       r"d:/composition/aprime/paper_draft.docx")
PDF = os.path.splitext(DOCX)[0] + ".pdf"

word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(DOCX, ReadOnly=False, AddToRecentFiles=False)
    # 更新域（标题编号/交叉引用若有）
    try:
        doc.Fields.Update()
    except Exception:
        pass
    doc.ExportAsFixedFormat(
        OutputFileName=PDF,
        ExportFormat=17,          # wdExportFormatPDF
        OpenAfterExport=False,
        OptimizeFor=0,            # wdExportOptimizeForPrint
        Item=0,                   # wdExportDocumentContent
        IncludeDocProps=True,
        KeepIRM=True,
        CreateBookmarks=1,        # wdExportCreateHeadingBookmarks
        DocStructureTags=True,
        BitmapMissingFonts=True,
        UseISO19005_1=False,
    )
    pages = doc.ComputeStatistics(2)   # wdStatisticPages
    doc.Close(SaveChanges=0)
    print(f"wrote {PDF}")
    print(f"pages={pages} size={os.path.getsize(PDF)/1024:.0f} KB")
finally:
    word.Quit()
