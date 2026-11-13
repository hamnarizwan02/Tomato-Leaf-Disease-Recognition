"""Render paper content (paper_content.py) to (1) IEEEtran LaTeX and (2) a two-column PDF with reportlab.

Content blocks: ("h1", text) ("h2", text) ("p", text) ("bullets", [text]) ("table", label, caption, header, rows,
col_fracs, wide) ("fig", label, path, caption, wide). Inline markup in text: <b>..</b>, <i>..</i>, citations [n] or
[n, m] / [n]–[m] (numbers into REFS), and figure/table references written plainly ("Fig. 2", "Table III").
"""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, ListFlowable, ListItem,
                                NextPageTemplate, PageTemplate, Paragraph, Spacer, Table, TableStyle)

FONTS = Path("F:/NUST/ML/code/mid/Tomato-Leaf-Disease-Recognition/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf")
for n, f in (("SR", "DejaVuSerif.ttf"), ("SB", "DejaVuSerif-Bold.ttf"), ("SI", "DejaVuSerif-Italic.ttf"),
             ("SBI", "DejaVuSerif-BoldItalic.ttf"), ("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf")):
    pdfmetrics.registerFont(TTFont(n, str(FONTS / f)))
pdfmetrics.registerFontFamily("SR", normal="SR", bold="SB", italic="SI", boldItalic="SBI")

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


# ------------------------------------------------------------------ LaTeX
def tex_escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    s = s.replace("±", r"$\pm$").replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("×", r"$\times$")
    s = s.replace("−", "--").replace("–", "--").replace("—", "---").replace("→", r"$\rightarrow$")
    s = s.replace("η²", r"$\eta^2$").replace("²", r"$^2$").replace("…", r"\ldots{}")
    return s


def tex_inline(s: str) -> str:
    s = tex_escape(s)
    s = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", s)
    s = re.sub(r"<i>(.*?)</i>", r"\\emph{\1}", s)

    def cites(m):
        nums = []
        for part in m.group(1).split(","):
            part = part.strip()
            if "--" in part:
                a, b = part.split("--")
                nums += list(range(int(a.strip(" []")), int(b.strip(" []")) + 1))
            else:
                nums.append(int(part.strip(" []")))
        return r"~\cite{" + ",".join(f"r{n}" for n in nums) + "}"
    s = re.sub(r"\s?\[(\d+(?:\]?\s?(?:,|--)\s?\[?\d+)*)\]", cites, s)
    return s


def to_latex(meta, blocks, refs, out: Path):
    L = [r"\documentclass[conference]{IEEEtran}", r"\usepackage{graphicx,booktabs,cite,amsmath}",
         r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\begin{document}",
         r"\title{" + tex_escape(meta["title"]) + "}",
         r"\author{" + r" \and ".join(r"\IEEEauthorblockN{" + tex_escape(a) + r"}\IEEEauthorblockA{" +
                                       tex_escape(meta["affil"]) + "}" for a in meta["authors"]) + "}",
         r"\maketitle", r"\begin{abstract}", tex_inline(meta["abstract"]), r"\end{abstract}",
         r"\begin{IEEEkeywords}", tex_escape(meta["keywords"]), r"\end{IEEEkeywords}"]
    for b in blocks:
        k = b[0]
        if k == "h1":
            L.append(r"\section{" + tex_escape(b[1]) + "}")
        elif k == "h2":
            L.append(r"\subsection{" + tex_escape(b[1]) + "}")
        elif k == "p":
            L.append(tex_inline(b[1]) + "\n")
        elif k == "bullets":
            L.append(r"\begin{itemize}" + "".join(r"\item " + tex_inline(x) for x in b[1]) + r"\end{itemize}")
        elif k == "table":
            _, label, cap, head, rows, fr, wide = b
            env = "table*" if wide else "table"
            cols = "l" + "c" * (len(head) - 1)
            L += [r"\begin{" + env + "}[t]", r"\centering", r"\caption{" + tex_inline(cap) + "}",
                  r"\label{" + label + "}", r"\footnotesize", r"\begin{tabular}{" + cols + "}", r"\toprule",
                  " & ".join(r"\textbf{" + tex_escape(h) + "}" for h in head) + r" \\", r"\midrule"]
            L += [" & ".join(tex_inline(str(c)) for c in r) + r" \\" for r in rows]
            L += [r"\bottomrule", r"\end{tabular}", r"\end{" + env + "}"]
        elif k == "fig":
            _, label, path, cap, wide = b
            env = "figure*" if wide else "figure"
            w = r"\textwidth" if wide else r"\columnwidth"
            L += [r"\begin{" + env + "}[t]", r"\centering", r"\includegraphics[width=" + w + "]{" +
                  Path(path).name + "}", r"\caption{" + tex_inline(cap) + "}", r"\label{" + label + "}",
                  r"\end{" + env + "}"]
    L.append(r"\begin{thebibliography}{99}")
    for i, r in enumerate(refs, 1):
        L.append(r"\bibitem{r" + str(i) + "} " + tex_escape(r))
    L += [r"\end{thebibliography}", r"\end{document}"]
    out.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ PDF
def pdf_inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<b>", "\x01").replace("</b>", "\x02").replace("<i>", "\x03") \
        .replace("</i>", "\x04").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("\x01", "<b>").replace("\x02", "</b>").replace("\x03", "<i>").replace("\x04", "</i>")


def to_pdf(meta, blocks, refs, out: Path):
    W, H = letter
    M, G = 0.62 * inch, 0.25 * inch
    colw = (W - 2 * M - G) / 2
    body = ParagraphStyle("b", fontName="SR", fontSize=8.6, leading=10.6, alignment=TA_JUSTIFY, spaceAfter=3.5,
                          firstLineIndent=9)
    plain = ParagraphStyle("p", parent=body, firstLineIndent=0)
    h1 = ParagraphStyle("h1", fontName="SR", fontSize=9.2, leading=12, alignment=TA_CENTER, spaceBefore=7,
                        spaceAfter=4, textColor=colors.black)
    h2 = ParagraphStyle("h2", fontName="SI", fontSize=8.8, leading=11, spaceBefore=5, spaceAfter=2.5)
    cap = ParagraphStyle("c", fontName="SR", fontSize=7.4, leading=9, alignment=TA_JUSTIFY, spaceAfter=6)
    cell = ParagraphStyle("t", fontName="SR", fontSize=7, leading=8.4)
    cellb = ParagraphStyle("tb", parent=cell, fontName="SB")
    ref = ParagraphStyle("r", fontName="SR", fontSize=6.9, leading=8.2, leftIndent=12, firstLineIndent=-12,
                         spaceAfter=1.2)
    title = ParagraphStyle("T", fontName="SR", fontSize=17, leading=20.5, alignment=TA_CENTER, spaceAfter=7)
    auth = ParagraphStyle("A", fontName="SR", fontSize=9, leading=11.5, alignment=TA_CENTER, spaceAfter=2)
    absst = ParagraphStyle("ab", parent=body, fontName="SB", fontSize=8.2, leading=10, firstLineIndent=0)

    story, n1, n2, tabn, fign = [], 0, 0, 0, 0
    labels = {}
    for b in blocks:                                          # number tables and figures first
        if b[0] == "table":
            tabn += 1
            labels[b[1]] = ROMAN[tabn - 1]
        elif b[0] == "fig":
            fign += 1
            labels[b[1]] = str(fign)
    tabn = fign = 0

    title_h = 1.55 * inch
    first = PageTemplate("first", [Frame(M, H - M - title_h, W - 2 * M, title_h, id="title", showBoundary=0),
                                   Frame(M, M, colw, H - 2 * M - title_h - 6, id="c1"),
                                   Frame(M + colw + G, M, colw, H - 2 * M - title_h - 6, id="c2")],
                         onPage=lambda c, d: _foot(c, d, W, M))
    later = PageTemplate("later", [Frame(M, M, colw, H - 2 * M, id="c1"), Frame(M + colw + G, M, colw, H - 2 * M,
                                                                                    id="c2")],
                         onPage=lambda c, d: _foot(c, d, W, M))
    story += [Paragraph(pdf_inline(meta["title"]), title),
              Paragraph(", ".join(meta["authors"]), auth), Paragraph(pdf_inline(meta["affil"]), auth),
              NextPageTemplate("later"), FrameBreak(),
              Paragraph("<i>Abstract</i>—" + pdf_inline(meta["abstract"]), absst),
              Paragraph("<i>Index Terms</i>—" + pdf_inline(meta["keywords"]), absst), Spacer(1, 4)]
    for b in blocks:
        k = b[0]
        if k == "h1":
            n1 += 1
            n2 = 0
            story.append(Paragraph(f"{ROMAN[n1 - 1]}. {pdf_inline(b[1]).upper()}", h1))
        elif k == "h2":
            n2 += 1
            story.append(Paragraph(f"{chr(64 + n2)}. {pdf_inline(b[1])}", h2))
        elif k == "p":
            story.append(Paragraph(pdf_inline(b[1]), body))
        elif k == "bullets":
            story.append(ListFlowable([ListItem(Paragraph(pdf_inline(x), plain), leftIndent=9) for x in b[1]],
                                      bulletType="bullet", start="•", leftIndent=9, bulletFontSize=6))
        elif k == "table":
            _, label, c, head, rows, fr, wide = b
            tabn += 1
            width = colw if not wide else colw
            data = [[Paragraph(pdf_inline(h), cellb) for h in head]] + \
                   [[Paragraph(pdf_inline(str(x)), cell) for x in r] for r in rows]
            t = Table(data, colWidths=[f * width for f in fr], repeatRows=1)
            t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                                   ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                                   ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
            story.append(KeepTogether([Paragraph(f"TABLE {ROMAN[tabn - 1]}<br/>{pdf_inline(c)}",
                                                 ParagraphStyle("tc", parent=cap, alignment=TA_CENTER)), t,
                                       Spacer(1, 6)]))
        elif k == "fig":
            _, label, path, c, wide = b
            fign += 1
            img = Image(path)
            r = img.imageHeight / img.imageWidth
            img.drawWidth, img.drawHeight = colw, colw * r
            story.append(KeepTogether([img, Paragraph(f"Fig. {fign}. {pdf_inline(c)}", cap)]))
    story.append(Paragraph("REFERENCES", h1))
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {pdf_inline(r)}", ref))
    doc = BaseDocTemplate(str(out), pagesize=letter, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                          title=meta["title"], author=", ".join(meta["authors"]))
    doc.addPageTemplates([first, later])
    doc.build(story)
    return labels


def _foot(c, d, W, M):
    c.saveState()
    c.setFont("SR", 7)
    c.drawCentredString(W / 2, M * 0.5, str(d.page))
    c.restoreState()
