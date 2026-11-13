"""Final project report (reportlab, A4). Phase 1-2 numbers are copied from docs/verified_results.md §§3-13; every
Phase 3 number is read from F:/NUST/ML/runs/p4_multisource/results (same loader as the paper).

    F:/NUST/ML/tools/report_venv/Scripts/python build_final_report.py
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).parent / "paper"))
import paper_content as pc  # noqa: E402

REPO = pc.REPO
RUN = pc.RUN
OUT = Path("F:/NUST/ML/code/mid/Final_Project_Report_2026-11-13.pdf")
OLD_FIG = Path("F:/NUST/ML/report/fig")
FIG = pc.FIG

FONTDIR = REPO / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONTDIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FONTDIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(FONTDIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, INK2, RULE, TINT = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d9d8d3", "#f3f2ef"))
ACCENT = colors.HexColor("#1c5cab")
base = ParagraphStyle("b", fontName="DV", fontSize=9.4, leading=13.2, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
small = ParagraphStyle("s", parent=base, fontSize=8, leading=10.5, textColor=INK2)
refst = ParagraphStyle("r", parent=base, fontSize=7.8, leading=10, leftIndent=14, firstLineIndent=-14, spaceAfter=2)
cell = ParagraphStyle("c", parent=base, fontSize=8.2, leading=10.6, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="DVB")
h1 = ParagraphStyle("h1", parent=base, fontName="DVB", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6,
                    textColor=ACCENT)
h2 = ParagraphStyle("h2", parent=base, fontName="DVB", fontSize=10.8, leading=14, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("t", parent=base, fontName="DVB", fontSize=19, leading=24, spaceAfter=4)
sub = ParagraphStyle("st", parent=base, fontSize=10, leading=14, textColor=INK2, spaceAfter=10)
P = lambda t, s=base: Paragraph(t, s)
W = 16.6


def bullets(items, style=base):
    return ListFlowable([ListItem(P(i, style), leftIndent=12, bulletColor=INK2) for i in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def table(rows, widths):
    data = [[P(str(c), cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    st += [("BACKGROUND", (0, r), (-1, r), TINT) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return t


def callout(text):
    t = Table([[P(text)]], colWidths=[W * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf1fb")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def fig(path, width_cm, caption):
    img = Image(str(path))
    r = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * r
    return KeepTogether([img, P(caption, small), Spacer(1, 6)])


def footer(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(INK2)
    c.drawString(2.2 * cm, 1.2 * cm, "Robust and explainable classical ML for tomato disease recognition: final report")
    c.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{d.page}")
    c.restoreState()


def md_inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"<i>\1</i>", t)
    return re.sub(r"`(.+?)`", r"<font name='DV'>\1</font>", t)


def strip_refs(t):
    """Remove the paper's table/figure numbers, e.g. ' (Table III, Fig. 3)'."""
    return re.sub(r" \((?:Table [IVX]+|Fig\. \d+)(?:, (?:Table [IVX]+|Fig\. \d+))*\)", "", t)


def literature_review():
    """docs/literature_review.md -> flowables (section numbering continues as 2.x)."""
    txt = (REPO / "docs" / "literature_review.md").read_text(encoding="utf-8")
    body, refs = txt.split("## References", 1)
    out = []
    for blk in re.split(r"\n\s*\n", body.split("\n", 1)[1]):
        blk = blk.strip()
        if not blk or blk.startswith("Written 2026"):
            continue
        if blk.startswith("## "):
            out.append(P("2." + blk[3:].strip(), h2))
        else:
            out.append(P(md_inline(" ".join(blk.split()))))
    out.append(P("References", h2))
    out += [P(md_inline(f"[{i}] {r}"), refst) for i, r in enumerate(pc.load_refs(), 1)]
    return out


def build():
    d = pc._load()
    f = lambda x: f"{x:.2f}"
    ov, st, sel = d["ov"], d["st"], d["sel"].groupby(["pool", "features", "model"])["balanced_accuracy"].mean()
    o = lambda task, s, m="lr": pc._ov(d, task, s, m)
    hl, cw, co, cl, rn = (o("bin", s) for s in ["hand+lesion", "colour_words", "concepts", "clip", "resnet"])
    hl6, co6, cl6, rn6 = (o("six", s) for s in ["hand+lesion", "concepts", "clip", "resnet"])
    best6 = ov[ov.task == "six"].sort_values("loso_mean").iloc[-1]
    six_nd = ov[(ov.task == "six") & (ov.arm != "deep")].loso_mean.max()
    les = st[st.feature.str.startswith("lesion_")]
    hand = st[(st.pool == "formula") & ~st.feature.str.startswith("lesion_")]
    nst = st[st.stable].groupby("pool").size().to_dict()
    rfb = o("bin", "hand+lesion", "rf")

    s = [P("Robust and Explainable Classical ML for Tomato Disease Recognition", title),
         P("Final project report. CS-871 Machine Learning, MS(AI), NUST SEECS, Fall 2026. Hamna Sadia Rizwan, "
           "Maria Ilyas Malik, Manahil Kamran. 13 November 2026.", sub)]

    # 1 summary
    s += [P("1. Summary", h1),
          callout(f"<b>Bottom line.</b> Classical models trained on one image source do not transfer, and neither do "
                  f"their explanations. Trained on seven pooled sources and tested on an eighth, they recognise "
                  f"healthy vs diseased leaves with balanced accuracy {f(hl.loso_mean)} (hand-crafted + lesion "
                  f"features, logistic regression) up to {f(co.loso_mean)} (24 CLIP concept scores from a frozen "
                  f"network). Six-class diagnosis transfers poorly with formula or k-means features (macro-F1 "
                  f"{f(six_nd)}) and moderately with frozen CLIP embeddings ({f(cl6.loso_mean)}). "
                  f"<b>Feature-level explanation works:</b> {int(st.stable.sum())} of {len(st)} named features relate "
                  f"to disease in the same direction on at least 6 of 7 independent sources, led by leaf-relative "
                  f"yellow, non-green and lesion-size measurements and by brown/olive k-means colour words. "
                  f"A model's importance ranking does not recover these features."),
          Spacer(1, 6),
          P("<b>What we did.</b>"),
          bullets(["Phase 1-2 (to 27 Oct): 105 hand-crafted colour, texture and shape features; RF, SVM, LR; grouped "
                   "splits; 50 perturbations, augmentation, illumination normalisation, shortcut controls.",
                   "Task 2 (12 Nov): grouped re-evaluation of the supplied 7,200-image set; cross-dataset test on "
                   "PlantDoc and two Bangladeshi field sets.",
                   "Phase 3 (12-13 Nov): an eight-source, de-duplicated benchmark (7,754 images); leave-one-source-"
                   "out evaluation; two learned-extractor arms compared with the formula features, all trained and "
                   "tested with classical ML: k-means visual words (classical, unsupervised) and frozen deep networks "
                   "(CLIP, ResNet-50, no training); feature-level explanation with bootstrap CIs per source, "
                   "glass-box models, explanation-guided selection."]),
          P("<b>Main findings.</b>"),
          bullets([f"Related images inflate benchmarks: supplied-set accuracy 0.96 (image split) vs 0.66 (grouped).",
                   f"About a third of our raw photos are also in PlantDoc (296 of 813 by content).",
                   f"Single-source models collapse on other sources (macro-F1 0.21-0.33); pooled training restores "
                   f"binary transfer ({f(hl.loso_mean)}-{f(co.loso_mean)}).",
                   f"For six-class diagnosis the frozen deep embeddings transfer clearly better than formula or "
                   f"k-means features ({f(cl6.loso_mean)} vs {f(six_nd)} macro-F1, higher on 6 of 7 sources).",
                   f"Linear models transfer better than RF and SVM; RF almost never predicts healthy on a new source "
                   f"(healthy recall {f(rfb.healthy_recall_mean)}).",
                   f"The best deep features (CLIP concepts) partly rely on capture context (plain background, hand, "
                   f"studio), shown by the glass-box weights.",
                   f"Leaf-relative lesion formulas are the most source-stable family ({int(les.stable.sum())} of "
                   f"{len(les)}), against {int(hand.stable.sum())} of {len(hand)} hand features and "
                   f"{nst.get('concepts', 0)} of 24 CLIP concepts."])]

    # 2 literature review
    s += [PageBreak(), P("2. Literature review", h1)] + literature_review()

    # 3 data
    t = pc.source_table()
    s += [P("3. Data", h1),
          P("Eight image sources, de-duplicated across sources with a rule fixed before Phase 3 (16x16 grey "
            "thumbnails, correlation at least 0.95 under 8 rotations and flips, or identical bytes). 425 duplicates "
            "were dropped and 72 label conflicts recorded; near-duplicates within a source are grouped and never "
            "split. Excluded: the supplied Augmented Dataset (derived from our photos) and the Dinajpur set (its "
            "labels mark lesions, and class ids could not be matched)."),
          table([t[3]] + t[4], [2.6, 6.2, 2.4, 2.4, 3.0]),
          P("Classes: six-class labels present + extra disease classes used only in the healthy-vs-diseased task.",
            small)]

    # 4 phase 1-2
    s += [P("4. Results on our own data and single-source transfer (Phases 1-2)", h1),
          table([["Experiment", "Result"],
                 ["Raw set, 10 grouped splits", "Macro-F1 RF 0.600, SVM 0.615, LR 0.596; not significantly different."],
                 ["Robustness (50 perturbations)", "Share of clean macro-F1 kept (RF): down-up scaling 0.42, blur 0.43, "
                  "contrast 0.61, colour cast 0.70, brightness 0.83, rotation 0.96."],
                 ["Augmentation / illumination", "Photometric augmentation helps brightness and contrast robustness; "
                  "Gray World costs 0.05-0.06 (it shrinks the leaf mask); CLAHE neutral."],
                 ["Source shortcut", "Holding out a source cluster lowers accuracy from about 0.61 to 0.47-0.50; "
                  "metadata alone scores 0.49."],
                 ["Task 2: supplied 7,200 set", "Image-level split 0.961 / 0.972 / 0.768 (RF/SVM/LR); supplied split "
                  "0.814 / 0.815 / 0.698; grouped 0.659 / 0.669 / 0.638."],
                 ["Cross-dataset (trained on raw)", "PlantDoc new images 0.29-0.33; Nazirpur 0.21-0.25; Brahmanbaria "
                  "healthy recall 3-17%. Domain AUC 0.85 (PlantDoc), 0.98-0.99 (Bangladesh)."],
                 ["Explanation transfer", "RF importance ranking agreement 0.87 within our data, 0.61 vs PlantDoc, "
                  "-0.01 vs Nazirpur. Only 2 of 105 features consistent across 4 datasets."]],
                [4.4, 12.2]),
          fig(OLD_FIG / "fig1_supplied_split_rule.png", 12.0,
              "Same images and models, different split rule (Task 2). Error bars: +/- 1 sd over 10 splits.")]

    # 5 phase 3
    tm = pc.results_blocks()
    tmain = next(b for b in tm if b[0] == "table" and b[1] == "tab:loso")
    s += [P("5. Multi-source evaluation: formula vs k-means vs frozen deep features", h1),
          P("Leave-one-source-out (LOSO): train on seven sources, choose hyper-parameters on 15% of every training "
            "source, test on the eighth. Seen-source reference: the same pooled data split 70/15/15 within every "
            "source, 5 seeds. Gap = seen minus unseen. All classifiers are classical (RF, RBF SVM, LR); the deep "
            "networks are frozen feature extractors."),
          table([tmain[3]] + tmain[4], [3.6, 2.4, 1.5, 1.5, 1.4, 1.6, 1.5, 3.1]),
          P(tmain[2].replace("Fig. 1 shows", "The next figure shows"), small),
          fig(FIG / "p3_arms_loso_vs_seen.png", 10.5, "Best classifier per feature set, unseen (bars) vs seen (ticks)."),
          fig(FIG / "p3_heatmap_bin.png", 11.5, "Healthy vs diseased per held-out source, best set of each arm."),
          P("Comparison of the two learned-extractor arms", h2),
          bullets([f"<b>K-means (classical):</b> colour words {f(cw.loso_mean)} unseen, smallest gap "
                   f"({f(cw.gap_mean)}) and lowest spread across sources; textons add little. Combined with the "
                   f"formula features they do not improve transfer ({f(o('bin', 'hand+lesion+kmeans').loso_mean)}).",
                   f"<b>Frozen deep:</b> CLIP concepts {f(co.loso_mean)}, CLIP embedding {f(cl.loso_mean)}, "
                   f"ResNet-50 {f(rn.loso_mean)} (healthy vs diseased, LR)."
                   + (f" Six classes: {f(co6.loso_mean)}, {f(cl6.loso_mean)}, {f(rn6.loso_mean)}." if co6 is not None
                      else ""),
                   f"<b>Formula:</b> hand + lesion {f(hl.loso_mean)} (binary), {f(hl6.loso_mean)} (six classes, "
                   f"above hand features alone on 7 of 7 sources, not significant after Holm correction)."])]
    if (FIG / "p3_heatmap_six.png").exists():
        s.append(fig(FIG / "p3_heatmap_six.png", 11.5, "Six classes (macro-F1) per held-out source, best set of each arm."))

    # 6 explanation
    tst = next(b for b in tm if b[0] == "table" and b[1] == "tab:stable")
    tse = next(b for b in tm if b[0] == "table" and b[1] == "tab:sel")
    expl_ps = [b[1] for b in tm if b[0] == "p"][-6:-3]
    s += [P("6. Feature-level explanation", h1),
          P("Per source with at least 30 healthy and 30 diseased images (7 sources), each feature's effect is "
            "AUC - 0.5 for diseased vs healthy with a 1,000-resample bootstrap 95% CI. A feature is source-stable if "
            "its CI excludes 0 with the majority sign in at least 75% of sources. No classifier is involved."),
          table([tst[3]] + tst[4], [3.6, 2.0, 11.0]),
          fig(FIG / "p3_stable_features.png", 14.5, "Per-source effects of the most source-stable features."),
          fig(FIG / "p3_colour_words.png", 12.5, "K-means colour words (learned without labels), ordered by effect."),
          P(strip_refs(expl_ps[0])),
          P(strip_refs(expl_ps[1])),
          table([tse[3]] + tse[4], [4.6, 3.0, 3.0, 3.0, 3.0]),
          P(tse[2], small),
          P(strip_refs(expl_ps[2]))]

    # 7 answers to RQs
    s += [P("7. Answers to the research questions", h1),
          table([["Question", "Answer"],
                 ["RQ1 Accuracy of classical ML", f"About 0.60 macro-F1 within our data; on unseen sources "
                  f"{f(hl.loso_mean)}-{f(co.loso_mean)} balanced accuracy (binary); six classes {f(six_nd)} macro-F1 "
                  f"with formula or k-means features, {f(cl6.loso_mean)} with frozen CLIP embeddings."],
                 ["RQ2 Robustness", "Resolution and blur hurt most; photometric augmentation helps lighting only. "
                  "The largest 'perturbation' is a change of source."],
                 ["RQ3 Shortcuts and leakage", "Strong evidence: related images inflate scores by 0.3; a third of our "
                  "data is PlantDoc; source clusters and capture-context concepts carry signal."],
                 ["RQ4 Useful, interpretable, stable features",
                  f"{int(st.stable.sum())} source-stable features, mainly leaf-relative yellow/brown/non-green "
                  f"fractions, lesion size and brown/olive colour words. They are interpretable and stable but, "
                  f"used alone, do not raise unseen-source accuracy."]],
                [4.6, 12.0])]

    # 8 limitations + next steps + reproducibility
    s += [P("8. Limitations", h1),
          bullets(["Labels come from different annotators; 72 cross-source conflicts are recorded, not resolved.",
                   "'Diseased' mixes different diseases per source; explanation covers healthy vs diseased only.",
                   "The HSV leaf mask is heuristic; deep extractors see the whole image, which favours context cues.",
                   "Only frozen, generic deep extractors; no fine-tuning (by design: classical ML only).",
                   "Overlap with PlantDoc: 296 of 813 raw photos by the content rule here vs 263 of 830 reported "
                   "earlier against six-class PlantDoc images; not yet reconciled image by image.",
                   "Runs were on a 4-core CPU machine; RF results differ slightly from the original machine "
                   "(in-domain 0.613 vs 0.600)."]),
          P("9. Next steps", h1),
          bullets(["Label review of the 72 cross-source conflicts and the 21 septoria/early-blight suspects.",
                   "Per-disease explanation (disease vs healthy for each disease) on sources that have it.",
                   "Leaf masking before the deep extractors, to test how much of their advantage is context.",
                   "Team: read the cited papers before submission (course rule: cite only sources read)."]),
          P("10. Reproducibility", h1),
          bullets(["Code: scripts/24 (sources) - 25/25b (features) - 26 (LOSO) - 27 (explanation) - 28 (summary) - "
                   "29 (figures); src/features_lesion.py, src/codebook.py; configs/sources.yaml, configs/concepts.yaml.",
                   "Environments: .venv (Python 3.11.9, requirements.txt); .venv-dl for the frozen extractors "
                   "(requirements-dl.txt).",
                   "Outputs: F:/NUST/ML/runs/p4_multisource (results/, logs/). Numbers: docs/verified_results.md "
                   "§§11-15. Commands: docs/experiment_log.md. Design: docs/phase3_design_2026-11-12.md."], small)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm,
                            bottomMargin=2.0 * cm, title="Final project report: tomato disease recognition",
                            author="CS-871 project team")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
