"""Mid-semester report (Deliverable 2), CS-871. Numbers are copied from docs/verified_results.md §§1-13 (Phase 1-2,
Task 2, cross-dataset, first feature-level analysis). Phase 3 results are deliberately kept for the final paper.

    F:/NUST/ML/tools/report_venv/Scripts/python build_midterm.py      (figures need matplotlib: run make_figs first
    with the project .venv: .venv/Scripts/python build_midterm.py --figs)
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = Path("F:/NUST/ML/code/mid/Tomato-Leaf-Disease-Recognition")
OUT = Path("F:/NUST/ML/code/mid/Mid_Semester_Report_CS871.pdf")
FIG = HERE / "fig"
OLD = Path("F:/NUST/ML/report/fig")


# ----------------------------------------------------------------------------------------------------------- figures
def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    FIG.mkdir(exist_ok=True)
    INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})

    # pipeline
    steps = [("Field photo", "813 images\n6 classes"), ("Resize", "256 x 256"), ("Leaf mask", "HSV hue gate\n+ morphology"),
             ("105 features", "colour 47\ntexture 38\nshape 20"), ("Grouped split", "70 / 15 / 15\n10 seeds"),
             ("RF / SVM / LR", "tuned on\nvalidation"), ("Test", "clean +\n50 perturbations")]
    fig, ax = plt.subplots(figsize=(9.5, 1.9))
    w, gap = 1.12, 0.26
    for i, (t, s) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#eaf1fb" if i not in (4, 6) else "#fdeee7",
                                    edgecolor="#2a78d6" if i not in (4, 6) else "#eb6834", lw=1.1))
        ax.text(x + w / 2, 0.98, t, ha="center", va="center", fontsize=8.2, weight="bold")
        ax.text(x + w / 2, 0.45, s, ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.25)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.62), xytext=(x + w + 0.02, 0.62),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.set_xlim(-0.05, len(steps) * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.1, 1.35)
    ax.axis("off")
    fig.savefig(FIG / "pipeline.png", bbox_inches="tight")
    plt.close(fig)

    # robustness (verified_results §4)
    rob = [("Rotation", .96, .96, .96), ("JPEG", .95, .87, .87), ("Background", .91, .89, .89),
           ("Gamma", .87, .84, .90), ("Occlusion", .87, .67, .70), ("Scale", .86, .89, .99),
           ("Brightness", .83, .70, .79), ("Shadow", .81, .77, .90), ("Noise", .78, .68, .80),
           ("Colour cast", .70, .61, .70), ("Contrast", .61, .59, .65), ("Combined pairs", .58, .46, .65),
           ("Motion blur", .52, .53, .73), ("Gaussian blur", .43, .31, .48), ("Down-then-up scaling", .42, .32, .47)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cols = {"RF": "#2a78d6", "SVM": "#eb6834", "LR": "#1baf7a"}
    for i, (k, *v) in enumerate(rob[::-1]):
        ax.hlines(i, min(v), max(v), color=GRID, lw=3, zorder=1)
        for (m, c), x, dy in zip(cols.items(), v, (0, 0, 0)):
            ax.scatter(x, i, s=34, color=c, edgecolors="white", linewidths=1, zorder=3, label=m if i == 0 else None)
    ax.set_yticks(range(len(rob)), [r[0] for r in rob[::-1]])
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_xlim(0.2, 1.05)
    ax.set_xlabel("Perturbed macro-F1 / clean macro-F1 (mean over levels and 10 splits)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("figures written")


# ----------------------------------------------------------------------------------------------------------- refs
def load_refs():
    txt = (REPO / "docs" / "literature_review.md").read_text(encoding="utf-8").split("## References", 1)[1]
    refs = {}
    for m in re.finditer(r"^\[(\d+)\]\s+(.*?)(?=^\[\d+\]|\Z)", txt, flags=re.M | re.S):
        refs[str(int(m.group(1)))] = " ".join(m.group(2).split())
    refs["pc"] = ('"PlantCity: A comprehensive image based on multi crop leaves in Pakistan," Data in Brief, vol. 63, '
                  'art. 112130, 2025, doi:10.1016/j.dib.2025.112130.')
    return refs


class Cite:
    """Citations are written as {2}, {2,3} or {10-12} with the literature-review numbers (and {pc} for PlantCity);
    they are renumbered in order of first appearance."""

    def __init__(self, refs):
        self.refs, self.order = refs, []

    def num(self, k):
        if k not in self.order:
            self.order.append(k)
        return self.order.index(k) + 1

    def __call__(self, text):
        def rep(m):
            keys = []
            for part in m.group(1).split(","):
                part = part.strip()
                if re.fullmatch(r"\d+-\d+", part):
                    a, b = map(int, part.split("-"))
                    keys += [str(i) for i in range(a, b + 1)]
                else:
                    keys.append(part)
            nums = sorted(self.num(k) for k in keys)
            out, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                out.append(f"[{nums[i]}]" if j == i else (f"[{nums[i]}], [{nums[j]}]" if j == i + 1
                                                          else f"[{nums[i]}]–[{nums[j]}]"))
                i = j + 1
            return ", ".join(out)
        return re.sub(r"\{([0-9a-z,\- ]+)\}", rep, text)


# ----------------------------------------------------------------------------------------------------------- pdf
def build():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    fd = REPO / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf"
    for n, f in (("SR", "DejaVuSerif.ttf"), ("SB", "DejaVuSerif-Bold.ttf"), ("SI", "DejaVuSerif-Italic.ttf"),
                 ("SBI", "DejaVuSerif-BoldItalic.ttf"), ("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(n, str(fd / f)))
    pdfmetrics.registerFontFamily("SR", normal="SR", bold="SB", italic="SI", boldItalic="SBI")
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

    INK, MUT, RULE, TINT = (colors.HexColor(c) for c in ("#111111", "#555555", "#cfcfcf", "#f2f2f2"))
    body = ParagraphStyle("b", fontName="SR", fontSize=10, leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                          spaceAfter=6)
    bl = ParagraphStyle("bl", parent=body, alignment=0, spaceAfter=2)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.6, leading=11.2, textColor=MUT, alignment=0, spaceAfter=8)
    cell = ParagraphStyle("c", fontName="SR", fontSize=8.6, leading=11, textColor=INK)
    cellb = ParagraphStyle("cb", parent=cell, fontName="SB")
    h1 = ParagraphStyle("h1", fontName="SB", fontSize=15, leading=19, spaceBefore=12, spaceAfter=6, textColor=INK,
                        keepWithNext=1)
    h2 = ParagraphStyle("h2", fontName="SB", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, textColor=INK,
                        keepWithNext=1)
    ttl = ParagraphStyle("t", fontName="SB", fontSize=19, leading=25, spaceAfter=6, textColor=INK)
    refst = ParagraphStyle("r", fontName="SR", fontSize=8.6, leading=11.4, leftIndent=20, firstLineIndent=-20,
                           spaceAfter=3, textColor=INK)
    W = 16.0

    C = Cite(load_refs())
    P = lambda t, s=body: Paragraph(C(t), s)

    def bullets(items):
        return ListFlowable([ListItem(P(i, bl), leftIndent=14) for i in items], bulletType="bullet", start="•",
                            leftIndent=14, bulletFontSize=8, spaceAfter=6)

    def table(rows, widths, caption=None, num=[0]):
        num[0] += 1
        data = [[Paragraph(C(str(c)), cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
        st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
              ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        st += [("BACKGROUND", (0, r), (-1, r), TINT) for r in range(2, len(rows), 2)]
        t.setStyle(TableStyle(st))
        parts = [Paragraph(f"<b>Table {num[0]}.</b> {C(caption)}", cap)] if caption else []
        return KeepTogether(parts + [t, Spacer(1, 8)])

    def fig(path, width, caption, num=[0]):
        num[0] += 1
        img = Image(str(path))
        img.drawWidth, img.drawHeight = width * cm, width * cm * img.imageHeight / img.imageWidth
        return KeepTogether([img, Paragraph(f"<b>Figure {num[0]}.</b> {C(caption)}", cap)])

    def footer(c, d):
        if d.page == 1:
            return
        c.saveState()
        c.setFont("SR", 8)
        c.setFillColor(MUT)
        c.drawString(2.5 * cm, 1.3 * cm, "CS-871 Machine Learning: Mid-Semester Report")
        c.drawRightString(A4[0] - 2.5 * cm, 1.3 * cm, str(d.page - 1))
        c.restoreState()

    s = []
    # ------------------------------------------------------------------ cover (same layout as the synopsis)
    cov = lambda t, size, lead, font="SB", col=INK, after=0: Paragraph(t, ParagraphStyle(
        "cv", fontName=font, fontSize=size, leading=lead, alignment=TA_CENTER, textColor=col, spaceAfter=after))
    logo = Image(str(HERE / "logo_0.png"))
    logo.drawWidth, logo.drawHeight = 7 * cm, 7 * cm * logo.imageHeight / logo.imageWidth
    s += [Spacer(1, 4.2 * cm), cov("MID-SEMESTER REPORT", 24, 30, after=18),
          cov("CS-871 Machine Learning<br/>MS(AI)", 22, 28, after=30),
          cov("Robust and Explainable Classical Machine Learning for Tomato Disease Recognition under "
              "Real-World Field Conditions", 12, 16, font="SI", col=MUT, after=2.2 * cm),
          cov("Hamna Sadia Rizwan 583282<br/>Maria Ilyas Malik 624045<br/>Manahil Kamran 582413", 12.5, 18,
              col=colors.HexColor("#444444"), after=2.6 * cm),
          logo, Spacer(1, 0.6 * cm), cov("NUST SEECS, Islamabad", 13, 17, font="SR"),
          cov("13 November 2026", 10, 14, font="SR", col=MUT), PageBreak()]

    # ------------------------------------------------------------------ title
    s += [Paragraph("Robust and Explainable Classical Machine Learning for Tomato Disease Recognition under "
                    "Real-World Field Conditions", ttl),
          Paragraph("Mid-semester progress report", ParagraphStyle("st", parent=body, fontName="SI", textColor=MUT,
                                                                    spaceAfter=10))]

    # ------------------------------------------------------------------ 1 introduction
    s += [Paragraph("1. Introduction", h1),
          P("Our synopsis set out to study how small, classical models cope with real field photos of tomato "
            "leaves. The models use hand-crafted colour, texture and shape features, are cheap to run, and their "
            "inputs are things a person can name and measure. We planned to measure how much accuracy they lose "
            "when lighting, contrast or background change, which feature groups carry the useful information, and "
            "how stable the feature importances are."),
          P("This report describes what we have built and found since then. The full pipeline is working and has "
            "been evaluated on our dataset with repeated, leakage-safe splits, under 50 controlled image "
            "perturbations, and on three external tomato datasets. The work also turned up problems we did not "
            "expect when we wrote the synopsis. The dataset is much smaller than its description says, about a third "
            "of it also appears in a public dataset, the supplied train/test split lets copies of the same photo sit "
            "on both sides, and models trained on our photos do poorly on other tomato datasets. These findings "
            "changed parts of the plan, and Section 7 explains each change."),
          P("Our research questions, refined from the synopsis objectives, are:"),
          bullets(["<b>RQ1</b>  How well can classical models classify the six classes from hand-crafted features when "
                   "copies of a photo are kept out of the test set?",
                   "<b>RQ2</b>  How robust are they to controlled changes in lighting, contrast, image quality and "
                   "background, and can simple remedies help?",
                   "<b>RQ3</b>  Do capture or source shortcuts inflate the scores, and do the models still work on "
                   "other tomato datasets?",
                   "<b>RQ4</b>  Which features carry the useful information, and how stable are the explanations?"]),
          table([["Synopsis objective", "Status at mid-semester"],
                 ["1. Baseline pipeline (colour, texture, shape; RF with other models as a check)",
                  "Done. RF, SVM and LR evaluated on 10 grouped splits (Section 5.1)."],
                 ["2. Loss under illumination, contrast and background perturbations",
                  "Done, widened to 50 conditions; two remedies tested (Sections 5.3, 5.4)."],
                 ["3. Feature-group ablation (colour, texture, shape, combined)", "Done (Section 5.1)."],
                 ["4. Stability of feature importance across folds and perturbations",
                  "First results (Section 5.7). Extended to stability across datasets."],
                 ["Out-of-distribution test (PlantCity, \"if there is time\")",
                  "Brought forward: tested on PlantDoc and two Bangladeshi field datasets (Section 5.6)."]],
                [7.6, 8.4], "Progress against the synopsis objectives.")]

    # ------------------------------------------------------------------ 2 literature
    s += [Paragraph("2. Literature survey", h1),
          Paragraph("2.1 Laboratory images versus field images", h2),
          P("Most image-based plant disease work builds on PlantVillage, about 54,000 photos of single leaves of 14 "
            "crops on a plain background {1}. Mohanty et al. trained convolutional networks on it and reached 99.35% "
            "accuracy on held-out PlantVillage images, but only about 31% on images collected under other conditions "
            "{2}. Ferentinos reported the same pattern on a larger collection {3}, and Barbedo listed the reasons such "
            "models struggle in practice, including the gap between training and deployment images and the difficulty "
            "of separating symptoms from the background {4}. Noyan showed how strong the capture bias in PlantVillage "
            "is: a model that sees only eight background pixels classifies it far above chance (49% against 2.6%) {5}."),
          P("Field and web datasets were collected in response. PlantDoc has 2,598 web images of 13 species with "
            "cluttered backgrounds {6}. Tomato-Village was photographed in fields in Rajasthan, India {7}. PlantWild "
            "is the largest in-the-wild collection, with 18,542 images in 89 classes {8}. Recent regional collections "
            "include our main dataset from Pakistan {9}, PlantCity {pc} and three Bangladeshi tomato datasets "
            "{10-12}. As far as we can find, none of the regional sets has a published benchmark, and each field "
            "dataset is usually evaluated only within itself. How a model trained on one of them behaves on another "
            "is rarely reported. This is the reason we evaluate our models on outside datasets as early as possible."),
          Paragraph("2.2 Hand-crafted features and classical classifiers", h2),
          P("Before deep learning, disease recognition usually meant segmenting the leaf and describing it with "
            "colour, texture and shape features {13}. Texture is commonly measured with grey-level co-occurrence "
            "matrices (GLCM) {14} and local binary patterns (LBP) {15}; colour with channel statistics and "
            "histograms. The usual classifiers are support vector machines {16} and random forests {17}. For tomato, "
            "Mokhtar et al. combined Gabor features with SVMs {18}, Hlaing and Zaw used SIFT-based texture statistics "
            "with colour moments {19}, Tan et al. compared SVM, k-NN and random forests on hand-crafted features "
            "with several CNNs {20}, and Singh and Misra combined segmentation with soft-computing classifiers {21}. "
            "These studies report high accuracy, but nearly all of them use PlantVillage images and random "
            "image-level splits. Our feature set follows this tradition (Section 4.3), and we test it on field photos "
            "with group-aware splits."),
          Paragraph("2.3 Robustness and distribution shift", h2),
          P("Hendrycks and Dietterich showed that image classifiers lose accuracy under common corruptions such as "
            "blur, noise and brightness changes {22}; our perturbation suite follows the same idea. Torralba and "
            "Efros showed that every dataset carries its own signature, so models trained on one dataset transfer "
            "poorly to another {23}. Work on domain generalisation evaluates by holding out whole domains; Gulrajani "
            "and Lopez-Paz found that, under a careful protocol of this kind, many specialised methods do no better "
            "than ordinary training {24}, and WILDS collected real distribution shifts for benchmarking {25}. "
            "Classical ways to reduce lighting effects include grey-world colour constancy {26} and contrast-limited "
            "adaptive histogram equalisation (CLAHE) {27}; data augmentation is the usual learning-based remedy {28}. "
            "We test one remedy of each kind in Section 4.8."),
          Paragraph("2.4 Shortcuts, leakage and duplicates", h2),
          P("Geirhos et al. describe shortcut learning, where a model uses cues that work on the benchmark but are "
            "unrelated to the real task {29}. In medical imaging, models picked up hospital and device signals {30}, "
            "and DeGrave et al. showed that good accuracy on an external test set does not prove that a model uses "
            "the pathology {31}. Kapoor and Narayanan list data leakage, including duplicates across training and "
            "test sets, as a main cause of irreproducible machine learning results {32}. Barz and Denzler found "
            "near-duplicates of 3.3% of CIFAR-10 and 10% of CIFAR-100 test images in the training sets {33}. Roberts "
            "et al. recommend grouped (blocked) cross-validation when data have structure {34}, and confident "
            "learning flags likely label errors from out-of-sample predictions {35}. These papers shaped our grouped "
            "splits (Section 4.4), our shortcut controls (Section 4.9) and our label audit."),
          Paragraph("2.5 Explaining models and features", h2),
          P("Feature importance explains a fitted model, and it has known weaknesses. Random forest impurity "
            "importance is biased towards some feature types {36}, permutation importance can be misleading when "
            "features are correlated {37}, and a feature's importance can change between equally accurate models "
            "{38}. Local explainers such as LIME {39} and SHAP {40} often disagree with each other {41}. Rudin argues "
            "that for high-stakes decisions an interpretable model is better than an explained black box {42}. In "
            "plant pathology, Toda and Okura looked inside a CNN trained on PlantVillage and found that it attends "
            "to lesion colour and texture, but only on that one laboratory dataset {43}. We therefore measure how "
            "stable explanations are across folds, perturbations and datasets, and we also look at each feature on "
            "its own, without a model (Section 4.12)."),
          Paragraph("2.6 Where our work fits", h2),
          P("Classical tomato studies report high accuracy on laboratory images with image-level splits. Field "
            "datasets are benchmarked within themselves. Explanations are usually shown for one model on one "
            "dataset. Our project evaluates classical models on field photos with grouped splits, measures how much "
            "related images inflate a benchmark, tests the models on other tomato datasets, and checks whether the "
            "features that explain a model on our data still mean the same thing elsewhere.")]

    # ------------------------------------------------------------------ 3 dataset
    s += [Paragraph("3. Dataset: what we received and what we found", h1),
          P("The synopsis described 7,200 balanced images (1,200 per class) with a fixed 840/240/120 split per class. "
            "When we opened the data we found two folders. The 7,200 images are an augmented set at 224 px with its "
            "own split (5,040 / 1,440 / 720). The original photos are in a separate raw folder of 830 images, with "
            "80 to 195 photos per class. We built the main experiments on the raw photos, because the augmented set "
            "is derived from them and its split has a leak (below)."),
          fig(REPO / "results" / "eda_samples.png", 13.0, "Sample raw photos, one row per class. Backgrounds, "
              "lighting and framing vary widely, as expected from field photos."),
          table([["Class", "Images used", "Class", "Images used"],
                 ["Early blight", "110", "Leaf mold", "185"],
                 ["Late blight", "110", "Yellow leaf curl virus", "80"],
                 ["Septoria leaf spot", "191", "Healthy", "137"]],
                [4.4, 3.6, 4.4, 3.6], "Raw images used after cleaning: 813 images in 769 source groups. The "
                "largest class is 2.4 times the smallest."),
          P("Our audit found the following:"),
          bullets(["<b>Same photo under two diseases.</b> Nine photos appear in two class folders. We removed them and "
                   "all their copies (17 images), leaving 813 images.",
                   "<b>Copies inside the dataset.</b> Resized, recompressed or flipped copies of a photo exist. We "
                   "group them so that a photo and its copies always go to the same side of a split (Section 4.4). "
                   "The 813 images form 769 groups.",
                   "<b>Overlap with PlantDoc.</b> 263 of the 830 raw images (31.7%) are identical to images in the "
                   "public PlantDoc dataset {6}, 257 of them byte for byte. For 25 of these the two datasets give "
                   "different labels, most often (19 images) early blight in PlantDoc and septoria in ours.",
                   "<b>Suspected label noise.</b> 21 septoria images have \"early blight\" in their file name. An "
                   "out-of-fold label audit with confident learning {35} flags 139 images as possibly mislabelled. "
                   "These are candidates only; we have not changed any label.",
                   "<b>The supplied split leaks.</b> 361 of the 720 test images of the augmented set have a related "
                   "image (same source file name or a near-identical rotated or flipped copy) in training or "
                   "validation."])]

    # ------------------------------------------------------------------ 4 methodology
    s += [Paragraph("4. Methodology", h1),
          P("Figure 2 shows the pipeline. Each step below names the course concepts it relies on and why we chose it."),
          fig(FIG / "pipeline.png", W, "The pipeline. Splits are made by source group, and hyper-parameters are tuned "
              "on the validation part of each split; the test part is used once per split."),
          Paragraph("4.1 Preprocessing and leaf segmentation", h2),
          P("Each photo is resized to 256 x 256 pixels. The leaf is separated from the background with a hue gate in "
            "HSV space (OpenCV hue 20 to 100, yellow to green) followed by morphological opening and closing. We use "
            "this simple rule because it has no parameters to learn and behaves the same way on every image; all "
            "features are computed inside the mask. GrabCut is used only to build the background-replacement test "
            "images, and we seed it so the masks are the same on every run."),
          Paragraph("4.2 Features (feature engineering, correlated features, curse of dimensionality)", h2),
          bullets(["<b>Colour (47):</b> mean, standard deviation and skewness of each channel in RGB, HSV and CIELAB, "
                   "plus a 12-bin hue histogram and an 8-bin saturation histogram.",
                   "<b>Texture (38):</b> uniform LBP histograms at two scales (8 points at radius 1, 16 points at "
                   "radius 2) {15} and GLCM statistics at distances 1 and 3 on 32 grey levels {14}.",
                   "<b>Shape, edge and lesion (20):</b> leaf area, circularity, solidity, extent and eccentricity, the "
                   "seven Hu moments, edge density and gradient statistics, and the count and size of lesion spots."]),
          P("That gives 105 features for about 570 training images per split, and many colour statistics are "
            "strongly correlated with each other. With this ratio of features to samples, a flexible model can fit "
            "noise, and correlated features make importance scores unstable {37}. Both points guide the model "
            "choices and the explainability tests below."),
          Paragraph("4.3 Splitting without leakage (generalisation, data leakage, variance of estimates)", h2),
          P("Near-duplicates are found with a perceptual hash (a DCT hash that also checks flipped versions) and "
            "confirmed when 16 x 16 thumbnails correlate at 0.98 or more. Linked images form one group, and each class "
            "is split 70/15/15 by whole groups, so a photo and its copies never appear in both training and test "
            "{32-34}. The test set has only 118 or 119 images, so one split gives a noisy estimate: on our first split "
            "RF scored 0.665 macro-F1, while the mean over ten splits is 0.600. We therefore repeat the whole "
            "protocol on ten grouped splits (seeds 42 to 51) and report the mean and standard deviation."),
          Paragraph("4.4 Models and why we chose them (bias-variance, ensembles, kernels, regularisation)", h2),
          bullets(["<b>Random forest (RF)</b> {17}, 300 trees. A single deep tree has low bias and high variance; "
                   "bagging over bootstrap samples and random feature subsets averages that variance away. RF handles "
                   "non-linear interactions and correlated features without scaling, and gives feature importances. "
                   "It stays our main model, as planned. Grid: maximum depth (none, 20, 40), features per split "
                   "(square root, 30%), minimum leaf size (1, 2).",
                   "<b>Support vector machine (SVM)</b> {16} with an RBF kernel. It maximises the margin, and the "
                   "kernel lets it draw non-linear boundaries. C and gamma trade bias against variance. Inputs are "
                   "standardised. Grid: C (1, 10, 100), gamma (scale, 0.01, 0.001).",
                   "<b>Logistic regression (LR)</b>, multinomial with an L2 penalty. A linear baseline with "
                   "regularisation strength C (0.01, 0.1, 1, 10), probabilities that tend to be better calibrated, "
                   "and coefficients that are easy to read.",
                   "All three use balanced class weights, because the classes are imbalanced by a factor of 2.4 "
                   "(the synopsis assumed balanced classes)."]),
          P("For every split, each hyper-parameter setting is fitted on the training part and scored on the "
            "validation part. The best setting is refitted on training plus validation and scored once on the test "
            "part. The test images never influence a choice."),
          Paragraph("4.5 Metrics and statistical testing (imbalanced metrics, calibration, multiple testing)", h2),
          P("As in the synopsis, macro-F1 is the main metric, because it weights every class equally. We also report "
            "accuracy, balanced accuracy, per-class precision, recall and F1, confusion matrices, and calibration "
            "(expected calibration error and Brier score). Two configurations are compared on the same ten splits "
            "with a paired Wilcoxon signed-rank test, and p-values are adjusted with the Holm method within each family "
            "of comparisons, so that running many tests does not produce false discoveries."),
          Paragraph("4.6 Robustness protocol", h2),
          P("We apply 50 controlled perturbations to the test images only: brightness (4 levels), contrast (4), "
            "gamma (4), colour cast (4), soft shadow (3), background replacement with grey, noise, soil or grass (4), "
            "Gaussian blur (3), motion blur (3), JPEG compression (3), noise (3), shrinking and re-enlarging the image "
            "(3), occlusion (3), rotation (3), zoom (3), and three combinations. Features and the leaf mask are "
            "recomputed on each perturbed image, as a deployed system would do. Robustness is the perturbed "
            "macro-F1 divided by the clean macro-F1."),
          Paragraph("4.7 Two remedies (regularisation through data, preprocessing for invariance)", h2),
          bullets(["<b>Experiment A, train-only augmentation.</b> Four extra copies of each training image: A1 uses "
                   "random flips and rotations up to 15 degrees; A2 adds brightness and contrast changes between 0.8 "
                   "and 1.2. Copies stay in their photo's group and never reach validation or test, which is checked "
                   "for every split. A zero-copy control reproduces the baseline exactly.",
                   "<b>Experiment B, illumination normalisation.</b> Every image (training and test) is normalised "
                   "before segmentation with either Gray World {26} or CLAHE on the CIELAB lightness channel {27} "
                   "(clip limit 2.0, 8 x 8 tiles). Nothing is fitted, so nothing can leak."]),
          Paragraph("4.8 Shortcut controls (unsupervised clustering, confounders)", h2),
          bullets(["A reference model that sees only image metadata: width, height, aspect ratio and file size.",
                   "Correlation of every feature with image size, and pruning of features that track it.",
                   "Re-processing all images to the same effective resolution, or to the same JPEG compression.",
                   "<b>Source clusters.</b> Images are clustered with k-means on image size, aspect ratio, bytes per "
                   "pixel and file-name pattern, with k chosen by the silhouette score. Each source group is kept in "
                   "one cluster. We then hold out one cluster at a time (leave-one-cluster-out)."]),
          Paragraph("4.9 Grouped re-evaluation of the supplied 7,200-image set", h2),
          P("To see how much the supplied split's leak matters, we grouped the augmented set with three rules fixed "
            "before running any model: our duplicate rule, the same source file name after removing augmentation "
            "suffixes, and near-identical thumbnails under any of the eight rotations and flips. Images whose group "
            "appears in two classes were dropped (183), leaving 7,017 images in 936 groups. We ran the same protocol "
            "twice on the same images, once split by group and once split image by image. Only the split rule "
            "differs."),
          Paragraph("4.10 Testing on other tomato datasets (generalisation, distribution shift)", h2),
          P("Models trained on our raw set (one per split) are applied unchanged to three external datasets: new "
            "PlantDoc images (all six classes) {6}, Nazirpur from Bangladesh (septoria and healthy) {10}, and "
            "Brahmanbaria from Bangladesh (healthy or diseased leaf crops) {11}. Nothing is tuned on external data. "
            "External images that duplicate our own are found first and scored separately. A third Bangladeshi set "
            "{12} was excluded because its labels mark individual lesions and its class numbers could not be matched "
            "to disease names."),
          Paragraph("4.11 Explainability", h2),
          bullets(["Permutation importance per feature and per feature group (colour, texture, shape), measured as "
                   "the drop in the probability of the true class.",
                   "Stability of the ranking across folds (Spearman rank correlation, overlap of the top 10) and "
                   "under perturbation, as planned in the synopsis.",
                   "Transfer: the same importances computed on the external datasets and compared with our own.",
                   "<b>Feature-level effects without a model.</b> For each feature and dataset, how well the feature "
                   "alone separates healthy from diseased leaves (ROC AUC minus 0.5). It needs no trained model, "
                   "which makes it usable on datasets where the models fail."])]

    # ------------------------------------------------------------------ 5 results
    s += [Paragraph("5. Preliminary results", h1),
          P("All numbers are test-set means over the ten grouped splits unless stated otherwise."),
          Paragraph("5.1 Clean performance and feature-group ablation (RQ1)", h2),
          table([["Features", "RF", "SVM", "LR"],
                 ["All (105)", "<b>0.600</b> ± 0.046", "<b>0.615</b> ± 0.026", "<b>0.596</b> ± 0.044"],
                 ["Colour + texture (85)", "0.611 ± 0.041", "0.630 ± 0.034", "0.598 ± 0.031"],
                 ["Colour (47)", "0.541 ± 0.050", "0.543 ± 0.048", "0.495 ± 0.037"],
                 ["Texture (38)", "0.483 ± 0.044", "0.515 ± 0.047", "0.518 ± 0.049"],
                 ["Shape (20)", "0.408 ± 0.042", "0.382 ± 0.046", "0.380 ± 0.041"],
                 ["Metadata only (reference)", "0.487 ± 0.035", "0.424 ± 0.034", "0.351 ± 0.042"]],
                [5.2, 3.6, 3.6, 3.6], "Test macro-F1, mean ± sd over 10 grouped splits. Chance is 0.17."),
          bullets(["The three models do not differ significantly (smallest adjusted p = 0.35). Accuracy is 0.61 to "
                   "0.63.",
                   "Colour and texture together are as good as all features, so shape adds nothing. Each group on its "
                   "own is significantly worse than all features (adjusted p 0.029 to 0.041).",
                   "RF fits its training data perfectly (training macro-F1 1.00), while validation macro-F1 levels "
                   "off at 0.54 to 0.58 from 40% of the training data upwards. The model is in a high-variance regime, "
                   "and more of the same kind of data helps little.",
                   "Early blight and septoria are confused in both directions by every model; this is the largest "
                   "error (for RF, 54 early blight test images predicted as septoria over the ten splits). Part of it "
                   "may come from the suspected label noise in Section 3."]),
          Paragraph("5.2 How much the supplied split inflates the score (RQ3)", h2),
          table([["Model", "Image-level split", "Supplied split", "Grouped split"],
                 ["RF", "0.961 ± 0.006", "0.814", "0.659 ± 0.058"],
                 ["SVM", "0.972 ± 0.004", "0.815", "0.669 ± 0.054"],
                 ["LR", "0.768 ± 0.011", "0.698", "0.638 ± 0.050"]],
                [3.2, 4.3, 4.2, 4.3], "Supplied augmented set, 7,017 images: test macro-F1 under three split rules. "
                "The supplied split is a single split."),
          fig(OLD / "fig1_supplied_split_rule.png", 12.5, "Same images, features and models; only the split rule "
              "changes. Error bars show one standard deviation over 10 splits."),
          P("With an image-level split, 98.7% of test images have a related image in training; with the supplied "
            "split, 50%; with grouping, none. The image-level score beats the grouped score on all ten seeds for every "
            "model, by 0.30 for RF and SVM. The grouped score (0.64 to 0.67) agrees with the accuracy on supplied-split "
            "test images that have no detected relative (0.66 to 0.69), which is an independent check. Copies changed "
            "beyond our rules could still cross the split, so 0.66 may still be slightly optimistic."),
          Paragraph("5.3 Robustness to image changes (RQ2)", h2),
          fig(FIG / "robustness.png", 12.5, "Share of clean macro-F1 kept under each perturbation family. A value of 1 "
              "means no loss."),
          bullets(["Loss of resolution and sharpness hurts most: shrinking and re-enlarging keeps 0.42 of clean "
                   "macro-F1 for RF and 0.32 for SVM, and Gaussian blur 0.43 and 0.31. A likely reason is that the "
                   "texture features depend on fine detail.",
                   "Contrast (0.59 to 0.65) and colour cast (0.61 to 0.70) come next. Rotation (0.96), JPEG (0.87 to "
                   "0.95) and background replacement (0.89 to 0.91) are nearly harmless. For the background this is "
                   "expected, since features are computed inside the leaf mask.",
                   "SVM is the least robust model in most families, and LR the most robust under blur."]),
          Paragraph("5.4 Do augmentation or normalisation help? (RQ2)", h2),
          bullets(["Augmentation does not change clean macro-F1 detectably (RF 0.600 to 0.613 with A2, not "
                   "significant).",
                   "Photometric augmentation (A2) improves robustness to brightness (+0.05 to +0.11) and contrast "
                   "(+0.09 to +0.12) for all three models, at test levels outside the training range. For RF, "
                   "contrast robustness rises from 0.61 to 0.78. It costs RF robustness to colour cast (-0.07) and "
                   "noise (-0.03).",
                   "Gray World lowers clean macro-F1 by 0.05 to 0.06 and makes colour casts harmless, as its design "
                   "implies. But it also changes segmentation: it shrinks the average leaf mask from 83% to 39% of the "
                   "image, so its effect cannot be read as colour normalisation alone.",
                   "CLAHE is roughly neutral and does not improve lighting robustness."]),
          Paragraph("5.5 Shortcuts (RQ3)", h2),
          bullets(["Metadata alone reaches macro-F1 0.487, about three times chance, so the dataset carries a source "
                   "signal. All features beat it on every split.",
                   "Equal resolution or equal JPEG compression leaves the scores unchanged (RF 0.598 and 0.603), so "
                   "pixel-level artefacts do not explain them. No feature correlates with image size above 0.43.",
                   "k-means found 8 source clusters (silhouette 0.465). One cluster is 87% leaf mold and is classified "
                   "with 0.91 to 0.96 accuracy. Holding out one cluster at a time drops accuracy from about 0.61 to "
                   "0.47 to 0.50. Part of the score is therefore tied to the image source. Holding out a cluster also "
                   "changes the class balance, so this is still preliminary."]),
          Paragraph("5.6 Other tomato datasets (RQ3)", h2),
          table([["Test data", "RF", "SVM", "LR"],
                 ["Our test split", "0.613 ± 0.046", "0.615 ± 0.026", "0.595 ± 0.044"],
                 ["PlantDoc, 310 new images (six classes)", "0.290 ± 0.018", "0.330 ± 0.028", "0.329 ± 0.012"],
                 ["Nazirpur, 378 images (septoria, healthy)", "0.245 ± 0.023", "0.234 ± 0.031", "0.213 ± 0.017"],
                 ["Brahmanbaria: healthy leaves called healthy", "3%", "17%", "12%"]],
                [6.4, 3.2, 3.2, 3.2], "Models trained on our raw set, applied unchanged. Macro-F1 over the classes "
                "present in each test set. The RF figure for our test split comes from a second machine and differs "
                "slightly from Table 3 (0.613 against 0.600); SVM and LR agree within 0.003."),
          fig(OLD / "fig2_cross_dataset.png", 13.5, "Cross-dataset macro-F1, mean ± sd over 10 splits."),
          bullets(["All three models drop by about the same amount.",
                   "Healthy leaves from other sources are mostly called diseased.",
                   "PlantDoc images that also appear in our data score far higher than new PlantDoc images (RF 0.81 "
                   "against 0.29), which confirms that the overlap would have inflated the result.",
                   "A classifier separates our images from the external ones using the same 105 features with ROC AUC "
                   "0.85 (PlantDoc) and 0.98 to 0.99 (Bangladesh), while the leaf masks look similar. The feature "
                   "distributions differ between sources."]),
          Paragraph("5.7 Explainability (RQ4)", h2),
          bullets(["Shuffling the colour group lowers the true-class probability by 0.164, texture by 0.083 and shape "
                   "by 0.039 (single split). The top features are LAB-b standard deviation, GLCM correlation and "
                   "HSV-S standard deviation.",
                   "Across folds the importance ranking is stable (rank correlation 0.90), but the top-10 list changes "
                   "(overlap 0.66 to 0.71). Under strong contrast changes the ranking changes a lot (rank correlation "
                   "0.17 at contrast 0.5).",
                   "Explanations do not carry over to other datasets. RF feature rankings agree 0.87 between our own "
                   "splits, 0.61 with PlantDoc and -0.01 with Nazirpur. SVM and LR rankings are unstable even between "
                   "our own splits (0.18 and 0.48). A model that fails on a dataset cannot explain that dataset."]),
          P("Because model-based explanations depend on a working model, we also measured each feature on its own. "
            "Figure 6 shows the result for healthy against diseased leaves, the one comparison available in all four "
            "datasets."),
          fig(OLD / "fig3_feature_effects.png", 12.5, "Effect of single features (ROC AUC minus 0.5; positive means "
              "higher in diseased leaves) in four datasets. Top: the two features consistent in all four. Bottom: the "
              "eight strongest features in our own data."),
          P("Only 2 of the 105 features point the same way with an effect of at least 0.1 in all four datasets: the "
            "share of yellow-orange leaf pixels (higher in diseased leaves) and the skewness of the green-red channel "
            "(lower in diseased leaves). Both describe visible yellowing and browning. The strongest features in our "
            "own data change direction elsewhere, and our dataset's effect profile correlates -0.51 with "
            "Brahmanbaria's. This is a first result without confidence intervals; Section 8 describes how we will "
            "make it rigorous.")]

    # ------------------------------------------------------------------ 6 challenges
    s += [Paragraph("6. Challenges and how we are handling them", h1),
          table([["Challenge", "Type", "What we did or will do"],
                 ["Dataset much smaller than described (830 photos, not 7,200), imbalanced, no usable split",
                  "Data", "Built our own grouped splits on the raw photos; class weights; supplied split audited and "
                  "kept only as a reference."],
                 ["Same photo filed under two diseases", "Data", "Removed the 9 photos and their copies (17 images)."],
                 ["A third of the data also in PlantDoc, 25 with different labels", "Data",
                  "Overlapping images excluded from the external test and scored separately; labels not changed "
                  "until reviewed."],
                 ["Suspected label noise (21 septoria files named early blight; 139 audit flags)", "Data",
                  "Blind visual review planned; results will be re-run with and without suspect images."],
                 ["Photos from one capture session, taken seconds apart, can fall on both sides of a split",
                  "Data", "Found from photo timestamps; a sensitivity run that groups by session is planned."],
                 ["A few rotated copies escape the flip-aware duplicate check", "Data",
                  "Documented; they affect all experiments equally, so comparisons stand. A rotation-aware rule is "
                  "already used for the supplied set."],
                 ["One test split too noisy (RF 0.665 on one split, 0.600 on average)", "Modelling",
                  "Ten repeated grouped splits with tuning inside each, paired tests with Holm correction."],
                 ["RF overfits (training 1.00, validation 0.54 to 0.58)", "Modelling",
                  "Depth and leaf-size grid; compared against regularised LR and SVM; robustness and transfer "
                  "measured, which is where overfitting shows."],
                 ["GrabCut masks changed between runs", "Modelling", "Seeded OpenCV before each call; masks now "
                  "identical across runs."],
                 ["Gray World shrinks the colour-based leaf mask", "Modelling",
                  "Reported as a confound; a run with the mask fixed from the original image is planned."],
                 ["Early blight and septoria confused both ways", "Error analysis",
                  "Linked to the suspected label noise; will be re-checked after the label review."],
                 ["Healthy leaves from other datasets called diseased", "Error analysis",
                  "Domain classifier shows the feature distributions shift; led to the multi-source plan in "
                  "Section 8."],
                 ["Small differences on a second machine (RF 0.613 against 0.600)", "Reproducibility",
                  "Every comparison is made within one machine; SVM and LR reproduce within 0.003."]],
                [5.6, 2.3, 8.1], "Challenges met so far.")]

    # ------------------------------------------------------------------ 7 revisions
    s += [Paragraph("7. Revisions to the original plan", h1),
          table([["Synopsis", "Now", "Reason"],
                 ["7,200 balanced images with the supplied 840/240/120 split; no class weights",
                  "813 raw photos, ten grouped 70/15/15 splits, balanced class weights",
                  "The 7,200 images are augmented copies of the raw photos and the supplied split leaks (Section 3)."],
                 ["RF as the main model, SVM and LR as a brief check", "All three evaluated in full; RF still used for "
                  "importances", "The three models score the same, and they differ in robustness and in how stable "
                  "their explanations are."],
                 ["Perturbations of illumination, contrast and background", "50 conditions, including blur, resolution, "
                  "noise and occlusion; augmentation and normalisation tested as remedies",
                  "Resolution and blur turned out to matter most; testing remedies answers RQ2 more fully."],
                 ["No shortcut analysis", "Metadata model, resolution and compression controls, source clusters",
                  "Image size alone predicts the class far above chance."],
                 ["PlantCity as an outside test \"if time allows\"", "Outside test done now on PlantDoc and two "
                  "Bangladeshi datasets; PlantCity remains a candidate", "The shortcut results made transfer the most "
                  "informative next experiment, and the PlantDoc overlap had to be checked anyway."],
                 ["Importance stability across folds and perturbations", "Also across datasets, plus feature effects "
                  "measured without a model", "Model-based explanations collapse where the model fails."],
                 ["Hand-drawn mask evaluation, extra model families", "Deferred", "Time goes to the experiments "
                  "that answer the research questions."]],
                [4.6, 5.6, 5.8], "Changes to the plan and why.")]

    # ------------------------------------------------------------------ 8 plan
    s += [Paragraph("8. Plan for the final paper (24 December)", h1),
          P("So far, models trained on one source mostly learned that source, and their explanations did too. For the "
            "final paper we will train on several sources at once and explain individual features."),
          bullets(["<b>Multi-source evaluation.</b> Pool our photos with the external tomato datasets, plus further "
                   "field, web and laboratory sources {6-8}, removing cross-source duplicates with our duplicate "
                   "rule. We have collected and de-duplicated these sources. Models will be trained on all sources but "
                   "one and tested on the held-out source (leave-one-source-out), the protocol used in domain "
                   "generalisation {24}.",
                   "<b>Better features.</b> Leaf-relative lesion features (the share of yellow, brown and dark tissue "
                   "on the leaf, and how lesions differ in colour from the healthy tissue of the same leaf), designed "
                   "to be less sensitive to camera colour.",
                   "<b>Learned feature extractors, still with classical classifiers.</b> Colour and texture "
                   "\"visual words\" learned with k-means {52-55}, and pretrained networks used only to extract "
                   "features {46, 49, 50}, with no network training. RF, SVM and LR stay the classifiers.",
                   "<b>Feature-level explanation.</b> The effect of every feature in every source, with bootstrap "
                   "confidence intervals, and a rule for which features keep the same effect across sources. We will "
                   "compare this with model-based importance and test whether models built only on stable features "
                   "transfer better.",
                   "<b>Open checks.</b> Label review, a capture-session sensitivity run, a matched random-holdout "
                   "control for leave-one-cluster-out, and Gray World with a fixed mask."]),
          table([["Period", "Work"],
                 ["14 Nov to 30 Nov", "Multi-source evaluation with the current features; label review."],
                 ["1 Dec to 12 Dec", "Lesion features, learned extractors, feature-level explanation with "
                  "confidence intervals."],
                 ["13 Dec to 24 Dec", "Sensitivity checks, error analysis, IEEE paper (5 to 6 pages) and "
                  "presentation."]],
                [4.0, 12.0], "Timeline to the final paper.")]

    # ------------------------------------------------------------------ references
    s += [Paragraph("References", h1)]
    for i, k in enumerate(C.order, 1):
        s.append(Paragraph(f"[{i}] " + C.refs[k].replace("&", "&amp;"), refst))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.3 * cm,
                            bottomMargin=2.3 * cm, title="Mid-Semester Report: Robust and Explainable Classical ML "
                            "for Tomato Disease Recognition", author="Hamna Sadia Rizwan, Maria Ilyas Malik, "
                            "Manahil Kamran")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT, "references:", len(C.order))


if __name__ == "__main__":
    if "--figs" in sys.argv:
        make_figs()
    else:
        build()
