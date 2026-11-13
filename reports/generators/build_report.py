"""Build the project status report PDF (reportlab). Numbers are copied from docs/verified_results.md §§3-13."""
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

FONTDIR = Path("F:/NUST/ML/code/mid/Tomato-Leaf-Disease-Recognition/.venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf")
pdfmetrics.registerFont(TTFont("DV", str(FONTDIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FONTDIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(FONTDIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

OUT = Path("F:/NUST/ML/code/mid/Project_Status_Report_2026-11-12.pdf")
FIG = Path("F:/NUST/ML/report/fig")
INK, INK2, RULE, TINT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d3"), \
    colors.HexColor("#f3f2ef")
ACCENT = colors.HexColor("#1c5cab")

base = ParagraphStyle("b", fontName="DV", fontSize=9.4, leading=13.2, textColor=INK, alignment=TA_LEFT,
                      spaceAfter=5)
small = ParagraphStyle("s", parent=base, fontSize=8, leading=10.5, textColor=INK2)
cell = ParagraphStyle("c", parent=base, fontSize=8.2, leading=10.6, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="DVB")
h1 = ParagraphStyle("h1", parent=base, fontName="DVB", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6,
                    textColor=ACCENT)
h2 = ParagraphStyle("h2", parent=base, fontName="DVB", fontSize=10.8, leading=14, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("t", parent=base, fontName="DVB", fontSize=19, leading=24, spaceAfter=4)
sub = ParagraphStyle("st", parent=base, fontSize=10, leading=14, textColor=INK2, spaceAfter=10)
box = ParagraphStyle("box", parent=base, fontSize=9.4, leading=13.2)

P = lambda t, s=base: Paragraph(t, s)


def bullets(items, style=base):
    return ListFlowable([ListItem(P(i, style), leftIndent=12, bulletColor=INK2) for i in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def table(rows, widths, header=True, bold_first_col=False, zebra=True):
    data = [[P(str(c), cellb if (header and r == 0) or (bold_first_col and j == 0) else cell)
             for j, c in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), TINT))
    t.setStyle(TableStyle(st))
    return t


def callout(text):
    t = Table([[P(text, box)]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf1fb")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def fig(name, width_cm, caption):
    img = Image(str(FIG / name))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return KeepTogether([img, P(caption, small), Spacer(1, 6)])


def footer(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(INK2)
    c.drawString(2.2 * cm, 1.2 * cm, "Tomato disease recognition with hand-crafted features: project status, 12 November 2026")
    c.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{d.page}")
    c.restoreState()


s = []
# ------------------------------------------------------------------ title + summary
s += [P("Robust and Explainable Classical ML for Tomato Disease Recognition", title),
      P("Project status: what we have done, what we found, and what to do next. "
        "CS-871, NUST SEECS. Status as of 12 November 2026.", sub)]

s += [P("1. Summary", h1),
      callout("<b>Bottom line.</b> The project has not failed, but its story has changed. Our models score about 0.60 "
              "macro-F1 on our own data and do not transfer to other tomato datasets (0.21–0.33). Explanations computed "
              "from those models therefore describe our dataset, not the disease. Explanation based on the "
              "<b>features themselves</b>, without any model, is feasible now and already gives a clear first result: "
              "only 2 of our 105 hand-crafted features separate healthy from diseased leaves in all four datasets we "
              "tested, and both are yellow/brown colour cues. We recommend making that feature-level, multi-dataset "
              "analysis the core of the final paper."),
      Spacer(1, 6),
      P("<b>What we did.</b> We built a classical pipeline (105 colour, texture and shape features; Random Forest, SVM, "
        "Logistic Regression) and evaluated it with leakage-safe grouped splits, 50 image perturbations, augmentation, "
        "illumination normalisation and source-shortcut controls. Since 27 October we added three experiments: a "
        "grouped re-evaluation of the supplied 7,200-image set (Task 2), a test on public tomato datasets from India "
        "(PlantDoc) and Bangladesh, and a model-free feature analysis across those datasets."),
      P("<b>Main findings.</b>"),
      bullets([
          "On our 813 raw photos, macro-F1 is about 0.60 for all three models (10 grouped splits); they do not differ "
          "significantly.",
          "The supplied 7,200-image benchmark (0.81) is inflated by related images on both sides of the split. "
          "Split by source photo, the same images score about 0.66; split image by image, 0.97.",
          "About a third of our raw dataset (263 of 830 images) is identical to images in the public PlantDoc dataset, "
          "and 25 of those carry a different label there.",
          "Trained on our data, the models reach only 0.21–0.33 macro-F1 on other datasets and call most healthy "
          "leaves from other sources diseased.",
          "Model-based explanations collapse on other datasets. Feature-level effects show that our dataset's "
          "strongest features flip direction elsewhere, while two colour features stay consistent everywhere."]),
      P("<b>Recommendation.</b> Keep classical ML and the hand-crafted features. Change the unit of analysis from "
        "\"one dataset, one model\" to \"several sources, each feature\": pool our data with the external datasets, "
        "evaluate by leaving one source out, and explain with per-feature effect sizes and their consistency across "
        "sources (Section 7). Before tomorrow's midterm deadline, the literature survey is the main open item."),
      ]

# ------------------------------------------------------------------ 2. project
s += [P("2. The project in brief", h1),
      P("Six classes of tomato leaves: healthy, early blight, late blight, septoria leaf spot, leaf mold and yellow "
        "leaf curl virus. Each image is resized to 256 × 256, the leaf is segmented with a colour (hue) gate, and 105 "
        "features are computed inside the leaf: colour (47), texture (38: LBP and GLCM) and shape/edge/lesion (20). "
        "Three classical models are tuned on a validation split, refitted on train + validation and scored once on "
        "test."),
      table([["Research question", "Status now"],
             ["RQ1: How accurately can classical ML classify the six classes?",
              "Answered on our data (about 0.60); answered across datasets (0.21–0.33)."],
             ["RQ2: How robust are the models to lighting and image quality?",
              "Largely answered: resolution and blur hurt most; photometric augmentation helps lighting only."],
             ["RQ3: Can shortcuts or related images inflate performance?",
              "Answered with strong evidence: supplied split, source clusters, PlantDoc overlap, cross-dataset drop."],
             ["RQ4: Which features are useful, interpretable and stable?",
              "First feature-level answer (Section 6.2); needs the multi-source design to be conclusive."]],
            [7.6, 8.8]),
      ]

# ------------------------------------------------------------------ 3. data
s += [P("3. Data", h1),
      table([["Dataset", "Role", "Images used", "Notes"],
             ["Pakistan raw set (Dataset (raw))", "Main training and test data", "813 of 830",
              "17 removed (same photo under two classes). 2.4× class imbalance. 31.7% identical to PlantDoc."],
             ["Supplied Augmented Dataset", "Benchmark; Task 2", "7,200 / 7,017",
              "Augmented copies of source photos; supplied split places related images on both sides."],
             ["PlantDoc (India, web images)", "External test", "310 new + 270 overlapping",
              "All six classes. Overlapping images scored separately."],
             ["Nazirpur, Bangladesh", "External test", "378 (+264 leaf curl)",
              "Septoria and healthy; “leaf curl” only as an uncertain match to yellow leaf curl virus."],
             ["Brahmanbaria, Bangladesh", "External test", "1,035 leaf crops", "Healthy vs diseased only."],
             ["Dinajpur, Bangladesh", "Excluded", "–",
              "Labels mark lesion spots, not leaves; class numbers cannot be matched to disease names."]],
            [3.9, 3.0, 2.7, 6.8]),
      ]

# ------------------------------------------------------------------ 4. results on our data
s += [P("4. Results on our own data (completed before 27 October)", h1),
      P("All numbers are means over 10 grouped 70/15/15 splits (seeds 42–51), so no source photo is in both training "
        "and test. Tuning happens inside each split."),
      P("4.1 Clean performance", h2),
      table([["Feature set", "RF", "SVM", "LR"],
             ["All 105 features", "0.600 ± 0.046", "0.615 ± 0.026", "0.596 ± 0.044"],
             ["Colour + texture (85)", "0.611 ± 0.041", "0.630 ± 0.034", "0.598 ± 0.031"],
             ["Colour only (47)", "0.541", "0.543", "0.495"],
             ["Texture only (38)", "0.483", "0.515", "0.518"],
             ["Shape only (20)", "0.408", "0.382", "0.380"],
             ["Metadata only (image size, file size)", "0.487", "0.424", "0.351"]],
            [6.4, 3.3, 3.3, 3.3]),
      Spacer(1, 4),
      bullets(["The three models are statistically indistinguishable (Wilcoxon, Holm-corrected).",
               "Shape adds nothing on top of colour + texture. Early blight and septoria are confused in both "
               "directions by every model.",
               "Image metadata alone reaches 0.49, three times chance: a source signal exists in the dataset."]),
      P("4.2 Robustness, augmentation and illumination", h2),
      table([["Experiment", "Main result"],
             ["50 test-time perturbations", "Share of clean macro-F1 kept (RF): down-then-up scaling 0.42, blur 0.43, "
              "contrast 0.61, colour cast 0.70, brightness 0.83, rotation 0.96."],
             ["A: train-only augmentation", "No clean change. Geometric + photometric (A2) improves brightness "
              "(+0.05 to +0.11) and contrast (+0.09 to +0.12) robustness for all models; costs colour cast and noise."],
             ["B: illumination normalisation", "Gray World lowers clean macro-F1 by 0.05–0.06 (confounded: it shrinks "
              "the colour-based leaf mask from 83% to 39%). CLAHE is roughly neutral."]],
            [4.6, 11.8]),
      P("4.3 Source shortcut", h2),
      P("Equalising resolution or JPEG compression does not change the scores, so pixel-level artefacts are not the "
        "explanation. But when a whole source cluster (images with similar size, aspect ratio and compression) is held "
        "out, accuracy falls from about 0.61 to 0.47–0.50. One cluster is 87% leaf mold and is classified with "
        "0.91–0.96 accuracy. Part of the score is source-specific. The cross-dataset test in Section 5.3 confirms "
        "this on real external data."),
      ]

# ------------------------------------------------------------------ 5. new experiments
s += [P("5. New experiments (12 November)", h1),
      P("5.1 Task 2: grouped evaluation of the supplied 7,200-image set", h2),
      P("We grouped related images with three rules fixed before any model was run (the pipeline's duplicate rule; "
        "the same source file name; near-identical thumbnails under any rotation or flip), dropped 183 images whose "
        "group appears under two classes, and kept 7,017 images in 936 groups. The number of groups per class "
        "(86–216) is close to the number of raw photos per class (80–195), so a group is roughly one source photo. "
        "We then ran the unchanged main protocol, and a control with the same images split image by image."),
      fig("fig1_supplied_split_rule.png", 15.5,
          "Figure 1. Same images, same features, same models: only the split rule differs. Bars are means over 10 "
          "splits (error bars: ± 1 sd); the supplied split is a single split. Share of test images with a related "
          "image in training: image-level 98.7%, supplied 50%, grouped 0%."),
      table([["Model", "Image-level split", "Supplied split", "Grouped split"],
             ["RF", "0.961 ± 0.006", "0.814", "0.659 ± 0.058"],
             ["SVM", "0.972 ± 0.004", "0.815", "0.669 ± 0.054"],
             ["LR", "0.768 ± 0.011", "0.698", "0.638 ± 0.050"]],
            [3.4, 4.3, 4.3, 4.4]),
      Spacer(1, 4),
      bullets(["The image-level score is higher than the grouped score on 10 of 10 seeds for every model.",
               "Grouped accuracy (0.64–0.67) agrees with the accuracy on supplied-split test images that have no "
               "detected sibling (0.66–0.69): two independent ways of removing related images give the same answer.",
               "Copies changed beyond our rules (crops, recolouring) could still cross splits, so 0.66 may still be "
               "slightly optimistic."]),
      P("5.2 Dataset provenance: a third of our data is PlantDoc", h2),
      P("Matching every PlantDoc tomato image against our 830 raw images by file content found 263 raw images (31.7%) "
        "that are identical to PlantDoc images, 257 of them byte for byte. For 25 of them the label differs; most often "
        "(19 images) PlantDoc says early blight where our dataset says septoria. This matches the 21 septoria images "
        "with early-blight file names we had already flagged. We have not changed any label: which dataset is right "
        "is not established."),
      P("5.3 Cross-dataset test", h2),
      P("For each of the 10 splits, the model trained on our data was applied unchanged to the external images. "
        "Nothing was tuned on external data. PlantDoc images that also appear in our data were scored separately."),
      fig("fig2_cross_dataset.png", 15.5,
          "Figure 2. Macro-F1 over the classes present in each test set, mean ± sd over 10 splits. Chance for six "
          "balanced classes is 0.17. Our-test numbers come from the second machine (Section 9)."),
      table([["Test data", "RF", "SVM", "LR"],
             ["Our test split", "0.613", "0.615", "0.595"],
             ["PlantDoc images also in our data (270)", "0.810", "0.653", "0.564"],
             ["PlantDoc new images (310)", "0.290", "0.330", "0.329"],
             ["Nazirpur (378)", "0.245", "0.234", "0.213"],
             ["Brahmanbaria: healthy leaves called healthy", "3%", "17%", "12%"]],
            [7.6, 2.9, 2.9, 3.0]),
      Spacer(1, 4),
      bullets(["The drop happens for every model, so model choice is not the problem.",
               "Healthy leaves from other sources are mostly called diseased. Yellow leaf curl (F1 0.50–0.59) and "
               "septoria (0.38–0.45) transfer partly to PlantDoc; healthy, early blight and leaf mold barely do.",
               "A classifier can tell our images from the external ones using the same 105 features (ROC AUC 0.85 for "
               "PlantDoc, 0.98–0.99 for the Bangladeshi sets), while leaf-mask size and failure rate are similar. The "
               "images differ in their feature distributions, not because segmentation broke."]),
      ]

# ------------------------------------------------------------------ 6. explainability
s += [P("6. Explainability: model-based vs feature-based", h1),
      P("6.1 Model-based explanations do not transfer", h2),
      P("Permutation importance measures how much a trained model's accuracy drops when a feature is shuffled. It "
        "therefore explains the model, and only where the model works."),
      bullets(["On our test split, colour matters most for RF (0.177 drop in true-class probability), then texture "
               "(0.092) and shape (0.042). On new PlantDoc images these fall to 0.039, 0.037 and 0.021; on Nazirpur "
               "to about 0.",
               "RF feature rankings agree 0.87 (rank correlation) between our own splits, 0.61 with PlantDoc and −0.01 "
               "with Nazirpur.",
               "SVM and LR rankings are unstable even between our own splits (0.18 and 0.48), so per-feature model "
               "explanations are only meaningful for RF."]),
      callout("As we suspected: <b>if the model does not work, its explanation is not "
              "meaningful.</b> Model-based explanations measured on our data describe a model that is specific to "
              "our dataset."),
      P("6.2 Feature-based explanation works without a model", h2),
      P("Feature-based explanation asks a different question: <i>does this feature itself differ between healthy and "
        "diseased leaves, and does that difference hold in every dataset?</i> No classifier is involved, so a failing "
        "model cannot invalidate it. For each feature and dataset we computed the effect as ROC AUC − 0.5 (positive: "
        "higher in diseased leaves). Healthy vs diseased is the one comparison available in all four datasets."),
      fig("fig3_feature_effects.png", 14.5,
          "Figure 3. Healthy-vs-diseased effect of each feature in each dataset. Top: the two features that are "
          "consistent (same direction, |effect| ≥ 0.1) in all four. Bottom: the eight strongest features in our raw "
          "set; most change direction in other datasets."),
      bullets(["<b>Consistent in all four datasets (2 of 105):</b> the share of yellow-orange leaf pixels (hue "
               "histogram bin 1, higher in diseased leaves) and the skewness of the green–red colour channel (LAB a*, "
               "lower in diseased leaves). Both describe visible yellowing and browning, a plausible disease cue.",
               "<b>Our dataset's strongest features are source-specific:</b> saturation skew, LAB-b mean, blue-channel "
               "mean and others change sign across datasets. Our raw set's effect profile correlates −0.51 with "
               "Brahmanbaria's, while the two Bangladeshi sets agree with each other (+0.40).",
               "By group: 2 of 47 colour, 0 of 38 texture and 0 of 20 shape features are consistent. Texture "
               "features shift most between our data and PlantDoc/Nazirpur.",
               "Six classes, our data vs PlantDoc: the per-feature effect rankings agree only weakly (0.30; top-10 "
               "overlap 0.11)."]),
      P("<b>Limits of this first analysis.</b> No confidence intervals yet, and the external subsets are small "
        "(PlantDoc has 46 healthy images). “Diseased” mixes different diseases in different datasets. It covers "
        "healthy vs diseased only. Section 7 turns this into a proper design.", small),
      ]

# ------------------------------------------------------------------ 7. is it optimal / options
s += [P("7. Is the current approach optimal? Options", h1),
      P("No. The current design trains and explains on one source and uses model-based explanations. Our results show "
        "both choices fail when the source changes. The table compares the options we considered."),
      table([["Option", "What it means", "Verdict"],
             ["A. Keep the current design", "One dataset, model-based importance.",
              "<b>Not recommended.</b> Explanations are dataset-specific (Section 6.1)."],
             ["B. Multi-source, feature-centred (recommended)",
              "Pool sources (Pakistan-only raw images, PlantDoc-derived raw images, new PlantDoc, Nazirpur, "
              "Brahmanbaria); evaluate leave-one-source-out; explain with per-feature effects and their consistency.",
              "<b>Recommended.</b> Uses data we already have, answers RQ3 and RQ4 directly, stays classical."],
             ["C. Switch the main dataset to PlantVillage (lab)",
              "Large, clean, all six classes, uniform background.",
              "High accuracy, but lab-only conflicts with our “real-world field” title. Useful at most as one extra "
              "source."],
             ["D. Make the supplied 7,200 set the main dataset",
              "Grouped as in Task 2.",
              "Not independent: it is derived from the same source photos. Use only as augmentation of the raw set."],
             ["E. Switch to deep learning (CNN)",
              "Learned features instead of hand-crafted ones.",
              "Out of scope for the course, and its explanations are harder to tie to named features. Does not solve "
              "the source problem."],
             ["F. Redesign features for invariance",
              "Lesion colour relative to the same leaf's healthy tissue, yellow/brown lesion area, ratios instead of "
              "raw colour values.",
              "<b>Recommended with B.</b> Section 6.2 suggests these are the cues that transfer."],
             ["G. Interpretable (glass-box) models",
              "Sparse logistic regression, shallow trees or additive models on the consistent features.",
              "<b>Recommended with B.</b> The model is its own explanation in feature terms."]],
            [3.9, 6.4, 6.1]),
      Spacer(1, 6),
      P("Recommended design for the final paper", h2),
      bullets(["<b>Sources:</b> split our raw set into its PlantDoc-derived part (263 images) and the rest, then add "
               "new PlantDoc, Nazirpur and Brahmanbaria. Each source is one group.",
               "<b>Tasks:</b> healthy vs diseased across all sources (the comparison every source supports); the "
               "six-class task on the sources that have those classes.",
               "<b>Evaluation:</b> leave one source out. Train on the others, test on the held-out source, and "
               "report each held-out source separately.",
               "<b>Explanation:</b> per-feature effect sizes with bootstrap confidence intervals in every source; a "
               "feature counts as explanatory only if its direction holds across sources. Select features on the "
               "training sources only, then check them on the held-out one (avoids selecting on test data).",
               "<b>Models:</b> RF (its importances are the only stable ones) plus one glass-box model on the "
               "consistent features. Compare their transfer with the full-feature models."]),
      ]

# ------------------------------------------------------------------ 8. next steps
s += [P("8. Next steps", h1),
      P("Before the midterm deadline (13 November, 23:59 PST)", h2),
      table([["#", "Task", "Why"],
             ["1", "Write the literature survey (§2 of the draft): lab vs field datasets, hand-crafted features, "
                   "robustness and shift, shortcut learning and dataset bias, explanation stability. Real, checked "
                   "sources only.", "Graded; still empty."],
             ["2", "Replace the old verified_results.md beside the midterm draft with the current project copy.",
              "The draft links to an out-of-date copy."],
             ["3", "Drop or mark as preliminary the unsaved counts in §6 of the draft (154 photos, 4 rotated pairs).",
              "No saved output supports them."],
             ["4", "Done 12 Nov: the feature-level result (Section 6.2) is in RQ4 of the draft, marked preliminary. Check it reads well.",
              "It answers the explainability question directly."]],
            [0.7, 11.2, 4.5]),
      P("For the final paper (24 December)", h2),
      table([["#", "Task", "Effort"],
             ["5", "Build the multi-source manifest (sources as groups; raw split into PlantDoc-derived and rest).",
              "Small: overlap data exists."],
             ["6", "Leave-one-source-out evaluation, healthy vs diseased and six classes.", "Medium"],
             ["7", "Feature-level explanation suite: effects with bootstrap CIs per source; consistency; selection "
                   "on training sources only.", "Medium"],
             ["8", "Invariant lesion features (relative colour, yellow/brown lesion fraction); test whether they "
                   "transfer better.", "Medium–large"],
             ["9", "Glass-box model on consistent features vs full-feature models.", "Small"],
             ["10", "Label review, including the 25 PlantDoc label conflicts and the 21 septoria / early blight "
                    "suspects.", "Small"],
             ["11", "Write the paper around the new story: inflated benchmarks, provenance, transfer, feature-level "
                    "explanation.", "Large"]],
            [0.7, 11.2, 4.5]),
      P("Risks: the external subsets are small (some classes have under 50 images); class definitions differ between "
        "datasets; Nazirpur's leaf-curl mapping is uncertain; the time to 24 December is limited, so tasks 5–7 and 9 "
        "should come before 8.", small),
      ]

# ------------------------------------------------------------------ 9. caveats
s += [P("9. Caveats and reproducibility", h1),
      bullets(["<b>Second machine.</b> The new experiments ran on a 4-core Windows machine (Python 3.11.9, same "
               "library versions). SVM and LR reproduce the committed results almost exactly (largest difference 0.003); Random Forest differs "
               "slightly (in-domain mean 0.613 vs 0.600), because it is sensitive to tiny floating-point differences "
               "in the features. Every comparison in Sections 5–6 was made on this one machine.",
               "<b>Statistics.</b> Our-data numbers are means over 10 grouped splits; the supplied split and the "
               "external sets are single fixed test sets scored by 10 models.",
               "<b>Unverified older figures</b> (no saved output): linkage percentages for the supplied set, the "
               "capture-session counts, the rotated-copy pairs. They are flagged in verified_results.md.",
               "<b>Where things are.</b> Numbers: docs/verified_results.md §§3–13. Commands: docs/experiment_log.md. "
               "New code: scripts/19–23, configs/supplied/, configs/external/. Run outputs: F:/NUST/ML/runs/. "
               "External data: F:/NUST/ML/external/."]),
      Spacer(1, 6),
      P("Dataset sources", h2),
      bullets(["PlantDoc: Singh et al., CODS-COMAD 2020; github.com/pratikkayal/PlantDoc-Dataset (CC BY 4.0).",
               "Nazirpur: Bapari, Zitu and Ummehani, Mendeley Data, DOI 10.17632/74zvtzx9pr.1 (CC BY 4.0).",
               "Brahmanbaria: Oni and Prama, Data in Brief 2025; Mendeley Data, DOI 10.17632/rnbsw72zb5.1.",
               "Dinajpur (excluded): Mendeley Data, DOI 10.17632/bpfd9cns5g.2.",
               "Pakistan tomato dataset: Malik, Ali and Ali, Mendeley Data, DOI 10.17632/3mbnb82mxd.2."], small),
      ]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title="Project status report: tomato disease recognition",
                        author="CS-871 project team")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
