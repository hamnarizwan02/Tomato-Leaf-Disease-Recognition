"""Paper content for render.py. Static sections are written here; every number in Results is read from the run's CSVs
(F:/NUST/ML/runs/p4_multisource/results and the Phase 1-2 verified results), so the paper cannot drift from the runs.

    F:/NUST/ML/tools/report_venv/Scripts/python paper_content.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
REPO = Path("F:/NUST/ML/code/mid/Tomato-Leaf-Disease-Recognition")
RUN = Path("F:/NUST/ML/runs/p4_multisource")

META = {
    "title": "Which Leaf Features Survive a Change of Camera? Multi-Source Evaluation and Feature-Level Explanation "
             "of Classical Tomato-Disease Classifiers",
    "authors": ["Hamna Sadia Rizwan", "Maria Ilyas Malik", "Manahil Kamran"],
    "affil": "School of Electrical Engineering and Computer Science (SEECS), NUST, Islamabad, Pakistan",
    "keywords": "plant disease recognition, domain shift, leave-one-source-out, hand-crafted features, "
                "bag of visual words, explainability, near-duplicate leakage",
}


def load_refs():
    """References [n] from docs/literature_review.md (entries may wrap over several lines)."""
    txt = (REPO / "docs" / "literature_review.md").read_text(encoding="utf-8").split("## References", 1)[1]
    refs = {}
    for m in re.finditer(r"^\[(\d+)\]\s+(.*?)(?=^\[\d+\]|\Z)", txt, flags=re.M | re.S):
        refs[int(m.group(1))] = " ".join(m.group(2).split())
    return [refs[i] for i in range(1, max(refs) + 1)]


INTRO = [
    ("h1", "Introduction"),
    ("p", "Image-based recognition of tomato leaf diseases is usually benchmarked on laboratory photographs, most "
          "often PlantVillage [1], where single leaves lie on a plain background. Accuracies close to 100% are "
          "routinely reported [2], [3], yet the same models lose most of their accuracy on photographs taken under "
          "other conditions [2], [3], [5]. Lightweight classical pipelines - segmentation, hand-crafted colour, "
          "texture and shape features, and a random forest (RF), support vector machine (SVM) or logistic "
          "regression (LR) - remain attractive for low-resource deployment because they are cheap and their inputs "
          "are named, measurable quantities [13], [18]-[21]. That second property suggests an explanation that "
          "does not depend on a particular model: if a feature truly reflects disease, its relation to the label "
          "should hold whatever camera, field or website the image came from."),
    ("p", "We started from 813 field photographs of six tomato classes from Pakistan and found two problems that "
          "together make single-dataset results uninformative. First, near-duplicate and rotated copies of the same "
          "leaf inflate image-level splits: on the supplied 7,200-image augmented set, an RF scores 0.96 accuracy "
          "with a random split but 0.66 when copies of one photograph are kept together. Second, models trained on "
          "one source fail on others (macro-F1 0.21-0.33 on PlantDoc and a Bangladeshi field set), and their "
          "feature-importance explanations collapse with them. We therefore pool eight public and private image "
          "sources and ask three questions:"),
    ("bullets", ["<b>RQ1</b> How well do classical models trained on several sources recognise disease on a source "
                 "they have never seen (leave-one-source-out, LOSO)?",
                 "<b>RQ2</b> How large is the gap between seen-source and unseen-source performance, and does it "
                 "depend on how the features were obtained: by formula, by a classical learned extractor (k-means "
                 "visual words), or by a frozen deep network?",
                 "<b>RQ3</b> Which individual features keep the same, statistically reliable relation to disease "
                 "across sources, and does restricting a model to those features improve transfer?"]),
    ("p", "All classifiers are classical (RF, SVM, LR). Deep networks appear only as frozen feature extractors, so the "
          "comparison isolates the representation. Our contributions are (i) a de-duplicated eight-source tomato "
          "benchmark with a grouped LOSO protocol, (ii) a matched comparison of formula, k-means and frozen deep "
          "features under identical classical training, and (iii) a model-free, source-stability criterion for "
          "feature-level explanation, checked against glass-box and permutation-importance explanations."),
]

RELATED = [
    ("h1", "Related Work"),
    ("h2", "Laboratory versus field data"),
    ("p", "Mohanty et al. [2] reached 99.35% on held-out PlantVillage images but about 31% on images from other "
          "sources; Ferentinos [3] and Barbedo [4] report the same pattern and attribute it to covariate shift and "
          "background. Noyan [5] classified PlantVillage far above chance from eight background pixels, a direct "
          "sign of capture bias. Field collections followed: PlantDoc [6], Tomato-Village [7], PlantWild [8] and "
          "recent regional sets from Pakistan [9] and Bangladesh [10]-[12]. Each is normally evaluated within "
          "itself."),
    ("h2", "Classical features"),
    ("p", "Classical pipelines describe texture with co-occurrence statistics [14] and local binary patterns [15], "
          "colour with moments and histograms, and classify with SVMs [16] or random forests [17]; tomato-specific "
          "examples include Gabor features [18], SIFT-based texture [19] and comparisons with CNNs [20], [21]. "
          "Bag-of-visual-words models [52], [53] learn colour and texture prototypes with k-means [54], [55] instead "
          "of fixing them by formula."),
    ("h2", "Distribution shift, shortcuts and leakage"),
    ("p", "Datasets carry their own signatures [23], and leave-domain-out evaluation shows that many domain-"
          "generalisation methods do not beat plain training [24], [25]. Shortcut learning [29] and source-specific "
          "signals in medical imaging [30], [31] show that external accuracy alone does not prove a model uses the "
          "pathology. Duplicates across training and test sets are a common leakage route [32], [33], and grouped "
          "splitting is the remedy for structured data [34]."),
    ("h2", "Explaining models and features"),
    ("p", "Impurity importance is biased [36], permutation importance extrapolates under correlated features [37], "
          "and importance changes across equally good models [38]; local explainers disagree [39]-[41]. Rudin [42] "
          "argues for inherently interpretable models. Concept-based methods express a model through named "
          "concepts [44], [45], and CLIP [46] makes concept scores available without concept labels [47], [48]. "
          "Toda and Okura [43] visualised a PlantVillage CNN but on one laboratory dataset. We instead define "
          "explanation at the level of features and require it to replicate across sources."),
]

DATA_METHODS = [
    ("h1", "Data"),
    ("p", "Table I lists the eight sources. Our own 813 photographs were compared with PlantDoc by content; 296 "
          "(36%) are identical or near-identical to PlantDoc images, so the raw set is split into <i>pd_raw</i> (with a PlantDoc "
          "copy) and <i>pk_raw</i> (without). Tomato-Village is the field variant (a); PlantVillage is a seeded "
          "sample of 150 images per class. Brahmanbaria images are leaf crops from its detection boxes. Classes "
          "outside the six (bacterial spot, mosaic, spider mites, leaf miner, spotted wilt, other Bangladeshi "
          "disease folders) and uncertain mappings enter the healthy-versus-diseased task only."),
    ("p", "<b>Cross-source de-duplication.</b> Every image is reduced to a 16x16 grey thumbnail; two images are "
          "duplicates if the Pearson correlation of the thumbnails is at least 0.95 under any of the eight "
          "rotations and flips, or if their bytes are identical. The threshold was fixed on the raw set before this "
          "study. Of 8,196 scanned images, 425 cross-source duplicates were dropped (277 PlantDoc, 146 PlantWild, 2 "
          "PlantVillage), keeping the copy in the higher-priority source; 72 pairs carry different labels and are "
          "recorded for review. Within a source, duplicates form groups that are never split; 7,754 images in 6,967 "
          "groups remain (largest group 12)."),
    ("h1", "Methods"),
    ("h2", "Features"),
    ("p", "All features except the deep ones are computed on a 256x256 copy of each image inside an HSV hue-gate "
          "leaf mask. <b>Formula arm.</b> <i>Hand</i>: 105 features (47 colour moments and histograms, 38 LBP and GLCM "
          "texture, 20 shape). <i>Lesion</i>: 20 leaf-relative features designed to be insensitive to camera colour: "
          "yellow, brown, dark, pale and non-green fractions of the leaf; lesion-minus-healthy-tissue differences in "
          "CIELAB; chromaticity statistics; spot density, size and circularity relative to the leaf. "
          "<b>K-means arm.</b> From 1,500 sampled leaf pixels per image, a CIELAB colour descriptor and an 8-d "
          "filter-bank texture descriptor (Sobel and Laplacian-of-Gaussian magnitudes at two scales, local standard "
          "deviation) are quantised with MiniBatch k-means into 24 colour words and 32 textons; an image is the "
          "share of its pixels nearest each word. In LOSO the codebooks are re-learned in every fold from the "
          "training sources only. <b>Deep arm (frozen).</b> CLIP ViT-B/32 [46], [49] image embeddings (512-d); "
          "24 named concept scores, the cosine similarity of the embedding with fixed text prompts (14 symptoms, "
          "10 context or shortcut cues such as plain background, hand, blur and watermark), fixed before any "
          "score was seen; and ImageNet ResNet-50 [50] pooled features (2,048-d, PCA to 256 inside each fold). "
          "No network is trained or fine-tuned."),
    ("h2", "Classifiers and protocols"),
    ("p", "RF (300 trees), RBF SVM and LR, all with balanced class weights and standardised inputs, with small "
          "fixed grids. Hyper-parameters are chosen on a validation set of 15% of every training source (grouped) "
          "and the model is refit on training plus validation. <b>LOSO</b>: train on seven sources, test on the "
          "eighth; nothing is tuned on it. <b>Seen-source reference</b>: the pooled data are split 70/15/15 within "
          "each source (grouped), five seeds, and scored per source. The <b>transfer gap</b> is seen minus LOSO. "
          "Task T1 is healthy versus diseased (balanced accuracy, AUC, healthy recall); task T2 is the six classes, "
          "scored with macro-F1 over the classes present in the held-out source. Every source counts once. "
          "Feature sets are compared with paired Wilcoxon signed-rank tests over held-out sources with Holm "
          "correction."),
    ("h2", "Feature-level explanation"),
    ("p", "For each feature and each source with at least 30 healthy and 30 diseased images, the effect is "
          "AUC - 0.5 of the feature alone for diseased versus healthy, with a 1,000-resample stratified bootstrap "
          "95% interval. A feature is <b>source-stable</b> if its interval excludes zero with the majority sign in "
          "at least 75% of eligible sources. This criterion uses no classifier. We relate it to three model-based "
          "views: the sign stability of L1-regularised LR coefficients across LOSO folds (glass box), RF "
          "permutation importance measured on the held-out source, and <b>explanation-guided selection</b>, in "
          "which source-stable features are identified from the training sources only and the models are refit "
          "on them before scoring the held-out source."),
]


def source_table():
    import pandas as pd
    c = pd.read_csv(RUN / "sources" / "counts.csv").set_index("source")
    desc = {"pk_raw": ("Own field photos, Pakistan", "6"), "pd_raw": ("Own photos also in PlantDoc", "6"),
            "plantdoc": ("Web images [6]", "6 + 3"), "plantwild": ("Web, in the wild [8]", "6 + 2"),
            "tomato_village": ("Field, Rajasthan [7]", "3 + 2"), "nazirpur": ("Field, Bangladesh [10]", "2 + bin"),
            "brahmanbaria": ("Field crops, Bangladesh [11]", "bin"), "plantvillage": ("Laboratory [1]", "6 + 4")}
    NAME = {"pk_raw": "PK raw", "pd_raw": "PD raw", "plantdoc": "PlantDoc", "plantwild": "PlantWild",
            "tomato_village": "T.-Village", "nazirpur": "Nazirpur", "brahmanbaria": "Brahmanb.",
            "plantvillage": "PlantVillage"}
    rows = [[NAME[s], desc[s][0], int(c.loc[s, "All"]), int(c.loc[s, "healthy"]), desc[s][1]]
            for s in desc]
    rows.append(["Total", "", int(c.loc["All", "All"]), int(c.loc["All", "healthy"]), ""])
    return ("table", "tab:sources", "Image sources after cross-source de-duplication. Classes: six-class labels "
            "present + extra disease classes used only in the binary task.",
            ["Source", "Origin", "Images", "Healthy", "Classes"], rows, [0.21, 0.37, 0.14, 0.14, 0.14], False)


FIG = RUN / "results" / "figures"
SETNAME = {"hand": "Hand (105)", "lesion": "Lesion (20)", "hand+lesion": "Hand + lesion", "colour_words":
           "Colour words (24)", "textons": "Textons (32)", "kmeans": "Colour words + textons",
           "hand+lesion+kmeans": "Hand + lesion + k-means", "concepts": "CLIP concepts (24)",
           "clip": "CLIP embedding (512)", "resnet": "ResNet-50 (PCA 256)"}
ARMNAME = {"formula": "Formula", "kmeans": "K-means", "formula+kmeans": "Formula + k-means", "deep": "Frozen deep",
           "concepts": "CLIP concepts"}
ORDER = ["hand", "lesion", "hand+lesion", "colour_words", "textons", "kmeans", "hand+lesion+kmeans", "concepts",
         "clip", "resnet"]


def _load():
    import pandas as pd
    s3, ex = RUN / "results" / "phase3_summary", RUN / "results" / "explain"
    return {"ov": pd.read_csv(s3 / "overview.csv"), "per": pd.read_csv(s3 / "per_source.csv"),
            "pv": pd.read_csv(s3 / "paired_vs_hand.csv"), "st": pd.read_csv(ex / "stability.csv"),
            "sel": pd.read_csv(ex / "selection_metrics.csv"), "cs": pd.read_csv(ex / "glassbox_coef_stability.csv"),
            "imp": pd.read_csv(ex / "rf_heldout_importance.csv")}


def _ov(d, task, set_, model):
    r = d["ov"][(d["ov"].task == task) & (d["ov"].set == set_) & (d["ov"].model == model)]
    return r.iloc[0] if len(r) else None


def _best(d, task, set_):
    o = d["ov"][(d["ov"].task == task) & (d["ov"].set == set_)]
    return o.loc[o.loso_mean.idxmax()] if len(o) else None


def _pv(d, task, set_):
    p = d["pv"]
    r = p[(p.task == task) & (p.model == "lr") & (p.set == set_)]
    return r.iloc[0] if len(r) else None


def results_blocks():
    d = _load()
    f = lambda x: f"{x:.2f}"
    rows = []
    for s in ORDER:
        b, m = _best(d, "bin", s), _best(d, "six", s)
        if b is None:
            continue
        lb, lm = _ov(d, "bin", s, "lr"), _ov(d, "six", s, "lr")
        rows.append([SETNAME[s], f(lb.loso_mean), f(lb.seen_mean), f(lb.gap_mean),
                     f(lm.loso_mean) if lm is not None else "-", f(lm.seen_mean) if lm is not None else "-",
                     f(lm.gap_mean) if lm is not None else "-"])
    t_main = ("table", "tab:loso", "Logistic regression, unseen source (U, LOSO) vs seen sources (S), mean over "
              "held-out sources; gap = S - U. T1: balanced accuracy, healthy vs diseased (8 sources; chance 0.5). "
              "T2: macro-F1 over the classes present (7 sources). Rows 1-3 formula, 4-6 k-means, 7 combined, 8-10 "
              "frozen deep. Fig. 1 shows the best of RF, SVM and LR per set.",
              ["Features", "T1 U", "T1 S", "T1 gap", "T2 U", "T2 S", "T2 gap"],
              rows, [0.34, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11], False)

    hl, cw, co, cl, rn = (_ov(d, "bin", s, "lr") for s in ["hand+lesion", "colour_words", "concepts", "clip", "resnet"])
    hl6, co6, cl6, rn6 = (_ov(d, "six", s, "lr") for s in ["hand+lesion", "concepts", "clip", "resnet"])
    best6 = d["ov"][d["ov"].task == "six"].sort_values("loso_mean").iloc[-1]
    pc, phl6 = _pv(d, "bin", "concepts"), _pv(d, "six", "hand+lesion")
    rfb = _ov(d, "bin", "hand+lesion", "rf")
    per = d["per"]
    naz = per[(per.task == "bin") & (per.set == "hand+lesion") & (per.model == "lr") & (per.source == "nazirpur")]
    st = d["st"]
    nst = st[st.stable].groupby("pool").size().to_dict()
    ntot = st.groupby("pool").size().to_dict()
    les = st[st.feature.str.startswith("lesion_")]
    hand = st[(st.pool == "formula") & ~st.feature.str.startswith("lesion_")]
    sel = d["sel"].groupby(["pool", "features", "model"])["balanced_accuracy"].mean()
    cs = d["cs"]
    ctx = cs[(cs.pool == "concepts") & cs.feature.str.startswith("context_") & cs.sign_stable]
    imp = d["imp"]
    rho = imp.groupby("heldout")[["importance", "stability"]].apply(
        lambda g: g.corr(method="spearman").iloc[0, 1])

    stab_rows = [["Hand (formula)", f"{int(hand.stable.sum())} / {len(hand)}",
                  "yellow-orange hue share (+), green-red skew (-), 3 LBP bins, GLCM correlation"],
                 ["Lesion (formula)", f"{int(les.stable.sum())} / {len(les)}",
                  "yellow and non-green fraction, lesion b* shift, largest spot (all +, 7/7 sources)"],
                 ["K-means words", f"{nst.get('kmeans', 0)} / {ntot.get('kmeans', 0)}",
                  "7 brown/olive/ochre colour words (+), 1 texton (-)"],
                 ["CLIP concepts", f"{nst.get('concepts', 0)} / {ntot.get('concepts', 0)}", "miner trails (-)"]]
    t_stab = ("table", "tab:stable", "Source-stable features (CI excludes 0 with the majority sign in at least 6 of "
              "7 eligible sources), healthy vs diseased.", ["Family", "Stable", "Examples"], stab_rows,
              [0.25, 0.15, 0.6], False)
    sel_rows = [[ARMNAME.get(p, p), f(sel[(p, "all", "l1lr")]), f(sel[(p, "selected", "l1lr")]),
                 f(sel[(p, "all", "rf")]), f(sel[(p, "selected", "rf")])]
                for p in ["formula", "kmeans", "formula+kmeans", "concepts"]]
    t_sel = ("table", "tab:sel", "Explanation-guided selection: LOSO balanced accuracy with all features vs only the "
             "features found source-stable on the training sources.",
             ["Pool", "L1-LR all", "L1-LR stable", "RF all", "RF stable"], sel_rows, [0.3, 0.18, 0.18, 0.17, 0.17],
             False)

    six_nd = d["ov"][(d["ov"].task == "six") & (d["ov"].arm != "deep")].sort_values("loso_mean").iloc[-1]
    pcl6 = _pv(d, "six", "clip")
    deep_p = d["pv"][(d["pv"].task == "six") & (d["pv"].model == "lr") & (d["pv"].arm == "deep")]
    return [
        ("h1", "Results"),
        ("h2", "Single-source baseline"),
        ("p", "On our own photographs, grouping near-duplicates lowers RF accuracy on the supplied augmented set from "
              "0.96 (random image-level split; 98.7% of test images have a relative in training) to 0.66 +/- 0.06 "
              "(grouped split, 10 seeds). Trained on our raw set alone, the models reach macro-F1 0.29-0.33 on "
              "independent PlantDoc images and 0.21-0.25 on Nazirpur, and recognise only 3-17% of healthy "
              "Brahmanbaria leaves. This motivates the multi-source design."),
        ("h2", "Unseen-source performance by extractor arm (RQ1, RQ2)"),
        t_main,
        ("fig", "fig:arms", str(FIG / "p3_arms_loso_vs_seen.png"),
         "Best classifier per feature set: mean over held-out sources (bars) and the same model on seen sources "
         "(ticks). Colour = extractor arm.", False),
        ("p", f"Training on seven sources and testing on the eighth, the formula arm reaches a balanced accuracy of "
              f"{f(hl.loso_mean)} (LR, hand + lesion) for healthy versus diseased, against {f(hl.seen_mean)} on "
              f"seen sources. The k-means colour words are lower ({f(cw.loso_mean)}) but the most even across "
              f"sources (SD {f(cw.loso_sd)}) and lose the least ({f(cw.gap_mean)}). Among the frozen deep "
              f"extractors, the 24 CLIP concept scores give the best unseen-source result ({f(co.loso_mean)}, "
              f"seen {f(co.seen_mean)}), higher than hand features on {int(pc.wins)} of {int(pc.n_sources)} "
              f"held-out sources (Wilcoxon p = {pc.p_wilcoxon:.3f}, Holm-adjusted {pc.p_holm:.2f}); the CLIP "
              f"embedding reaches {f(cl.loso_mean)} and ResNet-50 {f(rn.loso_mean)}, close to the formula "
              f"features. Linear models transfer best: with hand + lesion features the RF scores "
              f"{f(rfb.loso_mean)} unseen and {f(rfb.seen_mean)} seen, and it rarely predicts healthy on a new "
              f"source (healthy recall {f(rfb.healthy_recall_mean)} vs {f(hl.healthy_recall_mean)} for LR)."),
        ("p", f"Six-class diagnosis is where the arms separate. Formula and k-means features reach at most "
              f"{f(six_nd.loso_mean)} macro-F1 on an unseen source ({SETNAME[six_nd.set]}, "
              f"{six_nd.model.upper()}; seen {f(six_nd.seen_mean)}); hand + lesion features with LR beat hand "
              f"features on {int(phl6.wins)} of {int(phl6.n_sources)} sources (Holm-adjusted p = "
              f"{phl6.p_holm:.2f}). The frozen deep embeddings transfer clearly better: CLIP embedding "
              f"{f(cl6.loso_mean)} (seen {f(cl6.seen_mean)}), CLIP concepts {f(co6.loso_mean)} and ResNet-50 "
              f"{f(rn6.loso_mean)} with LR, each above hand features on {int(deep_p.wins.min())}-"
              f"{int(deep_p.wins.max())} of {int(pcl6.n_sources)} sources (Wilcoxon p = "
              f"{deep_p.p_wilcoxon.min():.3f}-{deep_p.p_wilcoxon.max():.3f}; not significant after Holm "
              f"correction with only {int(pcl6.n_sources)} sources). The 24 CLIP concepts lose the least between "
              f"seen and unseen sources ({f(co6.gap_mean)})."),
        ("fig", "fig:heat", str(FIG / "p3_heatmap_bin.png"),
         "Healthy vs diseased, per held-out source, best set of each arm.", False),
        ("fig", "fig:heat6", str(FIG / "p3_heatmap_six.png"),
         "Six classes (macro-F1 over the classes present), per held-out source, best set of each arm.", False),
        ("p", f"Performance depends strongly on the held-out source (Fig. 2). Nazirpur is the hardest for the "
              f"formula features ({f(naz.primary.iloc[0])}); a domain classifier separates it from our photographs "
              f"with AUC 0.98."),
        ("h2", "Feature-level explanation (RQ3)"),
        ("p", f"Of {len(st)} named features, {int(st.stable.sum())} are source-stable (Table III, Fig. 4). "
              f"Leaf-relative lesion features are the most reliable family ({int(les.stable.sum())} of {len(les)}): "
              f"the yellow and non-green fractions of the leaf, the lesion-minus-tissue shift in b* and the size of "
              f"the largest spot are higher in diseased leaves on all seven eligible sources. Of the 105 hand "
              f"features only {int(hand.stable.sum())} qualify, chiefly the yellow-orange hue share. The k-means "
              f"codebook, learned without labels, isolates the same signal in a form that can be shown rather than "
              f"defined: its stable words are browns, olives, ochres and near-black (Fig. 5), while the green "
              f"words have negative effects that are not stable. Only one CLIP concept is stable."),
        t_stab,
        ("fig", "fig:stable", str(FIG / "p3_stable_features.png"),
         "Per-source effect (AUC - 0.5) of the most source-stable features; hollow = 95% CI includes 0.", False),
        ("fig", "fig:words", str(FIG / "p3_colour_words.png"),
         "The 24 k-means colour words as swatches, ordered by signed median effect; below: effect and share of "
         "sources where it is significant with the majority sign.", False),
        ("p", f"Model-based explanations do not recover this set. The Spearman correlation between RF permutation "
              f"importance on the held-out source and feature stability ranges from {rho.min():.2f} to "
              f"{rho.max():.2f} across folds. The L1-LR on CLIP concepts, the strongest unseen-source model, keeps "
              f"sign-stable weights on {len(ctx)} context concepts in every LOSO fold - plain background (towards "
              f"healthy), in hand and studio (towards diseased) - alongside symptom concepts such as necrotic edges "
              f"and target spots. Part of its advantage comes from capture context that happens to be shared "
              f"between sources, not from symptoms."),
        t_sel,
        ("p", f"Restricting models to stable features does not improve transfer (Table IV): formula L1-LR "
              f"{f(sel[('formula', 'all', 'l1lr')])} to {f(sel[('formula', 'selected', 'l1lr')])}, k-means "
              f"{f(sel[('kmeans', 'all', 'l1lr')])} to {f(sel[('kmeans', 'selected', 'l1lr')])}, and CLIP concepts "
              f"lose {f(sel[('concepts', 'all', 'l1lr')] - sel[('concepts', 'selected', 'l1lr')])} because the "
              f"context concepts carry transferable signal. Stability identifies which features mean the same thing "
              f"everywhere; the other features still add predictive, if less interpretable, information."),
        ("h1", "Discussion"),
        ("p", "Three points stand out. First, pooling sources is what makes classical models transfer at all: "
              "unseen-source balanced accuracy of about 0.66 with formula features, against near-collapse when "
              "training on one source. Second, the two learned extractors behave differently. The k-means words "
              "transfer about as well as the formulas and add no accuracy, but they make the colour signal visible. "
              "Frozen deep features help little for healthy versus diseased (except the concept scores, which partly "
              "use context) but substantially for six-class diagnosis, where hand-crafted colour and texture "
              "statistics cannot separate diseases across cameras; the gap between seen and unseen sources remains "
              "for every arm. Third, feature-level explanation is possible and informative when it is required to "
              "replicate across sources: a short list of leaf-relative yellowing, browning and lesion-size "
              "measurements behaves the same way on laboratory, web and field images from three countries, which a "
              "model's importance ranking does not reveal. Accuracy and interpretability therefore pull apart: the "
              "most transferable diagnostic representation (a 512-d embedding) is the least interpretable."),
        ("p", "<b>Limitations.</b> Labels come from different annotators and protocols; 72 cross-source label "
              "conflicts were recorded but not resolved. 'Diseased' mixes different diseases per source. Sources "
              "differ in size and class mix, and several contribute only some classes. The HSV leaf mask is "
              "heuristic and is not used by the deep extractors, which see the whole image; this favours context "
              "cues. Only frozen, generic deep extractors were tested, and the explanation analysis covers the "
              "healthy-versus-diseased task only."),
        ("h1", "Conclusion"),
        ("p", "On a de-duplicated eight-source tomato benchmark with leave-one-source-out evaluation, classical models "
              "recognise disease on an unseen source with balanced accuracy of about 0.65-0.75. Six-class diagnosis "
              "transfers poorly with formula or k-means features (macro-F1 about 0.33) and moderately with frozen "
              "CLIP embeddings (about 0.52). A model-free stability criterion identifies a small set of named features - "
              "leaf-relative yellow, brown and non-green fractions, lesion size and k-means brown/olive colour "
              "words - whose relation to disease holds across sources. We recommend reporting unseen-source results "
              "and source-stable features alongside within-dataset accuracy."),
    ]


def build():
    from render import to_latex, to_pdf
    blocks = INTRO + RELATED + DATA_METHODS[:3] + [source_table()] + DATA_METHODS[3:] + results_blocks()
    refs = load_refs()
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    meta = dict(META, abstract=abstract())
    to_latex(meta, blocks, refs, out / "paper.tex")
    to_pdf(meta, blocks, refs, out / "paper.pdf")
    print("wrote", out / "paper.tex", out / "paper.pdf")


def abstract():
    d = _load()
    f = lambda x: f"{x:.2f}"
    o = lambda t, s: _ov(d, t, s, "lr")
    st = d["st"]
    six_nd = d["ov"][(d["ov"].task == "six") & (d["ov"].arm != "deep")].loso_mean.max()
    return (f"Classical tomato-disease classifiers are usually evaluated within one dataset. We pool eight "
            f"de-duplicated image sources (7,754 images: field photographs from Pakistan, India and Bangladesh, web "
            f"images and laboratory images) and evaluate random forest, SVM and logistic regression by "
            f"leave-one-source-out. We compare three feature arms under identical classical training: hand-crafted "
            f"and leaf-relative lesion formulas, k-means colour and texture words, and frozen deep extractors (CLIP, "
            f"ResNet-50) used without any training. Related near-duplicate images inflate a single-source benchmark "
            f"from 0.66 to 0.96 accuracy, and single-source models collapse on other sources. With pooled training, "
            f"healthy-versus-diseased balanced accuracy on an unseen source is {f(o('bin', 'hand+lesion').loso_mean)} "
            f"with formula features and {f(o('bin', 'concepts').loso_mean)} with CLIP concept scores, but six-class "
            f"macro-F1 is only {f(six_nd)} with formula or k-means features and {f(o('six', 'clip').loso_mean)} with "
            f"CLIP embeddings. A model-free criterion - a feature's effect must have the same sign and a bootstrap "
            f"interval excluding zero on at least 6 of 7 sources - finds {int(st.stable.sum())} of {len(st)} named "
            f"features stable, mainly leaf-relative yellow, non-green and lesion-size measurements and k-means "
            f"brown/olive colour words. Random-forest importance does not recover them, and a glass-box concept "
            f"model relies partly on capture context. Feature-level stability across sources is a practical basis "
            f"for explaining classical plant-disease models.")

if __name__ == "__main__":
    build()
