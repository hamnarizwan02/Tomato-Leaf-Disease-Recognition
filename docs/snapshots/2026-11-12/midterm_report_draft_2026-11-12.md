# Mid-semester report (draft)

**Robust and Explainable Classical Machine Learning for Tomato Disease Recognition under Real-World Field Conditions**
CS-871 Machine Learning, MS(AI), NUST SEECS, Fall 2026. Hamna Sadia Rizwan (583282), Maria Ilyas Malik (624045),
Manahil Kamran (582413). Due 13 November 2026, 23:59 PST.

> **Drafting notes (remove before submission).**
> - Graded on: literature survey, methodology tied to course concepts, preliminary results, challenges handled
>   honestly, and revisions to the plan with reasoning.
> - Every number comes from [verified_results.md](verified_results.md); do not copy numbers from the PDF findings log.
> - **Check the course brief for the page limit and template before formatting.**
> - Items marked TODO need team input.

---

## 1. Introduction and research questions

Tomato leaf diseases are usually recognised from photos. Most published benchmarks use laboratory images, which
overstate how well models work in real fields with uncontrolled light and backgrounds. We study lightweight classical
models: hand-crafted colour, texture and shape features with Random Forest (RF), RBF Support Vector Machine (SVM) and
Logistic Regression (LR). We use real field photos from Pakistan and ask:

1. How well can classical models classify six tomato leaf classes from hand-crafted features?
2. How robust are they to realistic changes in brightness, contrast, shadows, image quality and other conditions?
3. Do image-source or metadata shortcuts influence their predictions?
4. Which feature groups carry useful information, and how stable and interpretable are the explanations?

## 2. Literature survey

**TODO (team).** Our synopsis cited four works: PlantVillage, PlantDoc, the Pakistan tomato field dataset (Malik, Ali
and Ali, Mendeley Data v2, DOI 10.17632/3mbnb82mxd.2) and PlantCity. The survey should cover:
- (a) laboratory vs field plant-disease datasets and the lab-to-field gap;
- (b) hand-crafted-feature and classical-ML plant-disease recognition;
- (c) robustness and distribution shift in plant-disease recognition, including augmentation and colour constancy;
- (d) shortcut learning and dataset bias;
- (e) stability of feature-importance explanations.

Only cite sources the team has read and checked.

## 3. Dataset and what the audit found

We use the `Dataset (raw)` folder: 830 original photos in six classes (early blight, late blight, septoria leaf spot,
leaf mold, yellow leaf curl virus, healthy). The 7,200 images in the dataset description are a separately supplied
augmented set with its own train/val/test split (5,040 / 1,440 / 720). Our audit showed that this split places related
images on both sides: 361 of the 720 test images have a detected sibling (same source filename or a near-identical
rotated/flipped copy) in train or validation. We therefore do not use it for our main experiments. We benchmarked it
separately on its own split (Section 5) only as a reference.

Data preparation (Section 4.1) removed 17 images that are the same photo filed under two diseases, leaving 813 images
in 769 independent source groups. Classes are imbalanced by 2.4× (80 to 191 images per class).

**Provenance.** The dataset is described as smartphone photos from Pakistani fields, but 263 of the 830 raw images
(31.7%) are identical to images in the public PlantDoc dataset (257 byte-identical). For 25 of them the two datasets
disagree on the label, most often (19 images) PlantDoc "early blight" filed as septoria here. That matches the
early-blight file names we had already found in our septoria folder. We have not changed any label: which dataset is
right is not established.

## 4. Methodology, with the course concepts each step uses

### 4.1 Leakage-safe grouped splits (generalisation, data leakage)
- Near-duplicate photos (resized, recompressed, flipped or re-lit copies) are found with a flip-aware perceptual hash,
  confirmed by thumbnail correlation of at least 0.98, and merged into one group.
- Splits are made per class by whole groups (70/15/15), so a photo and its copies never sit in both training and test.
- The test set is 118–119 images, so we evaluate on **10 repeated grouped splits** (seeds 42–51) and report mean ± sd.
  A single split is not enough: the committed seed-42 split scores RF 0.665, the 10-split mean is 0.600.

### 4.2 Features (feature engineering, correlated features, curse of dimensionality)
- Each image is resized to 256 × 256 and the leaf is segmented with an HSV hue gate plus morphology.
- 105 features are computed inside the leaf mask:
  - colour (47): RGB/HSV/LAB moments, hue and saturation histograms;
  - texture (38): uniform LBP at two scales, GLCM statistics;
  - shape/edge/lesion (20).
- That is 105 features for about 570 training images, and many colour statistics are strongly correlated. This matters
  for both overfitting and explanation stability.

### 4.3 Models and tuning (bias–variance, regularisation, kernels, bagging)
- RF (bagged trees), RBF SVM (margin maximisation with a kernel) and L2-regularised multinomial LR, all with balanced
  class weights.
- Hyper-parameters are grid-searched on the validation split of each repeated split (fit on train), then the model is
  refit on train + validation and scored once on test. The test set never influences any choice.
- RF fits its training data perfectly (training macro-F1 1.00) while validation plateaus near 0.55–0.58. That is a
  high-variance regime.

### 4.4 Evaluation (metrics under imbalance, calibration, multiple testing)
- Headline metric: macro-F1, plus accuracy, balanced accuracy, MCC, per-class precision/recall/F1, expected
  calibration error (ECE) and Brier score.
- Configurations are compared on the same splits with paired Wilcoxon signed-rank tests and Holm correction within
  each family of comparisons.

### 4.5 Robustness protocol
- 50 controlled perturbations are applied to **test images only**: brightness, contrast, gamma, colour cast, shadow,
  background replacement, Gaussian/motion blur, JPEG, noise, down-then-up scaling, occlusion, rotation, scale, and
  three combinations.
- Features (and the leaf mask) are recomputed on each perturbed image.
- Robustness is reported as perturbed macro-F1 divided by clean macro-F1.

### 4.6 Source-shortcut controls
- A metadata-only model (image width, height, aspect, file size) is included as a reference row.
- Features correlated with image size are pruned.
- Images are re-processed to equal effective resolution or identical JPEG compression.
- Images are clustered by source characteristics, followed by leave-one-cluster-out evaluation.

### 4.7 Phase 2, experiment A: train-only augmentation (regularisation through data)
- Four augmented copies of each **training** image are generated:
  - A1: random flips and rotation within ±15°;
  - A2: A1 plus brightness and contrast scaling within ×[0.8, 1.2].
- Settings were fixed in advance. Features are extracted from each copy with the same 105-feature code.
- Copies inherit their source photo's group, are added only when that photo is in the training partition, and never
  reach validation or test. This is checked automatically for every split.
- A zero-copy control (A0) reproduces the baseline exactly.

### 4.8 Phase 2, experiment B: illumination normalisation (preprocessing, invariance)
- Every image (training, validation and test) is normalised independently before segmentation and features, after
  any test perturbation, as a deployed pipeline would do. No parameters are fitted, and there is no augmentation.
- Two methods:
  - Gray World colour constancy: equalise the three channel means;
  - CLAHE: contrast-limited adaptive histogram equalisation on the LAB lightness channel only, standard settings
    (clip limit 2.0, 8×8 tiles).
- Each method is a separate run with the same splits, grids and protocol as the baseline.

### 4.9 Grouped re-evaluation of the supplied augmented set (data leakage, evaluation design)
- The supplied 7,200-image set is re-split by source group instead of using its own folders. A group joins images
  linked by any of three rules fixed before any model was run:
  - the pipeline rule of Section 4.1;
  - the same source file name after removing augmentation suffixes;
  - near-identical 16×16 grey thumbnails (correlation ≥ 0.95) under any of the 8 rotations/flips.
- Images whose group also appears under another class are removed (183 images, the same policy as for the raw set).
  7,017 images remain in 936 groups. The number of groups per class (86–216) is close to the number of raw photos
  per class (80–195), which suggests the groups correspond to source photos.
- The main protocol is applied unchanged: 10 grouped 70/15/15 splits, tuning inside each split.
- **Control:** the same images, features and seeds, split at image level instead of by group. Only the split rule
  differs between the two runs.

### 4.10 Cross-dataset evaluation (generalisation, distribution shift)
- Models are trained on our raw set exactly as in Section 4.3, one per repeated split. Each fitted model is then
  applied unchanged to external tomato datasets: no tuning, refitting or threshold changes on external data.
- External sets:
  - PlantDoc (web-sourced field images, all six classes);
  - two Bangladeshi field datasets: Nazirpur (septoria and healthy; "leaf curl" is scored separately as an
    uncertain match to yellow leaf curl virus) and Brahmanbaria (healthy vs diseased only).
  - A third Bangladeshi set (Dinajpur) was excluded: its labels mark individual lesions, not leaves, and its class
    numbering could not be matched to disease names.
- **Overlap check first.** External images that duplicate any of our raw images are found with the same thumbnail
  rule (and identical file bytes). They are scored separately, never as "new" data.
- **Explanation transfer:** colour/texture/shape group importance and per-feature permutation importance are computed
  with the same fitted models on our test split and on the external images, and their rankings compared.

## 5. Preliminary results

All results use 10 grouped splits; full tables are in [verified_results.md](verified_results.md).

**RQ1, clean performance.** Macro-F1 with all features: RF 0.600 ± 0.046, SVM 0.615 ± 0.026, LR 0.596 ± 0.044.
- The three models are statistically indistinguishable.
- Colour + texture alone performs as well as all features (SVM 0.630 ± 0.034), so shape adds nothing.
- Each feature group alone is significantly worse than all features combined.
- Per class, leaf mold and healthy are recognised best and early blight worst. The most common confusion is early
  blight predicted as septoria.
- *Reference only, supplied augmented split (single split, not comparable with the numbers above):* test macro-F1
  RF 0.812, SVM 0.815, LR 0.696. Test images with a detected sibling in train/val are classified with about 93%
  accuracy by RF and SVM, those without one with about 69%. The supplied-split score should therefore not be read as
  accuracy on new leaves. The two groups also differ in class mix, so the gap is an association, not proof that
  overlap causes it.
- **Grouped re-evaluation of the supplied set (Section 4.9).** The same 7,017 images give very different scores
  depending only on how they are split (test macro-F1, mean ± sd over 10 splits):

  | Model | Image-level split | Supplied split | Grouped split |
  |---|---|---|---|
  | RF | 0.961 ± 0.006 | 0.814 | 0.659 ± 0.058 |
  | SVM | 0.972 ± 0.004 | 0.815 | 0.669 ± 0.054 |
  | LR | 0.768 ± 0.011 | 0.698 | 0.638 ± 0.050 |

  - In the image-level split, 98.7% of test images have a related image in training; in the supplied split, 50%; in
    the grouped split, none (under our detection rules).
  - The image-level score is higher than the grouped score on 10 of 10 seeds for every model (by 0.30 for RF and
    SVM). The supplied-split benchmark is therefore inflated by related images across the split.
  - The grouped accuracy (0.64–0.67) agrees with the accuracy on supplied-split test images without a detected
    sibling (0.66–0.69), an independent check.
  - Under grouping, the supplied set scores in the same range as our raw-set protocol (0.60–0.62), although the
    two are not directly comparable (different images, class balance, and augmented copies in training).
  - Undetected copies may still cross splits, so 0.66 may still be optimistic.

**RQ2, robustness.**
- Loss of resolution and sharpness hurts most: down-then-up scaling keeps only 0.42 of clean macro-F1 for RF, blur 0.43.
  Contrast (0.61) and colour cast (0.70) come next.
- Rotation (0.96), JPEG (0.95) and background replacement (0.91) are nearly harmless.
- *Preliminary (experiment A):* train-only augmentation does **not** change clean macro-F1 detectably (e.g. RF 0.600 →
  0.613 for A2, not significant).
- A2 significantly improves robustness to brightness (+0.05 to +0.11) and contrast (+0.09 to +0.12) for all three
  models. Those are the families it was trained on, although the test levels lie well outside the training range.
- For RF, A2 also improves blur, motion blur, down-then-up scaling, shadow and scale (+0.06 to +0.07), at a cost under
  colour cast (−0.07) and noise (−0.03).
- Comparing A2 with A1 shows that the brightness and contrast gains come from the photometric part. The source of the
  RF blur and rescaling gains is unresolved: they are not attributable to the photometric part, and A1 (geometric only)
  does not show them significantly either.
- *Preliminary (experiment B):* neither normalisation improves clean performance.
  - **Gray World** lowers clean macro-F1 by 0.05–0.06 with all features (baseline better on 8–9 of 10 splits; not
    significant after Holm correction). It gains exact invariance to channel-wise colour casts (+0.12 to +0.18), which follows from how
    Gray World works. It loses 0.07–0.15 under noise, gamma, background replacement and JPEG.
  - **Important confound:** because normalisation runs before the colour-based leaf mask, Gray World shrinks the masks
    from 83% to 39% of the image on average, and unevenly across classes. Its results therefore reflect colour
    normalisation *and* altered segmentation, not colour normalisation alone.
  - **CLAHE** leaves the masks unchanged and is roughly neutral. It does not improve brightness or contrast
    robustness, unlike photometric augmentation.

**RQ3, shortcuts.**
- Image metadata alone reaches macro-F1 0.487 ± 0.035, about three times chance. That is a real source signal in the
  dataset.
- All features beat it on every split.
- Equalising resolution or compression does not change the scores, so pixel-level artefacts do not explain them.
- But accuracy drops from about 0.61 to 0.47–0.50 when a whole source cluster is held out. One cluster is 87% leaf mold
  and is classified with 0.91–0.96 accuracy.
- Part of the score is therefore source-specific (preliminary: the clusters are proxies for source).
- **Cross-dataset test (Section 4.10).** Models trained on our raw set do not transfer to other tomato datasets
  (macro-F1 over the classes present, mean ± sd over 10 splits; our own test split for reference):

  | Test data | RF | SVM | LR |
  |---|---|---|---|
  | Our test split | 0.613 ± 0.046 | 0.615 ± 0.026 | 0.595 ± 0.044 |
  | PlantDoc, new images (310, six classes; chance 0.17) | 0.290 ± 0.018 | 0.330 ± 0.028 | 0.329 ± 0.012 |
  | Nazirpur, Bangladesh (378, septoria + healthy) | 0.245 ± 0.023 | 0.234 ± 0.031 | 0.213 ± 0.017 |

  - Healthy leaves from other sources are mostly called diseased: healthy recall 0.03–0.17 on Brahmanbaria
    (Bangladesh), 0.21–0.27 on Nazirpur.
  - PlantDoc images that also appear in our raw set score far higher than new PlantDoc images (RF 0.81 vs 0.29).
  - A classifier separating our images from external images on the same 105 features reaches ROC AUC 0.85
    (PlantDoc) and 0.98–0.99 (Bangladesh), while leaf-mask quality is similar. The models fail where the feature
    distribution shifts most, consistent with the source-specific signal found by leave-one-cluster-out.
  - The RF in-domain number here (0.613) was run on a second machine; it differs from the main run (0.600) because RF
    is sensitive to tiny floating-point differences in the features. SVM and LR reproduce almost exactly (largest
    difference 0.003).
  - Caveats: label definitions and photo styles differ between datasets, and one Bangladeshi mapping (leaf curl as
    yellow leaf curl virus) is uncertain and reported separately.

**RQ4, explainability (single split, preliminary).**
- Colour carries the most information when permuted as a group (0.164 drop in true-class probability), then texture
  (0.083) and shape (0.039).
- Feature rankings are stable across folds (rank correlation 0.90), but which features make the top 10 is not (Jaccard
  0.66–0.71).
- Under contrast changes the ranking changes substantially (rank correlation 0.17 at contrast 0.5).
- **Explanations do not transfer across datasets** (10 splits, same fitted models).
  - RF group importance falls from colour 0.177 / texture 0.092 / shape 0.042 on our test split to 0.039 / 0.037 /
    0.021 on new PlantDoc images, and to about 0 on Nazirpur.
  - RF feature rankings agree well between our own splits (rank correlation 0.87), less with PlantDoc (0.61), and not
    at all with Nazirpur (−0.01).
  - SVM and LR rankings are unstable even between our own splits (0.18 and 0.48), so per-feature explanations are
    only meaningful for RF.
  - A model that fails on a dataset has no meaningful explanation there. The explanations measured on our own data
    describe a dataset-specific model, not general disease cues.
- **Feature-level explanation, without a model (preliminary).** For each feature we measured how strongly it
  separates healthy from diseased leaves (ROC AUC − 0.5) in four datasets: ours, PlantDoc, Nazirpur and Brahmanbaria.
  - Only 2 of 105 features point the same way with an effect of at least 0.1 in all four: the share of yellow-orange
    leaf pixels (higher in diseased leaves, +0.11 to +0.32) and the skewness of the green–red colour channel (lower,
    −0.15 to −0.27). Both describe visible yellowing and browning.
  - The strongest features in our own data (e.g. saturation skew, LAB-b mean) change direction in other datasets.
    Our dataset's effect profile correlates −0.51 with Brahmanbaria's; the two Bangladeshi sets agree with each other
    (+0.40).
  - No confidence intervals yet, and the external subsets are small. We plan to make this multi-source,
    feature-level analysis the core of the final paper.

## 6. Challenges and how we handled them

| Challenge | What we did |
|---|---|
| Dataset smaller than stated (830 vs 7,200) and no usable split | Own grouped split; supplied augmented set audited and rejected |
| Same photo under two diseases | 9 pairs found; those photos and their copies removed (17 images) |
| Single test split too noisy | 10 repeated grouped splits, tuning inside each, paired tests with Holm |
| Image size predicts the class | Metadata baseline, resolution and compression controls, source clusters |
| Non-reproducible GrabCut masks | Seeded OpenCV before each call; masks now identical across runs |
| **Unresolved:** suspected label noise | 21 septoria images carry early-blight filenames; label audit flags 139 images. Blind visual review pending; nothing changed automatically |
| **Unresolved:** same-session photos | 154 photos from one afternoon; same-class shots seconds apart cross splits. Sensitivity analysis planned |
| **Unresolved:** rotated copies | 4 exact rotated-copy pairs escape the flip-aware duplicate check. Documented; shared by all experiments |
| Normalisation interacts with segmentation | Gray World recolours leaves so the colour-based mask shrinks (0.83 → 0.39). Reported as a confound; a diagnostic run keeping the original mask is possible but not yet run |
| Supplied benchmark inflated by related images | Re-split the supplied set by source group (Section 4.9): 0.81 → 0.66 macro-F1; image-level split gives 0.97 |
| A third of the raw set is PlantDoc | Found by exact image matching; PlantDoc images that overlap ours are excluded from the external test and scored separately |
| External labels differ from ours | Only unambiguous class names are mapped; uncertain mappings (leaf curl) are reported separately; one dataset whose labels mark lesions was excluded |
| Results differ slightly on a second machine | RF in-domain mean 0.613 vs 0.600 (SVM, LR identical). All comparisons are made within one machine |

## 7. Revisions to the original plan

The synopsis assumed 7,200 balanced images with a fixed split and no class weighting. Those assumptions described the
augmented set, so we re-split the raw photos, enabled class weights, and replaced the single split with 10 repeated
grouped splits. We added source-shortcut controls after finding that image size predicts the class. For Phase 2 we
generate our own train-only augmentation, because the supplied augmented split places related images in both training
and test.
Hand-drawn mask evaluation, extra model families and a CNN-embedding reference are deferred so the remaining time goes
to the experiments that answer RQ2 directly. After the supplied-split benchmark looked much better than our own
results, we added a grouped re-evaluation of the supplied set (Section 4.9) and a cross-dataset test on public
tomato datasets (Section 4.10). The plan's Phase 3 "external test set" was brought forward because the evidence
for source-specific learning (RQ3) made it the most informative next experiment.

## 8. Plan to the final paper (24 December)

1. Sensitivity analyses: label suspects after blind review (including the 25 PlantDoc label conflicts);
   capture-session grouping; scores on the PlantDoc-derived vs the remaining raw images.
2. Explain the cross-dataset failure: which features shift most between datasets, and whether training on more than
   one source (e.g. raw + supplied grouped, or adding an external set) improves transfer.
3. Leave-one-cluster-out with a matched random-holdout control.
4. Explainability across splits and perturbations.
5. Possibly a diagnostic Gray World run with a fixed mask. Combined augmentation + normalisation only if a
   normalisation helps on its own (experiment B suggests neither does).
6. Literature survey and write-up.
