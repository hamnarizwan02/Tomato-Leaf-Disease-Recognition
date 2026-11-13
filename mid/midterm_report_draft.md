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

### 2.1 Laboratory versus field datasets

Most image-based plant-disease work builds on **PlantVillage**, about 54,000 leaf images of 14 crops photographed one
leaf at a time on a plain background [1]. Mohanty et al. [2] trained convolutional networks on it and reached 99.35%
accuracy on a held-out PlantVillage test set, but only about 31% on images collected from trusted online sources under
different conditions. They concluded that more diverse training data is needed. Ferentinos [3] likewise reported
very high accuracy on a large laboratory-dominated collection together with a marked drop when laboratory-trained
models were applied to field images. Barbedo [4] listed the factors that limit real-world use, among them covariate
shift between training and deployment images and the difficulty of separating symptoms from the background. Noyan [5]
showed that PlantVillage carries capture-related bias: a model using only eight background pixels classified it far
above chance (49% vs 2.6%).

Field and in-the-wild datasets were built in response. **PlantDoc** [6] collected 2,598 web images of 13 species with
cluttered backgrounds. **Tomato-Village** [7] photographed tomato plants in fields in Rajasthan, India, motivated by the
observation that PlantVillage-trained models do not work well in real fields. **PlantWild** [8] is the largest
in-the-wild collection (18,542 images, 89 classes) and stresses small differences between diseases and large variation
within one disease. Regional field datasets for tomato from Pakistan [9] and Bangladesh [10]–[12] are recent and, to our
knowledge, have no published benchmark. A common pattern is that each dataset is evaluated **within itself**; how a
model trained on one source behaves on another is rarely reported.

### 2.2 Hand-crafted features and classical machine learning

Before deep learning, disease recognition relied on segmentation followed by colour, texture and shape descriptors
[13]. Texture is usually described with grey-level co-occurrence statistics [14] and local binary patterns [15]; colour
with channel moments and histograms. Classifiers are typically support vector machines [16] and random forests [17].
For tomato, Mokhtar et al. [18] combined Gabor wavelet features with SVMs, Hlaing and Zaw [19] modelled SIFT-based
texture statistics together with colour moments, and Tan et al. [20] compared SVM, k-NN and random forests on
hand-crafted features with several CNNs. Singh and Misra [21] combined segmentation with soft-computing classifiers.
These studies report high accuracy, but almost all use PlantVillage images with random image-level splits, so they
measure performance on a single laboratory source. Classical pipelines remain attractive for low-resource settings
because they are cheap and their inputs are named, measurable quantities.

### 2.3 Robustness and distribution shift

Hendrycks and Dietterich [22] showed that image classifiers degrade under common corruptions such as blur, noise and
brightness change, which motivates controlled perturbation tests. Torralba and Efros [23] showed that every dataset has
its own "signature" and that models trained on one dataset transfer poorly to another. Domain-generalisation work
evaluates by holding out whole domains; Gulrajani and Lopez-Paz [24] showed that, with a careful leave-domain-out
protocol, many specialised methods do not beat plain empirical risk minimisation, and WILDS [25] collected real
distribution shifts for benchmarking. Classical remedies for lighting variation include grey-world colour constancy
[26] and contrast-limited adaptive histogram equalisation (CLAHE) [27]; data augmentation is the usual learning-based
remedy [28].

### 2.4 Shortcut learning, leakage and duplicates

Geirhos et al. [29] describe shortcut learning: models exploit cues that work on the benchmark but not on the intended
task. In medical imaging, models learned hospital- or device-specific signals [30], [31], and DeGrave et al. [31]
showed that good external-test accuracy alone does not prove that a model uses the pathology. Kapoor and Narayanan [32]
catalogue data leakage as a main cause of irreproducible machine-learning results, including duplicates across training
and test sets. Barz and Denzler [33] found that 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates
in training. Roberts et al. [34] showed that when data are grouped or structured, blocked (grouped) cross-validation is
more appropriate than random splitting for predicting to new data. Label errors are a related problem; confident
learning [35] flags likely mislabelled examples from out-of-sample predicted probabilities.

### 2.5 Explanation of models and of features

Feature-importance measures explain a fitted model. Random-forest impurity importance is biased towards some feature
types [36], permutation importance can extrapolate to unrealistic inputs when features are correlated [37], and the
importance of a feature depends on which of many equally good models is chosen [38]. Local explainers such as LIME [39]
and SHAP [40] often disagree with each other [41]. Rudin [42] argues that for high-stakes decisions inherently
interpretable models should be preferred to post-hoc explanation. In plant pathology, Toda and Okura [43] visualised a
CNN trained on PlantVillage and found that it attends to lesion colour and texture, but on a single laboratory dataset.

**Concept-based explanation** expresses a model in human terms. TCAV [44] tests a network's sensitivity to
user-defined concepts, and concept bottleneck models [45] predict through a layer of named concepts. With
vision-language models such as CLIP [46], concept scores can be obtained without concept labels: Label-free CBM [47]
and LaBo [48] generate concepts with language models and score them with CLIP. In this project, models are trained and tested only with classical ML; we compare two kinds of learned
feature extractor. The classical one follows the bag-of-visual-words idea [52], [53]: codebooks of colour and
texture prototypes learned with k-means [54], [55]. The deep one uses frozen pretrained networks, OpenCLIP [49] and an
ImageNet ResNet-50 [50], purely as feature extractors.

### 2.6 Gap addressed by this project

(i) Classical tomato-disease studies report high accuracy on laboratory images with image-level splits; we evaluate on
field photos with grouped splits and quantify how much related images inflate a benchmark. (ii) Field datasets are
benchmarked within themselves; we test across eight image sources with leave-one-source-out evaluation and check
dataset provenance. (iii) Explanations are usually reported for one model on one dataset; we ask which features —
formula-based, learned by k-means, or extracted by a frozen deep network — keep the same effect across sources, and whether models restricted to
them transfer better.

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

### 4.11 Multi-source leave-one-source-out and feature-level explanation (generalisation, unsupervised learning, model selection)
- **Sources.** Eight image sources, each one group: our raw photos split into those with a PlantDoc copy (296) and
  the rest (517); PlantDoc; PlantWild (web, in the wild); Tomato-Village (field, India); Nazirpur and Brahmanbaria
  (Bangladesh); a sample of PlantVillage (laboratory). Cross-source duplicates are removed with the frozen thumbnail
  rule (425 dropped, 72 label conflicts recorded): 7,754 images.
- **Features, three arms, all classified with RF, SVM and LR.** (i) Formula: the 105 hand-crafted features plus 20
  new leaf-relative lesion features (yellow, brown, dark and non-green fractions; lesion-minus-tissue colour shift;
  spot size). (ii) K-means visual words (unsupervised learning): 24 colour words and 32 texture words learned with
  k-means on leaf pixels, re-learned inside every fold from the training sources only. (iii) Frozen deep extractors
  (no training): CLIP image embeddings, 24 named CLIP concept scores, ResNet-50 features.
- **Evaluation.** Leave-one-source-out: train on seven sources, choose hyper-parameters on 15% of each training
  source, test on the eighth. A seen-source reference splits every source 70/15/15. Tasks: healthy vs diseased
  (balanced accuracy) and six classes (macro-F1 over the classes present).
- **Feature-level explanation without a model.** Per source, each feature's effect (AUC − 0.5) with a bootstrap 95%
  CI; a feature is *source-stable* if the CI excludes 0 with the same sign in at least 6 of 7 sources. Compared with
  L1 logistic-regression coefficients (glass box), RF permutation importance, and selection of stable features on
  the training sources only.

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

**Phase 3, multi-source results (run 12–13 November; preliminary, full tables in verified_results §§14–15).**
- **Pooling sources makes binary transfer work.** Logistic regression on an unseen source, healthy vs diseased
  (balanced accuracy; chance 0.5): formula hand + lesion 0.66 (seen sources 0.82), k-means colour words 0.62 (most
  even across sources), CLIP concept scores 0.75, CLIP embedding 0.72, ResNet-50 0.67. Linear models transfer best;
  RF calls most healthy leaves of a new source diseased (healthy recall 0.22 vs 0.57 for LR).
- **Six-class diagnosis transfers poorly.** Unseen-source macro-F1: formula or k-means features at most 0.33 (seen
  0.55–0.69); frozen CLIP embeddings 0.53, higher on 6 of 7 sources (not significant after Holm correction with 7
  sources).
- **Feature-level explanation gives a positive answer to RQ4.** 24 of 205 named features are source-stable. Lesion
  formulas are the most reliable family (8 of 20): the yellow and non-green fractions of the leaf, the lesion b*
  shift and the largest spot size are higher in diseased leaves on all 7 sources. 7 of 105 hand features, 8 of 56
  k-means words (brown/olive/ochre colours) and 1 of 24 CLIP concepts qualify.
- Model-based explanations do not recover this set: RF importance on the held-out source is unrelated to stability
  (Spearman −0.22 to 0.16). The best concept model gives sign-stable weight to capture context (plain background,
  hand, studio). Restricting models to the stable features does not improve transfer (formula LR 0.64 → 0.63).

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
for source-specific learning (RQ3) made it the most informative next experiment. Because single-source models and their
explanations both failed on other datasets, we changed the unit of analysis from one dataset and one model to several
sources and each feature (Section 4.11): models are now evaluated on sources they have never seen, and explanation is
defined at the feature level and required to replicate across sources. At the team's request we compared two learned
feature extractors under the same classical training: k-means visual words and frozen deep networks used only to
extract features. CNN training remains out of scope.

## 8. Plan to the final paper (24 December)

Phase 3 (multi-source evaluation, both extractor arms, feature-level explanation) has been run; a paper draft and a
full report exist. Remaining:
1. Label review: the 72 cross-source label conflicts, the 25 PlantDoc conflicts and the 21 septoria/early-blight
   suspects, followed by a re-run of Phase 3.
2. Per-disease feature-level explanation (each disease vs healthy) on the sources that have it.
3. Mask the leaf before the frozen deep extractors, to measure how much of their advantage is context.
4. Sensitivity analyses: capture-session grouping; leave-one-cluster-out with a matched random-holdout control.
5. Final write-up; every cited paper read by the team.

## References


[1] D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of
mobile disease diagnostics," arXiv:1511.08060, 2015.
[2] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection,"
Frontiers in Plant Science, vol. 7, art. 1419, 2016, doi:10.3389/fpls.2016.01419.
[3] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," Computers and Electronics in
Agriculture, vol. 145, pp. 311–318, 2018.
[4] J. G. A. Barbedo, "Factors influencing the use of deep learning for plant disease recognition," Biosystems
Engineering, vol. 172, pp. 84–91, 2018, doi:10.1016/j.biosystemseng.2018.05.013.
[5] M. A. Noyan, "Uncovering bias in the PlantVillage dataset," arXiv:2206.04374, 2022.
[6] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease
detection," in Proc. ACM IKDD CoDS and COMAD, 2020, pp. 249–253, doi:10.1145/3371158.3371196.
[7] M. Gehlot, R. K. Saxena, and G. C. Gandhi, "'Tomato-Village': a dataset for end-to-end tomato disease detection in
a real-world environment," Multimedia Systems, vol. 29, pp. 3305–3328, 2023, doi:10.1007/s00530-023-01158-y.
[8] T. Wei, Z. Chen, Z. Huang, and X. Yu, "Benchmarking in-the-wild multimodal disease recognition and a versatile
baseline," in Proc. 32nd ACM Int. Conf. Multimedia (MM '24), 2024, pp. 1593–1601, doi:10.1145/3664647.3680599.
[9] M. N. Malik, S. H. Ali, and H. Ali, "Tomato leaf disease classification dataset in Pakistan," Mendeley Data, v2,
2026, doi:10.17632/3mbnb82mxd.2.
[10] P. Bapari, M. Z. R. Zitu, and M. Ummehani, "A high-resolution image dataset of tomato (Solanum lycopersicum) leaves
for multi-class disease detection and classification from Bangladesh," Mendeley Data, v1, 2025,
doi:10.17632/74zvtzx9pr.1.
[11] M. K. Oni and T. T. Prama, "A comprehensive dataset of tomato leaf images for disease analysis in Bangladesh," Data
in Brief, 2025; data: "Tomato leaf image dataset for disease analysis in real-world environment," Mendeley Data, v1,
2024, doi:10.17632/rnbsw72zb5.1.
[12] A. Imtiaz, F. B. I. Swapnil, S. R. Masud, and D. Karmaker, "Tomato leaf dataset: A dataset for multiclass disease detection and classification," Data in
Brief, 2025, doi:10.1016/j.dib.2025.111520; data: Mendeley Data, doi:10.17632/bpfd9cns5g.2.
[13] J. G. A. Barbedo, "Digital image processing techniques for detecting, quantifying and classifying plant
diseases," SpringerPlus, vol. 2, art. 660, 2013, doi:10.1186/2193-1801-2-660.
[14] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," IEEE Trans. Systems,
Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, 1973.
[15] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture
classification with local binary patterns," IEEE Trans. Pattern Anal. Mach. Intell., vol. 24, no. 7, pp. 971–987,
2002.
[16] C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, pp. 273–297, 1995.
[17] L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5–32, 2001.
[18] U. Mokhtar, M. A. S. Ali, A. E. Hassanien, and H. Hefny, "Tomato leaves diseases detection approach based on
support vector machines," in Proc. 11th Int. Computer Engineering Conf. (ICENCO), 2015,
doi:10.1109/ICENCO.2015.7416356.
[19] C. S. Hlaing and S. M. M. Zaw, "Tomato plant diseases classification using statistical texture feature and color
feature," in Proc. IEEE/ACIS 17th Int. Conf. Computer and Information Science (ICIS), 2018, pp. 439–444,
doi:10.1109/ICIS.2018.8466483.
[20] L. Tan, J. Lu, and H. Jiang, "Tomato leaf diseases classification based on leaf images: A comparison between
classical machine learning and deep learning methods," AgriEngineering, vol. 3, no. 3, pp. 542–558, 2021,
doi:10.3390/agriengineering3030035.
[21] V. Singh and A. K. Misra, "Detection of plant leaf diseases using image segmentation and soft computing
techniques," Information Processing in Agriculture, vol. 4, no. 1, pp. 41–49, 2017.
[22] D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to common corruptions and
perturbations," in Proc. ICLR, 2019.
[23] A. Torralba and A. A. Efros, "Unbiased look at dataset bias," in Proc. IEEE CVPR, 2011, pp. 1521–1528.
[24] I. Gulrajani and D. Lopez-Paz, "In search of lost domain generalization," in Proc. ICLR, 2021.
[25] P. W. Koh et al., "WILDS: A benchmark of in-the-wild distribution shifts," in Proc. ICML, PMLR 139, 2021,
pp. 5637–5664.
[26] G. Buchsbaum, "A spatial processor model for object colour perception," J. Franklin Institute, vol. 310, no. 1,
pp. 1–26, 1980.
[27] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in Graphics Gems IV, Academic Press, 1994,
pp. 474–485.
[28] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," J. Big Data, vol. 6,
art. 60, 2019.
[29] R. Geirhos et al., "Shortcut learning in deep neural networks," Nature Machine Intelligence, vol. 2,
pp. 665–673, 2020.
[30] J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest
radiographs: A cross-sectional study," PLoS Medicine, vol. 15, no. 11, e1002683, 2018.
[31] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over
signal," Nature Machine Intelligence, vol. 3, pp. 610–619, 2021, doi:10.1038/s42256-021-00338-7.
[32] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns,
vol. 4, no. 9, art. 100804, 2023, doi:10.1016/j.patter.2023.100804.
[33] B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of near-duplicates," J. Imaging, vol. 6, no. 6,
art. 41, 2020, doi:10.3390/jimaging6060041.
[34] D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure," Ecography, vol. 40, no. 8, pp. 913–929, 2017, doi:10.1111/ecog.02881.
[35] C. G. Northcutt, L. Jiang, and I. L. Chuang, "Confident learning: Estimating uncertainty in dataset labels,"
J. Artificial Intelligence Research, vol. 70, pp. 1373–1411, 2021.
[36] C. Strobl, A.-L. Boulesteix, A. Zeileis, and T. Hothorn, "Bias in random forest variable importance measures:
Illustrations, sources and a solution," BMC Bioinformatics, vol. 8, art. 25, 2007.
[37] G. Hooker, L. Mentch, and S. Zhou, "Unrestricted permutation forces extrapolation: Variable importance requires
at least one more model, or there is no free variable importance," Statistics and Computing, vol. 31, art. 82, 2021,
doi:10.1007/s11222-021-10057-z.
[38] A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a variable's importance
by studying an entire class of prediction models simultaneously," J. Machine Learning Research, vol. 20, no. 177,
pp. 1–81, 2019.
[39] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any
classifier," in Proc. ACM SIGKDD, 2016, pp. 1135–1144.
[40] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. NeurIPS, 2017.
[41] S. Krishna et al., "The disagreement problem in explainable machine learning: A practitioner's perspective,"
arXiv:2202.01602, 2022 (TMLR, 2024).
[42] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable
models instead," Nature Machine Intelligence, vol. 1, pp. 206–215, 2019.
[43] Y. Toda and F. Okura, "How convolutional neural networks diagnose plant disease," Plant Phenomics, vol. 2019,
art. 9237136, 2019, doi:10.34133/2019/9237136.
[44] B. Kim et al., "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors
(TCAV)," in Proc. ICML, PMLR 80, 2018, pp. 2668–2677.
[45] P. W. Koh et al., "Concept bottleneck models," in Proc. ICML, PMLR 119, 2020, pp. 5338–5348.
[46] A. Radford et al., "Learning transferable visual models from natural language supervision," in Proc. ICML,
PMLR 139, 2021, pp. 8748–8763.
[47] T. Oikarinen, S. Das, L. M. Nguyen, and T.-W. Weng, "Label-free concept bottleneck models," in Proc. ICLR, 2023.
[48] Y. Yang, A. Panagopoulou, S. Zhou, D. Jin, C. Callison-Burch, and M. Yatskar, "Language in a bottle: Language
model guided concept bottlenecks for interpretable image classification," in Proc. IEEE/CVF CVPR, 2023,
pp. 19187–19197.
[49] G. Ilharco et al., "OpenCLIP," Zenodo, 2021, doi:10.5281/zenodo.5143773.
[50] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016,
pp. 770–778.
[51] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Machine Learning Research, vol. 12,
pp. 2825–2830, 2011.
[52] G. Csurka, C. R. Dance, L. Fan, J. Willamowski, and C. Bray, "Visual categorization with bags of keypoints,"
in Proc. ECCV Workshop on Statistical Learning in Computer Vision, 2004.
[53] M. Varma and A. Zisserman, "A statistical approach to texture classification from single images," Int. J.
Computer Vision, vol. 62, pp. 61–81, 2005.
[54] D. Arthur and S. Vassilvitskii, "k-means++: The advantages of careful seeding," in Proc. ACM-SIAM SODA, 2007,
pp. 1027–1035.
[55] D. Sculley, "Web-scale k-means clustering," in Proc. WWW, 2010, pp. 1177–1178.
