# Verified results summary

The single source of numbers for the mid-semester report and the IEEE paper. Every number here was checked against the
file named next to it. Copy numbers from this page, not from the PDF findings log.

**Status labels**
- **Verified**: reproduced in the Phase 2 environment from the committed manifest, or checked against the source file.
- **Preliminary**: computed and verified, but not yet a conclusion (single split, untested mechanism, or pending checks).
- **Phase 1 env**: committed Phase 1 output, internally consistent, not re-run in the Phase 2 environment.
- **Stale, do not cite**: superseded numbers.

Run folders live outside Git in `~/Desktop/MLPROJECT/runs/`: `p2_base` (Phase 1 reproduction), `p2_augmentation`
(experiment A), `p2_illumination` (experiment B) and `p2_supplied` (supplied-split benchmark, §10). The commands for
each run are in [experiment_log.md](experiment_log.md).
Environment: Python 3.11.17, numpy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, scikit-image 0.26.0, OpenCV 5.0.0.93.

---

## 1. Dataset and splits (Verified)

| Item | Value | Source |
|---|---|---|
| Raw photos (`Dataset (raw)`) | 830 (119 / 137 / 113 / 195 / 80 / 186 per class folder) | directory count |
| Removed: same photo filed under two classes | 17 (9 conflicting pairs plus all their copies) | `results/dropped_cross_class.csv` |
| Images used | 813 in 769 source groups (726 singletons, 42 pairs, 1 triple) | `data/manifest.csv` |
| Per class | early blight 110, late blight 110, septoria 191, leaf mold 185, yellow leaf curl 80, healthy 137 (2.4× imbalance) | manifest |
| Committed split (seed 42) | train 570 / val 125 / test 118; no group crosses splits | manifest (reproduced exactly from groups) |
| Protocol | 10 grouped 70/15/15 splits, seeds 42–51; test sets of 118–119 images | `protocol_main/splits.csv` |
| Supplied "Augmented Dataset" | 7,200 images at 224 px with its own split. **Not used in the main protocol**; benchmarked separately on its own split (§10). Its split places related images on both sides: 133 source groups span splits (filename grouping + near-duplicate merging), 33.1% of test images have their source in train/val under that rule, and 361 / 720 test images have a detected sibling under the stricter §10 rules | `runs/p2_supplied/results/grouping_summary.json`, `dataset_audit.json`, `sibling_flags_test.csv` |

> **UNVERIFIED, do not cite (2026-10-27 audit).** Earlier linkage figures for the supplied set have no saved output:
> - 48% linked to a raw source by filename and 26% by content, "1 source in both train and test", and "83 files
>   descend from the 17 removed images" (2026-10-26). Their only record is one sentence in
>   `runs/p2_augmentation/NOTES.md`, and that analysis used `~/Downloads/Tomato Dataset part 2/`, not
>   `Tomato Dataset/Augmented Dataset`. Whether the two copies are identical has not been checked.
> - A full-set re-check (2026-10-27, read-only) gave 67% by filename and 25% by content; it was not saved either.
>
> Cite linkage percentages only after recomputing them with a saved script on the `Augmented Dataset` copy. Until
> then, use the saved figures in the row above.

## 2. Data-quality findings (Verified facts, unresolved)

The labels have not been changed and no image has been excluded beyond the 17 above.

| Finding | Evidence | Status |
|---|---|---|
| Suspected label noise | 21 septoria images have "early blight" in the filename (19 name only early blight, 2 name both); 4 are in the seed-42 test set. Also `early_blight/gray-mold-on-tomato-leaf.jpg`, `late_blight/tomato.early.blightH.jpg`, `healthy/curling-issues.jpg`. A filename is not proof | awaiting blind visual review |
| Label audit (RF, out-of-fold, 3 × 5 grouped CV) | 308 / 813 wrong out of fold; 139 confident-learning flags; 7 with margin > 0.5. Flags are candidates only | `label_audit_summary.json` (identical in both environments) |
| Capture-session bursts | 154 phone photos taken 2025-11-02 between 15:29 and 16:20 (healthy 66, early blight 38, yellow leaf curl 31, leaf mold 11, late blight 8). Rule: same class, different group, timestamps within 60 s gives 404 cross-split pairs; within 5 min gives 1,153. All 21 timestamped test images of split 0 have a same-class train/val image taken within 5 min. One pair inspected: same phone, 1 s apart, same plant | timestamp analysis; not yet a sensitivity run |
| Rotated copies in different groups | The duplicate check is flip-aware but not rotation-aware. 4 exact rotated-copy pairs (r ≥ 0.999) sit in different groups and are separated in 4–8 of the 10 splits | rotation-aware scan. Shared by all arms, so arm comparisons are unaffected |
| Non-deterministic GrabCut | Background-replacement masks varied between runs (26 of 40 images). Fixed by seeding OpenCV before each call (`src/segment.py`). Background conditions now differ from Phase 1 by at most ±0.007 macro-F1 averaged over splits | `runs/p2_base/NOTES.md` |

> **Source files missing.** The capture-session counts (154 photos; 404 and 1,153 cross-split pairs; the split-0
> check) and the rotated-copy scan (4 pairs, separated in 4–8 splits) came from read-only analyses whose outputs were
> not saved. The findings are reported as limitations; recompute them with a saved script before quoting exact counts
> in the paper.

## 3. Phase 1 baseline (Verified: reproduced exactly in the Phase 2 environment)

Test macro-F1, mean ± sd over 10 grouped splits, clean images. Source: `runs/p2_base/results/protocol_main/summary_clean.csv`.

| Feature set | RF | SVM | LR |
|---|---|---|---|
| All (105) | **0.600 ± 0.046** | **0.615 ± 0.026** | **0.596 ± 0.044** |
| Colour + texture (85) | 0.611 ± 0.041 | 0.630 ± 0.034 | 0.598 ± 0.031 |
| Colour (47) | 0.541 ± 0.050 | 0.543 ± 0.048 | 0.495 ± 0.037 |
| Texture (38) | 0.483 ± 0.044 | 0.515 ± 0.047 | 0.518 ± 0.049 |
| Shape (20) | 0.408 ± 0.042 | 0.382 ± 0.046 | 0.380 ± 0.041 |
| Metadata only (6) | 0.487 ± 0.035 | 0.424 ± 0.034 | 0.351 ± 0.042 |

- Accuracy (all features): RF 0.608, SVM 0.628, LR 0.609. Balanced accuracy 0.602 / 0.616 / 0.604. ECE 0.142 / 0.146 / 0.106.
- Paired tests (Wilcoxon over splits, Holm within family; `paired_tests.csv`):
  - models do not differ (smallest adjusted p 0.35);
  - each single feature group is worse than all features (adjusted p 0.029–0.041);
  - colour + texture vs all: not significant;
  - all features beat the metadata-only model on 10 of 10 splits (adjusted p 0.035).
- The single seed-42 split is favourable: RF 0.665 [95% bootstrap CI 0.565–0.736], SVM 0.656, LR 0.633
  (`baseline_test_metrics.csv`). **Do not use it as the headline number.**
- Overfitting: RF training macro-F1 1.00 at every learning-curve size; validation plateaus at 0.54–0.58 from 40% of
  the training data upwards (0.540, 0.564, 0.581, 0.558). Single seed-42 split (`learning_curve_rf.csv`).
- Pooled clean confusions (all features, 10 splits; the A0 rows of `runs/p2_augmentation/summary/confusion_pooled.csv`,
  which reproduce the baseline exactly). The largest off-diagonal counts:
  - RF: early blight → septoria 54, septoria → late blight 43, septoria → early blight 40;
  - SVM: early blight → septoria 41, septoria → early blight 39, septoria → late blight 38;
  - LR: septoria → early blight 49, early blight → septoria 32, septoria → late blight 30.

  Early blight and septoria are confused in both directions by every model. The dominant direction is early blight →
  septoria for RF and SVM, and the reverse for LR. Some of this may reflect the suspected label noise in §2 (21 septoria
  images with early-blight filenames); that is untested.

## 4. Robustness of the baseline (Verified)

Perturbed macro-F1 divided by clean macro-F1, mean over levels and 10 splits, all features
(`p2_base/.../robustness_by_kind.csv`). There are 50 conditions, applied to test images only.

| Kind | RF | SVM | LR |
|---|---|---|---|
| Down-then-up scaling | 0.42 | 0.32 | 0.47 |
| Gaussian blur | 0.43 | 0.31 | 0.48 |
| Motion blur | 0.52 | 0.53 | 0.73 |
| Combined pairs | 0.58 | 0.46 | 0.65 |
| Contrast | 0.61 | 0.59 | 0.65 |
| Colour cast | 0.70 | 0.61 | 0.70 |
| Noise | 0.78 | 0.68 | 0.80 |
| Shadow | 0.81 | 0.77 | 0.90 |
| Brightness | 0.83 | 0.70 | 0.79 |
| Scale | 0.86 | 0.89 | 0.99 |
| Occlusion | 0.87 | 0.67 | 0.70 |
| Gamma | 0.87 | 0.84 | 0.90 |
| Background (fixed GrabCut mask) | 0.91 | 0.89 | 0.89 |
| JPEG | 0.95 | 0.87 | 0.87 |
| Rotation | 0.96 | 0.96 | 0.96 |

The worst condition for RF is down-then-up scaling at 0.125 in 10 of 10 splits. The mean worst-case ratio is RF 0.245,
LR 0.146, SVM 0.083.

## 5. Source shortcut (Verified; interpretation Preliminary)

| Control | Result | Source |
|---|---|---|
| Metadata-only model (size, aspect, file size) | RF 0.487 ± 0.035, about 3× chance (0.167) | `summary_clean.csv` |
| Feature vs metadata correlation | 0 / 105 features reach \|Spearman\| 0.5; the highest is GLCM correlation d1 at 0.43. The pruned set equals the full set | `shortcut/feature_correlations.csv` |
| Source clusters | 8 clusters (silhouette 0.465); cluster 7 = 106 / 122 leaf mold; accuracy on cluster 7 under the main protocol 0.91–0.96 | `shortcut/cluster_summary.csv`, `accuracy_by_cluster.csv` |
| Leave-one-cluster-out (all features) | Mean accuracy over clusters RF 0.466, SVM 0.487, LR 0.498 (vs about 0.61 in-distribution); macro-F1 over present classes 0.34–0.37; metadata model 0.21–0.29 | `shortcut/loco_metrics.csv` |
| V1 same effective resolution / (d) same JPEG / V2 only ≥ 500 px (**Phase 1 env**) | RF all 0.598 / 0.603 / 0.569; SVM 0.616 / 0.612 / 0.608; LR 0.595 / 0.606 / 0.603. Unchanged within noise, and still above metadata | committed `results_res_matched`, `results_reencode`, `results/protocol_min_side500` |

What this shows: resolution and compression artefacts do not explain the scores, but part of the score does not
transfer to an unseen source cluster. What it cannot prove: that the models use the shortcut. Leave-one-cluster-out
also shifts class balance and has no matched random-holdout control yet.

## 6. Explainability, single split (Preliminary)

`runs/p2_base/results/importance_*`:
- Across 5 grouped CV folds: rank correlation of importances, impurity 0.905 ± 0.016 and permutation 0.902 ± 0.016.
  Top-10 overlap (Jaccard): impurity 0.66, permutation 0.71. So the ordering is stable, but which features make the top 10 is not.
- Grouped permutation importance (drop in true-class probability): colour 0.164, texture 0.083, shape 0.039.
  Group-level stability of 1.0 is trivial because there are only 3 groups.
- Top features: LAB-B std, GLCM correlation d3, HSV-S std, GLCM correlation d1.
- Rank agreement with the clean ranking under perturbation: 0.17 (contrast 0.5), 0.45 (contrast 2.0), 0.53 (brightness 0.5); ≥ 0.80 for shadows and backgrounds.
- Error analysis (split 0, RF): the most common error is early blight predicted as septoria (6 / 15 test images).

## 7. Experiment A: train-only augmentation (Verified runs; Preliminary conclusions)

Design: k = 4 copies per training image; validation and test never augmented; no copy of a validation or test image or
group is ever selected (checked per split). Arms: A0 control (0 copies, reproduces the baseline exactly); A1 geometric
(h/v flip p = 0.5, rotation ±15°); A2 = A1 + brightness and contrast ×[0.8, 1.2].
Sources: `runs/p2_augmentation/summary/` (each arm vs baseline) and `summary_A2_vs_A1/` (photometric part only).

**Clean macro-F1** (mean ± sd over 10 splits; paired Wilcoxon with Holm over 21 set × model comparisons):

| All features | Baseline | A1 | A2 |
|---|---|---|---|
| RF | 0.600 ± 0.046 | 0.612 ± 0.049 | 0.613 ± 0.046 |
| SVM | 0.615 ± 0.026 | 0.633 ± 0.036 | 0.615 ± 0.056 |
| LR | 0.596 ± 0.044 | 0.595 ± 0.041 | 0.597 ± 0.047 |

There is **no detectable clean change**: every adjusted p = 1.0, for all feature sets. Power is limited: with 10 splits
the smallest Wilcoxon p is 0.002, so a significant clean difference would need about 10 of 10 wins.

**Robustness** (change in macro-F1 vs baseline, mean over levels, all features; bold = Holm-adjusted p < 0.05 within
run × feature set × model; "(n)" = splits out of 10 where the arm won; † = family the arm was trained on):

| Kind | A2 RF | A2 SVM | A2 LR | A1 RF |
|---|---|---|---|---|
| Contrast † (A2) | **+0.110** (10) | **+0.119** (10) | **+0.086** (10) | −0.015 (2) |
| Brightness † (A2) | **+0.045** (10) | **+0.106** (10) | **+0.071** (10) | −0.007 (3) |
| Motion blur | **+0.072** (10) | +0.051 (6) | +0.032 (8) | +0.036 (8) |
| Gaussian blur | **+0.067** (9) | +0.022 (6) | **+0.042** (10) | +0.041 (7) |
| Down-then-up | **+0.065** (9) | +0.026 (6) | +0.033 (8) | +0.039 (7) |
| Shadow | **+0.057** (10) | +0.051 (9) | +0.005 (5) | +0.023 (7) |
| Scale | **+0.056** (10) | +0.034 (7) | +0.003 (5) | **+0.037** (10) |
| Rotation † | **+0.030** (9) | +0.009 (6) | +0.009 (5) | +0.041 (9) |
| Colour cast | **−0.069** (0) | −0.005 (5) | −0.019 (1) | **−0.041** (0) |
| Noise | **−0.034** (0) | −0.024 (4) | −0.040 (2) | −0.015 (4) |
| Background, gamma, JPEG, occlusion | within ±0.02, none significant | | | |

- RF robustness ratios, baseline → A2: contrast 0.61 → 0.78, brightness 0.83 → 0.89, shadow 0.81 → 0.89,
  blur 0.43 → 0.53, colour cast 0.70 → 0.57.
- **A2 vs A1** (photometric part only; `summary_A2_vs_A1/comparison_kinds.csv`):
  - brightness: RF +0.052, SVM +0.094, LR +0.066, all significant (10/10 wins);
  - contrast: RF +0.125 and SVM +0.100 significant (10/10 wins); LR +0.063 not significant (adjusted p 0.082);
  - shadow +0.033 for RF and +0.041 for SVM (significant);
  - colour cast −0.028 for RF (significant);
  - blur and down-then-up: RF +0.026 (not significant), SVM and LR −0.004 to −0.020. So the RF blur and rescaling gains
    of A2 are **not attributable to the photometric part**. A1 alone does not show them significantly either (RF blur
    +0.041, adjusted p 0.80; down-then-up +0.039, adjusted p 0.74), so their source is unresolved;
  - clean: no difference.
- Calibration (ECE, all features): SVM improves 0.146 → 0.110 (A2); LR worsens 0.106 → 0.125.

**Reading**:
- (i) Photometric augmentation reliably improves robustness to the lighting families it was trained on, at test levels
  that all lie outside the training range ×[0.8, 1.2] (brightness 0.5–1.6, contrast 0.5–2.0; 0.7 and 1.3 only just).
  The test perturbation and the augmentation use the same formulas (`src/perturb.py`, `src/augment.py`: brightness
  multiplies pixel values, contrast scales them about the image mean).
- (ii) It does not change clean accuracy detectably.
- (iii) It costs RF robustness to colour casts and noise.
- (iv) RF gains on blur and rescaling are significant in A2 but are not attributable to the photometric part, and A1
  alone does not reach significance. Possibly they come from the geometric part, e.g. rotation resampling slightly
  smoothing copies; this is a hypothesis.

## 8. Experiment B: illumination normalisation (Verified runs; Preliminary conclusions)

Design: every image (train, val and test) is normalised on its own, after any test perturbation and before
segmentation and features. No parameters are fitted and there is no augmentation. Same splits, grids and protocol;
hyper-parameters re-tuned per split. Variants:
- **GW**: Gray World, per-image channel-mean equalisation.
- **CL**: CLAHE on LAB lightness, clip limit 2.0, 8×8 tiles, fixed in advance.

Source: `runs/p2_illumination/summary/` (12/12 integrity checks; feature stores re-extracted byte-identically).

> **Gray World is a pipeline confound, not a colour-only test.** Normalisation runs before the HSV leaf mask
> (hue 20–100). Gray World recolours yellow/green leaves towards lavender, so those pixels leave the mask:
> - mean mask area falls from 0.83 to 0.39 of the image (median 0.47 × the baseline);
> - images with a mask under 20% of the frame rise from 0.7% to 13.4%;
> - the shrinkage differs by class (leaf mold 0.51; other classes 0.32–0.38).
>
> GW results therefore mix colour normalisation with altered segmentation, and must not be read as the effect of
> colour normalisation alone. CLAHE leaves masks unchanged (0.83).

**Clean macro-F1** (mean ± sd over 10 splits; Holm over 21 set × model comparisons):

| All features | Baseline | GW | CL |
|---|---|---|---|
| RF | 0.600 ± 0.046 | 0.548 ± 0.028 | 0.611 ± 0.037 |
| SVM | 0.615 ± 0.026 | 0.553 ± 0.036 | 0.612 ± 0.045 |
| LR | 0.596 ± 0.044 | 0.540 ± 0.058 | 0.610 ± 0.028 |

- **GW lowers clean macro-F1** by 0.05–0.06 for all models with all features (RF −0.052, SVM −0.062, LR −0.056;
  baseline wins 8–9 of 10 splits). Other feature sets range up to −0.091 (texture, LR). Raw p 0.006–0.010, but
  Holm-adjusted p 0.12–0.16; only colour + texture SVM survives correction (−0.072, adjusted p 0.041).
  So: consistent direction, but mostly not significant after correction.
- GW accuracy 0.567–0.569 vs baseline 0.608–0.628.
- **CL: no detectable clean change** (−0.003 to +0.014; all adjusted p = 1.0). The colour-only LR gain of +0.054
  (9/10 wins) has raw p 0.004 and adjusted p 0.08.

**Robustness** (change in macro-F1 vs baseline, mean over levels, all features; bold = Holm-adjusted p < 0.05 within
run × feature set × model; "(n)" = splits out of 10 where the variant won):

| Kind | GW RF | GW SVM | GW LR | CL RF | CL SVM | CL LR |
|---|---|---|---|---|---|---|
| Colour cast | **+0.127** (10) | **+0.175** (10) | **+0.117** (10) | +0.000 (5) | +0.004 (5) | −0.003 (5) |
| Occlusion | −0.010 (4) | **+0.093** (10) | **+0.080** (10) | −0.010 (2) | +0.016 (8) | **+0.043** (10) |
| Contrast | −0.035 (3) | −0.028 (2) | +0.032 (8) | +0.032 (9) | +0.012 (7) | +0.038 (8) |
| Shadow | −0.005 (4) | +0.010 (6) | **−0.044** (0) | **+0.027** (10) | +0.012 (7) | −0.011 (5) |
| Brightness | **−0.070** (0) | −0.013 (5) | −0.008 (4) | **−0.025** (0) | −0.023 (4) | −0.031 (3) |
| Noise | **−0.100** (0) | **−0.108** (0) | **−0.147** (0) | −0.000 (4) | −0.064 (1) | **−0.039** (0) |
| Background | **−0.097** (0) | **−0.091** (0) | **−0.087** (0) | +0.004 (4) | −0.021 (2) | +0.007 (6) |
| Gamma | **−0.074** (0) | **−0.082** (1) | **−0.100** (0) | +0.007 (6) | +0.018 (8) | +0.011 (7) |
| JPEG | **−0.083** (0) | −0.037 (2) | **−0.073** (1) | +0.004 (7) | +0.005 (5) | +0.032 (8) |
| Scale, rotation, motion blur | −0.035 to −0.077; 6 of 9 significant | | | −0.049 to +0.017; none significant | | |

- GW's colour-cast gain is largely **by construction**: the colour-cast perturbation scales each RGB channel by a
  constant, which Gray World's per-channel division cancels (up to clipping). RF colour-cast ratio 0.70 → 1.00.
- **CL is close to neutral.** It does not improve the lighting families it targets (contrast +0.01 to +0.04, not
  significant; brightness −0.02 to −0.03). Its only significant changes are RF shadow +0.027, LR occlusion +0.043,
  LR noise −0.039 and RF brightness −0.025.
- Per class, clean, all features: GW lowers F1 for most classes (e.g. RF yellow leaf curl 0.589 → 0.485, healthy
  0.755 → 0.667). CL changes are within ±0.06 and mixed.
- Calibration (ECE, all features): GW 0.126–0.150, CL 0.114–0.158, baseline 0.106–0.146. No consistent pattern.

**Reading**:
- (i) In this pipeline, neither normalisation improves clean performance.
- (ii) Gray World buys exact invariance to channel-wise colour casts, but costs clean accuracy and robustness to most
  other perturbations; its effect cannot be separated from the mask shrinkage it causes.
- (iii) CLAHE is roughly neutral and, unlike photometric augmentation (§7), does not improve brightness or contrast
  robustness.

Whether Gray World would help if the mask were kept fixed is untested; that is the pending diagnostic variant.

## 9. Stale, do not cite

PDF findings log §13 (v1/v2 runs of Oct 13–14), including RF 0.660, the v2 robustness drops (e.g. contrast 2.0
−0.391), the v2 importance numbers, the mask-comparison table, and stage 10 metadata 0.533. These came from an older
environment, and their output files were never committed. Use §§3–8 above instead (§10 is a separate benchmark, not a
replacement).

## 10. Supplied-split benchmark on the "Augmented Dataset" (Verified run; Preliminary interpretation)

> **These are results on the supplied split as published, not estimates of generalisation to unseen leaves.** About
> half of the test images have a detected sibling in train/val (below). They are also **not comparable with §3**:
> different images, protocol and split (see the comparison table at the end of this section). Do not report a
> "gain" over the raw-set results.

**Design.**
- Data: `Tomato Dataset/Augmented Dataset`, 7,200 RGB JPEGs, all 224 × 224, 0 corrupt files. The supplied split is
  preserved exactly: train 5,040 / val 1,440 / test 720, i.e. 840 / 240 / 120 per class. The manifest split equals the
  folder for every row.
- No image removed, relabelled, augmented or re-split. The raw-set label exclusions are not applied.
- Pipeline unchanged: the same 105 features (colour 47, texture 38, shape 20), HSV segmentation and 256 px cache (the
  224 px images are upsampled). The same RF / SVM / LR grids, `class_weight: balanced` (a no-op here, since classes are
  equal) and seed 42.
- Protocol (`04_baseline.py`):
  - tune on val, with models fitted on train only;
  - refit the best setting on train + val;
  - score test once;
  - 95% CI by bootstrap over test images (500 resamples).
  It is a single split, so no paired significance tests were run.
- Overlay `runs/p2_supplied/supplied_official.yaml`: official split, `drop_cross_class: false` (keep all images),
  no reference masks (no perturbations).
- Source: `runs/p2_supplied/results/` (`baseline_test_metrics.csv`, `export_metrics_test.csv`,
  `export_per_class_test.csv`, `confusion_*_test_counts.csv`, `sibling_*`).

**Test results** (n = 720):

| Model | Best params (val) | Val macro-F1 | Accuracy | Macro-F1 [95% CI] | Balanced acc. | MCC | ECE |
|---|---|---|---|---|---|---|---|
| RF | max depth 20, sqrt, min leaf 1 | 0.821 | 0.810 | 0.812 [0.784, 0.840] | 0.810 | 0.773 | 0.133 |
| SVM | C 10, gamma scale | 0.838 | 0.813 | 0.815 [0.787, 0.843] | 0.813 | 0.777 | 0.206 |
| LR | C 10 | 0.690 | 0.696 | 0.696 [0.664, 0.732] | 0.696 | 0.635 | 0.063 |

- RF and SVM differ by 0.003 macro-F1, and their CIs overlap almost entirely. **Neither is better on this evidence.**
  LR is clearly lower: its CI does not overlap theirs.
- LR's best C (10) is at the edge of the unchanged grid, so its score may be slightly underestimated.
- RF training macro-F1 is 1.00 at every learning-curve size (`learning_curve_rf.csv`).

**Per-class F1** (test, 120 images per class):

| Class | RF | SVM | LR |
|---|---|---|---|
| Healthy | 0.908 | 0.931 | 0.851 |
| Leaf mold | 0.840 | 0.871 | 0.741 |
| Yellow leaf curl | 0.849 | 0.806 | 0.843 |
| Late blight | 0.794 | 0.814 | 0.598 |
| Septoria | 0.736 | 0.758 | 0.560 |
| Early blight | 0.747 | 0.709 | 0.587 |

**Main confusions** (counts out of 120 per true class, `confusion_*_test_counts.csv`):
- Yellow leaf curl predicted as early blight or septoria: RF 15 + 10, SVM 19 + 14.
- Early blight predicted as septoria: RF 12, SVM 14, LR 26.
- Septoria predicted as late blight: SVM 14, LR 17. LR also confuses late blight with early blight and septoria
  (16 each).

**Detected-sibling analysis** (association, not proof of cause).
- **Definition.** A test image has a *detected sibling* if either rule holds:
  - **name:** same class and same source filename in train or val, after lower-casing and stripping both augmentation
    naming schemes and the Roboflow export id;
  - **content:** any train or val image whose 16 × 16 grey thumbnail correlates ≥ 0.95 with one of the 8
    rotations/flips of the test image.
- The rules are in `scripts/18_export_predictions.py` and `sibling_rules.json`. They were frozen at 01:36:42, before
  stage 4 was run (its metrics were written at 01:38:23).
- **Flags.** 361 / 720 test images have a detected sibling (name 321, content 241).
- **Class mix differs between groups:**

  | | Early blight | Healthy | Late blight | Leaf mold | Septoria | Yellow leaf curl |
  |---|---|---|---|---|---|---|
  | Detected sibling (361) | 47 | 87 | 87 | 9 | 75 | 56 |
  | No detected sibling (359) | 73 | 33 | 33 | 111 | 45 | 64 |

| Model | Detected sibling (n 361): acc. / bal. acc. / macro-F1 | No detected sibling (n 359): acc. / bal. acc. / macro-F1 | Accuracy gap [95% CI] |
|---|---|---|---|
| RF | 0.934 / 0.867 / 0.866 | 0.685 / 0.621 / 0.633 | +0.248 [0.193, 0.304] |
| SVM | 0.936 / 0.868 / 0.867 | 0.688 / 0.652 / 0.649 | +0.248 [0.193, 0.304] |
| LR | 0.734 / 0.682 / 0.658 | 0.657 / 0.591 / 0.594 | +0.077 [0.007, 0.146] |

**Reading.** Detected related-image overlap is strongly associated with higher accuracy: about 25 percentage points
for RF and SVM, and about 8 for LR (a linear model, with less capacity to fit individual images; this explanation is
untested). **Limitations:**
- **Matching is incomplete.** Copies under renamed files, or heavily transformed copies, are not detected. The
  "no detected sibling" subset is **not leakage-free**.
- **The groups differ in class mix** (table above), so part of the gap may reflect class difficulty. Balanced accuracy
  shows a similar gap.
- **Cross-class duplicates remain.** The pipeline finds 2,693 near-duplicate pairs, 61 of them across classes, and a
  read-only check (output not saved) found 2 byte-identical files filed under two classes. All were kept, so the labels
  themselves contain some noise.
- **It is an association.** The analysis does not isolate a mechanism.

**Other audit numbers.** With filename grouping plus near-duplicate merging, the pipeline finds:
- 1,516 source groups;
- 133 groups spanning splits;
- 33.1% of test images with their source in train/val.

That rule is weaker than the sibling rules above. A third of the images have source names not found in `Dataset (raw)`
(read-only check, output not saved: **unverified**). HSV mask mean area is 0.84 (whole-image fallback 7.6%).

**How this differs from §3–§8:**

| | Grouped experiments (§3–§8) | Supplied-split benchmark (§10) |
|---|---|---|
| Images | 813 raw photos (17 removed), sizes vary by class | 7,200 supplied images, about a third with source names not in the raw set (unverified, see above), all 224 px |
| Classes | imbalanced (80–191 per class) | balanced (1,200 per class) |
| Split | 10 grouped 70/15/15 splits; no source group crosses splits | 1 supplied 70/20/10 split; detected siblings across splits for about half of test |
| Statistics | mean ± sd over splits, paired Wilcoxon + Holm | single test set, bootstrap CI only |
| Shortcuts | image size predicts class (metadata model 0.49) | uniform size, so that shortcut is absent |

**Runtime** (wall clock, the team machine used for all Phase 2 runs):
- audit 7 s; manifest + 1.4 GB cache 10 s; EDA 17 s;
- features 100 s; sibling flags 10 s;
- stage 4 108 s (tuning: RF 66 s, SVM 14 s, LR 1 s); export 41 s;
- total about 5 min.

**Integrity** (`export_checks.csv`, `runs/p2_supplied/NOTES.md`):
- **23 / 23 export checks pass:**
  - the SVM probability refit gives exactly the stage-4 SVM predictions;
  - probabilities are finite and sum to 1;
  - accuracy, macro-F1, CI and per-class tables recomputed from the exported predictions equal stage 4's;
  - each train-only refit reproduces its best tuning score.
- **Protection check** before and after the run:
  - config, manifest, exclusions and `configs/` unchanged;
  - 687 existing output files identical;
  - 8,039 dataset files identical;
  - Git HEAD unchanged;
  - the only new repository file is the untracked `scripts/18_export_predictions.py`.

## 11. Task 2: grouped evaluation of the supplied set (Verified run, 2026-11-12; Preliminary interpretation)

**Machine.** A different machine from §§3–10 (Windows, 4 cores, Python 3.11.9, same library versions). It
reproduces the §10 supplied-split benchmark to within 0.002 macro-F1 (RF 0.814, SVM 0.815, LR 0.698;
`runs/p2_supplied_repro`) but not byte-identically, so every comparison in this section uses runs from this machine.
Commands and design: [experiment_log.md](experiment_log.md), 2026-11-12.

**Design (fixed before any model was run).** `scripts/19_group_supplied.py`, `configs/supplied/grouped.yaml`:
- groups = pipeline grouping + the frozen §10 name rule + the frozen §10 content rule (8 rotations/flips,
  r ≥ 0.95), same class only;
- images whose group also appears under another class are dropped: 163 (pipeline rule) + 20 (content rule) = 183;
- 7,017 images remain (1,080–1,200 per class) in 936 groups (86 singletons, largest 49 images);
- groups per class: early blight 133, late blight 121, septoria 207, leaf mold 216, yellow leaf curl 86,
  healthy 173. Close to the raw photo counts (119 / 113 / 195 / 186 / 80 / 137), so a group is approximately one
  source photo;
- protocol unchanged: 10 grouped 70/15/15 splits, seeds 42–51, tuning inside each split, clean images, feature sets
  `all` and `meta`;
- **control:** the same 7,017 images, features and seeds, split at image level (each image its own group).

Share of test images whose detected group has a member in train/val (`supplied_grouping/split_family_check.csv`):
grouped 0% on every split (by construction); image-level control 98.7%; supplied split 50% (361 / 720, §10).

**Test macro-F1, all features** (mean ± sd over 10 splits; test sets of 980–1,040 grouped, 1,051 image-level).
Sources: `runs/p2_supplied_grouped/results/protocol_{grouped,random}/summary_clean.csv`, §10 / this machine.

| Model | Image-level split (control) | Supplied split (§10, this machine) | **Grouped split** |
|---|---|---|---|
| RF | 0.961 ± 0.006 | 0.814 | **0.659 ± 0.058** |
| SVM | 0.972 ± 0.004 | 0.815 | **0.669 ± 0.054** |
| LR | 0.768 ± 0.011 | 0.698 | **0.638 ± 0.050** |
| Metadata only (RF) | 0.275 ± 0.014 | — | 0.265 ± 0.017 |

- The image-level score exceeds the grouped score on **10 of 10 seeds** for every model: by 0.30 for RF and SVM
  and 0.13 for LR (mean difference). The test sets differ between the two protocols, so this is not a paired
  test on the same images.
- Grouped accuracy (RF 0.662, SVM 0.672, LR 0.644) is close to the §10 accuracy on supplied-split test images
  without a detected sibling (0.685 / 0.688 / 0.657). Two independent ways of removing related images give
  similar numbers.
- Grouped: models do not differ after Holm correction (smallest adjusted p 0.15); all features beat metadata on
  10 of 10 splits (adjusted p 0.006).
- Metadata alone is weak here (0.27 vs chance 0.167), as expected: every supplied image is 224 × 224.
- Per-class F1, grouped, RF / SVM / LR: healthy 0.80 / 0.83 / 0.79; yellow leaf curl 0.76 / 0.79 / 0.78; leaf mold
  0.73 / 0.78 / 0.72; late blight 0.65 / 0.59 / 0.55; septoria 0.49 / 0.54 / 0.50; early blight 0.51 / 0.49 / 0.49.
  The same weak pair (early blight, septoria) as on the raw set.
- Grouped scores vary more across splits (sd 0.05–0.06) than image-level ones (0.004–0.011): with about 140
  test groups per split, which photos land in test matters.

**Reading.**
- (i) On this dataset, the split rule changes the headline from about 0.97 (image level) or 0.81 (supplied split)
  to about 0.66 (grouped). Related-image overlap between training and test, not disease recognition, accounts for
  most of the difference.
- (ii) Grouped supplied-set scores (0.64–0.67) are of the same order as the raw-set protocol (0.60–0.62, §3), but
  the two are **not directly comparable**: different images, balanced vs imbalanced classes, and about 7 augmented
  copies per source photo in the supplied training sets.
- (iii) Limitation: the grouping catches only what the three rules detect; copies changed beyond them (crops, strong
  recolouring) may still cross splits, so about 0.66 may still be optimistic. It is not a leak-free estimate.

## 12. Cross-dataset generalisation and explanation transfer (Verified run, 2026-11-12; Preliminary interpretation)

**Machine and in-domain check.** Same machine as §11. The unchanged `13_protocol.py` on our raw set gives here:
SVM and LR identical to §3 on 9 of 10 splits (largest difference 0.003); **RF differs on every split** (mean 0.613
vs 0.600 in §3, largest single-split difference 0.054) even where it chooses the same hyper-parameters. RF is the
model most sensitive to tiny feature differences (same pattern as the §11 reproduction). `scripts/20` reproduces this
machine's `13_protocol.py` exactly (difference 0 for every split and model). Source:
`runs/p3_cross_dataset/results/{protocol_check_all, cross_dataset/indomain_vs_protocol_main.csv}`.

**Design.** `scripts/20_cross_dataset.py`. For each of the 10 grouped splits of §3: tune on validation, refit on
train+val (all 105 features), then score our test split and every external image with the same fitted model. No
tuning, refitting or thresholds on external data. External images identical to any of our 830 raw images (same file
bytes, or 16×16 thumbnail r ≥ 0.95 under the 8 rotations/flips) are scored separately as "overlap".

**External sets** (configs in `configs/external/`; downloaded 2026-11-12):

| Set | Source | Images used | Classes scored |
|---|---|---|---|
| PlantDoc | Singh et al. 2020, GitHub, CC BY 4.0; web images | 580 (310 independent, 270 overlap) | all six |
| Nazirpur | Mendeley 10.17632/74zvtzx9pr.1, Bangladesh field, leaf crops | 378 | septoria 91, healthy 287 |
| Nazirpur leaf curl | same | 264 | "Leaf_Curl" as yellow leaf curl virus: **uncertain mapping** |
| Brahmanbaria | Mendeley 10.17632/rnbsw72zb5.1, Bangladesh field | 1,035 leaf crops (428 healthy, 607 diseased) | healthy vs any disease |
| Dinajpur (excluded) | Mendeley 10.17632/bpfd9cns5g.2 | — | boxes mark lesions, not leaves; class numbers could not be matched to names (box counts do not match the paper) |

**Finding 1: about a third of our raw set is PlantDoc.** 263 of the 830 raw images (31.7%) are identical to a PlantDoc
image (257 byte-identical). 25 of the 270 overlapping PlantDoc images carry a different label in our set, mostly
**19 PlantDoc "early blight" images filed as septoria** in ours, consistent with the 21 septoria images with
early-blight file names (§2). Which label is right is not established (PlantDoc labels are unverified too).
Sources: `cross_dataset/plantdoc/overlap_with_raw.csv`, `overlap_summary.json`, `overlap_label_conflicts.csv`
(derived from the former).

**Finding 2: performance does not transfer** (mean ± sd over 10 splits; `cross_dataset/<set>/summary.csv`,
`per_class_f1.csv`, `confusion_pooled.csv`). Macro-F1 is over the classes present in each set.

| Test data | RF | SVM | LR |
|---|---|---|---|
| Our test split (this machine) | 0.613 ± 0.046 | 0.615 ± 0.026 | 0.595 ± 0.044 |
| PlantDoc, overlapping our raw images (n 270) | 0.810 ± 0.016 | 0.653 ± 0.152 | 0.564 ± 0.077 |
| **PlantDoc, independent** (n 310, 6 classes; chance 0.17) | **0.290 ± 0.018** | **0.330 ± 0.028** | **0.329 ± 0.012** |
| **Nazirpur** (n 378, septoria + healthy) | **0.245 ± 0.023** | **0.234 ± 0.031** | **0.213 ± 0.017** |

- PlantDoc independent, accuracy 0.32–0.34. Per-class F1: yellow leaf curl 0.50–0.59 and septoria 0.38–0.45 transfer
  partly; healthy 0.17–0.31, early blight 0.17–0.19 and leaf mold 0.12–0.25 hardly at all.
- Nazirpur: healthy recall 0.21–0.27, septoria recall 0.02–0.10. RF predicts late blight for 49% of healthy and 47%
  of septoria leaves (pooled over splits).
- Nazirpur leaf curl (uncertain mapping): recall as yellow leaf curl 0.05–0.10; RF predicts late blight for 48%.
- Brahmanbaria: **healthy recall RF 0.03, SVM 0.17, LR 0.12**. "Disease detection" is 0.88–0.93 only because the models
  rarely predict healthy at all: RF calls 6.7% of diseased leaves healthy, more often than it calls healthy leaves
  healthy (2.8%).
- Overlapping PlantDoc images score far higher than independent ones (RF 0.81 vs 0.29): most of them were in the
  training pool of every split.

**Finding 3: the shift is in the feature distributions, not a segmentation failure** (`cross_dataset/domain_shift.csv`,
`scripts/22_domain_shift.py`). A random forest separating our images from external images on the 105 features
reaches ROC AUC 0.85 (PlantDoc independent) and 0.98–0.99 (Bangladeshi sets). Leaf-mask area (0.64–0.88 vs 0.83) and
the whole-image fallback rate (5–9% vs 6%) are similar to ours. Transfer is worst where the shift is largest.

**Finding 4: explanations do not transfer** (`importance_group.csv`, `importance_agreement.csv`,
`importance_own_split_agreement.csv`).
- Grouped permutation importance (drop in true-class probability), RF, our test → PlantDoc independent: colour
  0.177 → 0.039, texture 0.092 → 0.037, shape 0.042 → 0.021. On Nazirpur all groups are about 0 (−0.02 to 0.03).
  The features that carry the signal on our data carry little or none on external data.
- Per-feature ranking, RF only (SVM and LR rankings are unstable even between our own splits: Spearman 0.18 and
  0.48, so their transfer cannot be judged): agreement between our splits 0.87 (top-10 Jaccard 0.76); our test vs
  PlantDoc 0.61 ± 0.08 (Jaccard 0.39); vs Nazirpur −0.01 (Jaccard 0.26).
- RF top features on our test: LAB-B std, GLCM correlation d3/d1, hue histogram bin 1, HSV-S std. GLCM correlation and
  hue bins stay near the top on PlantDoc; LAB-B std drops to tenth.

**Reading.**
- (i) The in-dataset score (about 0.61) does not describe performance on other tomato datasets: 0.21–0.33 macro-F1,
  and healthy leaves from other sources are mostly classified as diseased.
- (ii) A model that does not work on a dataset has no meaningful explanation there: importances collapse towards 0.
  Explanations measured on our own split are therefore explanations of a dataset-specific model.
- (iii) Limitations: label definitions and photo style differ between datasets (PlantDoc labels unverified; Nazirpur
  images are tight leaf crops; leaf curl mapping uncertain); our training set already contains 263 PlantDoc images,
  so PlantDoc is not fully out-of-source; duplicates inside the external sets were not removed; clean images only.

## 13. Feature-level (model-free) explanation across datasets (Verified run, 2026-11-12; Preliminary)

`scripts/23_feature_explainability.py`; outputs `runs/p3_cross_dataset/results/feature_explain/`. No classifier is
trained: each of the 105 features is examined directly.

**Healthy vs diseased**, the one contrast present in all four datasets (our raw set 137 / 676; PlantDoc independent
46 / 264; Nazirpur 287 healthy / 355 septoria + leaf curl; Brahmanbaria 428 / 607). Effect = feature ROC AUC for
"diseased" − 0.5 (+ means higher in diseased leaves).
- **2 of 105 features are consistent** (same sign in all four datasets and |effect| ≥ 0.1 in each):
  `color_hue_hist_1`, the share of leaf pixels with OpenCV hue 15–30 (yellow-orange), +0.15 / +0.24 / +0.32 / +0.11;
  and `color_lab_A_skew` (skewness of the green–red channel), −0.15 / −0.16 / −0.16 / −0.27. Both correspond to
  visible yellowing/browning. 11 of 105 keep the same sign everywhere at any size.
- The strongest features in our raw set are **not** consistent: HSV-S skew (−0.26 raw, +0.20 PlantDoc), LAB-B mean
  (+0.22 raw, −0.18 PlantDoc, −0.15 Brahmanbaria), RGB-B mean, LAB-B std, saturation bin 2 all change sign.
- Rank agreement of the signed effect profiles: raw vs PlantDoc 0.22, raw vs Nazirpur 0.02, **raw vs Brahmanbaria
  −0.51**; Nazirpur vs Brahmanbaria 0.40. The two Bangladeshi sets agree with each other more than with ours.
- By group: colour 2 / 47 consistent, texture 0 / 38, shape 0 / 20.

**Six classes**, our raw set vs PlantDoc independent: Kruskal–Wallis η² per feature, rank agreement 0.30, top-10
overlap (Jaccard) 0.11. Raw top: LAB-B std, GLCM correlation d3, LAB-A mean, HSV-S skew. PlantDoc top: hue bin 1,
LBP P16R2 bins 12–13, LAB-A mean, GLCM dissimilarity d3.

**Distribution shift per feature** (KS statistic vs our raw set, mean by group): texture shifts most for PlantDoc
(0.28) and Nazirpur (0.26); colour shifts most for Brahmanbaria (0.30).

**Reading.** Feature-level explanation does not need a working model, and it shows which hand-crafted features carry
disease information that holds across sources (very few, both yellow/brown colour cues) and which are specific to our
dataset (the raw set's strongest features). Limitations: effect sizes on small external subsets are noisy (no CIs
yet); "diseased" mixes different diseases per dataset; healthy vs diseased only for the four-way comparison.
