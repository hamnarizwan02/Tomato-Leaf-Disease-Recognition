# Verified results summary

The single source of numbers for the mid-semester report and the IEEE paper. Every number here was checked against the
file named next to it. Copy numbers from this page, not from the PDF findings log.

**Status labels**
- **Verified**: reproduced in the Phase 2 environment from the committed manifest, or checked against the source file.
- **Preliminary**: computed and verified, but not yet a conclusion (single split, untested mechanism, or pending checks).
- **Phase 1 env**: committed Phase 1 output, internally consistent, not re-run in the Phase 2 environment.
- **Stale, do not cite**: superseded numbers.

Run folders live outside Git in `~/Desktop/MLPROJECT/runs/`: `p2_base` (Phase 1 reproduction) and `p2_augmentation`
(experiment A). The commands for each run are in [experiment_log.md](experiment_log.md).
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
| Supplied "Augmented Dataset" | 7,200 images at 224 px with its own split. **Not used**: only 48% link to a raw source by filename, 26% by image content (sample of 400); 1 source sits in both its train and test; 83 files descend from the 17 removed images | linkage analysis, 2026-10-26 |

## 2. Data-quality findings (Verified facts, unresolved)

The labels have not been changed and no image has been excluded beyond the 17 above.

| Finding | Evidence | Status |
|---|---|---|
| Suspected label noise | 21 septoria images have "early blight" in the filename (19 name only early blight, 2 name both); 4 are in the seed-42 test set. Also `early_blight/gray-mold-on-tomato-leaf.jpg`, `late_blight/tomato.early.blightH.jpg`, `healthy/curling-issues.jpg`. A filename is not proof | awaiting blind visual review |
| Label audit (RF, out-of-fold, 3 × 5 grouped CV) | 308 / 813 wrong out of fold; 139 confident-learning flags; 7 with margin > 0.5. Flags are candidates only | `label_audit_summary.json` (identical in both environments) |
| Capture-session bursts | 154 phone photos taken 2025-11-02 between 15:29 and 16:20 (healthy 66, early blight 38, yellow leaf curl 31, leaf mold 11, late blight 8). Rule: same class, different group, timestamps within 60 s gives 404 cross-split pairs; within 5 min gives 1,153. All 21 timestamped test images of split 0 have a same-class train/val image taken within 5 min. One pair inspected: same phone, 1 s apart, same plant | timestamp analysis; not yet a sensitivity run |
| Rotated copies in different groups | The duplicate check is flip-aware but not rotation-aware. 4 exact rotated-copy pairs (r ≥ 0.999) sit in different groups and are separated in 4–8 of the 10 splits | rotation-aware scan. Shared by all arms, so arm comparisons are unaffected |
| Non-deterministic GrabCut | Background-replacement masks varied between runs (26 of 40 images). Fixed by seeding OpenCV before each call (`src/segment.py`). Background conditions now differ from Phase 1 by at most ±0.007 macro-F1 averaged over splits | `runs/p2_base/NOTES.md` |

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
- Overfitting: RF training macro-F1 1.00 at every learning-curve size; validation plateaus at 0.54–0.58.

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
- **A2 vs A1** (photometric part only):
  - brightness +0.05 to +0.11 and contrast +0.06 to +0.13 (significant, mostly 10/10 wins);
  - shadow +0.033 for RF and +0.041 for SVM (significant);
  - colour cast −0.028 for RF (significant);
  - blur and down-then-up roughly 0, so **those gains come from the geometric part**;
  - clean: no difference.
- Calibration (ECE, all features): SVM improves 0.146 → 0.110 (A2); LR worsens 0.106 → 0.125.

**Reading**:
- (i) Photometric augmentation reliably improves robustness to the lighting families it was trained on, including
  test levels well outside the training range.
- (ii) It does not change clean accuracy detectably.
- (iii) It costs RF robustness to colour casts and noise.
- (iv) RF gains on blur and rescaling are real in A2 but come from the geometric part. The mechanism (rotation
  resampling slightly smoothing copies) is a hypothesis.

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

- **GW lowers clean macro-F1** by 0.05–0.07 for all models (baseline wins 8–9 of 10 splits). Raw p 0.006–0.010, but
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
environment, and their output files were never committed. Use §§3–7 above instead.
