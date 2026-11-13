# Findings log: Phase 1 update (Oct 15)

This text is meant to be pasted into *Project Findings & Methods Log*. Each part names its target section.
All numbers come from `results/protocol_main/`, `results/protocol_min_side500/`, `results/shortcut/`,
`results/label_audit*`, `results_res_matched/` and `results_reencode/` (runs of Oct 15).

---

## For section 13: new subsection "Phase 1 results (Oct 15)"

**What was run.** The 813 images and the committed v2 manifest were used. The new protocol uses 10 grouped splits
(70/15/15 per class, seeds 42 to 51; split 0 is the committed seed-42 split). In every split, hyper-parameters are
tuned on that split's validation set, the model is refitted on train+val and the test set is scored once. Each test
set is also scored under 50 perturbation conditions, and background replacement uses one fixed GrabCut mask per image.
Software: scikit-learn 1.9.1, CPU only. On split 0 the SVM (0.656) and logistic regression (0.633) reproduce v2 exactly;
the random forest gives 0.665 against 0.660 in v2, and stage 04 gives the same 0.665 on this machine, so the small
difference comes from the library version, not from the new code.

**Clean test scores (mean ± std over 10 splits).**

| Feature set | RF macro-F1 | SVM macro-F1 | LR macro-F1 |
|---|---|---|---|
| All (105) | 0.600 ± 0.046 | 0.615 ± 0.026 | 0.596 ± 0.044 |
| Colour + texture (85) | 0.611 ± 0.041 | 0.630 ± 0.034 | 0.598 ± 0.031 |
| Colour (47) | 0.541 ± 0.050 | 0.543 ± 0.048 | 0.495 ± 0.037 |
| Texture (38) | 0.483 ± 0.044 | 0.515 ± 0.047 | 0.518 ± 0.049 |
| Shape (20) | 0.408 ± 0.042 | 0.382 ± 0.046 | 0.380 ± 0.041 |
| Metadata only (6) | 0.487 ± 0.035 | 0.424 ± 0.034 | 0.351 ± 0.042 |

The other scores for all features (RF / SVM / LR) are: accuracy 0.608 / 0.628 / 0.609; balanced accuracy
0.602 / 0.616 / 0.604; MCC 0.527 / 0.552 / 0.530; expected calibration error 0.142 / 0.146 / 0.106; Brier score
0.529 / 0.562 / 0.520.

With tuning inside each split, the 10-split random forest mean (0.600) agrees with the earlier 5-split estimate
(0.606), and seed 42 (0.665) is again near the top of the range. That confirms that the single-split v2 number
flatters the model.

**Paired tests (Wilcoxon signed-rank on matched splits, Holm correction within each family).**
- *Models:* no difference between RF, SVM and LR survives the correction, for any feature set (smallest adjusted
  p = 0.35). The largest gap is the SVM against LR on all features (+0.020, SVM better on 8 of 10 splits).
- *Feature sets:* colour, texture and shape each fall below all features for every model (adjusted p 0.03 to 0.04,
  all features better on 9 or 10 of 10 splits). Colour + texture against all features is +0.011 (RF), +0.015 (SVM)
  and +0.003 (LR), all not significant. Shape still adds nothing, as in v2.
- *Against the metadata-only model (RF, 0.487):* all features score higher by 0.11 to 0.13 for every model, on 10 of
  10 splits (adjusted p 0.035). Texture alone is no better than metadata (RF 0.483 against 0.487), and shape is worse.

**Source shortcut.**
- *Metadata only, over 10 splits:* the random forest on image size, aspect ratio and compression reaches macro-F1
  0.487 ± 0.035. That is lower than the single-split 0.533 of stage 10 but still about three times chance (0.167).
- *(c) Features that track image size:* no feature reaches |Spearman| 0.5 with log pixel count or bytes per pixel.
  The highest are GLCM correlation at distance 1 (0.43), LBP bins (0.38 to 0.41) and GLCM correlation at distance 3
  (0.40). GLCM correlation, the suspect named in problem 14, is the most size-linked feature, but at the planned
  threshold of 0.5 the pruned set equals the full set, so this control removes nothing (problem 28).
- *(b) Source clusters:* KMeans on size, aspect ratio, compression and file-name pattern gives 8 clusters (k chosen by
  silhouette 0.47; every source group is kept in one cluster). Classes are unevenly spread over the clusters. Cluster 7
  is 106 of its 128 images leaf mold, mostly camera-named files of 0.8 MP with aspect ratio 0.75. Under the main
  protocol, test images in cluster 7 are classified with accuracy 0.91 to 0.96, against 0.28 to 0.70 in the other
  clusters.
- *Leave-one-cluster-out:* when a whole cluster is held out, accuracy falls to 0.47 to 0.50 and macro-F1 over the
  classes present to 0.34 to 0.38 (RF / SVM / LR, all features; 0.60 within distribution). Every class still appears
  in training for every held-out cluster. The metadata model falls to 0.14 to 0.18, as expected, since a source it has
  not seen gives it nothing to match. The models therefore lose about a third of their macro-F1 on an unseen source.
  This is the strongest evidence so far that part of the score is source-specific. Caveat: the clusters are a proxy
  for source, and holding one out also changes the class balance.
- *(a) and (d) Preprocessing variants:* these use the same manifest, the same 10 splits and tuning inside each split,
  and are scored on clean images only. Each variant rebuilds its image cache from the raw photos.

| Variant | RF all | SVM all | LR all | RF texture | RF metadata |
|---|---|---|---|---|---|
| Main run | 0.600 ± 0.046 | 0.615 ± 0.026 | 0.596 ± 0.044 | 0.483 ± 0.044 | 0.487 ± 0.035 |
| V1: same effective resolution (128 px, back to 256) | 0.598 ± 0.035 | 0.616 ± 0.056 | 0.595 ± 0.042 | 0.449 ± 0.046 | 0.487 ± 0.035 |
| (d) identical JPEG re-encoding (quality 85) | 0.603 ± 0.043 | 0.612 ± 0.042 | 0.606 ± 0.043 | 0.443 ± 0.049 | 0.487 ± 0.035 |
| V2: both sides at least 500 px (550 images) | 0.569 ± 0.030 | 0.608 ± 0.043 | 0.603 ± 0.041 | 0.500 ± 0.049 | 0.455 ± 0.056 |

  - Equalising resolution or compression does not change the score of any model on all features (every change is
    0.01 or less), and all features still beat the metadata model by 0.11 to 0.13 on 10 of 10 splits (adjusted
    p 0.035). Only texture alone loses 0.03 to 0.04, as expected for LBP and GLCM.
  - The 550 large images keep at least 52 images per class. On them the feature models score the same as before
    (SVM 0.608, LR 0.603; RF 0.569, within one standard deviation), and they beat the metadata model by 0.12 to 0.15
    on 10 of 10 splits.
  - The metadata model still reaches 0.455 on the large images alone, so the source signal is not only image size.
    Aspect ratio and compression still separate the classes.
  - In this subset about 18 features, mostly LBP bins, pass the 0.5 correlation threshold inside the training rows and
    are pruned. The pruned set scores the same or slightly higher (SVM 0.622, LR 0.602, RF 0.578).
  - *Reading:* the score is not explained by resolution or compression artefacts in the pixels. Together with the
    leave-one-cluster-out drop, this points to source-specific image content (background, framing, camera colour
    response) as the remaining shortcut. That is a hypothesis for Phase 2 (mask quality, colour constancy) and
    Phase 3 (external test set), not a result.

**Robustness (perturbed macro-F1 divided by clean, mean over 10 splits, all features).**

| Perturbation | RF | SVM | LR |
|---|---|---|---|
| Downscale then upscale (0.5, 0.25, 0.125) | 0.42 | 0.32 | 0.47 |
| Gaussian blur (sigma 1 to 3) | 0.43 | 0.31 | 0.48 |
| Motion blur (5 to 15 px) | 0.52 | 0.53 | 0.73 |
| Combined pairs | 0.58 | 0.46 | 0.65 |
| Contrast | 0.61 | 0.59 | 0.65 |
| Colour cast | 0.70 | 0.61 | 0.70 |
| Noise | 0.78 | 0.68 | 0.80 |
| Shadow | 0.81 | 0.77 | 0.90 |
| Brightness | 0.83 | 0.70 | 0.79 |
| Scale | 0.86 | 0.89 | 0.99 |
| Occlusion | 0.87 | 0.67 | 0.70 |
| Gamma | 0.87 | 0.84 | 0.90 |
| Background (fixed GrabCut mask) | 0.92 | 0.89 | 0.89 |
| JPEG quality 50 to 15 | 0.95 | 0.87 | 0.87 |
| Rotation | 0.96 | 0.96 | 0.96 |

- *Worst case:* downscale 0.125 is the worst condition for the random forest on all 10 splits (macro-F1
  0.146 ± 0.028, ratio 0.25). Gaussian blur sigma 3 gives 0.174, motion blur 15 gives 0.225 and contrast 0.5 gives
  0.263. The mean worst-case ratio is 0.25 (RF), 0.15 (LR) and 0.08 (SVM).
- *What breaks the models:* loss of resolution and sharpness hurts most, more than the lighting changes that
  dominated v1 and v2. Contrast and colour cast come next. This is consistent with texture features that depend on
  native resolution, and so with the source shortcut. The sensitivity to downscaling is what the resolution-matched
  variant (V1) is meant to test.
- *Background:* now that it is measured with one fixed GrabCut mask, it costs only 8 to 11% (problem 23 fixed).
  Rotation and JPEG compression are nearly harmless.
- *Model ranking under perturbation:* the SVM is the least robust model under most perturbations, a conclusion that
  now rests on 10 splits instead of one (problem 22).
- *Colour + texture* behaves like all features: its ratios are within about 0.05 for most perturbations.

**Calibration and selective prediction (all features).** Expected calibration error is 0.11 to 0.15. Accuracy on the
50% most confident test images is 0.81 (RF), 0.78 (SVM) and 0.80 (LR), against 0.61 to 0.63 on all images, so
confidence is informative even though the probabilities are not well calibrated.

**Per-class F1 (all features, mean over 10 splits, RF / SVM / LR).** Leaf mold 0.77 / 0.79 / 0.80; healthy
0.76 / 0.76 / 0.67; yellow leaf curl 0.59 / 0.60 / 0.61; septoria 0.52 / 0.59 / 0.56; late blight 0.53 / 0.49 / 0.51;
early blight 0.44 / 0.46 / 0.43. Leaf mold, the best class, is also the class that sits almost entirely in one source
cluster.

**Label audit.** Out-of-fold probabilities come from a random forest, averaged over three repeats of 5-fold grouped
CV. 308 of 813 images are misclassified out of fold. Confident learning flags 139 images: septoria 35, early blight
33, leaf mold 26, yellow leaf curl 17, late blight 14 and healthy 14. The most frequent suspected swap is septoria
labelled but predicted early blight (19). Only 7 flags have a margin above 0.5. With an out-of-fold accuracy near
0.62, most flags are expected to be model errors, not label errors, so nothing was removed. The 50 most confidently
wrong images are in `label_audit_top.png` for review. Confirmed errors go in `data/label_exclusions.csv`.

---

## For section 11: changes to the problems table

| # | New resolution text |
|---|---|
| 13 | v2 removed the 9 cross-class pairs. Phase 1 adds a systematic audit (stage 15: 139 confident-learning flags, 7 with margin above 0.5). Review by eye is pending; confirmed errors are listed in `data/label_exclusions.csv`, which drops the whole source group. |
| 14 | Controls done in Phase 1. Over 10 splits metadata alone gives macro-F1 0.487 ± 0.035, and all features beat it by 0.11 on every split. Equalising resolution (V1) or compression (d) and restricting to images at least 500 px (V2) leave the feature scores unchanged, so resolution and compression artefacts do not explain them. Leave-one-cluster-out still lowers macro-F1 from 0.60 to 0.34 to 0.38, and leaf mold is concentrated in one source cluster: a source shortcut remains, most likely through image content. Open: test it with colour constancy, mask quality and an external test set. |
| 16 | Done. Background replacement uses one cached GrabCut mask per image, whatever mask the features use. The background now costs 8 to 11% of macro-F1. |
| 21 | Done. 10 repeated grouped splits with tuning inside each split, mean ± std for every number, paired Wilcoxon tests with Holm correction. No model difference is significant. |
| 22 | Done. Perturbed features are extracted for all images, so robustness is scored on all 10 splits. The SVM is the least robust model, and downscaling is the worst perturbation on every split. |
| 23 | Done (see 16). |

New rows:

| # | Problem | Seen on | Why it matters | Resolution |
|---|---|---|---|---|
| 28 | On all 813 images no feature reaches |Spearman| 0.5 with image size or compression (highest: GLCM correlation d1, 0.43), so the planned pruned set equals the full set. | Real data (Phase 1) | Control (c) as specified removes nothing on the full set, which does not prove the features are free of the shortcut. | Report the correlation table. On the images at least 500 px the same rule prunes about 18 features (mostly LBP) with no loss in score. The resolution-matched and re-encoded variants also show no loss. |
| 29 | Performance drops on an unseen source cluster: macro-F1 0.34 to 0.38 against 0.60 within distribution. | Real data (Phase 1) | Part of the in-distribution score does not transfer, and per-class results partly reflect the source (leaf mold). | Report leave-one-cluster-out next to the main protocol. Run the variants. Use this as the motivation for the external test set in Phase 3. |

---

## For section 14: reproducibility additions

- New scripts: `12_extract_perturbed.py` (perturbed features for all images), `13_protocol.py` (repeated nested
  protocol), `14_shortcut_controls.py` (clusters, leave-one-cluster-out, correlations) and `15_label_audit.py`.
- New modules: `src/protocol.py`, `src/metadata.py` and `src/stats.py`. `src/perturb.py`, `src/evaluate.py` and
  `src/features.py` were extended.
- `work/features/clean_all.npy` and `work/features/perturbed/all/` hold features for every image in manifest order.
  They record what they were built from and are refused when stale.
- New settings in `config.yaml`: `protocol`, `shortcut`, `perturbations_extended`, `perturbation_combos` and
  `perturb_settings.reference_mask`. New variant configs: `configs/res_matched.yaml` and `configs/reencode.yaml`.
- `python run_pipeline.py variants` runs the three shortcut controls. Results: `results_res_matched/protocol_res_matched/`,
  `results_reencode/protocol_reencode/` and `results/protocol_min_side500/`.
- Run times on an 8-core laptop: stage 12 about 9 minutes, stage 13 about 7 minutes (about 40 s per split), stage 14
  about 2 minutes, stage 15 under a minute, and the three variants about 25 minutes together.

---

## For section 15: status and deviations from the plan

**Done in Phase 1:**
- stage 1, the evaluation protocol (10 grouped splits; 5x3 grouped CV is available with `--mode repeated_cv`);
- stage 2, all four source-shortcut controls: (a) V1 and V2, (b) source clusters, (c) correlation pruning and
  (d) re-encoding, plus the metadata reference row;
- stage 3, the audit (manual review pending);
- stage 4, the perturbations (50 conditions, fixed reference mask);
- stage 11, the scores.

**Still open from Phase 1:** reviewing the label-audit flags by eye. The variants were scored on clean images only;
their robustness under perturbation was not measured.

**Deviations from the plan:**
- *Reference mask:* GrabCut after the HSV gate, computed once and cached; no hand-drawn masks yet.
- *SVM probabilities:* Platt scaling fitted on internal cross-validated decision values. The predictions are the
  plain SVM's, so the SVM macro-F1 is unchanged.
- *Source clusters:* each source group is given the majority cluster of its images, so resized copies of a photo
  cannot leak across leave-one-cluster-out folds.
- *Correlation pruning:* computed on the train+val rows of each split only.
- *Paired tests:* the splits share training images, so the Wilcoxon p-values are approximate. With 10 splits the
  smallest attainable p is 0.002.
