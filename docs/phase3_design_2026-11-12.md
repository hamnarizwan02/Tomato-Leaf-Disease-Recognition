# Phase 3 design: multi-source evaluation and feature-level explanation (fixed 2026-11-12, before any Phase 3 run)

Motivation: verified_results §§11–13. Models trained on one source do not transfer (0.21–0.33 macro-F1 elsewhere),
model-based explanations collapse with them, and only 2 of 105 hand-crafted features separate healthy from diseased
leaves consistently across four datasets. Phase 3 asks which features — formula-based or model-extracted — carry
disease information that holds across image sources, and whether models built on them transfer better.

## Research questions (revised)
- **RQ1′** How well do classical models trained on several sources recognise disease on an unseen source?
- **RQ3′** How large is the gap between seen-source and unseen-source performance, per feature family?
- **RQ4′** Which features (hand-crafted formulas, lesion-relative formulas, model-extracted concepts) have effects that
  are consistent across sources, and does restricting a model to them improve transfer?

## Sources (each is one group; `scripts/24_build_sources.py`, `configs/sources.yaml`)
| Source | Origin | Six-class labels | Binary (healthy / diseased) |
|---|---|---|---|
| `pk_raw` | our raw set, images NOT identical to any PlantDoc image | all six | yes |
| `pd_raw` | our raw set, images identical to a PlantDoc image (our labels kept) | all six | yes |
| `plantdoc` | PlantDoc tomato, images not identical to any image of a higher-priority source | all six | + bacterial spot, mosaic, spider mites as diseased |
| `plantwild` | PlantWild v1 tomato (web, in the wild), same dedup rule | all six | + bacterial spot, mosaic |
| `tomato_village` | Tomato-Village variant a (field, Rajasthan), seeded sample | early blight, late blight, healthy | + leaf miner, spotted wilt virus (nutrient-deficiency classes excluded) |
| `nazirpur` | Bangladesh, Nazirpur (field, leaf crops) | septoria, healthy | all disease folders as diseased |
| `brahmanbaria` | Bangladesh, Brahmanbaria leaf crops (scripts/21) | — | yes |
| `plantvillage` | PlantVillage (lab), seeded sample of 150 per class (50 per other disease) | all six | + 4 other diseases |
Excluded: the supplied Augmented Dataset (derived from our raw photos, not an independent source); Dinajpur (labels are
lesions; class ids unmatched). Uncertain mappings (Nazirpur leaf curl, Nazirpur "Blight_Leaf") enter the binary task only.

**Cross-source duplicates.** 16×16 grey thumbnails, 8 rotations/flips, r ≥ 0.95, or identical bytes (the frozen §10
rule). A duplicate pair across sources keeps the copy in the higher-priority source (pk_raw/pd_raw > plantdoc >
plantwild > tomato_village > nazirpur > brahmanbaria > plantvillage) and drops the other. Label conflicts are recorded,
not resolved.

## Features (all computed from the same 256×256 cache and HSV leaf mask unless stated)
1. **hand** — the existing 105 colour/texture/shape features.
2. **lesion** — new leaf-relative features (`src/features_lesion.py`): yellow, brown, dark and pale fractions of the
   leaf; lesion-minus-healthy-tissue differences in LAB; chromaticity statistics; lesion count and size relative to
   the leaf. Designed to be insensitive to global illumination and camera colour. Definitions fixed in the module.
3. **clip_concepts** — model-extracted, named features: cosine similarity between the CLIP ViT-B/32 image embedding of
   the whole image and 24 fixed text concepts (`configs/concepts.yaml`): 14 symptom concepts and 10 context/shortcut
   concepts (background, hand, blur, lighting, watermark…). Concept list fixed here, before any score is seen.
4. **clip_embed** — the 512-d CLIP image embedding (not interpretable; performance reference).
5. **resnet_embed** — ImageNet ResNet-50 global-pooled 2,048-d embedding (not interpretable; reference).
Deep features are computed on the whole image (no mask), resized/centre-cropped as each model expects.

## Evaluation (`scripts/26_loso.py`)
- **Tasks**: T1 healthy vs diseased (all 8 sources); T2 six classes (sources with six-class labels; a held-out source is
  scored on its present classes).
- **Leave-one-source-out (LOSO)**: train on all other sources, test on the held-out one. Hyper-parameters chosen on a
  validation set made of 15% of each training source (grouped by within-source near-duplicate group), refit on train +
  validation. Nothing is tuned on the held-out source.
- **Seen-source reference**: the same pooled data split 70/15/15 within every source (grouped), 5 seeds. The
  seen − unseen difference is the transfer gap.
- Models: RF, RBF SVM, logistic regression (class_weight balanced), small grids fixed in the script.
- Metrics: balanced accuracy and macro-F1 over present classes; T1 also ROC AUC, healthy recall, disease recall.

## Explanation (`scripts/27_explain_multisource.py`)
1. **Formula-level effects** (hand + lesion): per source, healthy-vs-diseased effect (AUC − 0.5) with 1,000-resample
   bootstrap 95% CI. A feature is **source-stable** if its CI excludes 0 with the same sign in ≥ 75% of the sources
   that have ≥ 30 images of each class.
2. **Model-extracted concepts** (clip_concepts): the same analysis per concept; symptom vs context concepts compared.
3. **Glass-box model**: L1 logistic regression on concepts (and on hand + lesion); coefficients per LOSO fold; sign
   stability across folds is the model-level explanation.
4. **Explanation-guided selection**: in each LOSO fold, select source-stable features using the training sources
   only, fit RF and L1-LR on them, score the held-out source. Compared with the same models on all features.
5. **Model importance vs formula stability**: RF permutation importance on the held-out source, correlated with the
   formula-level stability of each feature.

## What would change the conclusions
- If unseen-source performance with deep embeddings is far above hand-crafted features, the transfer failure is a
  feature problem, not a data problem.
- If concept and lesion features close most of that gap, interpretable features are sufficient.
- If explanation-guided selection improves unseen-source scores, feature-level explanation is useful, not only
  descriptive.

## Amendment 1 (2026-11-12, before any Phase 3 model was trained): two extractor arms
Requested by the team: compare a **classical learned extractor** with a **deep extractor**, with classical ML for all
training and testing. Deep networks are used **frozen, as feature extractors only** (no training, no fine-tuning).
- Formula features (hand 105 + lesion 20) unchanged.
- **K-means arm (new, classical unsupervised ML; `src/codebook.py`)**: per image a seeded sample of 1,500 leaf pixels;
  colour descriptor = CIELAB; texture descriptor (8-d) = |Sobel x/y| and Laplacian-of-Gaussian at sigma 1 and 3 and
  local standard deviation (5×5, 9×9) of the contrast-normalised L channel. MiniBatchKMeans codebooks: 24 colour words,
  32 textons; an image becomes the share of its sampled pixels nearest each centroid. In LOSO the codebooks are
  re-learned in every fold from the training rows only (label-free). For the descriptive effect analysis one codebook is
  learned on all images without labels. Colour words are shown as colour swatches.
- **Deep arm**: CLIP ViT-B/32 concepts (the 24 concepts above, unchanged), CLIP embedding, ResNet-50 embedding (PCA 256).
- Feature sets in `26_loso.py`: hand, lesion, hand+lesion, colour_words, textons, kmeans, hand+lesion+kmeans, concepts,
  clip, resnet. Explanation pools in `27`: formula, kmeans, concepts, formula+kmeans, all.
- Comparison of interest: unseen-source (LOSO) performance and transfer gap per arm, and whether named features
  (formula, colour words, concepts) keep consistent effects across sources.
