# Experiment log

What was run, how, with which seeds, and what went wrong. Detailed per-run notes (hashes, every check) are in each run
folder's `NOTES.md`. Numbers are summarised in [verified_results.md](verified_results.md).

**Conventions**
- Code: branch `phase2` (uncommitted).
- Data: `~/Desktop/MLPROJECT/Tomato Dataset/Dataset (raw)`. Read-only, never in Git.
- Outputs: `~/Desktop/MLPROJECT/runs/<run>/{work,results,logs}`, outside Git. Committed `results/` is never written.
- Manifest: committed `data/manifest.csv` (sha256 `b819d160…da77f`, unchanged).
- Seeds: global 42; protocol splits 42–51; perturbations and augmentations seeded per (seed, image, condition/copy).

Environment variables used for every run:
```bash
export TOMATO_DATA_ROOT="$HOME/Desktop/MLPROJECT/Tomato Dataset/Dataset (raw)"
export TOMATO_WORK_DIR="$HOME/Desktop/MLPROJECT/runs/<run>/work"
export TOMATO_RESULTS_DIR="$HOME/Desktop/MLPROJECT/runs/<run>/results"
# TOMATO_CONFIG=configs/<overlay>.yaml only for variant runs; TOMATO_MANIFEST never set
```

## Phase 1 (Oct 13–15, previous environment, see the PDF findings log)
v1 and v2 runs of stages 00–11, then Phase 1 stages 12–15 and the shortcut variants. Committed outputs: `results/`,
`results_res_matched/`, `results_reencode/`. The outputs of stages 07–11 and of the mask comparison were never
committed.

## 2026-10-25: audit and Phase 1 reproduction (`runs/p2_base`)
| Step | Command | Outcome |
|---|---|---|
| Image cache | `01_prepare_data.py --cache-only` | (813, 256, 256, 3); manifest unchanged |
| Clean features | `03_extract_features.py` | (813, 105), 0 non-finite |
| Baseline | `04_baseline.py` | identical to committed Phase 1 files |
| Perturbed features | `12_extract_perturbed.py` | 50 × (813, 105) |
| **Fix** | `src/segment.py`: `cv2.setRNGSeed(0)` before `cv2.grabCut` | GrabCut masks were non-deterministic (26 / 40 images differed between runs). After the fix: 0 / 813 differ across orderings, threads and processes |
| Regenerate | `ensure_ref_masks(force=True)`; `12 --kinds background --force` | only the mask file and the 4 background matrices changed |
| Protocol | `13_protocol.py` | identical to Phase 1 except background rows (±0.007 mean) and 3 ECE ties at a bin edge |
| Remaining stages | `05`, `06`, `07`, `08`, `09`, `14`, `15` | all exit 0; 05, 14, 15 identical to Phase 1; 07–09 are new |

## 2026-10-26: experiment A, train-only augmentation (`runs/p2_augmentation`)
| Step | Command | Outcome |
|---|---|---|
| Supplied augmented set | linkage analysis (filenames + image content), run on `~/Downloads/Tomato Dataset part 2/` | rejected for the main protocol (see verified_results §1). No output was saved, so its figures are unverified |
| Base inputs | `01 --cache-only`, `03`, `12` | 53 / 53 files byte-identical to `p2_base` |
| Stores | `TOMATO_CONFIG=configs/aug_<arm>.yaml 16_extract_augmented.py` | none (0), geo (3,252), geo_photo (3,252); re-extraction byte-identical |
| A0 control | `TOMATO_CONFIG=configs/aug_none.yaml 13_protocol.py --augment` | attempt 1 failed (pandas 3 dtype of an empty index, fixed); attempt 2 meta-set probabilities off by ≤ 2e-6 (array memory order, fixed); attempt 3 predictions byte-identical to `p2_base` |
| A1 smoke | `… --augment --n 1 --sets all --models rf --tag smoke_aug_geo` | 2,280 copies added; leakage 0 |
| A1, A2 full | `TOMATO_CONFIG=configs/aug_geo.yaml …`, `…aug_geo_photo.yaml 13_protocol.py --augment` | 1,667 s and 1,615 s; leakage check 0 on every split |
| Summary | `17_summarise_run.py --base base=runs/p2_base:main --arm A0=… --arm A1=… --arm A2=… --out summary` | 23 / 23 integrity checks pass |
| Photometric part | `17_summarise_run.py --base "A1=${R}:aug_geo" --arm "A2=${R}:aug_geo_photo" --out summary_A2_vs_A1` | first attempt mangled by zsh (`$R:a` is a path modifier); quote as `"${R}"` |

## Changes from the original plan
| Plan | What we did | Why |
|---|---|---|
| 7,200 images, provided split | 830 raw photos, our own grouped split | the 7,200 are augmented copies whose split leaks sources |
| No class weights | `class_weight: balanced` | 2.4× imbalance in the raw set |
| Single test split | 10 grouped splits, tuning inside each, paired tests with Holm | single-split differences were within noise |
| Use the supplied augmented set | own train-only augmentation; the supplied set benchmarked separately on its own split (2026-10-27, reference only) | the supplied set cannot be reliably linked to raw sources, and its split places related images on both sides (133 source groups span splits; 361 / 720 test images have a detected sibling). Earlier linkage percentages are unverified (verified_results §1) |
| Phase 2: masks, illumination, features, models, CNN reference, augmentation | augmentation and illumination (Gray World, CLAHE) done; the rest deferred | time to the mid-semester deadline; augmentation and illumination test RQ2 most directly |

## Open items
- Blind visual review of the label suspects (21 septoria/early-blight filenames, 3 others, 7 strong audit flags). Decisions go in `data/label_exclusions.csv` with a reason.
- Sensitivity runs: capture-session grouping (60 s / 5 min), and rotation-aware grouping of the 4 pairs.
- Decide on the diagnostic Gray World variant (mask from the un-normalised image, colours normalised for features only). Not implemented.
- Combined augmentation + normalisation: not started. Experiment B gives no normalisation that helps on its own, so a combination is not currently motivated.
- Leave-one-cluster-out with a matched random-holdout control; literature survey.
- Supplied set, planned for the final-paper phase (not started):
  - grouped evaluation of the supplied set. Grouping rule, handling of the 61 cross-class pairs and of images with
    untraceable sources to be fixed in writing before any run; it needs a new split, which needs team approval;
  - if linkage percentages are to be cited, recompute them with a saved script on the `Augmented Dataset` copy
    (verified_results §1);
  - LR's best C sits at the grid edge (optional).
- Saved scripts for the capture-session and rotated-copy analyses, whose outputs were not saved (verified_results §2).

## 2026-10-26: experiment B, illumination normalisation (`runs/p2_illumination/{grayworld,clahe}`)
| Step | Command | Outcome |
|---|---|---|
| Code | `src/illumination.py`; `normalize()` in the feature and augmentation workers; `configs/illum_{grayworld,clahe}.yaml` | no-op with the base config: re-extracted clean, perturbed and augmented features byte-identical |
| Smoke | `01 --cache-only`, `03`, `12 --kinds brightness contrast shadow`, `13 --n 1 --sets all --models rf --tag smoke_illum_<m>` | exit 0. Gray World shrinks mean leaf-mask area 0.83 → 0.39 (recoloured pixels leave the hue gate); CLAHE leaves masks unchanged |
| Full runs | `TOMATO_CONFIG=configs/illum_<m>.yaml`: `12_extract_perturbed.py` (remaining 39 conditions), then `13_protocol.py --tag illum_<m>` | 07:10 → 07:42; protocol 527 s (GW) and 534 s (CL); exit 0; 50/50 conditions; no failures |
| Checks | stores re-validated; 40 rows re-extracted (clean + contrast 2.0) | 50/50 keys valid, 0 non-finite, byte-identical |
| Summary | `17_summarise_run.py --base "base=${R}/p2_base:main" --arm "GW=${R}/p2_illumination/grayworld:illum_grayworld" --arm "CL=${R}/p2_illumination/clahe:illum_clahe" --out "${R}/p2_illumination/summary"` | 12/12 integrity checks pass. Results in verified_results §8 |
| Mask diagnostic | leaf-area feature by class and condition | GW mean mask 0.83 → 0.39, class-dependent (leaf mold 0.51, others 0.32–0.38); masks under 20% of the frame 0.7% → 13.4% |

## 2026-10-27: supplied-split benchmark (`runs/p2_supplied`)
Existing pipeline (105 features, RF/SVM/LR grids, seed 42) on the supplied `Augmented Dataset` with its own
train/val/test folders. It is a reference benchmark on the supplied split, **not** a generalisation estimate.
Results: verified_results §10. Per-step notes and hashes: `runs/p2_supplied/NOTES.md`.

**Setup**
- Environment: `source runs/p2_supplied/env.sh`. It sets `TOMATO_DATA_ROOT` (the Augmented Dataset),
  `TOMATO_WORK_DIR`, `TOMATO_RESULTS_DIR`, `TOMATO_MANIFEST=runs/p2_supplied/manifest_supplied.csv`,
  `TOMATO_CONFIG=runs/p2_supplied/supplied_official.yaml` and `PYTHONDONTWRITEBYTECODE=1`.
- Overlay settings:
  - `split_strategy: official`;
  - `drop_cross_class: false` (keep all 7,200);
  - `reference_mask: null`;
  - label exclusions pointed at a file that does not exist.
- New code: `scripts/18_export_predictions.py` (untracked). No existing file was modified.

| Step | Command | Outcome |
|---|---|---|
| Snapshot | sha256 of protected files, 687 output files and 8,039 dataset files; `git status` | `runs/p2_supplied/logs/*_before*` |
| Audit | `00_audit_dataset.py --hash` | 7 s; 1,516 source groups, 133 spanning splits; 33.1% of test has its source in train/val (built-in rule) |
| Manifest + cache | `01_prepare_data.py` | 10 s; 7,200 rows, split equals folder for every row; 2,693 near-duplicate pairs, 61 cross-class (kept); 1.42 GB cache |
| EDA | `02_eda_duplicates.py --skip-dups` | 17 s; skips only the duplicate-flag step (no built-in "no-leak" score in stage 4); nothing is removed in stage 02 |
| Features | `03_extract_features.py` (after `--benchmark 100`, 35 ms/img) | 100 s; (7,200, 105), 0 non-finite |
| Sibling flags | `18_export_predictions.py --flags-only` | 10 s; rules frozen before stage 4 existed; 361 / 720 test images with a detected sibling |
| Baseline | `04_baseline.py` | 108 s; RF 0.812, SVM 0.815, LR 0.696 test macro-F1 |
| Export | `18_export_predictions.py` | 41 s; 23 / 23 checks pass; predictions with class probabilities, confusion counts and the sibling breakdown |
| Protection check | the snapshot commands again, diffed | all identical; HEAD unchanged; only new file `scripts/18_export_predictions.py` |

Warnings:
- the expected pipeline warning that 133 source photos span splits;
- LR's best C (10) is at the grid edge;
- RF and SVM are over-confident (ECE 0.133 and 0.206).

## 2026-11-12: new machine, supplied-split reproduction, Task 2 grouped evaluation (`F:/NUST/ML/runs/`)
**Environment.** Windows 10, 4 cores, 8.5 GB RAM. `.venv` in the repo: Python 3.11.9 (3.11.17 not available as a
Windows build), numpy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, scikit-image 0.26.0, OpenCV 5.0.0.93 (pinned to the
Phase 2 versions). Data: `F:/NUST/ML/Tomato Dataset/{Dataset (raw), Augmented Dataset}` (830 / 7,200 files).

**Reproduction of the supplied-split benchmark** (`runs/p2_supplied_repro`, `configs/supplied/official.yaml`,
stages 01, 03, 04). Grouping identical (2,693 pairs, 61 cross-class, 1,516 groups, 133 spanning splits). Test
macro-F1 RF 0.814 / SVM 0.815 / LR 0.698 (2026-10-27: 0.812 / 0.815 / 0.696); RF chose max_depth None instead of 20.
Close but **not byte-identical**, so feature values differ slightly on this machine (cause not isolated: CPU
instruction set or Python patch level). Consequence: Task 2 compares runs made on this machine only.

**Task 2 design (fixed before any model was run; `scripts/19_group_supplied.py`, `configs/supplied/grouped.yaml`).**
- Grouping = union of (i) the stage-01 pipeline grouping, (ii) the frozen name rule of stage 18 (same class, same
  source name), (iii) the frozen content rule of stage 18 (16×16 grey thumbnail, 8 rotations/flips, r ≥ 0.95),
  same class only.
- Cross-class: every image whose group contains an image in a cross-class pipeline or content pair is dropped (raw-set
  policy). Same name under two classes is reported only (8 names), since short names can collide.
- Untraceable sources: no special handling; singletons unless a rule links them.
- Protocol: main protocol unchanged (10 grouped 70/15/15 splits, seeds 42–51, tuning inside each split), clean only,
  feature sets `all` and `meta`. Control: the identical image set split at image level (`manifest_random.csv`,
  group = path), same seeds, same features.
- Grouping result (before any model): 7,037 images after the pipeline drop (163) → 7,017 after the content drop (20);
  936 groups (86 singletons, largest 49 = 0.7%); groups per class 86–216, close to the raw photo counts (80–195).
  Test images with a detected relative in train/val: grouped 0% (by construction), random 98.7%, supplied split 50%.

**Task 2 runs.** `01 --cache-only`, `03` (6 min), `13_protocol.py --no-perturbed --sets all meta --tag grouped`
(62 min) and the same with `TOMATO_MANIFEST=…/manifest_random.csv --tag random` (49 min). Exit 0. Results:
verified_results §11.

## 2026-11-12: cross-dataset evaluation (`F:/NUST/ML/runs/p3_cross_dataset`)
| Step | Command | Outcome |
|---|---|---|
| Downloads | PlantDoc tomato folders via GitHub (file by file; `?` in names replaced, originals in `file_map.csv`); Nazirpur, Brahmanbaria, Dinajpur via browser from Mendeley (script downloads were refused by Mendeley and not pursued) | `F:/NUST/ML/external/` |
| Label check | contact sheets of 12 random YOLO box crops per class id | Brahmanbaria 0 = healthy, 1 = diseased. Dinajpur: boxes are lesions (only id 2 = whole healthy leaves), box counts do not match the paper: **excluded** |
| Leaf crops | `21_crop_yolo_leaves.py` (Brahmanbaria, 10% margin, PNG) | 1,035 crops from 1,028 labelled photos (2 photos without labels) |
| Raw features | `01 --cache-only`, `03` on `Dataset (raw)`, committed manifest (sha256 `b819d160…`) | (813, 105) |
| Cross-dataset | `20_cross_dataset.py --ext configs/external/{plantdoc,nazirpur,nazirpur_leafcurl,brahmanbaria}.yaml` | 59 min, exit 0 |
| In-domain check | `13_protocol.py --no-perturbed --sets all --tag check_all` | script 20 = protocol on this machine (difference 0); vs committed §3: SVM/LR match, RF differs (mean 0.613 vs 0.600) |
| Diagnostics | `22_domain_shift.py` | domain AUC, mask area, own-split importance agreement |

Results: verified_results §12.

## 2026-11-12: Phase 3 setup (multi-source; design in [phase3_design_2026-11-12.md](phase3_design_2026-11-12.md))
- **Snapshot** of all docs, the midterm draft, the status PDF and every result/log of runs p2_supplied_repro,
  p2_supplied_grouped and p3_cross_dataset: `docs/snapshots/2026-11-12/`.
- **New data** (`F:/NUST/ML/external/`, logs in `external/logs/`):
  - PlantWild v1 tomato folders (1,966 images): only the `images/tomato*` members were read from the 2.7 GB zip on
    Hugging Face (`uqtwei2/PlantWild`, CC BY-NC-ND 4.0) through HTTP range requests;
  - Tomato-Village variant a (GitHub `mamta-joshi-gehlot/Tomato-Village`): all early blight (496) and healthy (216),
    seeded samples (seed 42) of late blight 400, leaf miner 200, spotted wilt virus 200; nutrient deficiencies not used;
  - PlantVillage (GitHub `spMohanty/PlantVillage-Dataset`, raw/color): seeded samples of 150 per class for our six
    classes and 50 each of bacterial spot, target spot, mosaic virus, spider mites (contents API lists at most 1,000
    files per folder; samples are drawn from that listing);
  - per-file maps (`file_map_*.csv`) record original paths.
- **Environments**: `.venv` unchanged; `.venv-deep` (Python 3.11.9, torch 2.5.1+cpu, torchvision 0.20.1,
  open_clip_torch 2.29.0, numpy 2.1.3) for model-based features; model caches in `F:/NUST/ML/model_cache`.
- **Literature review** written from checked sources: [literature_review.md](literature_review.md); also §2 of the
  midterm draft.

## 2026-11-12: Phase 3 runs (run dir `F:/NUST/ML/runs/p4_multisource`, logs in `logs/`)
- **Environment change**: `.venv-deep` was abandoned (partially deleted, files locked) and removed; the frozen
  extractors run in `.venv-dl` (same versions; pinned in `requirements-dl.txt`), caches in
  `F:/NUST/ML/model_cache_dl/`. `.gitignore` now lists `.venv-dl/` and `.venv-deep/`.
- **Stage 24** (`24_build_sources.py`): 8,196 images scanned, 2,335 duplicate pairs, 425 cross-source duplicates
  dropped (PlantDoc 277, PlantWild 146, PlantVillage 2), 72 label conflicts recorded (`sources/label_conflicts.csv`),
  7,754 images in 6,967 within-source groups (largest 12). Raw manifest: 813 images, of which 296 have a PlantDoc
  copy under the content rule (pd_raw) and 517 do not (pk_raw). Note: verified_results §12 reports 263 of 830
  (31.7%) against the six-class PlantDoc images; the two figures use different reference sets and denominators and
  the difference has not been reconciled image by image.
- **Stage 25** (`25_extract_multisource.py`): 256 px cache 491 s; hand (105) + lesion (20) features and k-means
  descriptors 805 s.
- **Stage 25b** (`25b_deep_features.py`, `.venv-dl`, CPU, frozen): ResNet-50 (2,048), CLIP ViT-B/32 embedding
  (512) and 24 concept scores for all 7,754 images; 77.5 min including weight download.
- **Stage 26** (`26_loso.py`): classical arms (`--tag _classical`: hand, lesion, hand+lesion, colour_words,
  textons, kmeans, hand+lesion+kmeans) and deep arm (`--tag _deep`: concepts, clip, resnet) run in parallel.
- **Stage 29** (`29_figures_phase3.py`) written for the Phase 3 figures; four-arm palette
  (#2a78d6/#eb6834/#1baf7a/#eda100) passes the CVD validator; bars carry value labels because two hues are below
  3:1 contrast.

## 2026-10-27: documentation audit (no runs)
Read-only check of every number in the midterm draft against the saved outputs, followed by corrections to
verified_results, this log and the midterm draft. No code, configuration, data or run output was changed.
- Corrected in verified_results:
  - A2-vs-A1 ranges (§7): brightness up to +0.094, not +0.11; contrast not significant for LR;
  - the source of the blur gains is now unresolved (§7);
  - Gray World clean drop is 0.05–0.06 with all features (§8);
  - the §9 cross-reference now reads §§3–8.
- Added to verified_results: the learning-curve values and pooled confusions (§3).
- Marked unverified (no saved output): the supplied-set linkage percentages (§1); the capture-session and rotated-copy
  counts (§2); "a third untraceable" and the 2 byte-identical cross-class files (§10).
- Resolved by reading code: the test brightness/contrast perturbations and the A2 augmentation use the same formulas.
