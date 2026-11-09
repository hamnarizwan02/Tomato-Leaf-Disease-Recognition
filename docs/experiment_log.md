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
