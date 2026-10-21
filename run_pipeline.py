"""Local runner for the tomato-robust-ml pipeline (converted from 00_colab_runner_fixed.ipynb).

Thin launcher only: all logic lives in src/ and scripts/. Run from the repo root (the folder with config.yaml).

    python run_pipeline.py setup --zip ~/Downloads/Tomato_Leaf_Disease_Dataset_Pakistan.zip
    python run_pipeline.py stage 00 01 02        # run selected stages
    python run_pipeline.py all                   # stages 00-11 in order
    python run_pipeline.py phase1                # Phase 1: stages 12-15 (perturbed features, protocol, shortcut, labels)
    python run_pipeline.py variants              # shortcut controls: resolution-matched, re-encoded, min side 500
    python run_pipeline.py seg-compare           # HSV vs GrabCut vs no mask
    python run_pipeline.py archive               # v2 re-run: archive old results, clear derived files
    python run_pipeline.py export                # zip results/ (+ results_* variants) and the manifest
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO, "data", "raw")          # dataset zip is unpacked here
WORK_DIR = os.path.join(REPO, "work")                 # big derived files (features, image cache)
RESULTS_DIR = os.path.join(REPO, "results")
ENV_FILE = os.path.join(REPO, ".pipeline_env.json")   # remembers data_root between runs
DEFAULT_VARIANT = "Dataset (raw)"                     # folder inside the zip used as data_root

# stage id -> (script, extra args). 03 runs a 100-image benchmark first, as in the notebook.
STAGES = {
    "00": [("00_audit_dataset.py", ["--hash"])],
    "01": [("01_prepare_data.py", [])],
    "02": [("02_eda_duplicates.py", [])],
    "03": [("03_extract_features.py", ["--benchmark", "100"]), ("03_extract_features.py", [])],
    "04": [("04_baseline.py", [])],
    "05": [("05_ablation.py", [])],
    "06": [("06_robustness.py", [])],
    "07": [("07_importance_stability.py", [])],
    "08": [("08_error_analysis.py", [])],
    "09": [("09_figures.py", [])],
    "10": [("10_source_shortcut_check.py", [])],
    "11": [("11_repeated_splits.py", ["--seeds", "5"])],
    # Phase 1 (section 15 of the findings log)
    "12": [("12_extract_perturbed.py", ["--benchmark", "20"]), ("12_extract_perturbed.py", [])],
    "13": [("13_protocol.py", [])],
    "14": [("14_shortcut_controls.py", [])],
    "15": [("15_label_audit.py", [])],
}
CORE_STAGES = [f"{i:02d}" for i in range(12)]
PHASE1_STAGES = ["12", "13", "14", "15"]

SEG_VARIANTS = {"none": "configs/seg_none.yaml", "grabcut": "configs/seg_grabcut.yaml"}
# source-shortcut controls (a) V1 and (d): each rebuilds its own image cache from the raw photos
SHORTCUT_VARIANTS = {"res_matched": "configs/res_matched.yaml", "reencode": "configs/reencode.yaml"}


def run(cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def pipeline_env():
    if not os.path.isfile(ENV_FILE):
        sys.exit("run `python run_pipeline.py setup --zip <dataset zip>` first")
    with open(ENV_FILE) as f:
        saved = json.load(f)
    return dict(os.environ,
                TOMATO_DATA_ROOT=saved["data_root"],
                TOMATO_WORK_DIR=WORK_DIR,
                TOMATO_RESULTS_DIR=RESULTS_DIR)


def cmd_setup(args):
    for need in ("config.yaml", "scripts", "requirements.txt"):
        if not os.path.exists(os.path.join(REPO, need)):
            sys.exit(f"{need} not found in {REPO}: put the tomato-robust-ml code here first")
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

    done = os.path.join(DATA_DIR, ".done")
    if not os.path.exists(done):
        zip_path = os.path.expanduser(args.zip or "")
        if not os.path.isfile(zip_path):
            sys.exit("dataset not unpacked yet: pass --zip <path to Tomato_Leaf_Disease_Dataset_Pakistan.zip>")
        os.makedirs(DATA_DIR, exist_ok=True)
        print("unzipping", zip_path, "->", DATA_DIR)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        open(done, "w").close()

    hits = [d for d, _, _ in os.walk(DATA_DIR) if os.path.basename(d) == args.variant]
    if not hits:
        sys.exit(f"no folder named {args.variant!r} under {DATA_DIR}")
    with open(ENV_FILE, "w") as f:
        json.dump({"data_root": hits[0]}, f, indent=2)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("data_root:", hits[0], "| work:", WORK_DIR, "| results:", RESULTS_DIR, "| cpus:", os.cpu_count())


def cmd_stage(args):
    env = pipeline_env()
    for sid in args.ids:
        sid = sid.zfill(2)
        if sid not in STAGES:
            sys.exit(f"unknown stage {sid!r}; choose from {', '.join(STAGES)}")
        for script, extra in STAGES[sid]:
            print(f"===== stage {sid}: {script}", flush=True)
            run([sys.executable, f"scripts/{script}", *extra], env=env)


def cmd_all(args):
    args.ids = CORE_STAGES
    cmd_stage(args)


def cmd_phase1(args):
    args.ids = PHASE1_STAGES
    cmd_stage(args)


def cmd_variants(args):
    """Same manifest and split; only the image preprocessing changes. Clean protocol only (no perturbations)."""
    base = pipeline_env()
    for name, cfgfile in SHORTCUT_VARIANTS.items():
        env = dict(base, TOMATO_CONFIG=cfgfile, TOMATO_WORK_DIR=os.path.join(REPO, f"work_{name}"),
                   TOMATO_RESULTS_DIR=os.path.join(REPO, f"results_{name}"))
        for script, extra in [("01_prepare_data.py", ["--cache-only"]), ("03_extract_features.py", []),
                              ("13_protocol.py", ["--no-perturbed", "--tag", name])]:
            print(f"===== {name}: {script}", flush=True)
            run([sys.executable, f"scripts/{script}", *extra], env=env)
    print("===== min side 500: 13_protocol.py", flush=True)
    run([sys.executable, "scripts/13_protocol.py", "--no-perturbed", "--min-side", "500"], env=base)


def cmd_seg_compare(args):
    """Same manifest/split and cached images; only the segmentation setting changes."""
    base = pipeline_env()
    for name, cfgfile in SEG_VARIANTS.items():
        wd = os.path.join(REPO, f"work_{name}")
        rd = os.path.join(REPO, f"results_seg_{name}")
        os.makedirs(wd, exist_ok=True)
        os.makedirs(rd, exist_ok=True)
        for f in glob.glob(os.path.join(WORK_DIR, "images_cache*")):
            shutil.copy(f, wd)
        env = dict(base, TOMATO_CONFIG=cfgfile, TOMATO_WORK_DIR=wd, TOMATO_RESULTS_DIR=rd)
        for s in ["03_extract_features.py", "04_baseline.py", "06_robustness.py"]:
            print(f"===== {name}: {s}", flush=True)
            run([sys.executable, f"scripts/{s}"], env=env)


def cmd_archive(args):
    """v2 re-run only: keep the first run's results, then clear work/ and results/ so nothing stale is reused."""
    arch = os.path.join(REPO, "archive_v1")
    os.makedirs(arch, exist_ok=True)
    if os.path.isdir(RESULTS_DIR) and os.listdir(RESULTS_DIR):
        shutil.copytree(RESULTS_DIR, os.path.join(arch, "results_v1"), dirs_exist_ok=True)
    if os.path.isdir(WORK_DIR):
        shutil.copytree(WORK_DIR, os.path.join(arch, "work_v1"), dirs_exist_ok=True)
        shutil.rmtree(WORK_DIR)
    for f in glob.glob(os.path.join(RESULTS_DIR, "*")):
        os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)
    print("archived v1 ->", arch, "| work/ and results/ cleared")


def cmd_export(args):
    stage = os.path.join(REPO, "results_export")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    variants = [d for d in sorted(glob.glob(os.path.join(REPO, "results_*"))) if d != stage]
    for d in [RESULTS_DIR] + variants:
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(stage, os.path.basename(d)))
    manifest = os.path.join(REPO, "data", "manifest.csv")
    if os.path.exists(manifest):
        shutil.copy(manifest, stage)
    out = shutil.make_archive(os.path.join(REPO, "results"), "zip", stage)
    print("exported:", stage, "| zip:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="install requirements and unpack the dataset")
    s.add_argument("--zip", help="path to Tomato_Leaf_Disease_Dataset_Pakistan.zip")
    s.add_argument("--variant", default=DEFAULT_VARIANT, help="folder inside the zip used as data_root")
    s.add_argument("--skip-install", action="store_true", help="don't pip install requirements.txt")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("stage", help="run one or more stages, e.g. `stage 03 04`")
    s.add_argument("ids", nargs="+")
    s.set_defaults(func=cmd_stage)

    sub.add_parser("all", help="run stages 00-11 in order").set_defaults(func=cmd_all)
    sub.add_parser("phase1", help="run Phase 1 stages 12-15").set_defaults(func=cmd_phase1)
    sub.add_parser("variants", help="shortcut controls (resolution-matched, re-encoded, min side 500)"
                   ).set_defaults(func=cmd_variants)
    sub.add_parser("seg-compare", help="HSV vs GrabCut vs no mask").set_defaults(func=cmd_seg_compare)
    sub.add_parser("archive", help="archive the first run and clear derived files").set_defaults(func=cmd_archive)
    sub.add_parser("export", help="collect results into results.zip").set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#pushing first commit    
