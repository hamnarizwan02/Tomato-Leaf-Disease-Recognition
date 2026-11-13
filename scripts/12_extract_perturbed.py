"""Stage 12 (Member A): perturbed features for ALL images, core + extended perturbations + combos (section 15, stage 4).

    python scripts/12_extract_perturbed.py --benchmark 20     # time one image batch per condition, save nothing
    python scripts/12_extract_perturbed.py                    # resumable: skips conditions already extracted
    python scripts/12_extract_perturbed.py --kinds downup jpeg    # only some kinds
    python scripts/12_extract_perturbed.py --force

Every image is perturbed once per condition, so any protocol split can be scored without re-extraction. Background
replacement uses the cached reference mask (perturb_settings.reference_mask), not the feature mask.
Output: work/features/perturbed/all/{condition}.npy (manifest order) + index.json; results/perturbation_conditions.csv
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import cache_file, ensure_ref_masks, load_cache, load_manifest  # noqa: E402
from src.features import (extract_matrix, extraction_key, perturbed_index, save_perturbed_all,  # noqa: E402
                          worker_cfg)
from src.perturb import all_conditions, condition_name  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kinds", nargs="*", help="only these perturbation kinds (default: all)")
    ap.add_argument("--benchmark", type=int, default=0, help="time N images per condition and exit")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    df = load_manifest(cfg)
    load_cache(cfg, df)
    ensure_ref_masks(cfg, df)
    sub = worker_cfg(cfg)
    conds = [(k, v) for k, v in all_conditions(cfg, extended=True) if not args.kinds or k in args.kinds]
    pd.DataFrame([{"condition": condition_name(k, v), "kind": k, "level": v} for k, v in conds]).to_csv(
        cfg["paths"]["results_dir"] / "perturbation_conditions.csv", index=False)
    rows = np.arange(len(df))

    if args.benchmark:
        pick = np.random.default_rng(cfg["seed"]).choice(rows, min(args.benchmark, len(rows)), replace=False)
        total = 0.0
        for kind, level in conds:
            t0 = time.time()
            extract_matrix(cache_file(cfg), pick, sub, cfg["n_jobs"], cond=(kind, level))
            dt = time.time() - t0
            total += dt
            print(f"{condition_name(kind, level):40s} {1000 * dt / len(pick):6.0f} ms/img")
        print(f"Estimated full run: {total / len(pick) * len(df) / 60:.1f} min for {len(conds)} conditions")
        return

    done = perturbed_index(cfg)
    for i, (kind, level) in enumerate(conds, 1):
        name = condition_name(kind, level)
        if done.get(name) == extraction_key(cfg, df, perturbed=True, condition=name) and not args.force:
            print(f"[{i}/{len(conds)}] {name}: up to date")
            continue
        t0 = time.time()
        X = extract_matrix(cache_file(cfg), rows, sub, cfg["n_jobs"], cond=(kind, level))
        save_perturbed_all(cfg, df, name, X)
        print(f"[{i}/{len(conds)}] {name}: {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
