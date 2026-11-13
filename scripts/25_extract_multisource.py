"""Stage 25: image cache, formula features (hand 105 + lesion 20) and codebook descriptors for the Phase 3 manifest.

    python scripts/25_extract_multisource.py --run F:/NUST/ML/runs/p4_multisource

Uses the main config (256 px, HSV mask, same feature settings) with data_root = sources.yaml data_root.
Outputs (<run>/work/): images_cache.npy, features_hand.npy (+ names), features_lesion.npy (+ names),
desc_colour.npy / desc_texture.npy (seeded leaf-pixel samples for the k-means codebooks, src/codebook.py)
"""
import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import build_image_cache, cache_file  # noqa: E402
from src.features import extract_named, worker_cfg  # noqa: E402
from src.features_lesion import lesion_features  # noqa: E402
from src.codebook import descriptors  # noqa: E402
from src.segment import segment_leaf  # noqa: E402
from src.utils import save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _work(cache_path, idx, sub):
    arr = np.load(cache_path, mmap_mode="r")
    H, L, DC, DT = [], [], [], []
    for i in idx:
        img = np.ascontiguousarray(arr[i])
        mask = segment_leaf(img, sub["segmentation"])
        H.append(np.fromiter(extract_named(img, sub).values(), np.float32))
        L.append(np.fromiter(lesion_features(img, mask).values(), np.float32))
        c, t = descriptors(img, mask, int(i))
        DC.append(c)
        DT.append(t)
    return np.stack(H), np.stack(L), np.stack(DC), np.stack(DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    work = run / "work"
    work.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(open(ROOT / "configs" / "sources.yaml"))
    cfg = load_config()
    c = copy.deepcopy(cfg)
    c["paths"]["data_root"] = Path(spec["data_root"])
    c["paths"]["work_dir"] = work
    df = pd.read_csv(run / "sources" / "manifest_sources.csv")
    t0 = time.time()
    build_image_cache(c, df)
    print(f"cache: {time.time() - t0:.0f}s")
    sub = worker_cfg(cfg)
    from src.features import feature_schema
    from src.features_lesion import lesion_schema
    hn, ln = feature_schema(sub), lesion_schema()
    idx = np.arange(len(df))
    chunks = [x for x in np.array_split(idx, 64) if len(x)]
    t0 = time.time()
    parts = Parallel(n_jobs=-1)(delayed(_work)(str(cache_file(c)), ch, sub) for ch in chunks)
    Xh = np.nan_to_num(np.concatenate([p[0] for p in parts]), nan=0.0, posinf=0.0, neginf=0.0)
    Xl = np.nan_to_num(np.concatenate([p[1] for p in parts]), nan=0.0, posinf=0.0, neginf=0.0)
    np.save(work / "features_hand.npy", Xh)
    np.save(work / "features_lesion.npy", Xl)
    np.save(work / "desc_colour.npy", np.concatenate([p[2] for p in parts]))
    np.save(work / "desc_texture.npy", np.nan_to_num(np.concatenate([p[3] for p in parts])))
    save_json(hn, work / "features_hand.names.json")
    save_json(ln, work / "features_lesion.names.json")
    save_json({"manifest_paths_md5": __import__("hashlib").md5("|".join(df["path"]).encode()).hexdigest(),
               "n": len(df)}, work / "features_formula.meta.json")
    print(f"features: hand {Xh.shape}, lesion {Xl.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
