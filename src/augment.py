"""Train-only augmentation (Phase 2, experiment A).

augment(img, row, copy, aug, seed) -> uint8 RGB image, deterministic given (seed, manifest row, copy number).
Settings come from config["augmentation"] (set in configs/aug_*.yaml, absent from config.yaml):
    name      store name (work/features/augmented/<name>.npy)
    copies    augmented copies per image (0 = control: the store is empty)
    hflip_p, vflip_p   probability of a horizontal / vertical flip
    rotate_deg         rotation drawn uniformly from [-rotate_deg, rotate_deg], reflected border
    brightness, contrast   [lo, hi] multiplicative factor, or null to switch off
Every random value is drawn in the same order whatever is switched on, so copy c of an image has the same flips and
angle in the geometric and the geometric + photometric arm (the second differs only by the photometric change).

Copies are made for EVERY manifest row once and stored with their source row; scripts/13_protocol.py --augment adds to
each split only the copies whose source is in that split's train partition, so nothing derived from a validation or
test image is ever trained on.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import extract_named, extraction_key, feature_dir, feature_schema
from .illumination import normalize
from .perturb import _affine
from .utils import load_json, save_json

AUG_TAG = 0xA06                     # separates augmentation draws from the perturbation draws for the same image


def augment(img: np.ndarray, row: int, copy: int, aug: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng([int(seed), AUG_TAG, int(row), int(copy)])
    hflip, vflip = rng.random() < aug.get("hflip_p", 0.0), rng.random() < aug.get("vflip_p", 0.0)
    angle = rng.uniform(-1.0, 1.0) * float(aug.get("rotate_deg") or 0.0)
    b, c = rng.random(), rng.random()
    x = img
    if hflip:
        x = x[:, ::-1]
    if vflip:
        x = x[::-1]
    x = np.ascontiguousarray(x).astype(np.float32)
    if angle:
        x = _affine(x, angle, 1.0)
    if aug.get("brightness"):
        lo, hi = aug["brightness"]
        x = x * (lo + b * (hi - lo))
    if aug.get("contrast"):
        lo, hi = aug["contrast"]
        m = x.mean()
        x = (x - m) * (lo + c * (hi - lo)) + m
    return np.clip(x, 0, 255).astype(np.uint8)


def _worker(cache_path: str, pairs, sub: dict, aug: dict):
    arr = np.load(cache_path, mmap_mode="r")
    # same normalisation as the original images (no-op unless preprocess.illumination is set)
    return np.stack([np.fromiter(extract_named(normalize(augment(np.ascontiguousarray(arr[r]), r, c, aug, sub["seed"]),
                                                         sub.get("illumination")), sub).values(), dtype=np.float32)
                     for r, c in pairs])


def extract_augmented(cache_path, rows, aug: dict, sub: dict, n_jobs: int = -1):
    """(X, index) for `copies` augmented versions of each cache row; index has columns source_row, copy."""
    index = pd.DataFrame([(int(r), c) for r in rows for c in range(int(aug["copies"]))], columns=["source_row", "copy"])
    if index.empty:
        return np.zeros((0, len(feature_schema(sub))), np.float32), index
    n_workers = (os.cpu_count() or 1) if n_jobs in (-1, None) else n_jobs
    pairs = index.values
    chunks = [c for c in np.array_split(pairs, max(1, min(len(pairs), n_workers * 8))) if len(c)]
    X = np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_worker)(str(cache_path), ch, sub, aug) for ch in chunks))
    bad = ~np.isfinite(X)
    if bad.any():
        print(f"  note: replaced {int(bad.sum())} non-finite feature values with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, index


# --------------------------------------------------------------------------- store
def aug_key(cfg, df) -> str:
    """Clean extraction key (manifest, preprocessing, segmentation, features) + seed + augmentation settings."""
    blob = {"clean": extraction_key(cfg, df), "seed": cfg["seed"], "augmentation": cfg["augmentation"]}
    return hashlib.md5(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def aug_files(cfg):
    d = feature_dir(cfg) / "augmented"
    d.mkdir(parents=True, exist_ok=True)
    name = cfg["augmentation"]["name"]
    return d / f"{name}.npy", d / f"{name}_index.csv", d / f"{name}.meta.json"


def save_augmented(cfg, df, X: np.ndarray, index: pd.DataFrame) -> None:
    fx, fi, fm = aug_files(cfg)
    np.save(fx, X)
    index.to_csv(fi, index=False)
    save_json({"key": aug_key(cfg, df), "n": len(X), "augmentation": cfg["augmentation"], "seed": cfg["seed"]}, fm)


def load_augmented(cfg, df) -> dict:
    """{"X": (n_aug, n_features), "src": manifest row of each copy, "copy": copy number}; refused when stale."""
    fx, fi, fm = aug_files(cfg)
    if not (fx.exists() and fi.exists() and fm.exists()):
        raise FileNotFoundError(f"{fx} missing: run scripts/16_extract_augmented.py with this config first")
    if load_json(fm)["key"] != aug_key(cfg, df):
        raise RuntimeError(f"{fx.name} was built from another manifest/config/augmentation: re-run scripts/16 --force")
    index = pd.read_csv(fi)
    X = np.load(fx)
    if len(X) != len(index):
        raise RuntimeError(f"{fx.name} has {len(X)} rows, its index {len(index)}")
    # explicit int: an empty index (copies: 0) is read back without a dtype
    return {"X": X, "src": index["source_row"].to_numpy(dtype=np.int64), "copy": index["copy"].to_numpy(dtype=np.int64)}
